#!/usr/bin/env python3

import argparse
import asyncio
import contextlib
import logging
import signal
from concurrent.futures import ThreadPoolExecutor
from functools import partial

from wyoming.server import AsyncServer

from const import BASE_DIR, AppState
from gemma4_recognizer import Gemma4Recognizer
from hass_api import HomeAssistant
from intent_server import Gemma4EventHandler
from web_server import make_web_server, run_web_server

_LOGGER = logging.getLogger(__name__)


# -----------------------------------------------------------------------------


async def main() -> None:
    """Run app."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--uri", required=True, help="unix:// or tcp://")
    #
    parser.add_argument("--http-host", default="127.0.0.1")
    parser.add_argument("--http-port", type=int, default=5000)
    #
    parser.add_argument("--hass-token", required=True)
    parser.add_argument("--hass-api", default="http://homeassistant.local:8123")
    #
    parser.add_argument(
        "--hf-repo",
        default="bartowski/google_gemma-4-E2B-it-GGUF",
        help="Hugging Face repo for Gemma 4 (official: ggml-org/gemma-4-E2B-it-GGUF)",
    )
    parser.add_argument(
        "--hf-filename",
        default="google_gemma-4-E2B-it-Q5_K_M.gguf",
        help="Gemma 4 model filename (official: gemma-4-E2B-it-Q8_0.gguf)",
    )
    parser.add_argument(
        "--tool-call-cache-size",
        type=int,
        default=100,
        help="Number of sentences to remember for tool calls",
    )
    parser.add_argument(
        "--llama-state", required=True, help="Path to save llama.cpp state"
    )
    parser.add_argument(
        "--n-ctx", type=int, default=0, help="Size of model context (0 = auto)"
    )
    parser.add_argument(
        "--n-ctx-overhead",
        type=int,
        default=128,
        help="Number of tokens expected beyond system prompt (only when n_ctx = auto)",
    )
    parser.add_argument(
        "--n-threads",
        type=int,
        default=0,
        help="CPU threads for the model (0 = llama.cpp default). "
        "Throughput is memory-bandwidth bound, so leave headroom for Home "
        "Assistant on the same box rather than using every core.",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=64,
        help="Maximum number of tokens the model can generate",
    )
    parser.add_argument(
        "--benchmark-fixture",
        default=None,
        help="Path to benchmark fixture YAML (default: bundled benchmark.yaml)",
    )
    #
    parser.add_argument(
        "--debug", action="store_true", help="Print DEBUG messages to console"
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.DEBUG if args.debug else logging.INFO)
    _LOGGER.debug(args)

    _LOGGER.info("Loading scripts from Home Assistant")
    hass = HomeAssistant(token=args.hass_token, api_url=args.hass_api)
    hass_info = await hass.get_home_info()

    all_tools = await hass.get_script_tools(hass_info)
    # Only scripts exposed to voice are given to the model. The rest are kept for
    # the web UI, which shows what targeting they would get if exposed.
    script_tools = [tool for tool in all_tools if tool.exposed]
    _LOGGER.debug(
        "Loaded %s script(s), %s exposed to voice", len(all_tools), len(script_tools)
    )

    if not script_tools:
        _LOGGER.warning(
            "No scripts have been exposed to voice. Agent will not function."
        )

    _LOGGER.info(
        "Loading Gemma 4 (repo=%s, filename=%s)", args.hf_repo, args.hf_filename
    )
    recognizer = Gemma4Recognizer(
        repo_id=args.hf_repo.strip(),
        filename=args.hf_filename.strip(),
        state_path=args.llama_state,
        cache_size=args.tool_call_cache_size,
        n_ctx=args.n_ctx if args.n_ctx > 0 else None,
        n_ctx_overhead=args.n_ctx_overhead,
        n_threads=args.n_threads if args.n_threads > 0 else None,
        max_tokens=args.max_tokens,
        debug=args.debug,
    )

    # Single worker serializes all model access (live recognition + benchmark).
    llama_executor = ThreadPoolExecutor(max_workers=1)

    state = AppState(
        hass=hass,
        hass_info=hass_info,
        tools={t.name: t for t in script_tools},
        recognizer=recognizer,
        llama_executor=llama_executor,
        all_tools=all_tools,
    )

    # Start the web UI before the model loads: the first boot can spend many
    # minutes downloading the model and building the cached state, and the UI
    # shows a "model loading" notice until it is ready. This also lets the
    # container's health check pass while the download is in progress.
    benchmark_fixture_path = args.benchmark_fixture or (BASE_DIR / "benchmark.yaml")
    flask_app = make_web_server(state, benchmark_fixture_path)
    flask_thread = run_web_server(flask_app, host=args.http_host, port=args.http_port)
    flask_thread.start()

    # Load on the model worker thread so all model access stays on one thread
    # (a benchmark request arriving during load queues behind it).
    loop = asyncio.get_running_loop()
    tools_list = [t.tool for t in script_tools]
    await loop.run_in_executor(llama_executor, recognizer.load, tools_list)

    # Only serve Wyoming once the model is ready: the discovery service waits
    # for this port before telling Home Assistant about the agent.
    server = AsyncServer.from_uri(args.uri)
    _LOGGER.info("Ready")

    # Handle graceful termination
    stop_event = asyncio.Event()

    def request_stop():
        stop_event.set()

    for sig in (signal.SIGTERM, signal.SIGINT):
        loop.add_signal_handler(sig, request_stop)

    server_task = asyncio.create_task(server.run(partial(Gemma4EventHandler, state)))

    try:
        await stop_event.wait()
    finally:
        _LOGGER.info("Shutting down")

        server_task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await server_task


# -----------------------------------------------------------------------------

if __name__ == "__main__":
    asyncio.run(main())
