#!/usr/bin/env python3

import argparse
import asyncio
import logging
import time
import threading
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from collections.abc import Mapping
from typing import Any, Dict, Optional, List, Tuple

import voluptuous as vol
from voluptuous.humanize import humanize_error
from jinja2 import BaseLoader, StrictUndefined
from jinja2.nativetypes import NativeEnvironment
from ruamel.yaml import YAML, YAMLError
from flask import Flask, jsonify, render_template, request, url_for
from werkzeug.middleware.proxy_fix import ProxyFix
from llama_cpp import Llama
from huggingface_hub import hf_hub_download
from wyoming.event import Event
from wyoming.server import AsyncEventHandler, AsyncServer
from wyoming.info import Attribution, Describe, Info, HandleModel, HandleProgram
from wyoming.asr import Transcript
from wyoming.handle import Handled, NotHandled

from commands import Command, ToolsAndCommandsSchema
from tools import parse_tool_calls
from hass_api import HomeAssistant, InfoForRecognition

_LOGGER = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parent

_yaml = YAML()

# -----------------------------------------------------------------------------


@dataclass
class AppState:
    tools_path: Optional[Path]
    tools: List[Dict[str, Any]]
    commands: Dict[str, Command]
    prime_event: Optional[asyncio.Event] = None

    def load(self):
        self.tools = []
        self.commands = {}
        if (self.tools_path is None) or (not self.tools_path.exists()):
            _LOGGER.debug("Missing tools file: %s", self.tools_path)
            return

        _LOGGER.debug("Loading tools from YAML: %s", self.tools_path)
        with open(self.tools_path, "r", encoding="utf-8") as tools_file:
            tools_and_commands = _yaml.load(tools_file)

        self.tools = tools_and_commands.get("tools", [])
        commands_list = tools_and_commands.get("commands")
        if commands_list:
            # Parse commands
            for command_dict in commands_list:
                command = Command.from_dict(command_dict)
                self.commands[command.id] = command

        if self.tools:
            _LOGGER.debug("Loaded %s tool(s)", len(self.tools))

        if self.commands:
            _LOGGER.debug("Loaded %s command(s)", len(self.commands))


def prime_model(
    state: AppState, args: argparse.Namespace, llm: Llama, text: str = "test"
) -> None:
    try:
        response = llm.create_chat_completion(
            messages=[{"role": "user", "content": text}],
            tools=state.tools,
            temperature=args.temperature,
            top_p=args.top_p,
            max_tokens=args.max_tokens,
            tool_choice=args.tool_choice,
        )
        _LOGGER.debug("Priming complete")
    finally:
        if state.prime_event:
            state.prime_event.set()


# -----------------------------------------------------------------------------


async def main() -> None:
    """Run app."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--uri", required=True, help="unix:// or tcp://")
    #
    parser.add_argument("--hass-token")
    parser.add_argument("--hass-api", default="http://homeassistant.local:8123")
    parser.add_argument("--device-id", help="Id of device to mimic (debugging only)")
    parser.add_argument(
        "--satellite-id", help="Id of satellite to mimic (debugging only)"
    )
    #
    parser.add_argument("--tools", help="Path to read/write YAML with tool definitions")
    #
    parser.add_argument("--repo-id", default="ggml-org/gemma-4-E2B-it-GGUF")
    parser.add_argument("--filename", default="gemma-4-E2B-it-Q8_0.gguf")
    parser.add_argument("--n-ctx", type=int, default=2048)
    parser.add_argument("--max-tokens", type=int, default=64)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--enable-thinking", action="store_true")
    parser.add_argument("--tool-choice", default="auto")
    parser.add_argument("--no-tools", action="store_true", help="Disable tools")
    parser.add_argument(
        "--prime-model",
        action="store_true",
        help="Run test prompt through model after loading or updating tools",
    )
    #
    parser.add_argument("--http-host", default="127.0.0.1", help="Host for web UI")
    parser.add_argument("--http-port", default=5000, type=int, help="Port for web UI")
    #
    parser.add_argument(
        "--debug", action="store_true", help="Print DEBUG messages to console"
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.DEBUG if args.debug else logging.INFO)
    _LOGGER.debug(args)

    if args.no_tools:
        _LOGGER.info("Tools have been disabled")

    state = AppState(
        tools_path=Path(args.tools) if (args.tools and (not args.no_tools)) else None,
        tools=[],
        commands={},
    )
    state.load()

    hass: Optional[HomeAssistant] = None
    if args.hass_token and args.hass_api:
        hass = HomeAssistant(token=args.hass_token, api_url=args.hass_api)

    _LOGGER.info("Downloading: %s", args.repo_id)
    model_path = hf_hub_download(
        repo_id=args.repo_id,
        filename=args.filename,
    )

    _LOGGER.info("Loading model: %s", model_path)
    llm = Llama(
        model_path=model_path,
        chat_template_kwargs={"enable_thinking": args.enable_thinking},
        n_ctx=args.n_ctx,
        verbose=args.debug,
    )

    if args.prime_model:
        _LOGGER.info("Priming model")
        state.prime_event = asyncio.Event()
        loop = asyncio.get_running_loop()
        loop.run_in_executor(None, prime_model, state, args, llm)

    # Run web UI
    flask_app = get_app(state, args, llm)

    def run_flask():
        flask_app.run(host=args.http_host, port=args.http_port, use_reloader=False)

    flask_thread = threading.Thread(target=run_flask, daemon=True)
    flask_thread.start()

    server = AsyncServer.from_uri(args.uri)
    _LOGGER.info("Ready")

    try:
        await server.run(partial(WyomingEventHandler, args, llm, state, hass))
    except KeyboardInterrupt:
        pass


# -----------------------------------------------------------------------------


class WyomingEventHandler(AsyncEventHandler):
    """Event handler for clients."""

    def __init__(
        self,
        cli_args: argparse.Namespace,
        llm: Llama,
        state: AppState,
        hass: Optional[HomeAssistant],
        *args,
        **kwargs,
    ) -> None:
        """Initialize event handler."""
        super().__init__(*args, **kwargs)

        self.client_id = str(time.monotonic_ns())
        self.cli_args = cli_args
        self.llm = llm
        self.state = state
        self.hass = hass

        self._info_event: Optional[Event] = None
        self._env = NativeEnvironment(loader=BaseLoader(), undefined=StrictUndefined)

    async def handle_event(self, event: Event) -> bool:
        """Handle Wyoming event."""
        try:
            return await self._handle_event(event)
        except Exception:
            _LOGGER.exception("Error handling event")

        return True

    async def _handle_event(self, event: Event) -> bool:
        """Handle Wyoming event."""
        if Describe.is_type(event.type):
            await self._write_info()
            return True

        if Transcript.is_type(event.type):
            transcript = Transcript.from_event(event)
            _LOGGER.debug("Handling: %s", transcript)

            handled = False
            response_text: Optional[str] = None
            hass_info: Optional[InfoForRecognition] = None
            device_id: Optional[str] = self.cli_args.device_id
            satellite_id: Optional[str] = self.cli_args.satellite_id

            if transcript.context:
                device_id = transcript.context.get("device_id", device_id)
                satellite_id = transcript.context.get("satellite_id", satellite_id)

            if self.hass:
                hass_info = await self.hass.get_info(device_id, satellite_id)
                _LOGGER.debug("Home Assistant info: %s", hass_info)
            else:
                _LOGGER.warning("No Home Assistant device/area information loaded")

            if transcript.text:
                if self.state.prime_event:
                    _LOGGER.debug("Waiting for priming to finish")
                    await self.state.prime_event.wait()
                    self.state.prime_event = None

                start_time = time.monotonic()
                response = self.llm.create_chat_completion(
                    messages=[{"role": "user", "content": transcript.text}],
                    tools=self.state.tools,
                    temperature=self.cli_args.temperature,
                    top_p=self.cli_args.top_p,
                    max_tokens=self.cli_args.max_tokens,
                    tool_choice=self.cli_args.tool_choice,
                )
                end_time = time.monotonic()
                _LOGGER.debug(
                    "Response in %s second(s): %s", end_time - start_time, response
                )

                content = response["choices"][0]["message"]["content"]
                if self.cli_args.no_tools:
                    # Return LLM response directly
                    response_text = content
                    handled = True
                else:
                    # Handle tool calls
                    tool_calls = parse_tool_calls(content)
                    response_lines: List[str] = []

                    for command_id, command_args in tool_calls:
                        command = self.state.commands.get(command_id)
                        if command is None:
                            _LOGGER.warning("Command not found: %s", command_id)
                            continue

                        command_success, command_response = await self.handle_command(
                            command, command_args, hass_info, transcript.language
                        )
                        if command_response:
                            response_lines.append(command_response)

                        if command_success:
                            handled = True
                        else:
                            handled = False
                            response_text = (
                                f"Unexpected error while handling command: {command.id}"
                            )
                            break

                    if handled:
                        response_text = "\n".join(response_lines)
                    else:
                        response_text = response_text or "No tools were called."

            if handled:
                await self.write_event(
                    Handled(text=response_text, context=transcript.context).event()
                )
            else:
                await self.write_event(
                    NotHandled(
                        text=response_text
                        or "Unexpected error. Check log for details.",
                        context=transcript.context,
                    ).event()
                )

            return True

        return True

    async def handle_command(
        self,
        command: Command,
        command_args: Dict[str, Any],
        hass_info: Optional[InfoForRecognition],
        language: Optional[str],
    ) -> Tuple[bool, Optional[str]]:
        try:
            command_response = await self._handle_command(
                command, command_args, hass_info, language
            )
            return (True, command_response)
        except Exception:
            _LOGGER.exception("Unexpected error while handling command: %s", command)

        return (False, None)

    async def _handle_command(
        self,
        command: Command,
        command_args: Dict[str, Any],
        hass_info: Optional[InfoForRecognition],
        language: Optional[str],
    ) -> Optional[str]:
        slots: Dict[str, Any] = {}
        variables: Dict[str, Any] = {"args": command_args, "slots": slots}

        if language:
            variables["language"] = language

        if hass_info:
            variables["satellite"] = {
                "device_id": hass_info.current_device_id,
                "satellite_id": hass_info.current_satellite_id,
                "area_id": hass_info.current_area_id,
                "area_name": hass_info.current_area_name,
                "floor_id": hass_info.current_floor_id,
            }

        if command.intent and command.intent.slots:
            slots.update(command.intent.slots)

        if command.action:
            # Run action in Home Assistant.
            # Targets and data values are rendered as templates in Home
            # Assistant.
            if not self.hass:
                raise RuntimeError(
                    "Home Assistant token and URL must be set to run actions"
                )

            # Actions themselves can be templates
            action = command.action
            action_name = action.action
            if is_template_string(action_name):
                action_name = await self.render_template(action_name, variables)

            domain, service = action_name.split(".", maxsplit=1)
            action_data: Dict[str, Any] = {}
            action_target: Dict[str, Any] = {}

            if action.data:
                action_data = await self.render_templates_recursive(
                    action.data, variables
                )

            if action.target:
                action_target = await self.render_templates_recursive(
                    action.target, variables
                )

            _LOGGER.debug(
                "Running action: %s with target=%s, data=%s",
                action_name,
                action_target,
                action_data,
            )

            await self.hass.trigger_service(
                domain,
                service,
                service_data=action_data,
                target=action_target,
            )

        if command.intent:
            # Intent recognized
            if not self.hass:
                raise RuntimeError(
                    "Home Assistant token and URL must be set to handle intents"
                )

            intent_name = command.intent.name
            if is_template_string(intent_name):
                intent_name = await self.render_template(intent_name, variables)

            intent_slots: Optional[Dict[str, Any]] = None
            if command.intent.slots:
                intent_slots = await self.render_templates_recursive(
                    command.intent.slots, variables
                )

                if isinstance(intent_slots, dict):
                    # Update template variables
                    for key, value in intent_slots.items():
                        slots[key] = value

                    # Remove keys with null values
                    intent_slots = {
                        key: value
                        for key, value in intent_slots.items()
                        if value is not None
                    }

            _LOGGER.debug(
                "Handling intent in Home Assistant: name=%s, slots=%s",
                intent_name,
                intent_slots,
            )
            await self.hass.handle_intent(
                intent_name=intent_name,
                language=command.intent.language or language or "en",
                data=intent_slots or {},
                device_id=hass_info.current_device_id if hass_info else None,
                satellite_id=hass_info.current_satellite_id if hass_info else None,
            )

        # Render response
        response: Optional[str] = None
        if command.response:
            _LOGGER.debug(
                "Rendering response: text=%s, variables=%s", command.response, variables
            )
            try:
                response = await self.render_template(command.response, variables)
            except Exception:
                # TODO: error message
                _LOGGER.exception("Unexpected error while rendering response")
                raise

        return response

    async def render_templates_recursive(
        self, data: Any, variables: Mapping[str, Any]
    ) -> Any:
        # Template string handling
        if isinstance(data, str) and is_template_string(data):
            return await self.render_template(data, variables)

        # Mapping (dict-like)
        if isinstance(data, Mapping):
            return {
                k: await self.render_templates_recursive(v, variables)
                for k, v in data.items()
            }

        # Sequence (but not str/bytes)
        if isinstance(data, (list, tuple)):
            rendered = [
                await self.render_templates_recursive(v, variables) for v in data
            ]
            return rendered if isinstance(data, list) else tuple(rendered)

        return data

    async def render_template(self, data: str, variables: Mapping[str, Any]) -> Any:
        if self.hass:
            return await self.hass.render_template(data, variables)

        return self._env.from_string(data).render(variables)

    async def _write_info(self) -> None:
        if self._info_event is not None:
            await self.write_event(self._info_event)
            return

        info = Info(
            handle=[
                HandleProgram(
                    name="intent-gemma4",
                    attribution=Attribution("", ""),
                    installed=True,
                    description="Gemma 4 Agent",
                    version="",
                    models=[
                        HandleModel(
                            name="gemma4",
                            attribution=Attribution("", ""),
                            installed=True,
                            description="gemma4",
                            version="",
                            languages=[],
                        )
                    ],
                )
            ]
        )

        self._info_event = info.event()
        await self.write_event(self._info_event)


def is_template_string(maybe_template: str) -> bool:
    """Check if the input is a Jinja2 template."""
    return "{" in maybe_template and (
        "{%" in maybe_template or "{{" in maybe_template or "{#" in maybe_template
    )


# -----------------------------------------------------------------------------


def get_app(state: AppState, args: argparse.Namespace, llm: Llama) -> Flask:
    flask_app = Flask(__name__, template_folder=str(BASE_DIR / "templates"))
    flask_app.secret_key = "90a238ad-7e69-4438-85dc-eee0a68c7435"

    flask_app.wsgi_app = ProxyFix(flask_app.wsgi_app, x_proto=1, x_host=1)  # type: ignore[method-assign]
    flask_app.wsgi_app = IngressPrefixMiddleware(flask_app.wsgi_app)  # type: ignore[method-assign]

    @flask_app.context_processor
    def inject_url_for():
        return dict(url_for=url_for)  # pylint: disable=use-dict-literal

    @flask_app.route("/", methods=["GET"])
    def index():
        if state.tools_path:
            content = state.tools_path.read_text(encoding="utf-8")
        else:
            content = "# No tools path was set. Tools will not be saved!\n"

        return render_template("index.html", content=content)

    @flask_app.route("/save", methods=["POST"])
    def save():
        if not state.tools_path:
            return (
                jsonify({"ok": False, "error": "Tools path was not set"}),
                400,
            )

        text = request.get_data(as_text=True)

        try:
            data = _yaml.load(text)
        except YAMLError as err:
            return (
                jsonify(
                    {
                        "ok": False,
                        "error": f"YAML parse error: {err}",
                    }
                ),
                400,
            )

        try:
            data = ToolsAndCommandsSchema(data)
        except vol.Invalid as err:
            err_text = (humanize_error(data, err),)
            _LOGGER.error(
                "Invalid tools: %s",
                err_text,
            )
            return (
                jsonify({"ok": False, "error": err_text}),
                400,
            )

        state.tools_path.write_text(text, encoding="utf-8")
        _LOGGER.debug("Wrote: %s", state.tools_path)

        state.load()

        return jsonify({"ok": True, "message": "Saved successfully."})

    return flask_app


class IngressPrefixMiddleware:
    def __init__(self, app):
        self.app = app

    def __call__(self, environ, start_response):
        ingress_path = environ.get("HTTP_X_INGRESS_PATH", "")
        if ingress_path:
            environ["SCRIPT_NAME"] = ingress_path
            path_info = environ.get("PATH_INFO", "")
            if path_info.startswith(ingress_path):
                environ["PATH_INFO"] = path_info[len(ingress_path) :] or "/"
        return self.app(environ, start_response)


# -----------------------------------------------------------------------------

if __name__ == "__main__":
    asyncio.run(main())
