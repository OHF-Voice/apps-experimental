import hashlib
import io
import json
import logging
import math
import pickle
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union, cast

from huggingface_hub import hf_hub_download
from llama_cpp import Llama

from tools import parse_tool_calls
from util import LRUCache

TOOL_ARGS = Dict[str, Any]
TOOL_CALL = Tuple[str, TOOL_ARGS]
MODEL_RESPONSE = Tuple[List[TOOL_CALL], str]


DEFAULT_REPO = "ggml-org/gemma-4-E2B-it-GGUF"
DEFAULT_FILENAME = "gemma-4-E2B-it-Q8_0.gguf"
DEFAULT_SYSTEM_PROMPT = """
Call a tool for the following sentence.
If no tool applies, say you don't understand in the following language.
"""
DEFAULT_USER_PROMPT = 'Sentence: "{text}"\nLanguage: "{language}"'

_LOGGER = logging.getLogger(__name__)


class Gemma4Recognizer:
    """Gemma 4 tool caller with a persisted, cached system+tools prefix.

    The expensive part of each request is evaluating the (large, fixed) system
    prompt + tools prefix. We build that prefix once, snapshot llama.cpp's KV
    cache state to disk keyed by a hash of (model, prompt, tools), and restore
    it on startup so the prefix is never re-evaluated across restarts. At
    runtime llama.cpp reuses the in-memory prefix automatically, so only the
    per-utterance user turn is evaluated.
    """

    def __init__(
        self,
        state_path: Union[str, Path],
        repo_id: str = DEFAULT_REPO,
        filename: str = DEFAULT_FILENAME,
        system_prompt: str = DEFAULT_SYSTEM_PROMPT,
        user_prompt: str = DEFAULT_USER_PROMPT,
        cache_size: int = 0,
        n_ctx: Optional[int] = None,
        n_ctx_overhead: int = 128,
        n_threads: Optional[int] = None,
        max_tokens: int = 64,
        enable_thinking: bool = False,
        debug: bool = False,
    ) -> None:
        self.llm: Optional[Llama] = None
        self.state_path = Path(state_path)
        self.repo_id = repo_id
        self.filename = filename
        self.user_prompt = user_prompt
        self.n_ctx = n_ctx
        self.n_ctx_overhead = n_ctx_overhead
        self.n_threads = n_threads
        self.max_tokens = max_tokens
        self.temperature = 0.0
        self.top_p = 1.0
        self.top_k = 1
        self.enable_thinking = enable_thinking
        self.tool_choice: str = "auto"
        self.tools: Optional[List[Dict[str, Any]]] = None
        self.debug = debug

        self.cache: Optional[LRUCache] = None
        if cache_size > 0:
            self.cache = LRUCache(cache_size)

        self.system_prompt = system_prompt
        self.system_message = {
            "role": "system",
            "content": system_prompt,
        }

    def load(self, tools: List[Dict[str, Any]]) -> None:
        self.tools = sorted(tools, key=lambda t: t["function"]["name"])
        if self.llm is None:
            if self.n_ctx is None:
                # Estimate context size
                with io.StringIO() as prompt_file:
                    print(self.system_prompt, file=prompt_file)
                    json.dump(self.tools, prompt_file)
                    num_chars = len(prompt_file.getvalue())

                # Round up to multiple of 64.
                # Assume chars / 3 is a conservative upper bound on tokens.
                # User prompt and templating is covered in overhead.
                n_ctx = (
                    math.ceil(
                        (
                            math.ceil(num_chars / 3)
                            + self.max_tokens
                            + self.n_ctx_overhead
                        )
                        / 64
                    )
                    * 64
                )
            else:
                n_ctx = self.n_ctx

            try:
                model_path = hf_hub_download(
                    repo_id=self.repo_id,
                    filename=self.filename,
                    local_files_only=True,
                )
            except OSError:
                model_path = hf_hub_download(
                    repo_id=self.repo_id,
                    filename=self.filename,
                    local_files_only=False,
                )
            _LOGGER.debug("Loading gemma4: %s (n_ctx=%s)", model_path, n_ctx)
            self.llm = Llama(
                model_path=model_path,
                chat_template_kwargs={"enable_thinking": self.enable_thinking},
                n_ctx=n_ctx,
                n_threads=self.n_threads,
                verbose=self.debug,
            )

        actual_tools_hash = _get_tools_hash(
            f"{self.repo_id}/{self.filename}", self.tools, self.system_prompt
        )
        state_metadata_path = self.state_path.with_suffix(".sha256")
        rebuild_state = True
        if state_metadata_path.exists() and self.state_path.exists():
            expected_tools_hash = state_metadata_path.read_text(
                encoding="utf-8"
            ).strip()
            if expected_tools_hash == actual_tools_hash:
                _LOGGER.debug("Cache hit. Loading state: %s", self.state_path)
                with open(self.state_path, "rb") as state_file:
                    state = pickle.load(state_file)

                self.llm.load_state(state)
                rebuild_state = False

        if rebuild_state:
            _LOGGER.info("Building tools prefix...")
            if self.cache is not None:
                self.cache.clear()

            start_time = time.monotonic()
            self.llm.create_chat_completion(
                messages=[self.system_message],  # type: ignore
                tools=self.tools,  # type: ignore
                max_tokens=0,
            )
            end_time = time.monotonic()
            _LOGGER.debug("Built prefix in %s second(s)", end_time - start_time)
            _LOGGER.debug("Saving state: %s", self.state_path)
            self.state_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.state_path, "wb") as state_file:
                state = self.llm.save_state()
                pickle.dump(state, state_file)

            state_metadata_path.write_text(actual_tools_hash, encoding="utf-8")

    def get_tool_calls(self, text: str, language: str = "en") -> MODEL_RESPONSE:
        assert self.llm, "Not loaded"

        text = text.strip()
        cache_key = f"{language}: {text}"
        if self.cache is not None:
            cached_response = self.cache.get(cache_key)
            if cached_response is not None:
                _LOGGER.debug("Returning cached response (key='%s')", cache_key)
                return cached_response

        start_time = time.monotonic()
        response = cast(
            Dict[str, Any],
            self.llm.create_chat_completion(
                messages=[
                    self.system_message,  # type: ignore
                    {
                        "role": "user",
                        "content": self.user_prompt.format(
                            text=text, language=language
                        ),
                    },
                ],
                tools=self.tools,  # type: ignore
                temperature=self.temperature,
                top_p=self.top_p,
                top_k=self.top_k,
                max_tokens=self.max_tokens,
                tool_choice=self.tool_choice,  # type: ignore
            ),
        )
        end_time = time.monotonic()
        _LOGGER.debug("Response in %s second(s): %s", end_time - start_time, response)

        content = response["choices"][0]["message"]["content"] or ""
        model_response = (parse_tool_calls(content), content)
        if self.cache is not None:
            _LOGGER.debug("Caching '%s' -> %s", cache_key, model_response)
            self.cache.set(cache_key, model_response)

        return model_response

    def describe(self) -> Dict[str, Any]:
        """Return the effective runtime config (for benchmark reports)."""
        n_threads: Optional[int] = None
        n_ctx: Optional[int] = self.n_ctx
        draft_model: Optional[str] = None
        if self.llm is not None:
            n_threads = getattr(self.llm.context_params, "n_threads", None)
            n_ctx = self.llm.n_ctx()
            if getattr(self.llm, "draft_model", None) is not None:
                draft_model = type(self.llm.draft_model).__name__

        return {
            "repo_id": self.repo_id,
            "filename": self.filename,
            "n_ctx": n_ctx,
            "n_threads": n_threads,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "draft_model": draft_model,
        }

    def run_sentences(
        self,
        tools: List[Dict[str, Any]],
        sentences: List[Dict[str, Any]],
        passes: int = 3,
        default_language: str = "en",
    ) -> Dict[str, Any]:
        """Benchmark a fixed tool/sentence fixture without disturbing live state.

        Snapshots the running assistant's state, builds a temporary prefix from
        ``tools``, runs each sentence ``passes`` times (resetting to the warm
        prefix before each timed call so every measurement mirrors a fresh
        per-utterance start in production), then restores the live state. The
        response cache is bypassed entirely. Must be called on the same single
        thread that serves recognition (the model is not thread-safe).
        """
        assert self.llm, "Not loaded"
        passes = max(1, passes)
        sorted_tools = sorted(tools, key=lambda t: t["function"]["name"])

        # Snapshot live state so benchmarking never disturbs the running assistant.
        prod_state = self.llm.save_state()
        try:
            # Build the temporary static prefix for the benchmark tools.
            start_time = time.monotonic()
            self.llm.create_chat_completion(
                messages=[self.system_message],  # type: ignore
                tools=sorted_tools,  # type: ignore
                max_tokens=0,
            )
            rebuild_seconds = time.monotonic() - start_time

            # Snapshot the warm prefix so each sentence starts from the same point.
            prefix_state = self.llm.save_state()

            results: List[Dict[str, Any]] = []
            for sentence in sentences:
                text = str(sentence["text"]).strip()
                language = str(sentence.get("language") or default_language)
                sentence_passes: List[Dict[str, Any]] = []
                content = ""

                for _ in range(passes):
                    # Reset to warm prefix (untimed): mirrors a production
                    # per-utterance start where only the user turn is evaluated.
                    self.llm.load_state(prefix_state)

                    start_time = time.monotonic()
                    response = cast(
                        Dict[str, Any],
                        self.llm.create_chat_completion(
                            messages=[
                                self.system_message,  # type: ignore
                                {
                                    "role": "user",
                                    "content": self.user_prompt.format(
                                        text=text, language=language
                                    ),
                                },
                            ],
                            tools=sorted_tools,  # type: ignore
                            temperature=self.temperature,
                            top_p=self.top_p,
                            top_k=self.top_k,
                            max_tokens=self.max_tokens,
                            tool_choice=self.tool_choice,  # type: ignore
                        ),
                    )
                    latency = time.monotonic() - start_time

                    content = response["choices"][0]["message"]["content"] or ""
                    usage = response.get("usage") or {}
                    sentence_passes.append(
                        {
                            "latency": latency,
                            "prompt_tokens": usage.get("prompt_tokens"),
                            "completion_tokens": usage.get("completion_tokens"),
                        }
                    )

                tool_calls = parse_tool_calls(content)
                results.append(
                    {
                        "text": text,
                        "language": language,
                        "passes": sentence_passes,
                        "content": content,
                        "tool_calls": [
                            {"name": name, "args": args} for name, args in tool_calls
                        ],
                    }
                )
        finally:
            # Always restore the live assistant's state.
            self.llm.load_state(prod_state)

        return {
            "rebuild_seconds": rebuild_seconds,
            "passes": passes,
            "num_tools": len(sorted_tools),
            "config": self.describe(),
            "sentences": results,
        }


# -----------------------------------------------------------------------------


def _get_tools_hash(
    model_id: str, tools: List[Dict[str, Any]], system_prompt: str
) -> str:
    hasher = hashlib.sha256()
    hasher.update(model_id.encode())
    hasher.update(system_prompt.encode())
    hasher.update(json.dumps(tools).encode())
    return hasher.hexdigest()
