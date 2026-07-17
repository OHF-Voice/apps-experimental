import hashlib
import io
import json
import logging
import math
import pickle
import re
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


# Hermes-style tool calls (Qwen, Hermes, many fine-tunes):
# <tool_call>{"name": ..., "arguments": {...}}</tool_call>
HERMES_TOOL_CALL_RE = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.DOTALL)

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
        chat_format: Optional[str] = None,
        flash_attn: bool = False,
        prompt_lookup: int = 0,
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
        # Optional llama-cpp-python chat handler. Leave None for Gemma 4 (tool
        # calls are parsed from message content); set e.g.
        # "chatml-function-calling" for models that emit structured tool_calls.
        self.chat_format = chat_format
        # Latency knobs (no accuracy cost): flash attention, and prompt-lookup
        # speculative decoding (drafts n-grams from the prompt -- effective when
        # the output echoes input tokens like names/areas).
        self.flash_attn = flash_attn
        self.prompt_lookup = prompt_lookup
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

    def load(self, tools: List[Dict[str, Any]], build_prefix: bool = True) -> None:
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
            draft_model = None
            if self.prompt_lookup > 0:
                from llama_cpp.llama_speculative import LlamaPromptLookupDecoding

                draft_model = LlamaPromptLookupDecoding(
                    num_pred_tokens=self.prompt_lookup
                )

            _LOGGER.debug("Loading gemma4: %s (n_ctx=%s)", model_path, n_ctx)
            self.llm = Llama(
                model_path=model_path,
                chat_template_kwargs={"enable_thinking": self.enable_thinking},
                chat_format=self.chat_format,
                n_ctx=n_ctx,
                n_threads=self.n_threads,
                flash_attn=self.flash_attn,
                draft_model=draft_model,
                verbose=self.debug,
            )

        if not build_prefix:
            # Benchmark-only use (e.g. the sweep): skip the persisted prefix and
            # its build/save. run_sentences() builds its own warm prefix, so this
            # avoids paying the (slow on a Pi) prefix eval twice.
            return

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

        message = response["choices"][0]["message"]
        content = message.get("content") or ""
        model_response = (_extract_tool_calls(message), content)
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
            "chat_format": self.chat_format,
            "flash_attn": self.flash_attn,
            "prompt_lookup": self.prompt_lookup,
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
        # Some configs (e.g. prompt-lookup speculative decoding) have state that
        # llama.cpp can't serialize; fall back to relying on its automatic
        # longest-common-prefix reuse, which means we can't reset between passes
        # of the same sentence -- so force a single pass (each distinct sentence
        # is still a fresh user turn against the warm prefix).
        can_snapshot = True
        prod_state = None
        try:
            prod_state = self.llm.save_state()
        except Exception:  # noqa: BLE001
            can_snapshot = False
            _LOGGER.warning(
                "save_state unavailable; benchmarking without per-pass reset"
            )

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
            prefix_state = None
            if can_snapshot:
                try:
                    prefix_state = self.llm.save_state()
                except Exception:  # noqa: BLE001
                    can_snapshot = False

            effective_passes = passes if can_snapshot else 1
            results: List[Dict[str, Any]] = []
            for sentence in sentences:
                text = str(sentence["text"]).strip()
                language = str(sentence.get("language") or default_language)
                sentence_passes: List[Dict[str, Any]] = []
                content = ""

                for _ in range(effective_passes):
                    # Reset to warm prefix (untimed): mirrors a production
                    # per-utterance start where only the user turn is evaluated.
                    if can_snapshot:
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

                    message = response["choices"][0]["message"]
                    content = message.get("content") or ""
                    usage = response.get("usage") or {}
                    sentence_passes.append(
                        {
                            "latency": latency,
                            "prompt_tokens": usage.get("prompt_tokens"),
                            "completion_tokens": usage.get("completion_tokens"),
                        }
                    )

                tool_calls = _extract_tool_calls(message)
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
            # Restore the live assistant's state when we were able to snapshot it.
            if can_snapshot and prod_state is not None:
                try:
                    self.llm.load_state(prod_state)
                except Exception:  # noqa: BLE001
                    pass

        return {
            "rebuild_seconds": rebuild_seconds,
            "passes": effective_passes,
            "num_tools": len(sorted_tools),
            "config": self.describe(),
            "sentences": results,
        }


# -----------------------------------------------------------------------------


# Duration slots where a 0 means "absent" (smaller models, e.g. Qwen, tend to
# fill every field: minutes:5, hours:0, seconds:0). Safe to drop because no
# non-timer intent uses these keys.
_DURATION_ZERO_KEYS = frozenset(
    {"hours", "minutes", "seconds", "start_hours", "start_minutes", "start_seconds"}
)


def _clean_args(args: Dict[str, Any]) -> Dict[str, Any]:
    """Drop hallucinated empty/zero-duration slot values some models emit.

    Removes ``None``/``""`` values (e.g. ``area:""``, ``name:""``) and
    zero-valued duration slots. Home Assistant treats these as absent, and Gemma
    doesn't emit them, so this is a no-op for the tuned baseline.
    """
    cleaned: Dict[str, Any] = {}
    for key, value in args.items():
        if value is None or value == "":
            continue
        if key in _DURATION_ZERO_KEYS and value == 0:
            continue
        cleaned[key] = value
    return cleaned


def _extract_tool_calls(message: Dict[str, Any]) -> List[TOOL_CALL]:
    """Tool calls from a chat message, model-agnostically.

    Prefers llama.cpp's native structured ``tool_calls``; otherwise parses the
    content as Hermes-style ``<tool_call>{...}</tool_call>`` (Qwen etc.) or
    Gemma 4's ``<|tool_call>call:NAME{...}<tool_call|>`` syntax. Argument values
    are cleaned of hallucinated empties/zeros.
    """
    raw_calls = message.get("tool_calls") or []
    calls: List[TOOL_CALL] = []
    for raw_call in raw_calls:
        function = raw_call.get("function") or {}
        name = function.get("name")
        if not name:
            continue

        arguments = function.get("arguments")
        args: Dict[str, Any] = {}
        if isinstance(arguments, dict):
            args = arguments
        elif isinstance(arguments, str) and arguments.strip():
            try:
                args = json.loads(arguments)
            except json.JSONDecodeError:
                _LOGGER.debug("Could not parse tool arguments: %s", arguments)
                args = {}
        calls.append((name, args))

    if not calls:
        content = message.get("content") or ""
        # Hermes-style (Qwen etc.) is prefix-cacheable, unlike the
        # chatml-function-calling handler; fall back to Gemma's content syntax.
        calls = _parse_hermes_tool_calls(content) or parse_tool_calls(content)

    return [(name, _clean_args(args)) for name, args in calls]


def _loads_tool_object(raw: str) -> Optional[Dict[str, Any]]:
    try:
        obj = json.loads(raw)
    except json.JSONDecodeError:
        # Some models (e.g. Qwen) double-wrap the object: {{ ... }}.
        if raw.startswith("{{"):
            try:
                obj = json.loads(raw[1:-1])
            except json.JSONDecodeError:
                return None
        else:
            return None
    return obj if isinstance(obj, dict) else None


def _parse_hermes_tool_calls(content: str) -> List[TOOL_CALL]:
    calls: List[TOOL_CALL] = []
    for match in HERMES_TOOL_CALL_RE.finditer(content):
        obj = _loads_tool_object(match.group(1))
        if obj is None:
            continue
        name = obj.get("name")
        if not name:
            continue
        args = obj.get("arguments")
        if isinstance(args, str):
            try:
                args = json.loads(args)
            except json.JSONDecodeError:
                args = {}
        calls.append((name, args if isinstance(args, dict) else {}))
    return calls


def _get_tools_hash(
    model_id: str, tools: List[Dict[str, Any]], system_prompt: str
) -> str:
    hasher = hashlib.sha256()
    hasher.update(model_id.encode())
    hasher.update(system_prompt.encode())
    hasher.update(json.dumps(tools).encode())
    return hasher.hexdigest()
