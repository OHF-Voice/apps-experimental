import pickle
import sys
import tempfile
import unittest
from pathlib import Path
from types import ModuleType
from unittest.mock import patch

try:
    import llama_cpp  # noqa: F401
except ModuleNotFoundError:
    llama_cpp_stub = ModuleType("llama_cpp")
    llama_cpp_stub.Llama = object
    llama_cpp_stub.__version__ = "test"
    sys.modules["llama_cpp"] = llama_cpp_stub

from gemma4_recognizer import (  # noqa: E402
    LLAMA_CPP_VERSION,
    Gemma4Recognizer,
    _get_tools_hash,
    _parse_tool_calls,
)


class _FakeLlama:
    def __init__(self, n_ctx=64, load_error=None):
        self._n_ctx = n_ctx
        self.load_error = load_error
        self.create_calls = 0
        self.loaded_states = []

    def n_ctx(self):
        return self._n_ctx

    def create_chat_completion(self, **kwargs):
        self.create_calls += 1
        return {}

    def save_state(self):
        return {"state": "good"}

    def load_state(self, state):
        self.loaded_states.append(state)
        if self.load_error:
            raise self.load_error


class ToolCallParserTests(unittest.TestCase):
    def test_single_quoted_value_can_contain_comma(self):
        text = (
            "<|tool_call>call:play{artist:<|'|>Earth, Wind & Fire<|'|>}" "<tool_call|>"
        )

        self.assertEqual(
            [("play", {"artist": "Earth, Wind & Fire"})],
            _parse_tool_calls(text),
        )

    def test_malformed_argument_drops_call_instead_of_raising(self):
        text = "<|tool_call>call:demo{broken}<tool_call|>"

        self.assertEqual([], _parse_tool_calls(text))


class StateCacheTests(unittest.TestCase):
    def test_corrupt_matching_state_is_rebuilt(self):
        tools = [{"type": "function", "function": {"name": "demo"}}]
        with tempfile.TemporaryDirectory() as temp_dir:
            state_path = Path(temp_dir) / "state.bin"
            state_path.write_bytes(b"not a pickle")

            recognizer = Gemma4Recognizer(state_path=state_path)
            recognizer.tools = tools
            recognizer.llm = _FakeLlama()
            recognizer.model_path = Path("/model")
            runtime_model_id = (
                f"llama-cpp-python/{LLAMA_CPP_VERSION};"
                "n_ctx=64;flash_attn=1;model_path=/model;"
                f"{recognizer.repo_id}/{recognizer.filename}"
            )
            state_path.with_suffix(".sha256").write_text(
                _get_tools_hash(runtime_model_id, tools, recognizer.system_prompt),
                encoding="utf-8",
            )

            recognizer._restore_or_build_state()  # pylint: disable=protected-access

            self.assertEqual(1, recognizer.llm.create_calls)
            with open(state_path, "rb") as state_file:
                self.assertEqual({"state": "good"}, pickle.load(state_file))

    def test_failed_context_growth_restores_old_model(self):
        old_tools = [{"type": "function", "function": {"name": "old"}}]
        new_tools = [{"type": "function", "function": {"name": "new"}}]
        recognizer = Gemma4Recognizer(state_path="unused.bin")
        old_llm = _FakeLlama(n_ctx=64)
        recognizer.llm = old_llm
        recognizer.tools = old_tools
        recognizer.ready = True

        def create_larger(_n_ctx):
            recognizer.llm = _FakeLlama(n_ctx=128)

        with patch.object(recognizer, "required_n_ctx", return_value=128), patch.object(
            recognizer, "_create_llm", side_effect=create_larger
        ), patch.object(
            recognizer,
            "_restore_or_build_state",
            side_effect=RuntimeError("prefix failed"),
        ):
            with self.assertRaisesRegex(RuntimeError, "prefix failed"):
                recognizer.reload(new_tools)

        self.assertIs(old_llm, recognizer.llm)
        self.assertEqual(old_tools, recognizer.tools)
        self.assertTrue(recognizer.ready)
        self.assertEqual([{"state": "good"}], old_llm.loaded_states)


if __name__ == "__main__":
    unittest.main()
