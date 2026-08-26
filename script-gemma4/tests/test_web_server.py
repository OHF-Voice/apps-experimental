import unittest
from concurrent.futures import ThreadPoolExecutor

from const import AppState
from hass_api import HomeAssistantInfo, Tool
from overrides import AREA, ENTITY, FLOOR, NameOverrides, Overrides
from web_server import _apply_candidate, _replacement_global_names


def _info():
    return HomeAssistantInfo(states={}, entities={}, areas={}, floors={})


def _tool(name):
    return Tool(
        name=name,
        exposed=True,
        tool={"type": "function", "function": {"name": name}},
    )


class _Recognizer:
    def __init__(self, error=None):
        self.error = error
        self.reload_calls = []

    def reload(self, tools):
        self.reload_calls.append(tools)
        if self.error:
            raise self.error


class NameReplacementTests(unittest.TestCase):
    def test_global_names_preserve_per_script_names(self):
        current = NameOverrides(
            per_script={"demo": {"target": {"light.desk": ["Desk only"]}}}
        )
        known = {
            ENTITY: {"light.desk"},
            AREA: set(),
            FLOOR: set(),
        }

        replacement = _replacement_global_names(
            current,
            {ENTITY: {"light.desk": ["Desk"]}, AREA: {}, FLOOR: {}},
            known,
        )

        self.assertEqual(
            {"demo": {"target": {"light.desk": ["Desk only"]}}},
            replacement.per_script,
        )
        self.assertEqual(["Desk"], replacement.by_kind[ENTITY]["light.desk"])


class TransactionTests(unittest.TestCase):
    def test_failed_model_reload_does_not_commit_candidate_state(self):
        old_info = _info()
        old_tool = _tool("old")
        old_overrides = Overrides()
        executor = ThreadPoolExecutor(max_workers=1)
        state = AppState(
            hass=object(),  # type: ignore[arg-type]
            hass_info=old_info,
            tools={old_tool.name: old_tool},
            recognizer=_Recognizer(RuntimeError("reload failed")),  # type: ignore[arg-type]
            llama_executor=executor,
            all_tools=[old_tool],
            overrides=old_overrides,
        )
        new_tool = _tool("new")
        candidate = Overrides()

        try:
            with self.assertRaisesRegex(RuntimeError, "reload failed"):
                _apply_candidate(state, _info(), [new_tool], candidate)

            self.assertIs(old_info, state.hass_info)
            self.assertIs(old_overrides, state.overrides)
            self.assertEqual([old_tool], state.all_tools)
            self.assertEqual({"old": old_tool}, state.tools)
        finally:
            executor.shutdown()

    def test_successful_model_reload_commits_one_snapshot(self):
        old_tool = _tool("old")
        recognizer = _Recognizer()
        executor = ThreadPoolExecutor(max_workers=1)
        state = AppState(
            hass=object(),  # type: ignore[arg-type]
            hass_info=_info(),
            tools={old_tool.name: old_tool},
            recognizer=recognizer,  # type: ignore[arg-type]
            llama_executor=executor,
            all_tools=[old_tool],
            overrides=Overrides(),
        )
        new_info = _info()
        new_tool = _tool("new")
        candidate = Overrides()

        try:
            targeted = _apply_candidate(state, new_info, [new_tool], candidate)

            self.assertEqual([new_tool], targeted)
            self.assertIs(new_info, state.hass_info)
            self.assertIs(candidate, state.overrides)
            self.assertEqual({"new": new_tool}, state.tools)
        finally:
            executor.shutdown()


if __name__ == "__main__":
    unittest.main()
