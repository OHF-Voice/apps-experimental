import unittest

from hass_api import SatelliteInfo, Tool
from tool_mapping import HomeGeometry, map_tool_call


def _tool(required=None, name_map=None):
    return Tool(
        name="demo",
        tool={
            "type": "function",
            "function": {
                "name": "demo",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "target": {"type": "string"},
                        "note": {"type": "string"},
                    },
                    "required": required or [],
                },
            },
        },
        name_map=name_map or {},
    )


class ToolMappingTests(unittest.TestCase):
    def test_omitted_required_field_blocks_script(self):
        call = map_tool_call(_tool(required=["target"]), {})

        self.assertFalse(call.can_run)
        self.assertEqual(["target"], call.missing_required)

    def test_unknown_model_argument_is_not_forwarded(self):
        call = map_tool_call(_tool(), {"note": "ok", "invented": "unsafe"})

        self.assertEqual({"note": "ok"}, call.variables)

    def test_ambiguous_name_uses_satellite_area(self):
        tool = _tool(
            required=["target"],
            name_map={
                "target": {"Speaker": ["media_player.kitchen", "media_player.office"]}
            },
        )
        geometry = HomeGeometry(
            entity_areas={
                "media_player.kitchen": "kitchen",
                "media_player.office": "office",
            }
        )

        call = map_tool_call(
            tool,
            {"target": "Speaker"},
            SatelliteInfo(area_id="office"),
            geometry,
        )

        self.assertTrue(call.can_run)
        self.assertEqual("media_player.office", call.variables["target"])


if __name__ == "__main__":
    unittest.main()
