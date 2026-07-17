"""Maps recognized tool calls to Home Assistant intents.

Tools are auto-generated from the intent metadata (``intent_tools``). Each tool
name is either a short, generation-friendly alias for one Home Assistant intent
(``TOOL_NAMES``) or one of the split sub-tools (``TOOL_OVERRIDES``) that fold
back into a single intent (open/close -> HassTurnOn/HassTurnOff, ...). Slots
pass through unchanged. Tool names that are already intent names -- e.g. from a
manual ``--tools`` override file -- pass through as-is.
"""

from typing import Any, Dict, Tuple

from intent_tools import MEDIA_INTENTS, TOOL_TO_INTENT

# The model occasionally fills a media tool's `name` with a generic noun from
# the utterance ("pause the music" -> name="music"). Under live HA the name
# enum anchors it to real players; scrub the known generics so the no-enum
# case degrades to "whatever is currently playing" instead of a failed entity
# match in Home Assistant.
_GENERIC_MEDIA_NAMES = {"music", "the music", "media", "song", "the song", "audio"}


def tool_to_intent(
    tool_name: str, tool_args: Dict[str, Any]
) -> Tuple[str, Dict[str, Any]]:
    intent_name = TOOL_TO_INTENT.get(tool_name, tool_name)

    if intent_name in MEDIA_INTENTS:
        name = tool_args.get("name")
        if isinstance(name, str) and name.strip().lower() in _GENERIC_MEDIA_NAMES:
            tool_args = {k: v for k, v in tool_args.items() if k != "name"}

    # The invented increase_/decrease_volume aliases carry their direction in
    # the tool name; fold it into volume_step's sign.
    if tool_name in ("increase_volume", "decrease_volume"):
        step = str(tool_args.get("volume_step") or "").lstrip("+-")
        if not step.isdigit():
            step = "up" if tool_name == "increase_volume" else "down"
        elif tool_name == "decrease_volume":
            step = f"-{step}"
        tool_args = {**tool_args, "volume_step": step}

    return (intent_name, tool_args)
