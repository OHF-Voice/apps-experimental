"""Maps recognized tool calls to Home Assistant intents.

The model is prompted with generic, easy-to-call tools (e.g. ``start_timer``,
``control_media``). This module translates each tool call into the
corresponding Home Assistant intent name + slots that get emitted as a Wyoming
``Intent`` event. Tools whose names are already Home Assistant intents (e.g.
``HassGetWeather``) pass through unchanged.
"""

from typing import Any, Dict, Tuple


def _duration_slots(
    total_seconds: int,
    hours_key: str = "hours",
    minutes_key: str = "minutes",
    seconds_key: str = "seconds",
) -> Dict[str, Any]:
    slots: Dict[str, Any] = {}
    hours = total_seconds // 3600
    if hours > 0:
        slots[hours_key] = hours
    minutes = (total_seconds % 3600) // 60
    if minutes > 0:
        slots[minutes_key] = minutes
    seconds = total_seconds % 60
    if seconds > 0:
        slots[seconds_key] = seconds

    return slots


def tool_to_intent(
    tool_name: str, tool_args: Dict[str, Any]
) -> Tuple[str, Dict[str, Any]]:
    if tool_name == "start_timer":
        total_seconds = tool_args.get("total_seconds", 0)
        timer_args = _duration_slots(total_seconds)

        timer_name = tool_args.get("name")
        if timer_name:
            timer_args["name"] = timer_name

        return ("HassStartTimer", timer_args)

    if tool_name == "control_timer":
        timer_action = tool_args.get("action")
        if timer_action == "cancel_all":
            return ("HassCancelAllTimers", {})

        timer_args: Dict[str, Any] = {}
        total_seconds = tool_args.get("total_seconds")
        if total_seconds is not None:
            timer_args = _duration_slots(
                total_seconds,
                hours_key="start_hours",
                minutes_key="start_minutes",
                seconds_key="start_seconds",
            )

        timer_name = tool_args.get("name")
        if timer_name:
            timer_args["name"] = timer_name

        if timer_action == "pause":
            return ("HassPauseTimer", timer_args)

        if timer_action == "resume":
            return ("HassUnpauseTimer", timer_args)

        if timer_action == "cancel":
            return ("HassCancelTimer", timer_args)

        raise ValueError(f"Unexpected action for control_timer: {timer_action}")

    if tool_name == "update_timer":
        timer_action = tool_args.get("action")
        total_update_seconds = tool_args.get("total_update_seconds", 0)
        timer_args = _duration_slots(total_update_seconds)

        timer_name = tool_args.get("name")
        if timer_name:
            timer_args["name"] = timer_name

        if timer_action == "add":
            return ("HassIncreaseTimer", timer_args)

        if timer_action == "remove":
            return ("HassDecreaseTimer", timer_args)

        raise ValueError(f"Unexpected action for update_timer: {timer_action}")

    if tool_name == "control_media":
        media_action = tool_args.get("action")
        if media_action == "pause":
            return ("HassMediaPause", {})

        if media_action == "resume":
            return ("HassMediaUnpause", {})

        if media_action == "next":
            return ("HassMediaNext", {})

        raise ValueError(f"Unexpected action for control_media: {media_action}")

    return (tool_name, tool_args)
