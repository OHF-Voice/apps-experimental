"""Command matcher that uses sentence transformers (fuzzy)."""

import datetime
import itertools
import logging
import re
from collections.abc import Iterable
from dataclasses import dataclass, replace
from enum import Enum, auto
from typing import Any, Collection, Dict, List, Optional, Set, Union

import voluptuous as vol

_LOGGER = logging.getLogger()


# ---------------------------------------------------------------------------

IntentSchema = vol.Any(
    str,
    {
        vol.Required("name"): str,
        vol.Optional("language"): str,
        vol.Optional("slots"): {str: vol.Any(str, int, float, bool)},
    },
)


ActionSchema = {
    vol.Required("action"): str,
    vol.Optional("target"): {
        vol.Optional("entity_id"): vol.Any(str, [str]),
        vol.Optional("area_id"): vol.Any(str, [str]),
        vol.Optional("floor_id"): vol.Any(str, [str]),
    },
    vol.Optional("data"): {str: object},
}

ErrorsSchema = {
    vol.Optional("unknown_command"): str,
}

CommandSchema = {
    vol.Required("id"): str,
    vol.Optional("intent"): IntentSchema,
    vol.Optional("action"): ActionSchema,
    vol.Optional("description"): str,
    vol.Optional("response"): str,
}


ToolsAndCommandsSchema = vol.Schema(
    {
        vol.Required("commands"): [CommandSchema],
        # OpenAI function schema
        vol.Required("tools"): [dict],
    },
    # extra=vol.ALLOW_EXTRA,
)

# -----------------------------------------------------------------------------


@dataclass
class CommandIntent:
    name: str
    language: Optional[str] = None
    slots: Optional[Dict[str, Any]] = None


@dataclass
class CommandAction:
    action: str
    target: Optional[Dict[str, Any]] = None
    data: Optional[Dict[str, Any]] = None


@dataclass
class Command:
    id: str
    intent: Optional[CommandIntent] = None
    action: Optional[CommandAction] = None
    description: Optional[str] = None
    response: Optional[str] = None

    @staticmethod
    def from_dict(command_dict: Dict[str, Any]) -> "Command":
        # Parse intent
        intent: Optional[CommandIntent] = None
        intent_value = command_dict.get("intent")
        if intent_value:
            if isinstance(intent_value, str):
                intent = CommandIntent(name=intent_value)
            else:
                intent = CommandIntent(
                    name=intent_value["name"],
                    language=intent_value.get("language"),
                    slots=intent_value.get("slots"),
                )

        # Parse action
        action: Optional[CommandAction] = None
        action_value = command_dict.get("action")
        if action_value:
            if isinstance(action_value, str):
                action = CommandAction(action=action_value)
            else:
                action = CommandAction(
                    action=action_value["action"],
                    target=action_value.get("target"),
                    data=action_value.get("data"),
                )

        return Command(
            id=command_dict["id"],
            description=command_dict.get("description"),
            intent=intent,
            action=action,
            response=command_dict.get("response"),
        )
