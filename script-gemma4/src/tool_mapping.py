"""Map model tool calls back to Home Assistant script calls."""

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Set

from hass_api import Tool

_LOGGER = logging.getLogger(__name__)


@dataclass
class MappedCall:
    """One tool call resolved against the tool that produced it."""

    tool_id: str
    # Variables to pass to the script, with names resolved to ids.
    variables: Dict[str, Any] = field(default_factory=dict)
    # Field -> value the model produced that is not in that field's enum.
    unresolved: Dict[str, Any] = field(default_factory=dict)
    # Unresolved fields that the script marks as required.
    missing_required: List[str] = field(default_factory=list)

    @property
    def script_id(self) -> str:
        return f"script.{self.tool_id}"

    @property
    def can_run(self) -> bool:
        """False when a field the script requires could not be resolved."""
        return not self.missing_required


def required_fields(tool: Tool) -> Set[str]:
    """Field names the script marks as required."""
    params = (tool.tool.get("function") or {}).get("parameters") or {}
    return set(params.get("required") or [])


def map_tool_call(tool: Tool, tool_args: Dict[str, Any]) -> MappedCall:
    """Resolve the names a model produced into Home Assistant ids.

    Fields with no name map (free text, numbers, dates, ...) pass through
    unchanged. Fields with a name map hold a display name that has to resolve to
    an entity/area/floor id, and enums steer generation without constraining it,
    so the model can invent a name that is not in the list. An unresolvable name
    is dropped rather than passed along: forwarding the display name would make
    the script fail on its own target, and forwarding ``None`` would look to the
    script like a deliberately empty value.
    """
    call = MappedCall(tool_id=tool.name)
    required = required_fields(tool)

    for var_key, var_value in tool_args.items():
        # An enum and its name map are always generated together, so an empty
        # map means this field is not name-mapped at all.
        name_map = tool.name_map.get(var_key)
        if not name_map:
            call.variables[var_key] = var_value
            continue

        mapped_value = name_map.get(var_value)
        if mapped_value is None:
            _LOGGER.debug(
                "No mapping for %s=%r in tool %s", var_key, var_value, tool.name
            )
            call.unresolved[var_key] = var_value
            if var_key in required:
                call.missing_required.append(var_key)
            continue

        _LOGGER.debug("Mapping '%s' -> '%s'", var_value, mapped_value)
        call.variables[var_key] = mapped_value

    call.missing_required.sort()
    return call
