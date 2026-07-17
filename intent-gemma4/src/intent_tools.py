"""Generate the LLM tool set from the OHF intent metadata + Home Assistant state.

Most *supported* intents in ``intents.yaml`` become one OpenAI-style function
tool. Tool names are short snake_case aliases (``TOOL_NAMES``) rather than the
``HassCamelCase`` intent names: latency is generation-bound and the tool name is
emitted in every tool call, so short names that tokenize well are a direct
latency win. A few broad intents are additionally split into several
language-specific tools that all map back to the same intent (see
``TOOL_OVERRIDES``) -- e.g. ``HassTurnOn`` becomes ``turn_on``/``open``/``lock``
because "turn on the light", "open the blinds" and "lock the door" are phrased
very differently even though Home Assistant handles them with one intent.
``TOOL_TO_INTENT`` maps every generated tool name back to its intent.

The tool set is minimized for a specific home:

1. Domain gating (via ``hassil.filter_intents`` at the intent level, then a
   per-tool domain check): a tool is dropped when it is scoped to entity domains
   and none of those domains has an exposed entity.
2. Enum injection / trimming: ``name``/``domain``/``device_class`` are
   constrained to the exposed entity names (by the tool's domains), the tool's
   inferred domains, and the device classes actually present -- and the slot is
   dropped when its enum would be empty. ``area``/``floor`` enumerate all areas
   and floors. All include aliases.

Without Home Assistant info, all supported intents are emitted with free-string
name/area/floor/device_class slots (useful for local development).
"""

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, FrozenSet, List, Optional, Set, Union

from hassil import filter_intents
from hassil.intents import Intent, IntentData, Intents
from ruamel.yaml import YAML

from hass_api import HomeInfo

_LOGGER = logging.getLogger(__name__)

_IMPORTANCES = ("required", "usable", "complete", "optional")

MEDIA_CLASSES = [
    "album", "app", "artist", "channel", "composer", "contributing_artist",
    "directory", "episode", "game", "genre", "image", "movie", "music",
    "playlist", "podcast", "season", "track", "tv_show", "url", "video",
]

# Type/range for slots whose JSON schema can't be derived from the metadata.
# Descriptions are overridden by the per-intent slot description when present.
STATIC_SLOT_SCHEMAS: Dict[str, Dict[str, Any]] = {
    "brightness": {"type": "integer", "minimum": 0, "maximum": 100},
    "temperature": {"type": "number"},
    "color": {"type": "string"},
    "position": {"type": "integer", "minimum": 0, "maximum": 100},
    "percentage": {"type": "integer", "minimum": 0, "maximum": 100},
    "volume_level": {"type": "integer", "minimum": 0, "maximum": 100},
    "volume_step": {
        "type": "string",
        "description": "'up', 'down', or a percentage to change by, e.g. '20'",
    },
    "state": {
        "type": "string",
        "description": "State to check for, e.g. on, off, open, closed, locked, home",
    },
    "device_class": {
        "type": "string",
        "description": (
            "Device class, e.g. window, blind, curtain, shade, garage, door, "
            "shutter, awning, gate"
        ),
    },
    "item": {"type": "string"},
    "search_query": {"type": "string"},
    "media_class": {"type": "string", "enum": MEDIA_CLASSES},
    "message": {"type": "string"},
    "response": {"type": "string"},
    "hours": {"type": "integer", "minimum": 0},
    "minutes": {"type": "integer", "minimum": 0},
    "seconds": {"type": "integer", "minimum": 0},
    "start_hours": {"type": "integer", "minimum": 0},
    "start_minutes": {"type": "integer", "minimum": 0},
    "start_seconds": {"type": "integer", "minimum": 0},
    "conversation_command": {"type": "string"},
}


# -----------------------------------------------------------------------------
# Tuning knobs
# -----------------------------------------------------------------------------

MEDIA_INTENTS = (
    "HassMediaPause",
    "HassMediaUnpause",
    "HassMediaNext",
    "HassMediaPrevious",
    "HassMediaPlayerMute",
    "HassMediaPlayerUnmute",
    "HassSetVolume",
    "HassSetVolumeRelative",
    "HassMediaSearchAndPlay",
)

_MEDIA_NAME_DESCRIPTION = (
    "Name of a specific media player to target (e.g. 'Living Room TV'). Leave "
    "unset to control whatever is currently playing; never use a generic word "
    "like 'music', 'media', or 'song' as the name."
)

# Per-(intent, slot) description overrides, applied over the metadata
# description (an intent's override applies to all of its sub-tools).
# Media `name`: the model otherwise fills generic nouns ("pause the music" ->
# name="music"); steer it to real player names. (Excluding the slot instead was
# worse -- it perturbed the prefix and made transport controls route to Mute.)
SLOT_DESCRIPTION_OVERRIDES: Dict[tuple, str] = {
    (intent, "name"): _MEDIA_NAME_DESCRIPTION for intent in MEDIA_INTENTS
}

# Per-tool (function) description overrides, applied over the metadata
# description for non-split intents. HassSetPosition otherwise loses
# "set the curtains to 50%" to the `close` tool, so spell out that it sets a
# specific position.
DESCRIPTION_OVERRIDES: Dict[str, str] = {
    "HassSetPosition": (
        "Set a cover or valve to a specific position / percentage open "
        "(0 = fully closed, 100 = fully open), e.g. 'set the curtains to 50%' "
        "or 'open the blinds halfway'. Use this whenever a position or "
        "percentage is given, instead of the open/close tools."
    ),
}

# Short, generation-friendly tool names for the non-split intents. Latency is
# generation-bound, so a 2-token name beats an 8-token HassCamelCase name in
# every tool call. Keep names unique across this dict + TOOL_OVERRIDES; timer
# tools keep a _timer suffix so generic verbs (pause/cancel) stay with media
# and HassNevermind. Intents without an entry keep their intent name (e.g.
# tools from a manual --tools override file).
TOOL_NAMES: Dict[str, str] = {
    "HassGetState": "get_state",
    "HassNevermind": "nevermind",
    "HassSetPosition": "set_position",
    "HassGetCurrentDate": "get_date",
    "HassGetCurrentTime": "get_time",
    "HassLightSet": "set_light",
    "HassClimateSetTemperature": "set_temperature",
    "HassClimateGetTemperature": "get_temperature",
    "HassGetWeather": "get_weather",
    "HassListAddItem": "add_item",
    "HassListCompleteItem": "complete_item",
    "HassListRemoveItem": "remove_item",
    "HassMediaPause": "pause",
    "HassMediaUnpause": "resume",
    "HassMediaNext": "next_track",
    "HassMediaPrevious": "previous_track",
    "HassSetVolume": "set_volume",
    "HassMediaPlayerMute": "mute",
    "HassMediaPlayerUnmute": "unmute",
    "HassSetVolumeRelative": "change_volume",
    "HassMediaSearchAndPlay": "play_media",
    # "set_timer" over "start_timer": matches the dominant "set a timer for..."
    # phrasing, and "start_" otherwise attracts "run/start X" utterances that
    # belong to turn_on (scripts) and start_vacuum.
    "HassStartTimer": "set_timer",
    "HassCancelTimer": "cancel_timer",
    "HassIncreaseTimer": "increase_timer",
    "HassDecreaseTimer": "decrease_timer",
    "HassPauseTimer": "pause_timer",
    "HassUnpauseTimer": "resume_timer",
    "HassTimerStatus": "timer_status",
    "HassFanSetSpeed": "set_fan_speed",
}

# Off-prompt aliases: tool names the model predictably invents by analogy with
# the real ones (compositional snake_case). Never shown in the prompt, but
# accepted on the way back so a near-miss call still maps to the right intent.
# increase_/decrease_volume also get their direction folded into volume_step in
# intent_mapping.tool_to_intent.
TOOL_ALIASES: Dict[str, str] = {
    "start_timer": "HassStartTimer",
    "increase_volume": "HassSetVolumeRelative",
    "decrease_volume": "HassSetVolumeRelative",
}

_TIMER_INTENTS = (
    "HassStartTimer",
    "HassCancelTimer",
    "HassPauseTimer",
    "HassUnpauseTimer",
    "HassIncreaseTimer",
    "HassDecreaseTimer",
    "HassTimerStatus",
)

# Per-(intent, slot) pairs to omit from the generated tool(s) entirely.
#
# `name` is dropped where the model mis-extracts a generic domain noun as the
# entity name (e.g. "set a timer..." -> name="timer", "what is the weather" ->
# name="weather") and naming/targeting by name is uncommon: every timer intent,
# plus HassGetWeather.
#
# `start_*` (targeting a timer by its original duration, "the 10 minute
# timer"): the model never fills these -- 0% across every model/quant in the
# coverage runs -- so they are dead schema weight; bare timer-edit tools work
# for the common one-timer case.
#
# `conversation_command` (deferred commands, "turn off the lights in 10
# minutes"): never filled correctly (the model emits a second immediate tool
# call instead), and its description attracts "run X" utterances into
# set_timer.
#
# `media_class`: all its combinations are optional and its 20-item enum is the
# single largest slot in the prefix; "play Pink Floyd playlist" degrades fine
# to a plain search_query.
_TIMER_START_SLOTS = ("start_hours", "start_minutes", "start_seconds")
SLOT_EXCLUDES: Set[tuple] = (
    {(intent, "name") for intent in _TIMER_INTENTS}
    | {
        (intent, slot)
        for intent in _TIMER_INTENTS
        for slot in _TIMER_START_SLOTS
    }
    | {
        ("HassGetWeather", "name"),
        ("HassStartTimer", "conversation_command"),
        ("HassMediaSearchAndPlay", "media_class"),
    }
)

# Intents to omit entirely (never exposed as tools).
INTENT_EXCLUDES: Set[str] = {
    "HassCancelAllTimers",
    # Open-ended intents that the model fires on arbitrary/declarative speech.
    "HassBroadcast",
    "HassRespond",
    # Shopping list is handled via the todo intents (HassList*); todo only.
    "HassShoppingListAddItem",
    "HassShoppingListCompleteItem",
    # Out of scope for v1. The vacuum metadata is also self-ambiguous ("vacuum
    # the kitchen" is the example for both VacuumStart and VacuumCleanArea).
    "HassVacuumStart",
    "HassVacuumReturnToBase",
    "HassVacuumCleanArea",
    "HassLawnMowerStartMowing",
    "HassLawnMowerDock",
}


@dataclass(frozen=True)
class _ToolSpec:
    """One generated tool: how it is shown to the model + how it maps back."""

    tool_name: str
    intent_name: str
    description: str
    slots: FrozenSet[str]
    name_domains: FrozenSet[str] = frozenset()
    inferred_domains: FrozenSet[str] = frozenset()
    device_class_domains: FrozenSet[str] = frozenset()

    @property
    def gate_domains(self) -> Set[str]:
        """Domains that must have an exposed entity for this tool to apply."""
        return (
            set(self.name_domains)
            | set(self.inferred_domains)
            | set(self.device_class_domains)
        )


# Broad intents split into language-specific tools. Each tool maps back to the
# intent_name; the domains partition the intent's domains across the tools so
# enum injection and gating stay correct (e.g. `open` only sees cover/valve
# entities). Keep tool_name in sync with intent_mapping.SUBTOOL_TO_INTENT.
TOOL_OVERRIDES: Dict[str, List[_ToolSpec]] = {
    "HassTurnOn": [
        _ToolSpec(
            tool_name="turn_on",
            intent_name="HassTurnOn",
            description="Turn on, start, or activate a device, scene, or script",
            slots=frozenset({"name", "area", "floor", "domain"}),
            name_domains=frozenset(
                {"light", "switch", "fan", "media_player", "input_boolean",
                 "scene", "script"}
            ),
            inferred_domains=frozenset({"light", "fan"}),
        ),
        _ToolSpec(
            tool_name="open",
            intent_name="HassTurnOn",
            description=(
                "Fully open a cover such as blinds, curtains, shades, a garage "
                "door, or a valve (for a specific position use set_position)"
            ),
            slots=frozenset({"name", "area", "floor", "domain", "device_class"}),
            name_domains=frozenset({"cover", "valve"}),
            inferred_domains=frozenset({"cover"}),
            device_class_domains=frozenset({"cover", "valve"}),
        ),
        _ToolSpec(
            tool_name="lock",
            intent_name="HassTurnOn",
            description="Lock a lock",
            slots=frozenset({"name", "area", "floor"}),
            name_domains=frozenset({"lock"}),
        ),
    ],
    "HassTurnOff": [
        _ToolSpec(
            tool_name="turn_off",
            intent_name="HassTurnOff",
            description="Turn off, stop, or deactivate a device",
            slots=frozenset({"name", "area", "floor", "domain"}),
            name_domains=frozenset(
                {"light", "switch", "fan", "media_player", "input_boolean"}
            ),
            inferred_domains=frozenset({"light", "fan"}),
        ),
        _ToolSpec(
            tool_name="close",
            intent_name="HassTurnOff",
            description=(
                "Fully close a cover such as blinds, curtains, shades, a garage "
                "door, or a valve (for a specific position use set_position)"
            ),
            slots=frozenset({"name", "area", "floor", "domain", "device_class"}),
            name_domains=frozenset({"cover", "valve"}),
            inferred_domains=frozenset({"cover"}),
            device_class_domains=frozenset({"cover", "valve"}),
        ),
        _ToolSpec(
            tool_name="unlock",
            intent_name="HassTurnOff",
            description="Unlock a lock",
            slots=frozenset({"name", "area", "floor"}),
            name_domains=frozenset({"lock"}),
        ),
    ],
}

# tool name -> Home Assistant intent, for every generated tool (imported by
# intent_mapping so the mapping has a single source of truth).
TOOL_TO_INTENT: Dict[str, str] = {
    spec.tool_name: spec.intent_name
    for specs in TOOL_OVERRIDES.values()
    for spec in specs
}
TOOL_TO_INTENT.update({tool: intent for intent, tool in TOOL_NAMES.items()})
assert len(TOOL_TO_INTENT) == len(TOOL_NAMES) + sum(
    len(specs) for specs in TOOL_OVERRIDES.values()
), "Tool name collision between TOOL_NAMES and TOOL_OVERRIDES"
assert not set(TOOL_ALIASES) & set(TOOL_TO_INTENT), "Alias shadows a real tool"
TOOL_TO_INTENT.update(TOOL_ALIASES)


# -----------------------------------------------------------------------------


def load_metadata(path: Union[str, Path]) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as metadata_file:
        return YAML(typ="safe").load(metadata_file) or {}


def unknown_tool_names(
    tools: List[Dict[str, Any]], metadata: Dict[str, Any]
) -> List[str]:
    """Tool names that do not map to a supported Home Assistant intent.

    Used to reject stale manual override files (e.g. saved by an older version
    whose generic tools no longer exist) instead of silently mis-mapping them.
    """
    valid = set(TOOL_TO_INTENT) | {
        name for name, info in metadata.items() if info.get("supported")
    }
    names = {
        str((tool.get("function") or {}).get("name") or "?") for tool in tools
    }
    return sorted(names - valid)


def _collect_domains(info: Dict[str, Any], key: str) -> Set[str]:
    """Union of inferred_domains/name_domains across an intent's combinations."""
    result: Set[str] = set()
    for comb in (info.get("slot_combinations") or {}).values():
        groups = comb.get(key) or {}
        for importance in _IMPORTANCES:
            for domain in groups.get(importance) or []:
                result.add(domain)
    return result


def _name_domains(info: Dict[str, Any]) -> Set[str]:
    return _collect_domains(info, "name_domains")


def _inferred_domains(info: Dict[str, Any]) -> Set[str]:
    return _collect_domains(info, "inferred_domains")


def _gate_domains(info: Dict[str, Any]) -> Set[str]:
    """Entity domains the intent is scoped to (empty = always applicable)."""
    return _name_domains(info) | _inferred_domains(info)


def _slots_used(info: Dict[str, Any]) -> Set[str]:
    slots: Set[str] = set()
    for comb in (info.get("slot_combinations") or {}).values():
        for slot in comb.get("slots") or []:
            slots.add(slot)
    return slots


def _default_spec(intent_name: str, info: Dict[str, Any]) -> _ToolSpec:
    """The single passthrough tool for an intent without a TOOL_OVERRIDES entry."""
    name_domains = frozenset(_name_domains(info))
    inferred_domains = frozenset(_inferred_domains(info))
    slots = frozenset(_slots_used(info))
    return _ToolSpec(
        tool_name=TOOL_NAMES.get(intent_name, intent_name),
        intent_name=intent_name,
        description=DESCRIPTION_OVERRIDES.get(intent_name, info.get("description", "")),
        slots=slots,
        name_domains=name_domains,
        inferred_domains=inferred_domains,
        device_class_domains=(
            name_domains | inferred_domains if "device_class" in slots else frozenset()
        ),
    )


def _specs_for(intent_name: str, info: Dict[str, Any]) -> List[_ToolSpec]:
    return TOOL_OVERRIDES.get(intent_name) or [_default_spec(intent_name, info)]


def _viable_intents(
    metadata: Dict[str, Any], available_domains: Optional[Set[str]]
) -> Set[str]:
    """Use hassil's filter to decide which intents are applicable to this home."""
    intents_map: Dict[str, Intent] = {}
    for name, info in metadata.items():
        if not info.get("supported"):
            continue
        gate = sorted(_gate_domains(info))
        data = IntentData(
            sentence_texts=[],
            slots=({"domain": gate} if gate else {}),
        )
        intents_map[name] = Intent(name=name, data=[data])

    intents = Intents(language="en", intents=intents_map)
    if available_domains is None:
        return set(intents_map)

    filtered = filter_intents(
        intents, domains=available_domains, keep_domainless=True
    )
    return set(filtered.intents)


def _slot_schema(
    spec: _ToolSpec,
    slot: str,
    slot_meta: Dict[str, Any],
    home_info: Optional[HomeInfo],
    available_domains: Optional[Set[str]],
) -> Optional[Dict[str, Any]]:
    """Build the JSON schema for one slot, or None to omit it."""
    if (spec.intent_name, slot) in SLOT_EXCLUDES:
        return None

    description = SLOT_DESCRIPTION_OVERRIDES.get(
        (spec.intent_name, slot), slot_meta.get("description")
    )

    if slot == "name":
        if home_info is None:
            schema: Dict[str, Any] = {"type": "string"}
        else:
            domains = set(spec.name_domains)
            if available_domains is not None:
                domains &= available_domains
            names = home_info.entity_names(domains or None)
            if not names:
                return None
            schema = {"type": "string", "enum": names}
    elif slot == "area":
        if home_info is None:
            schema = {"type": "string"}
        else:
            area_names = home_info.area_names()
            if not area_names:
                # No areas defined: trim the slot instead of leaving a free
                # string the model could hallucinate into.
                return None
            schema = {"type": "string", "enum": area_names}
    elif slot == "floor":
        if home_info is None:
            schema = {"type": "string"}
        else:
            floor_names = home_info.floor_names()
            if not floor_names:
                return None
            schema = {"type": "string", "enum": floor_names}
    elif slot == "domain":
        domains = set(spec.inferred_domains)
        if available_domains is not None:
            domains &= available_domains
        if domains:
            schema = {"type": "string", "enum": sorted(domains)}
        elif available_domains is None and spec.inferred_domains:
            schema = {"type": "string", "enum": sorted(spec.inferred_domains)}
        else:
            return None
    elif slot == "device_class":
        dc_description = SLOT_DESCRIPTION_OVERRIDES.get(
            (spec.tool_name, slot), STATIC_SLOT_SCHEMAS["device_class"]["description"]
        )
        if home_info is None:
            return {"type": "string", "description": dc_description}
        domains = set(spec.device_class_domains)
        if available_domains is not None:
            domains &= available_domains
        classes = home_info.device_classes(domains or None)
        if not classes:
            # No exposed entity in scope has a device class: trim the slot.
            return None
        return {"type": "string", "enum": classes, "description": dc_description}
    else:
        base = STATIC_SLOT_SCHEMAS.get(slot)
        if base is None:
            _LOGGER.debug("No schema for slot '%s'; using free string", slot)
            schema = {"type": "string"}
        else:
            schema = dict(base)

    if description:
        schema["description"] = description

    return schema


def build_tools(
    metadata: Dict[str, Any], home_info: Optional[HomeInfo] = None
) -> List[Dict[str, Any]]:
    """Generate the filtered, enum-injected OpenAI tool list."""
    available_domains = home_info.available_domains if home_info is not None else None
    viable = _viable_intents(metadata, available_domains)

    dropped: List[str] = []
    tools: List[Dict[str, Any]] = []
    for name, info in metadata.items():
        if not info.get("supported"):
            continue
        if name in INTENT_EXCLUDES:
            continue
        if name not in viable:
            dropped.append(name)
            continue

        slot_meta = info.get("slots") or {}
        for spec in _specs_for(name, info):
            gate = spec.gate_domains
            if available_domains is not None and gate and not (gate & available_domains):
                # This (sub-)tool's domains aren't present in the home.
                continue

            props: Dict[str, Any] = {}
            required: List[str] = []
            for slot in sorted(spec.slots):
                schema = _slot_schema(
                    spec, slot, slot_meta.get(slot) or {}, home_info, available_domains
                )
                if schema is None:
                    continue
                props[slot] = schema
                if (slot_meta.get(slot) or {}).get("required"):
                    required.append(slot)

            # Always include a parameters object (empty when no slots): some chat
            # handlers (e.g. chatml-function-calling) render
            # `parameters | tojson` and choke on a missing key.
            parameters: Dict[str, Any] = {"type": "object", "properties": props}
            if required:
                parameters["required"] = sorted(required)

            function: Dict[str, Any] = {
                "name": spec.tool_name,
                "description": spec.description,
                "parameters": parameters,
            }
            tools.append({"type": "function", "function": function})

    _LOGGER.info(
        "Built %s tool(s); dropped %s intent(s) with no available domain: %s",
        len(tools),
        len(dropped),
        sorted(dropped),
    )
    return tools
