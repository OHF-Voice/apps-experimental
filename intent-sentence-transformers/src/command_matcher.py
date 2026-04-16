"""Command matcher that uses sentence transformers (fuzzy)."""

import datetime
import logging
from dataclasses import dataclass
from enum import Enum, auto
from typing import Any, Collection, Dict, List, Optional, Union

import lingua_franca
import numpy as np
import voluptuous as vol
from lingua_franca.parse import extract_duration, extract_number
from sentence_transformers import SentenceTransformer

# Commands with scores below are rejected.
DEFAULT_THRESHOLD = 0.9

# If the two command scores' difference is below this margin, the match is
# rejected.
DEFAULT_MARGIN = 0.015

_LOGGER = logging.getLogger()


# ---------------------------------------------------------------------------

IntentSchema = vol.Any(
    str,
    {
        vol.Required("name"): str,
        vol.Optional("slots"): {str: vol.Any(str, int, float, bool)},
    },
)

DurationSchema = vol.Any(
    str,
    {
        vol.Optional("seconds_slot"): str,
        vol.Optional("minutes_slot"): str,
        vol.Optional("hours_slot"): str,
    },
)

PercentageSchema = {
    vol.Required("slot"): str,
}

CommandSchema = {
    vol.Required("id"): str,
    vol.Required("sentences"): [str],
    vol.Optional("intent"): IntentSchema,
    vol.Optional("description"): str,
    vol.Optional("threshold"): float,
    vol.Optional("margin"): float,
    vol.Optional("duration"): bool,
    vol.Optional("current_area"): bool,
    vol.Optional("percentage"): PercentageSchema,
    vol.Optional("response"): str,
    vol.Optional("hass_response"): str,
}


Schema = vol.Schema(
    {
        vol.Required("language"): str,
        vol.Required("commands"): [CommandSchema],
    },
    # extra=vol.ALLOW_EXTRA,
)

# -----------------------------------------------------------------------------

for module in ("sentence_transformers", "transformers", "torch"):
    logging.getLogger(module).setLevel(logging.WARNING)


class ParseError(Exception):
    pass


@dataclass
class CurrentAreaInfo:
    slot: str = "area"


@dataclass
class DurationInfo:
    seconds_slot: str = "seconds"
    minutes_slot: str = "minutes"
    hours_slot: str = "hours"


@dataclass
class PercentageInfo:
    slot: str


@dataclass
class CommandIntent:
    name: str
    slots: Optional[Dict[str, Any]] = None


@dataclass
class CommandAction:
    action: str
    target: Optional[Dict[str, Any]] = None
    data: Optional[Dict[str, Any]] = None


# ---------------------------------------------------------------------------


@dataclass
class Command:
    id: str
    sentences: List[str]
    intent: Optional[CommandIntent] = None
    action: Optional[CommandAction] = None
    description: Optional[str] = None
    current_area: Optional[CurrentAreaInfo] = None
    duration: Optional[DurationInfo] = None
    percentage: Optional[PercentageInfo] = None
    response: Optional[str] = None
    hass_response: Optional[str] = None
    score_threshold: Optional[float] = None
    score_margin: Optional[float] = None

    @staticmethod
    def from_dict(command_dict: Dict[str, Any]) -> "Command":
        current_area: Optional[CurrentAreaInfo] = None
        duration: Optional[DurationInfo] = None
        percentage: Optional[PercentageInfo] = None

        current_area_value = command_dict.get("current_area")
        if current_area_value is True:
            # Defaults
            current_area = CurrentAreaInfo()
        elif current_area_value:
            current_area = CurrentAreaInfo(slot=current_area_value["slot"])

        duration_value = command_dict.get("duration")
        if duration_value is True:
            # Defaults
            duration = DurationInfo()
        elif duration_value:
            duration = DurationInfo(
                seconds_slot=duration_value.get("seconds_slot", "seconds"),
                minutes_slot=duration_value.get("minutes_slot", "minutes"),
                hours_slot=duration_value.get("hours_slot", "hours"),
            )

        percentage_value = command_dict.get("percentage")
        if percentage_value:
            percentage = PercentageInfo(slot=percentage_value["slot"])

        # Parse intent
        intent: Optional[CommandIntent] = None
        intent_value = command_dict.get("intent")
        if intent_value:
            if isinstance(intent_value, str):
                intent = CommandIntent(name=intent_value)
            else:
                intent = CommandIntent(
                    name=intent_value["name"], slots=intent_value.get("slots")
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
            sentences=command_dict["sentences"],
            intent=intent,
            action=action,
            current_area=current_area,
            duration=duration,
            percentage=percentage,
            response=command_dict.get("response"),
            hass_response=command_dict.get("hass_response"),
            score_threshold=command_dict.get("score_threshold"),
            score_margin=command_dict.get("score_margin"),
        )


@dataclass
class CommandMatch:
    command: Command
    slots: Optional[Dict[str, Any]] = None

    score: Optional[float] = None
    margin: Optional[float] = None
    """How close the best command was to the second best."""


class CommandMatchFailureReason(Enum):
    """Reason for command match failure."""

    NO_COMMANDS = auto()
    NO_MATCH = auto()
    BELOW_THRESHOLD = auto()
    BELOW_MARGIN = auto()


@dataclass
class CommandMatchFailure:
    reason: CommandMatchFailureReason
    best_command: Optional[Command] = None
    best_score: Optional[float] = None
    second_best_command: Optional[Command] = None
    second_best_score: Optional[float] = None
    threshold: Optional[float] = None
    min_margin: Optional[float] = None
    margin: Optional[float] = None

    def to_string(self) -> str:
        text = ""
        if self.reason == CommandMatchFailureReason.NO_COMMANDS:
            text += "no commands"
        elif self.reason == CommandMatchFailureReason.NO_MATCH:
            text += "no match"
        elif self.reason == CommandMatchFailureReason.BELOW_THRESHOLD:
            text += "matched command was below threshold"
            if self.best_command:
                text += f", command={self.best_command.id}"
            if self.best_score is not None:
                text += f", score={self.best_score}"
            if self.threshold is not None:
                text += f", threshold={self.threshold}"
        elif self.reason == CommandMatchFailureReason.BELOW_MARGIN:
            text += "matched command was too close to another command"
            if self.best_command:
                text += f", command={self.best_command.id}"
            if self.second_best_command:
                text += f", other_command={self.second_best_command.id}"
            if self.margin is not None:
                text += f", margin={self.margin}"
            if self.min_margin is not None:
                text += f", min_margin={self.min_margin}"

        return text


# ---------------------------------------------------------------------------


class CommandMatcher:
    def __init__(self, model: SentenceTransformer) -> None:
        self.model = model
        self.commands: List[Command] = []
        self.centroids: Optional[np.ndarray] = None

    def reset(self) -> None:
        self.commands: List[Command] = []
        self.centroids: Optional[np.ndarray] = None

    def add(self, command: Command) -> None:
        if not command.sentences:
            raise ValueError("Invalid command")

        # Encode all examples for this command
        # normalize_embeddings=True gives unit vectors out of the box,
        # but we'll L2-normalize again after averaging just to be safe.
        emb = self.model.encode(
            command.sentences,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )

        # Average to get a centroid for the command
        centroid = emb.mean(axis=0, keepdims=True)
        centroid = l2_normalize(centroid)  # re-normalize after averaging

        # Must match 1-to-1 with self.centroids
        self.commands.append(command)

        if self.centroids is None:
            self.centroids = centroid
        else:
            self.centroids = np.vstack((self.centroids, centroid[0]))

    def match(
        self,
        language: str,
        text: str,
        *,
        current_area_id: Optional[str] = None,
        threshold: float = DEFAULT_THRESHOLD,
        min_margin: float = DEFAULT_MARGIN,
        disabled_commands: Optional[Collection[str]] = None,
    ) -> Union[CommandMatch, CommandMatchFailure]:
        if (not self.commands) or (self.centroids is None):
            return CommandMatchFailure(CommandMatchFailureReason.NO_COMMANDS)

        _LOGGER.debug(
            "Matching '%s' (language=%s, threshold=%s, min_margin=%s, current_area_id=%s, disabled_commands=%s)",
            text,
            language,
            threshold,
            min_margin,
            current_area_id,
            disabled_commands,
        )

        # Encode query
        q = self.model.encode(
            [text],
            convert_to_numpy=True,
            normalize_embeddings=True,
        )  # shape (1, dim)

        # Cosine similarity = dot product since vectors are normalized
        # intent_centroids: (N, dim), q.T: (dim, 1) → sims: (N, 1)

        sims = (self.centroids @ q.T).squeeze(-1)  # shape (N,)
        order = np.argsort(-sims)  # descending

        for order_idx, best_idx in enumerate(order):
            best_score = float(sims[best_idx])
            best_command = self.commands[best_idx]

            if best_command.score_threshold is not None:
                # Command has specific threshold
                best_command_threshold = best_command.score_threshold
            else:
                best_command_threshold = threshold

            if best_score < best_command_threshold:
                # Below threshold
                _LOGGER.debug(
                    "Command was below threshold: text=%s, score=%s, threshold=%s",
                    text,
                    best_score,
                    best_command_threshold,
                )
                return CommandMatchFailure(
                    CommandMatchFailureReason.BELOW_THRESHOLD,
                    best_command=best_command,
                    best_score=best_score,
                    threshold=best_command_threshold,
                )

            if disabled_commands and (best_command.id in disabled_commands):
                # Command is disabled
                _LOGGER.debug("Matched '%s' but is was disabled", best_command.id)
                continue

            if best_command.current_area and (not current_area_id):
                # Command requires current area, which is not present
                _LOGGER.debug("Matched '%s' but no current area", best_command.id)
                continue

            # TODO: return error if number/duration missing
            match_slots: Dict[str, Any] = {}
            try:
                if best_command.duration:
                    match_slots.update(
                        parse_duration(language, text, best_command.duration)
                    )
                elif best_command.percentage:
                    match_slots.update(
                        parse_number(language, text, best_command.percentage.slot)
                    )
            except ParseError as err:
                _LOGGER.debug("Matched '%s' but %s", best_command.id, err)
                continue

            # Check how close this command was to the next one
            if best_command.score_margin is not None:
                # Command has specific margin
                best_min_margin = best_command.score_margin
            else:
                best_min_margin = min_margin

            margin: Optional[float] = None
            second_order_idx = order_idx + 1
            while second_order_idx < len(order):
                second_best_idx = order[second_order_idx]
                second_best_command = self.commands[second_best_idx]
                second_order_idx += 1
                if disabled_commands and (second_best_command.id in disabled_commands):
                    # Command is disabled
                    continue

                if second_best_command.current_area and (not current_area_id):
                    # Command requires current area, which is not present
                    continue

                second_best_score = float(sims[second_best_idx])
                margin = best_score - second_best_score
                if margin < best_min_margin:
                    # Too close to other command
                    _LOGGER.debug(
                        "Matched '%s' but it was too close to '%s' (margin=%s, min_margin=%s)",
                        best_command.id,
                        second_best_command.id,
                        margin,
                        best_min_margin,
                    )
                    return CommandMatchFailure(
                        CommandMatchFailureReason.BELOW_MARGIN,
                        best_command=best_command,
                        best_score=best_score,
                        second_best_command=second_best_command,
                        second_best_score=second_best_score,
                        threshold=best_command_threshold,
                        margin=margin,
                        min_margin=best_min_margin,
                    )

            # Run extractors
            command_match = CommandMatch(
                best_command, slots=match_slots, score=best_score, margin=margin
            )

            return command_match

        return CommandMatchFailure(CommandMatchFailureReason.NO_MATCH)


def l2_normalize(x: np.ndarray, axis: int = -1, eps: float = 1e-12) -> np.ndarray:
    norm = np.linalg.norm(x, axis=axis, keepdims=True)
    return x / np.maximum(norm, eps)


# ---------------------------------------------------------------------------

_LOADED_LANGS = set()


def parse_duration(
    language: str,
    text: str,
    info: DurationInfo,
) -> Dict[str, Any]:
    """Parse duration from text using lingua_franca."""
    data: Dict[str, Any] = {}

    if language not in _LOADED_LANGS:
        lingua_franca.load_language(language)
        _LOADED_LANGS.add(language)

    # Hack because lingua_franca fails on "twenty-five"
    text = text.replace("-", " ")

    result = extract_duration(text, lang=language)
    if (not result) or (not result[0]):
        raise ParseError("no duration")

    duration: datetime.timedelta = result[0]
    total_seconds = int(duration.total_seconds())
    hours, remainder = divmod(total_seconds, 3600)
    minutes, seconds = divmod(remainder, 60)

    if hours > 0:
        data[info.hours_slot] = hours

    if minutes > 0:
        data[info.minutes_slot] = minutes

    if seconds > 0:
        data[info.seconds_slot] = seconds

    return data


def parse_number(language: str, text: str, slot: str) -> Dict[str, Any]:
    """Parse number from text using lingua_franca."""
    if language not in _LOADED_LANGS:
        lingua_franca.load_language(language)
        _LOADED_LANGS.add(language)

    # Hack to fix lingua_franca not recognizing "twenty-five"
    text = text.replace("-", " ")

    result = extract_number(text, lang=language)
    if result is False:
        raise ParseError("no number")

    return {slot: result}
