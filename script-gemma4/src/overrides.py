"""Which scripts this app targets, overriding Home Assistant's exposure.

Home Assistant's "expose to voice" is the default: a script exposed there is
targeted here. The overrides file records only the deliberate exceptions, as two
sets, so exposing something new in Home Assistant is still picked up
automatically::

    scripts:
      enabled:  [peppa_pig_sound]   # not exposed in HA, but targeted here
      disabled: [get_date]          # exposed in HA, but not targeted here

Storing the exceptions rather than the resulting list matters: a stored list
would freeze Home Assistant's exposure state at whatever it was when the user
last touched the page.
"""

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Dict, List, Optional, Set

import yaml

if TYPE_CHECKING:
    from hass_api import Tool

_LOGGER = logging.getLogger(__name__)


@dataclass
class ScriptOverrides:
    """Deliberate exceptions to Home Assistant's voice exposure."""

    # Scripts targeted even though they are not exposed to voice.
    enabled: Set[str] = field(default_factory=set)
    # Scripts not targeted even though they are exposed to voice.
    disabled: Set[str] = field(default_factory=set)

    def targets(self, tool: "Tool") -> bool:
        """True when this script should be given to the model."""
        if tool.name in self.disabled:
            return False

        return tool.exposed or (tool.name in self.enabled)

    def select(self, all_tools: List["Tool"]) -> List["Tool"]:
        """The tools the model should be given, in the given order."""
        return [tool for tool in all_tools if self.targets(tool)]

    def as_dict(self) -> Dict[str, Dict[str, List[str]]]:
        return {
            "scripts": {
                "enabled": sorted(self.enabled),
                "disabled": sorted(self.disabled),
            }
        }

    def prune(self, known_scripts: Set[str]) -> Set[str]:
        """Drop entries naming scripts that no longer exist.

        Returns the names that were dropped. Callers must only pass a script
        list they know to be complete: an empty or partial list (Home Assistant
        still starting, an integration not yet loaded) is indistinguishable from
        every script having been deleted, and pruning against it would quietly
        discard the user's choices.
        """
        dropped = (self.enabled | self.disabled) - known_scripts
        if dropped:
            self.enabled -= dropped
            self.disabled -= dropped

        return dropped


def load(path: Optional[Path]) -> ScriptOverrides:
    """Read the overrides file, falling back to "no exceptions"."""
    if (path is None) or (not path.exists()):
        _LOGGER.debug("No script overrides file: %s", path)
        return ScriptOverrides()

    try:
        with open(path, "r", encoding="utf-8") as overrides_file:
            data = yaml.safe_load(overrides_file) or {}
    except (OSError, yaml.YAMLError) as err:
        # A corrupt file must not stop the app from starting: fall back to
        # Home Assistant's exposure, which is always a safe answer.
        _LOGGER.warning("Ignoring unreadable overrides file %s: %s", path, err)
        return ScriptOverrides()

    scripts = data.get("scripts") or {}
    overrides = ScriptOverrides(
        enabled=_name_set(scripts.get("enabled")),
        disabled=_name_set(scripts.get("disabled")),
    )

    # Disabling wins, so a name in both sets is meaningless. Keep the file
    # honest rather than silently applying a rule the user cannot see.
    both = overrides.enabled & overrides.disabled
    if both:
        _LOGGER.warning(
            "Script(s) %s are both enabled and disabled in %s; treating as disabled",
            sorted(both),
            path,
        )
        overrides.enabled -= both

    _LOGGER.debug(
        "Loaded overrides: %s enabled, %s disabled",
        len(overrides.enabled),
        len(overrides.disabled),
    )
    return overrides


def save(path: Path, overrides: ScriptOverrides) -> None:
    """Write the overrides file, replacing it atomically."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(".tmp")
    with open(tmp_path, "w", encoding="utf-8") as overrides_file:
        yaml.safe_dump(
            overrides.as_dict(),
            overrides_file,
            default_flow_style=False,
            sort_keys=True,
        )

    tmp_path.replace(path)
    _LOGGER.debug("Saved overrides: %s", path)


def _name_set(value: object) -> Set[str]:
    """A set of script names from whatever the YAML held."""
    if not isinstance(value, list):
        return set()

    return {str(name).strip() for name in value if str(name).strip()}
