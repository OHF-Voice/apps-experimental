import asyncio
import logging
import re
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional

from home_assistant_intents import get_intents
from ruamel.yaml import YAML

from intent_tools import unknown_tool_names

if TYPE_CHECKING:
    from gemma4_recognizer import Gemma4Recognizer
    from hass_api import HomeAssistant, HomeInfo

BASE_DIR = Path(__file__).resolve().parent

__version__ = "1.0.0"

_LOGGER = logging.getLogger(__name__)
_yaml = YAML()


@dataclass
class AppState:
    tools_path: Optional[Path]
    tools: List[Dict[str, Any]]
    recognizer: "Gemma4Recognizer"
    # Single worker serializes all access to the (non-thread-safe) model, shared
    # between live recognition and the benchmark endpoint.
    llama_executor: ThreadPoolExecutor
    hass: Optional["HomeAssistant"] = None
    # Intent metadata used to (re)generate tools, and the main asyncio loop the
    # web thread schedules HA fetches on. Set when tools are auto-generated.
    metadata: Dict[str, Any] = field(default_factory=dict)
    # Last Home Assistant snapshot the tools were built from (None when no HA
    # or in manual override mode). Shown in the web UI.
    home_info: Optional["HomeInfo"] = None
    loop: Optional[asyncio.AbstractEventLoop] = None
    lang_intents: Dict[str, Any] = field(default_factory=dict)

    def load(self) -> bool:
        """Load tool definitions from the manual override YAML.

        Returns False (leaving ``tools`` empty, so the caller falls back to
        auto-generation) when the file is missing, or when it contains tool
        names that no longer map to intents -- e.g. a file saved by an older
        version of this app with a different tool scheme.
        """
        self.tools = []
        if (self.tools_path is None) or (not self.tools_path.exists()):
            _LOGGER.debug("Missing tools file: %s", self.tools_path)
            return False

        _LOGGER.debug("Loading tools from YAML: %s", self.tools_path)
        with open(self.tools_path, "r", encoding="utf-8") as tools_file:
            data = _yaml.load(tools_file) or {}

        tools = data.get("tools") or []
        unknown = unknown_tool_names(tools, self.metadata)
        if unknown:
            _LOGGER.warning(
                "Ignoring tools override %s: unmappable tool name(s) %s "
                "(saved by an older version?); using auto-generated tools",
                self.tools_path,
                unknown,
            )
            return False

        self.tools = tools
        _LOGGER.debug("Loaded %s tool(s)", len(self.tools))
        return bool(self.tools)

    def get_lang_intents(self, language: str) -> Dict[str, Any]:
        if language not in self.lang_intents:
            original_language = language
            lang_intents = get_intents(language)
            if not lang_intents:
                # Normalize and try again
                lang_parts = re.split(r"[-_]", language, maxsplit=1)
                lang_family: Optional[str] = None
                if len(lang_parts) == 1:
                    language = language.lower()
                else:
                    lang_family = lang_parts[0]
                    assert lang_family
                    language = lang_family.lower() + "-" + language[0].upper()

                lang_intents = get_intents(language)
                if (not lang_intents) and lang_family:
                    lang_intents = get_intents(lang_family)

            if lang_intents:
                self.lang_intents[language] = lang_intents
                self.lang_intents[original_language] = lang_intents

        return self.lang_intents.get(language, {})

    def get_intent_response(
        self, language: str, intent_name: str, default_key: str = "default"
    ) -> Optional[str]:
        responses = (
            self.get_lang_intents(language)
            .get("responses", {})
            .get("intents", {})
            .get(intent_name, {})
        )
        if len(responses) == 1:
            return next(iter(responses.values()))

        return responses.get(default_key)

    def get_error_response(
        self, language: str, key: str = "no_intent"
    ) -> Optional[str]:
        return (
            self.get_lang_intents(language)
            .get("responses", {})
            .get("errors", {})
            .get(key)
        )
