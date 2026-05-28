from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Dict

from hass_api import HomeAssistant, HomeAssistantInfo, Tool

BASE_DIR = Path(__file__).parent

if TYPE_CHECKING:
    from gemma4_recognizer import Gemma4Recognizer


@dataclass
class AppState:
    hass: HomeAssistant
    hass_info: HomeAssistantInfo
    tools: Dict[str, Tool]
    recognizer: "Gemma4Recognizer"
