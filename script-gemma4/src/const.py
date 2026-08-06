from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Dict, List

from hass_api import HomeAssistant, HomeAssistantInfo, Tool

BASE_DIR = Path(__file__).parent

if TYPE_CHECKING:
    from gemma4_recognizer import Gemma4Recognizer


@dataclass
class AppState:
    hass: HomeAssistant
    hass_info: HomeAssistantInfo
    # Tools the model can call, by tool name. Only targeted scripts belong here:
    # this is what the Wyoming handler looks a tool call up in before running it.
    tools: Dict[str, Tool]
    recognizer: "Gemma4Recognizer"
    # Single worker serializes all access to the (non-thread-safe) model, shared
    # between live recognition and the benchmark endpoint.
    llama_executor: ThreadPoolExecutor
    # Every script in Home Assistant, targeted or not, for the web UI to show.
    all_tools: List[Tool] = field(default_factory=list)
