import asyncio
import logging
import time
from typing import Optional

from wyoming.asr import Transcript
from wyoming.event import Event
from wyoming.info import Attribution, Describe, HandleModel, HandleProgram, Info
from wyoming.intent import Entity, Intent, NotRecognized
from wyoming.server import AsyncEventHandler

from const import AppState, __version__
from intent_mapping import tool_to_intent
from intent_tools import needs_context_area

_LOGGER = logging.getLogger(__name__)


class IntentEventHandler(AsyncEventHandler):
    """Wyoming event handler that recognizes intents with Gemma 4."""

    def __init__(
        self,
        state: AppState,
        *args,
        **kwargs,
    ) -> None:
        """Initialize event handler."""
        super().__init__(*args, **kwargs)

        self.client_id = str(time.monotonic_ns())
        self.state = state

        self._info_event: Optional[Event] = None

    async def handle_event(self, event: Event) -> bool:
        """Handle Wyoming event."""
        try:
            return await self._handle_event(event)
        except Exception:
            _LOGGER.exception("Error handling event")

        return True

    async def _handle_event(self, event: Event) -> bool:
        """Handle Wyoming event."""
        if Describe.is_type(event.type):
            await self._write_info()
            return True

        if not Transcript.is_type(event.type):
            return True

        transcript = Transcript.from_event(event)
        _LOGGER.debug("Handling: %s", transcript)

        if not transcript.text:
            return True

        language = transcript.language or "en"

        # Resolve the satellite's area concurrently with recognition; it is
        # needed when the matched slot combination is context_area (e.g.
        # "turn on the lights" targets the area the command came from).
        context = transcript.context or {}
        satellite_task: Optional[asyncio.Task] = None
        if self.state.hass is not None:
            device_id = context.get("device_id")
            satellite_id = context.get("satellite_id")
            if device_id or satellite_id:
                satellite_task = asyncio.create_task(
                    self.state.hass.get_satellite_info(device_id, satellite_id)
                )

        # Serialize on the single model worker (the model is not thread-safe and
        # is shared with the benchmark endpoint).
        loop = asyncio.get_running_loop()
        tool_calls, _content = await loop.run_in_executor(
            self.state.llama_executor,
            self.state.recognizer.get_tool_calls,
            transcript.text,
            language,
        )
        _LOGGER.debug("Tool call(s): %s", tool_calls)

        intent_name: Optional[str] = None
        intent_slots = {}
        if tool_calls:
            # TODO: multiple tool calls
            tool_name, tool_args = tool_calls[0]
            intent_name, intent_slots = tool_to_intent(tool_name, tool_args)
            # Copy before mutating: the args dict may be shared with the
            # recognizer's response cache.
            intent_slots = dict(intent_slots)

        if intent_name and (satellite_task is not None):
            satellite_info = None
            try:
                satellite_info = await satellite_task
            except Exception:
                _LOGGER.exception("Failed to get satellite info")
            if (
                satellite_info is not None
                and satellite_info.area_id
                and "area" not in intent_slots
                and needs_context_area(
                    self.state.metadata, intent_name, set(intent_slots)
                )
            ):
                _LOGGER.debug(
                    "Using satellite area for %s: %s",
                    intent_name,
                    satellite_info.area_id,
                )
                intent_slots["area"] = satellite_info.area_id
        elif satellite_task is not None:
            satellite_task.cancel()

        if not intent_name:
            await self.write_event(
                NotRecognized(
                    text=self.state.get_error_response(language),
                    context=transcript.context,
                ).event()
            )
            return True

        _LOGGER.debug("Sending intent: name=%s, slots=%s", intent_name, intent_slots)
        await self.write_event(
            Intent(
                name=intent_name,
                entities=[
                    Entity(name=arg_name, value=arg_value)
                    for arg_name, arg_value in intent_slots.items()
                ],
                text=self.state.get_intent_response(language, intent_name),
                context=transcript.context,
            ).event()
        )
        return True

    async def _write_info(self) -> None:
        if self._info_event is not None:
            await self.write_event(self._info_event)
            return

        info = Info(
            intent=[
                HandleProgram(
                    name="intent-gemma4",
                    attribution=Attribution(
                        "Open Home Foundation Voice", "https://github.com/OHF-Voice"
                    ),
                    installed=True,
                    description="Gemma 4 Intent Recognizer",
                    version=__version__,
                    models=[
                        HandleModel(
                            name="gemma4",
                            attribution=Attribution(
                                "Google DeepMind",
                                "https://deepmind.google/models/gemma/gemma-4/",
                            ),
                            installed=True,
                            description="gemma4",
                            version="",
                            languages=[],  # all languages
                        )
                    ],
                )
            ],
        )

        self._info_event = info.event()
        await self.write_event(self._info_event)
