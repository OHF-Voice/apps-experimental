import asyncio
import logging
import time
from typing import Dict, Optional

from wyoming.asr import Transcript
from wyoming.error import Error
from wyoming.event import Event
from wyoming.handle import Handled, NotHandled
from wyoming.info import Attribution, Describe, HandleModel, HandleProgram, Info
from wyoming.server import AsyncEventHandler

from const import AppState

_LOGGER = logging.getLogger(__name__)


class Gemma4EventHandler(AsyncEventHandler):
    """Event handler for clients."""

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

        if Transcript.is_type(event.type):
            transcript = Transcript.from_event(event)
            _LOGGER.debug("Handling: %s", transcript)

            try:
                language = transcript.language or "en"
                satellite_info: Dict[str, str] = {}
                if transcript.context:
                    device_id = transcript.context.get("device_id")
                    if device_id:
                        satellite_info["device_id"] = device_id
                    satellite_id = transcript.context.get("satellite_id")
                    if satellite_id:
                        satellite_info["entity_id"] = satellite_id
                    area_id = await self.state.hass.get_current_area(
                        device_id, satellite_id
                    )
                    if area_id:
                        satellite_info["area_id"] = area_id

                tool_calls, response_text = self.state.recognizer.get_tool_calls(
                    transcript.text, language
                )
                if not tool_calls:
                    await self.write_event(
                        NotHandled(
                            text=response_text, context=transcript.context
                        ).event()
                    )
                    return True

                for tool_id, tool_args in tool_calls:
                    tool = self.state.tools[tool_id]
                    script_id = f"script.{tool_id}"
                    variables = {}

                    # Map names to ids
                    for var_key, var_value in tool_args.items():
                        name_map = tool.name_map.get(var_key, {})
                        variables[var_key] = name_map.get(var_value, var_value)

                    if satellite_info:
                        variables["satellite"] = satellite_info

                    _LOGGER.debug(
                        "Calling script %s with variables %s", script_id, variables
                    )
                    asyncio.create_task(
                        self.state.hass.call_service(
                            "script",
                            "turn_on",
                            service_data={"variables": variables},
                            target={"entity_id": script_id},
                        )
                    )

                await self.write_event(
                    Handled(text="", context=transcript.context).event()
                )
            except Exception:
                _LOGGER.exception("Unexpected error during handling")
                await self.write_event(
                    Error(
                        text="Unexpected error during handling", code="handle-error"
                    ).event()
                )

            return True

        return True

    async def _write_info(self) -> None:
        if self._info_event is not None:
            await self.write_event(self._info_event)
            return

        info = Info(
            handle=[
                HandleProgram(
                    name="script-gemma4",
                    attribution=Attribution(
                        "Open Home Foundation Voice", "https://github.com/OHF-Voice"
                    ),
                    installed=True,
                    description="Gemma 4 Script Runner",
                    version="0.0.1",
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
                    supports_home_control=True,
                )
            ],
        )

        self._info_event = info.event()
        await self.write_event(self._info_event)
