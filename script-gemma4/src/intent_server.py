import asyncio
import logging
import time
from typing import Any, Dict, Optional

from wyoming.asr import Transcript
from wyoming.error import Error
from wyoming.event import Event
from wyoming.handle import Handled, NotHandled
from wyoming.info import Attribution, Describe, HandleModel, HandleProgram, Info
from wyoming.server import AsyncEventHandler

from const import AppState
from hass_api import SatelliteInfo
from tool_mapping import map_tool_call

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
                satellite_info: Optional[SatelliteInfo] = None
                satellite_info_task: Optional[asyncio.Task] = None
                if transcript.context:
                    device_id = transcript.context.get("device_id")
                    satellite_id = transcript.context.get("satellite_id")
                    satellite_info_task = asyncio.create_task(
                        self.state.hass.get_satellite_info(device_id, satellite_id)
                    )

                loop = asyncio.get_running_loop()
                tool_calls, response_text = await loop.run_in_executor(
                    self.state.llama_executor,
                    self.state.recognizer.get_tool_calls,
                    transcript.text,
                    language,
                )
                if not tool_calls:
                    await self.write_event(
                        NotHandled(
                            text=response_text, context=transcript.context
                        ).event()
                    )
                    return True

                if satellite_info_task:
                    satellite_info = await satellite_info_task

                ran_script = False
                for tool_id, tool_args in tool_calls:
                    # Only tools in the targeted set may run: the model can emit
                    # a name that is not a tool at all, and (once scripts can be
                    # enabled individually) one that is a script we must not run.
                    tool = self.state.tools.get(tool_id)
                    if tool is None:
                        _LOGGER.warning("Ignoring unknown tool: %s", tool_id)
                        continue

                    call = map_tool_call(tool, tool_args)
                    if call.unresolved:
                        _LOGGER.warning(
                            "Dropping unresolvable field(s) of %s: %s",
                            call.script_id,
                            call.unresolved,
                        )

                    if not call.can_run:
                        _LOGGER.warning(
                            "Not running %s: required field(s) %s could not be "
                            "resolved to anything in this home",
                            call.script_id,
                            call.missing_required,
                        )
                        continue

                    variables: Dict[str, Any] = dict(call.variables)
                    if satellite_info:
                        satellite_info_dict = satellite_info.as_dict()
                        satellite_info_dict["language"] = language
                        variables["satellite"] = satellite_info_dict

                    _LOGGER.debug(
                        "Calling script %s with variables %s",
                        call.script_id,
                        variables,
                    )
                    ran_script = True
                    asyncio.create_task(
                        self.state.hass.call_service(
                            "script",
                            "turn_on",
                            service_data={"variables": variables},
                            target={"entity_id": call.script_id},
                        )
                    )

                if not ran_script:
                    # Every call was unknown or unresolvable, so nothing ran.
                    await self.write_event(
                        NotHandled(
                            text=response_text, context=transcript.context
                        ).event()
                    )
                    return True

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
