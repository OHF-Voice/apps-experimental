#!/usr/bin/env python3

import argparse
import asyncio
import logging
import time
from functools import partial
from pathlib import Path
from typing import TYPE_CHECKING, Optional

import yaml
from jinja2 import BaseLoader, Environment
from wyoming.asr import Transcript
from wyoming.event import Event
from wyoming.info import Attribution, Describe, Info, IntentModel, IntentProgram
from wyoming.intent import Entity, Intent, NotRecognized
from wyoming.server import AsyncEventHandler, AsyncServer
from wyoming.error import Error

from hass_api import HomeAssistant

if TYPE_CHECKING:
    from command_matcher import CommandMatcher

_LOGGER = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parent

# -----------------------------------------------------------------------------


async def main() -> None:
    """Run app."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--uri", required=True, help="unix:// or tcp://")
    parser.add_argument(
        "--sentences", required=True, help="Path to sentences YAML file"
    )
    parser.add_argument("--model", help="HuggingFace id of sentence transformers model")
    #
    parser.add_argument(
        "--hass-token", required=True, help="Long-lived access token for Home Assistant"
    )
    parser.add_argument(
        "--hass-api",
        default="http://homeassistant.local:8123/api",
        help="URL of Home Assistant API",
    )
    #
    parser.add_argument(
        "--model-cache-dir", help="Directory to cache sentence transformer models"
    )
    parser.add_argument(
        "--model-local-files",
        action="store_true",
        help="Only look for model files locally",
    )
    #
    parser.add_argument(
        "--satellite-id", help="Satellite id to use if not provided by Home Assistant"
    )
    #
    parser.add_argument(
        "--debug", action="store_true", help="Print DEBUG messages to console"
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.DEBUG if args.debug else logging.INFO)
    _LOGGER.debug(args)

    hass = HomeAssistant(token=args.hass_token, api_url=args.hass_api)

    from sentence_transformers import SentenceTransformer

    from command_matcher import Command, CommandMatcher

    _LOGGER.debug("Loading sentences from %s", args.sentences)
    with open(args.sentences, "r", encoding="utf-8") as sentences_file:
        sentences_dict = yaml.safe_load(sentences_file)
        language = sentences_dict["language"]

    if args.model:
        model_name = args.model
    elif language == "en":
        model_name = "intfloat/e5-small-v2"
    else:
        model_name = "intfloat/multilingual-e5-small"

    _LOGGER.debug("Loading model: %s", model_name)
    model = SentenceTransformer(
        model_name,
        cache_folder=(
            str(Path(args.model_cache_dir).resolve()) if args.model_cache_dir else None
        ),
        local_files_only=args.model_local_files,
    )
    _LOGGER.debug("Loaded model: %s", model_name)

    _LOGGER.debug("Training command matcher")
    matcher = CommandMatcher(model)

    for command_dict in sentences_dict["commands"]:
        matcher.add(Command.from_dict(command_dict))

    server = AsyncServer.from_uri(args.uri)
    _LOGGER.info("Ready")

    try:
        await server.run(
            partial(
                WyomingEventHandler,
                language,
                matcher,
                model_name,
                hass,
                args.satellite_id,
            )
        )
    except KeyboardInterrupt:
        pass


# -----------------------------------------------------------------------------


class WyomingEventHandler(AsyncEventHandler):
    """Event handler for clients."""

    def __init__(
        self,
        language: str,
        matcher: "CommandMatcher",
        model_name: str,
        hass: HomeAssistant,
        satellite_id: Optional[str],
        *args,
        **kwargs,
    ) -> None:
        """Initialize event handler."""
        super().__init__(*args, **kwargs)

        self.client_id = str(time.monotonic_ns())
        self.language = language
        self.matcher = matcher
        self.model_name = model_name
        self.hass = hass
        self.satellite_id = satellite_id

        self._info_event: Optional[Event] = None
        self._env = Environment(loader=BaseLoader())

    async def handle_event(self, event: Event) -> bool:
        """Handle Wyoming event."""
        try:
            return await self._handle_event(event)
        except Exception:
            _LOGGER.exception("Error handling event")

        return True

    async def _handle_event(self, event: Event) -> bool:
        """Handle Wyoming event."""
        from command_matcher import CommandMatch, CommandMatchFailure

        if Describe.is_type(event.type):
            await self._write_info()
            return True

        if Transcript.is_type(event.type):
            transcript = Transcript.from_event(event)
            current_area_id: Optional[str] = None
            satellite_id: Optional[str] = None
            if transcript.context:
                satellite_id = transcript.context.get("satellite_id")

            satellite_id = satellite_id or self.satellite_id

            if satellite_id:
                hass_info = await self.hass.get_info(satellite_id=satellite_id)
                current_area_id = hass_info.current_area_id

            try:
                command_match = self.matcher.match(
                    self.language, transcript.text, current_area_id=current_area_id
                )
            except Exception as err:
                _LOGGER.exception(err)
                await self.write_event(
                    NotRecognized(
                        text=f"Unexpected error during recognition: {err}",
                        context=transcript.context,
                    ).event()
                )
                return True

            _LOGGER.debug(command_match)
            if isinstance(command_match, CommandMatchFailure):
                # Match failed
                await self.write_event(
                    NotRecognized(
                        text=command_match.to_string(),
                        context=transcript.context,
                    ).event()
                )
            elif isinstance(command_match, CommandMatch):
                # Match succeeded
                command = command_match.command
                slots = command.intent_slots or {}
                if command_match.slots:
                    slots.update(command_match.slots)

                if command.current_area:
                    slots.setdefault(command.current_area.slot, current_area_id)

                response: Optional[str] = None
                variables = {"slots": slots}
                if command.response:
                    response = self._env.from_string(command.response).render(variables)
                elif command.hass_response:
                    response = await self.hass.render_template(
                        command.hass_response, variables
                    )

                await self.write_event(
                    Intent(
                        name=command.intent_name,
                        entities=[Entity(name=k, value=v) for k, v in slots.items()],
                        text=response,
                        context=transcript.context,
                    ).event()
                )

            return True

        return True

    async def _write_info(self) -> None:
        if self._info_event is not None:
            await self.write_event(self._info_event)
            return

        info = Info(
            intent=[
                IntentProgram(
                    "sentence-transformers",
                    attribution=Attribution(
                        name="OHF Voice", url="https://github.com/OHF-Voice"
                    ),
                    installed=True,
                    description="",
                    version="",
                    models=[
                        IntentModel(
                            self.model_name,
                            attribution=Attribution(
                                name="HuggingFace",
                                url="https://huggingface.co/",
                            ),
                            installed=True,
                            description=self.model_name,
                            version="",
                            languages=[self.language],
                        )
                    ],
                )
            ]
        )

        self._info_event = info.event()
        await self.write_event(self._info_event)


# -----------------------------------------------------------------------------

if __name__ == "__main__":
    asyncio.run(main())
