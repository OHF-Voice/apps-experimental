#!/usr/bin/env python3

import argparse
import asyncio
import logging
import time
import os
import tempfile
import wave
from functools import partial
from pathlib import Path
from typing import cast, Optional, TYPE_CHECKING

from wyoming.asr import Transcript, Transcribe
from wyoming.audio import AudioChunk, AudioStop
from wyoming.event import Event
from wyoming.info import AsrModel, AsrProgram, Attribution, Describe, Info
from wyoming.server import AsyncEventHandler, AsyncServer

if TYPE_CHECKING:
    from nemo.collections.asr.models import ASRModel

_LOGGER = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parent

SUPPORTED_LANGUAGES = [
    "bg",  # Bulgarian
    "hr",  # Croatian
    "cs",  # Czech
    "da",  # Danish
    "nl",  # Dutch
    "en",  # English
    "et",  # Estonian
    "fi",  # Finnish
    "fr",  # French
    "de",  # German
    "el",  # Greek
    "hu",  # Hungarian
    "it",  # Italian
    "lv",  # Latvian
    "lt",  # Lithuanian
    "mt",  # Maltese
    "pl",  # Polish
    "pt",  # Portuguese
    "ro",  # Romanian
    "sk",  # Slovak
    "sl",  # Slovenian
    "es",  # Spanish
    "sv",  # Swedish
    "ru",  # Russian
    "uk",  # Ukrainian
]


# -----------------------------------------------------------------------------


async def main() -> None:
    """Runs fallback ASR server."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--uri", required=True, help="unix:// or tcp://")
    parser.add_argument("--model", required=True, help="NeMo model id to use")
    parser.add_argument("--cache-dir", help="Path to HuggingFace cache")
    parser.add_argument(
        "--debug", action="store_true", help="Print DEBUG messages to console"
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.DEBUG if args.debug else logging.INFO)
    _LOGGER.debug(args)

    logging.getLogger("numba").setLevel(logging.ERROR)

    if args.cache_dir:
        cache_dir = str(Path(args.cache_dir).resolve())
        os.environ["HF_HUB_CACHE"] = cache_dir
        os.environ["NEMO_CACHE_DIR"] = cache_dir
        _LOGGER.debug("Set cache directory: %s", cache_dir)

    # Have to import after setting cache dir
    from nemo.collections.asr.models import ASRModel

    _LOGGER.info("Loading %s", args.model)
    model = cast(ASRModel, ASRModel.from_pretrained(model_name=args.model))
    model.eval()
    _LOGGER.debug("Loaded model")

    server = AsyncServer.from_uri(args.uri)
    _LOGGER.info("Ready")

    try:
        with tempfile.TemporaryDirectory() as temp_dir_str:
            wav_dir = Path(temp_dir_str)
            await server.run(partial(SttEventHandler, model, wav_dir))
    except KeyboardInterrupt:
        pass


# -----------------------------------------------------------------------------


class SttEventHandler(AsyncEventHandler):
    """Event handler for clients."""

    def __init__(
        self,
        model: "ASRModel",
        wav_dir: Path,
        *args,
        **kwargs,
    ) -> None:
        """Initialize event handler."""
        super().__init__(*args, **kwargs)

        self.client_id = str(time.monotonic_ns())
        self.model = model
        self.wav_dir = wav_dir
        self._wav_file: Optional[wave.Wave_write] = None
        self._wav_path = wav_dir / f"{self.client_id}.wav"

        self._language: Optional[str] = None
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

        if AudioChunk.is_type(event.type):
            chunk = AudioChunk.from_event(event)
            if self._wav_file is None:
                _LOGGER.debug("Receiving audio")
                self._wav_file = wave.open(str(self._wav_path), "wb")
                self._wav_file.setframerate(chunk.rate)
                self._wav_file.setsampwidth(chunk.width)
                self._wav_file.setnchannels(chunk.channels)

            self._wav_file.writeframes(chunk.audio)
        elif AudioStop.is_type(event.type):
            assert self._wav_file is not None
            self._wav_file.close()

            _LOGGER.debug("Transcribing %s", self._wav_path)
            start_time = time.monotonic()
            results = self.model.transcribe(
                [str(self._wav_path)],
                source_lang=self._language,
                target_lang=self._language,
            )
            end_time = time.monotonic()
            _LOGGER.debug("Results in %s second(s): %s", end_time - start_time, results)

            text = results[0].text
            language = self._language
            _LOGGER.debug("Transcript (%s): %s", language, text)

            await self.write_event(Transcript(text=text, language=language).event())

            # Reset
            self._wav_path.unlink(missing_ok=True)
            self._wav_file = None
            self._language = None
        elif Transcribe.is_type(event.type):
            transcribe = Transcribe.from_event(event)
            self._language = transcribe.language

        return True

    async def _write_info(self) -> None:
        if self._info_event is not None:
            await self.write_event(self._info_event)
            return

        info = Info(
            asr=[
                AsrProgram(
                    name="stt-canary",
                    attribution=Attribution(
                        name="The Home Assistant Authors",
                        url="http://github.com/OHF-voice",
                    ),
                    description="Canary 1b-v2",
                    installed=True,
                    version="0.0.1",
                    models=[
                        AsrModel(
                            name="canary-1b-v2",
                            attribution=Attribution(
                                name="NVIDIA",
                                url="https://huggingface.co/nvidia/canary-1b-v2",
                            ),
                            installed=True,
                            description="Canary 1b-v2",
                            version=None,
                            languages=SUPPORTED_LANGUAGES,
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
