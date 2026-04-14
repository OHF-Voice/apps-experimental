#!/usr/bin/env python3

import argparse
import asyncio
import logging
import time
import os
from functools import partial
from pathlib import Path
from typing import Optional, TYPE_CHECKING, List

# Needed by PhonMatchNet
os.environ["TF_USE_LEGACY_KERAS"] = "1"


import numpy as np
from wyoming.event import Event
from wyoming.server import AsyncEventHandler, AsyncServer
from wyoming.audio import AudioChunk, AudioChunkConverter, AudioStart, AudioStop
from wyoming.info import Attribution, Describe, Info, WakeModel, WakeProgram
from wyoming.wake import Detect, Detection, NotDetected

if TYPE_CHECKING:
    from ukws import UniversalKeywordSearch

_LOGGER = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parent

# -----------------------------------------------------------------------------


async def main() -> None:
    """Run app."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--uri", required=True, help="unix:// or tcp://")
    parser.add_argument(
        "--wake-word",
        required=True,
        help="Wake words to listen for separate by ',' in the form input:output or just input",
    )
    parser.add_argument(
        "--checkpoint", required=True, help="Model checkpoint directory"
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Detection threshold from 0-1 (higher reduces false positives)",
    )
    parser.add_argument(
        "--trigger-count",
        type=int,
        default=1,
        help="Number of windows beyond threshold for detection (higher reduces false positives)",
    )
    parser.add_argument(
        "--window-length",
        type=float,
        default=1.25,
        help="Seconds of audio in each window",
    )
    parser.add_argument(
        "--hop-length",
        type=float,
        default=0.5,
        help="Seconds of audio between each hop",
    )
    parser.add_argument(
        "--debug", action="store_true", help="Print DEBUG messages to console"
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.DEBUG if args.debug else logging.INFO)
    _LOGGER.debug(args)

    # input/output
    ww_spoken = []
    ww_phrases = []
    for wake_word in args.wake_word.split(","):
        wake_word = wake_word.strip()
        if ":" in wake_word:
            spoken, phrase = wake_word.split(":", maxsplit=1)
        else:
            spoken, phrase = (wake_word, wake_word)

        ww_spoken.append(spoken)
        ww_phrases.append(phrase)

    from ukws import UniversalKeywordSearch, find_latest_checkpoint

    _LOGGER.debug("Looking for model in %s", args.checkpoint)
    checkpoint_path = find_latest_checkpoint(args.checkpoint)

    _LOGGER.debug("Loading model: %s", checkpoint_path)
    model = UniversalKeywordSearch(
        checkpoint_path,
        ww_spoken,
        detect_threshold=args.threshold,
        trigger_count=args.trigger_count,
        window_s=args.window_length,
        hop_s=args.hop_length,
    )
    _LOGGER.debug("Loaded model")

    server = AsyncServer.from_uri(args.uri)
    _LOGGER.info("Ready")

    try:
        await server.run(partial(WyomingEventHandler, model, ww_phrases))
    except KeyboardInterrupt:
        pass


# -----------------------------------------------------------------------------


class WyomingEventHandler(AsyncEventHandler):
    """Event handler for clients."""

    def __init__(
        self,
        model: "UniversalKeywordSearch",
        wake_words: List[str],
        *args,
        **kwargs,
    ) -> None:
        """Initialize event handler."""
        super().__init__(*args, **kwargs)

        self.client_id = str(time.monotonic_ns())
        self.model = model
        self.wake_words = wake_words
        self.converter = AudioChunkConverter(rate=16000, width=2, channels=1)
        self.audio_timestamp = 0
        self.detected = False

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

        if Detect.is_type(event.type):
            # TODO: multiple wake words
            pass
        elif AudioStart.is_type(event.type):
            _LOGGER.debug("Receiving audio")
            self.model.reset()
            self.audio_timestamp = 0
            self.detected = False
        elif AudioChunk.is_type(event.type):
            chunk = self.converter.convert(AudioChunk.from_event(event))
            audio_array = (
                np.frombuffer(chunk.audio, dtype=np.int16).astype(np.float32) / 32768.0
            )
            detected_idxs = self.model.process_chunk(audio_array)
            if detected_idxs:
                self.detected = True
                for ww_idx in detected_idxs:
                    ww_phrase = self.wake_words[ww_idx]
                    await self.write_event(
                        Detection(
                            name=ww_phrase, timestamp=self.audio_timestamp
                        ).event()
                    )
                    _LOGGER.debug("Detected %s at %s", ww_phrase, self.audio_timestamp)
                self.model.reset()

            self.audio_timestamp += chunk.milliseconds
        elif AudioStop.is_type(event.type):
            # Inform client if no detections occurred
            if not self.detected:
                # No wake word detections
                await self.write_event(NotDetected().event())

                _LOGGER.debug(
                    "Audio stopped without detection from client: %s. Max score was: %s",
                    self.client_id,
                    max(self.model.scores) if self.model.scores else "n/a",
                )

        return True

    async def _write_info(self) -> None:
        if self._info_event is not None:
            await self.write_event(self._info_event)
            return

        info = Info(
            wake=[
                WakeProgram(
                    name="phonmatchnet",
                    description="An open vocabulary keyword detector",
                    attribution=Attribution(
                        name="ncsoft", url="https://github.com/ncsoft/PhonMatchNet"
                    ),
                    installed=True,
                    version="1.0.0",
                    models=[
                        WakeModel(
                            name=wake_word,
                            description=wake_word,
                            phrase=wake_word,
                            attribution=Attribution(
                                name="The Home Assistant Authors",
                                url="http://github.com/OHF-voice",
                            ),
                            installed=True,
                            languages=["en"],
                            version="v1",
                        )
                        for wake_word in self.wake_words
                    ],
                )
            ],
        )

        self._info_event = info.event()
        await self.write_event(self._info_event)


# -----------------------------------------------------------------------------

if __name__ == "__main__":
    asyncio.run(main())
