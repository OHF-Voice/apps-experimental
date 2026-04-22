from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Deque, List, Optional

import numpy as np
from pymicro_vad import MicroVad


@dataclass
class VadFrame:
    start_sample: int
    end_sample: int
    prob: float


class VadWindowGate:
    """
    Streaming gate for 16 kHz audio that uses a frame-based VAD to decide
    whether a full analysis window should be processed.

    Assumptions:
    - Input audio is mono, 16 kHz.
    - Incoming chunks are float32/float64 in [-1, 1] or int16.
    - The VAD object provides:
        - chunk_samples() -> int
        - process_chunk(bytes) -> float
      where process_chunk expects raw PCM16 bytes for exactly chunk_samples().

    Behavior:
    - Audio may arrive in arbitrary chunk sizes.
    - Internally, audio is buffered and sliced into exact VAD-sized frames.
    - A full output window is considered every `hop_seconds`.
    - A window is emitted only if enough of its covered VAD frames are speech.

    Typical usage:
        gate = VadWindowGate(vad, window_seconds=1.5, hop_seconds=0.1)

        while streaming:
            windows = gate.process_chunk(audio_chunk)
            for w in windows:
                # score with PhonMatchNet
                ...
    """

    def __init__(  # pylint: disable=too-many-positional-arguments
        self,
        vad: MicroVad,
        window_seconds: float,
        hop_seconds: float,
        sample_rate: int = 16000,
        speech_threshold: float = 0.5,
        min_speech_fraction: float = 0.2,
        min_speech_frames: int = 1,
        pad_windows: bool = False,
    ) -> None:
        if sample_rate != 16000:
            raise ValueError("This class currently expects 16 kHz audio.")

        if window_seconds <= 0:
            raise ValueError("window_seconds must be > 0")
        if hop_seconds <= 0:
            raise ValueError("hop_seconds must be > 0")
        if not (0.0 <= speech_threshold <= 1.0):
            raise ValueError("speech_threshold must be in [0, 1]")
        if not (0.0 <= min_speech_fraction <= 1.0):
            raise ValueError("min_speech_fraction must be in [0, 1]")
        if min_speech_frames < 0:
            raise ValueError("min_speech_frames must be >= 0")

        self.vad = vad
        self.sample_rate = sample_rate
        self.window_samples = int(round(window_seconds * sample_rate))
        self.hop_samples = int(round(hop_seconds * sample_rate))
        self.vad_frame_samples = 160  # 10ms

        self.speech_threshold = float(speech_threshold)
        self.min_speech_fraction = float(min_speech_fraction)
        self.min_speech_frames = int(min_speech_frames)
        self.pad_windows = bool(pad_windows)

        # Audio buffer holding recent audio as float32 mono.
        self.audio_buffer = np.zeros(0, dtype=np.float32)

        # Leftover samples waiting to make a full VAD frame.
        self.vad_input_buffer = np.zeros(0, dtype=np.float32)

        # Recent VAD results with absolute sample positions.
        self.vad_frames: Deque[VadFrame] = deque()

        # Absolute sample count seen so far.
        self.total_samples_seen = 0

        # Absolute sample index where next output window ends.
        self.next_window_end_sample: Optional[int] = None

    def reset(self) -> None:
        self.audio_buffer = np.zeros(0, dtype=np.float32)
        self.vad_input_buffer = np.zeros(0, dtype=np.float32)
        self.vad_frames.clear()
        self.total_samples_seen = 0
        self.next_window_end_sample = None

    def process_chunk(self, chunk: np.ndarray) -> List[np.ndarray]:
        """
        Feed an arbitrary audio chunk and return a list of full windows
        that passed the VAD gate.

        Each returned window is float32, mono, length = window_samples.
        """
        x = self._normalize_input_chunk(chunk)

        if x.size == 0:
            return []

        old_total = self.total_samples_seen
        self.total_samples_seen += x.size

        # Append to audio buffers.
        self.audio_buffer = np.concatenate([self.audio_buffer, x])
        self.vad_input_buffer = np.concatenate([self.vad_input_buffer, x])

        # Run VAD on as many exact-sized frames as possible.
        self._consume_vad_frames(start_sample_of_new_chunk=old_total)

        # Initialize first evaluation point once we have some samples.
        if self.next_window_end_sample is None:
            self.next_window_end_sample = self.window_samples

        emitted: List[np.ndarray] = []

        # Consider every hop whose window end is now available.
        while self.next_window_end_sample <= self.total_samples_seen:
            window_end = self.next_window_end_sample
            window_start = window_end - self.window_samples

            if window_start < 0:
                if self.pad_windows:
                    window = self._extract_window_with_left_pad(
                        window_start, window_end
                    )
                else:
                    self.next_window_end_sample += self.hop_samples
                    continue
            else:
                window = self._extract_window(window_start, window_end)

            if self._should_process_window(window_start, window_end):
                window = self._normalize_window(window, window_start, window_end)
                emitted.append(window)

            self.next_window_end_sample += self.hop_samples

        # Discard old VAD metadata and old audio no longer needed.
        self._trim_state()

        return emitted

    def _normalize_input_chunk(self, chunk: np.ndarray) -> np.ndarray:
        x = np.asarray(chunk)

        if x.ndim != 1:
            x = x.reshape(-1)

        if np.issubdtype(x.dtype, np.integer):
            # Assume PCM16-like integer input.
            x = x.astype(np.float32) / 32768.0
        else:
            x = x.astype(np.float32, copy=False)

        return x

    def _consume_vad_frames(self, start_sample_of_new_chunk: int) -> None:
        """
        Consume self.vad_input_buffer in exact VAD frame sizes and append
        absolute-positioned VAD results to self.vad_frames.
        """
        # Absolute sample position of the first sample currently in vad_input_buffer
        # after appending the latest chunk.
        buffer_start_abs = self.total_samples_seen - self.vad_input_buffer.size

        while self.vad_input_buffer.size >= self.vad_frame_samples:
            frame = self.vad_input_buffer[: self.vad_frame_samples]
            self.vad_input_buffer = self.vad_input_buffer[self.vad_frame_samples :]

            frame_start = buffer_start_abs
            frame_end = frame_start + self.vad_frame_samples
            buffer_start_abs = frame_end

            pcm16 = self._float_to_pcm16_bytes(frame)
            prob = self.vad.process_10ms(pcm16)

            self.vad_frames.append(
                VadFrame(
                    start_sample=frame_start,
                    end_sample=frame_end,
                    prob=prob,
                )
            )

    def _float_to_pcm16_bytes(self, x: np.ndarray) -> bytes:
        y = np.clip(x, -1.0, 1.0)
        y = (y * 32767.0).astype(np.int16)
        return y.tobytes()

    def _extract_window(self, start_sample: int, end_sample: int) -> np.ndarray:
        """
        Extract [start_sample:end_sample) from audio_buffer.
        """
        buffer_start_abs = self.total_samples_seen - self.audio_buffer.size
        rel_start = start_sample - buffer_start_abs
        rel_end = end_sample - buffer_start_abs

        if rel_start < 0 or rel_end > self.audio_buffer.size:
            raise RuntimeError("Requested window is not fully present in audio buffer")

        return self.audio_buffer[rel_start:rel_end].copy()

    def _extract_window_with_left_pad(
        self, start_sample: int, end_sample: int
    ) -> np.ndarray:
        """
        Extract [start_sample:end_sample), left-padding with zeros if start_sample < 0.
        """
        needed = end_sample - start_sample
        if needed != self.window_samples:
            raise RuntimeError("Unexpected padded window length request")

        left_pad = max(0, -start_sample)
        real_start = max(0, start_sample)

        real = self._extract_window(real_start, end_sample)
        if left_pad == 0:
            return real

        return np.concatenate([np.zeros(left_pad, dtype=np.float32), real], axis=0)

    def _should_process_window(self, window_start: int, window_end: int) -> bool:
        """
        Decide whether the window [window_start:window_end) has enough speech
        based on overlapping VAD frames.
        """
        overlapping = [
            vf
            for vf in self.vad_frames
            if vf.end_sample > window_start and vf.start_sample < window_end
        ]

        if not overlapping:
            return False

        speech_frames = sum(1 for vf in overlapping if vf.prob >= self.speech_threshold)
        speech_fraction = speech_frames / len(overlapping)

        if speech_frames < self.min_speech_frames:
            return False

        if speech_fraction < self.min_speech_fraction:
            return False

        return True

    def _trim_state(self) -> None:
        """
        Keep only the audio and VAD metadata needed for future windows.
        """
        # Oldest sample that might still be needed by a future window.
        # Future windows start at next_window_end_sample - window_samples.
        if self.next_window_end_sample is None:
            keep_from = max(0, self.total_samples_seen - self.window_samples)
        else:
            keep_from = max(0, self.next_window_end_sample - self.window_samples)

        # Trim audio buffer.
        buffer_start_abs = self.total_samples_seen - self.audio_buffer.size
        drop_audio = max(0, keep_from - buffer_start_abs)
        if drop_audio > 0:
            self.audio_buffer = self.audio_buffer[drop_audio:]

        # Trim VAD frames that end before keep_from.
        while self.vad_frames and self.vad_frames[0].end_sample <= keep_from:
            self.vad_frames.popleft()

    def _normalize_window(  # pylint: disable=too-many-positional-arguments
        self,
        window: np.ndarray,
        window_start: int,
        window_end: int,
        target_rms: float = 0.08,
        max_gain: float = 6.0,
        min_gain: float = 0.5,
        eps: float = 1e-8,
    ) -> np.ndarray:
        """
        Normalize using speech-only RMS from VAD frames.
        """

        # Remove DC offset
        x = window - np.mean(window)

        speech_samples = []

        for vf in self.vad_frames:
            if vf.prob < self.speech_threshold:
                continue

            if vf.end_sample <= window_start:
                continue

            if vf.start_sample >= window_end:
                continue

            start = max(vf.start_sample, window_start)
            end = min(vf.end_sample, window_end)

            rel_start = start - window_start
            rel_end = end - window_start

            speech_samples.append(x[rel_start:rel_end])

        if speech_samples:
            speech = np.concatenate(speech_samples)
            rms = np.sqrt(np.mean(speech**2) + eps)
        else:
            rms = np.sqrt(np.mean(x**2) + eps)

        gain = target_rms / max(rms, eps)
        gain = np.clip(gain, min_gain, max_gain)

        x = np.clip(x * gain, -1.0, 1.0)

        return x
