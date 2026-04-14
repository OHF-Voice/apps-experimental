"""Inference for PhonMatchNet model.

See: https://github.com/ncsoft/PhonMatchNet
"""

import logging
import os
from typing import List

import tensorflow as tf
import numpy as np

from ukws_model import ukws
from dataset.g2p.g2p_en.g2p import G2p

_LOGGER = logging.getLogger(__name__)

# NOTE: Needs env: TF_USE_LEGACY_KERAS=1


class UniversalKeywordSearch:

    def __init__(
        self,
        checkpoint_path: str,
        wake_word: str,
        vocab_size: int = 72,
        sample_rate: int = 16000,
        window_s=1.25,
        hop_s=0.5,
        detect_threshold=0.5,
        trigger_count=1,
    ) -> None:
        model_kwargs = {
            "vocab": vocab_size,
            "text_input": "g2p_embed",
            "audio_input": "both",
            "frame_length": 400,
            "hop_length": 160,
            "num_mel": 40,
            "sample_rate": sample_rate,
            "log_mel": False,
            "stack_extractor": True,
        }

        self.model = ukws.BaseUKWS(**model_kwargs)
        self.wake_word = wake_word
        self.g2p = G2p()
        _LOGGER.debug("Phonemes for '%s': %s", wake_word, self.g2p(wake_word))
        self.ww_embedding = self.g2p.embedding(wake_word)

        if self.ww_embedding.ndim == 2:
            self.ww_embedding = np.expand_dims(self.ww_embedding, 0)

        ckpt = tf.train.Checkpoint(model=self.model)
        ckpt_prefix = find_latest_checkpoint(checkpoint_path)
        ckpt.restore(ckpt_prefix).expect_partial()

        self.sample_rate = sample_rate
        self.window_samples = int(window_s * sample_rate)
        self.hop_samples = int(hop_s * sample_rate)
        self.detect_threshold = detect_threshold
        self.trigger_count = trigger_count

        self.buffer = np.zeros(0, dtype=np.float32)
        self.samples_since_last_score = 0
        self.consecutive_hits = 0
        self.total_samples_seen = 0
        self.scores: List[float] = []

    def reset(self) -> None:
        self.buffer = np.zeros(0, dtype=np.float32)
        self.samples_since_last_score = 0
        self.consecutive_hits = 0
        self.total_samples_seen = 0
        self.scores = []

    def score_window(self, window_audio: np.ndarray) -> float:
        x = np.expand_dims(window_audio.astype(np.float32), 0)  # [1, T]
        prob = self.model(x, self.ww_embedding, training=False)[0]
        return float(prob.numpy().reshape(-1)[0])

    def process_chunk(self, chunk: np.ndarray) -> bool:
        """
        Process an incoming audio chunk of arbitrary size.

        Returns:
            True if triggered, else False.
        """
        chunk = np.asarray(chunk, dtype=np.float32).reshape(-1)

        self.buffer = np.concatenate([self.buffer, chunk])
        self.samples_since_last_score += len(chunk)
        self.total_samples_seen += len(chunk)

        fired = False

        while self.samples_since_last_score >= self.hop_samples:
            self.samples_since_last_score -= self.hop_samples

            # This scoring point is self.samples_since_last_score samples before
            # the end of the buffer.
            end_offset = self.samples_since_last_score
            end_idx = len(self.buffer) - end_offset
            start_idx = end_idx - self.window_samples

            if start_idx < 0:
                continue

            window = self.buffer[start_idx:end_idx]
            score = self.score_window(window)
            self.scores.append(score)

            _LOGGER.debug(score)

            detected = score >= self.detect_threshold

            if detected:
                self.consecutive_hits += 1
            else:
                self.consecutive_hits = 0

            if self.consecutive_hits >= self.trigger_count:
                self.consecutive_hits = 0
                fired = True
                _LOGGER.debug("Detected %s with score %s", self.wake_word, score)
                break

        max_buffer = self.window_samples + self.hop_samples
        if len(self.buffer) > max_buffer:
            self.buffer = self.buffer[-max_buffer:]

        return fired


def find_latest_checkpoint(path: str) -> str:
    if os.path.isdir(path):
        ckpt = tf.train.latest_checkpoint(path)
        if ckpt is None:
            raise FileNotFoundError(f"No TensorFlow checkpoint found in: {path}")
        return ckpt
    return path
