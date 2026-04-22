"""Inference for PhonMatchNet model.

See: https://github.com/ncsoft/PhonMatchNet
"""

import logging
import os
from collections.abc import Collection
from typing import List, Optional

import numpy as np
import tensorflow as tf

from dataset.g2p.g2p_en.g2p import G2p
from ukws_model import ukws

_LOGGER = logging.getLogger(__name__)

# NOTE: Needs env: TF_USE_LEGACY_KERAS=1


class UniversalKeywordSearch:

    def __init__(  # pylint: disable=too-many-positional-arguments
        self,
        checkpoint_path: str,
        wake_words: List[str],
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
        self.wake_words = wake_words

        ckpt = tf.train.Checkpoint(model=self.model)
        ckpt_prefix = find_latest_checkpoint(checkpoint_path)
        ckpt.restore(ckpt_prefix).expect_partial()

        # Pre-compute text encoding
        self.g2p = G2p()
        self.emb_t_batch = self._encode_wake_words()
        self.num_wake_words = self.emb_t_batch.shape[0]
        assert self.num_wake_words == len(self.wake_words)

        self.sample_rate = sample_rate
        self.window_samples = int(window_s * sample_rate)
        self.hop_samples = int(hop_s * sample_rate)
        self.detect_threshold = detect_threshold
        self.trigger_count = trigger_count

        self.buffer = np.zeros(0, dtype=np.float32)
        self.samples_since_last_score = 0
        self.consecutive_hits = np.zeros(
            self.num_wake_words,
            dtype=np.int32,
        )
        self.total_samples_seen = 0
        self.scores: List[float] = []

    def reset(self) -> None:
        self.buffer = np.zeros(0, dtype=np.float32)
        self.samples_since_last_score = 0
        self.consecutive_hits = np.zeros(
            self.num_wake_words,
            dtype=np.int32,
        )
        self.total_samples_seen = 0
        self.scores = []

    def score_window(self, window_audio: np.ndarray):
        x = tf.convert_to_tensor(window_audio[None, :], dtype=tf.float32)  # [1, T]

        features = self.model.extract_audio_features(x, training=False)
        emb_s, _ = self.model.encode_speech_features(features, training=False)

        emb_s_batch = tf.repeat(
            emb_s,
            repeats=self.num_wake_words,
            axis=0,
        )

        # pylint: disable=protected-access
        if hasattr(emb_s, "_keras_mask") and emb_s._keras_mask is not None:
            # pylint: disable=protected-access
            emb_s_batch._keras_mask = tf.repeat(
                emb_s._keras_mask,
                repeats=self.num_wake_words,
                axis=0,
            )

        prob, *_ = self.model.score_from_features_and_text_encoding(
            emb_s_batch,
            self.emb_t_batch,
        )

        return prob.numpy().reshape(-1)

    def process_chunk(
        self,
        chunk: np.ndarray,
    ) -> Optional[Collection[int]]:
        """
        Process an incoming audio chunk of arbitrary size.

        Returns:
            None if no wake words detected,
            otherwise a collection of wake word indexes.
        """

        chunk = np.asarray(chunk, dtype=np.float32).reshape(-1)

        self.buffer = np.concatenate([self.buffer, chunk])
        self.samples_since_last_score += len(chunk)
        self.total_samples_seen += len(chunk)

        fired_indexes = set()

        while self.samples_since_last_score >= self.hop_samples:
            self.samples_since_last_score -= self.hop_samples

            end_offset = self.samples_since_last_score
            end_idx = len(self.buffer) - end_offset
            start_idx = end_idx - self.window_samples

            if start_idx < 0:
                continue

            window = self.buffer[start_idx:end_idx]

            # scores shape: [num_wake_words]
            scores = self.score_window(window)

            self.scores.append(scores)

            _LOGGER.debug("Scores: %s", scores)

            detected = scores >= self.detect_threshold

            # Update per-word counters
            self.consecutive_hits[detected] += 1
            self.consecutive_hits[~detected] = 0

            # Check which fired
            fired_now = np.where(self.consecutive_hits >= self.trigger_count)[0]

            if len(fired_now) > 0:
                fired_indexes.update(fired_now.tolist())

                for idx in fired_now:
                    self.consecutive_hits[idx] = 0

                _LOGGER.debug(
                    "Detected wake words: %s",
                    [self.wake_words[i] for i in fired_now],
                )

                break

        max_buffer = self.window_samples + self.hop_samples

        if len(self.buffer) > max_buffer:
            self.buffer = self.buffer[-max_buffer:]

        if fired_indexes:
            return sorted(fired_indexes)

        return None

    def _encode_wake_words(self):
        emb_w = []

        for wake_word in self.wake_words:
            if _LOGGER.isEnabledFor(logging.DEBUG):
                phonemes = self.g2p(wake_word)
                _LOGGER.debug(
                    "Phonemes for '%s': %s",
                    wake_word,
                    phonemes,
                )

            emb = self.g2p.embedding(wake_word)

            # Ensure batch dimension exists
            if emb.ndim == 2:
                emb = emb[None, :, :]

            emb_w.append(emb)

        # Find maximum sequence length
        max_len = max(e.shape[1] for e in emb_w)

        padded = []

        for e in emb_w:
            pad_len = max_len - e.shape[1]

            if pad_len > 0:
                e = np.pad(
                    e,
                    ((0, 0), (0, pad_len), (0, 0)),
                    mode="constant",
                )

            padded.append(e)

        # Stack into batch
        text_batch = tf.convert_to_tensor(
            np.concatenate(padded, axis=0),
            dtype=tf.float32,
        )

        # Encode once
        emb_t = self.model.encode_text(
            text_batch,
            training=False,
        )

        return emb_t


def find_latest_checkpoint(path: str) -> str:
    if os.path.isdir(path):
        ckpt = tf.train.latest_checkpoint(path)
        if ckpt is None:
            raise FileNotFoundError(f"No TensorFlow checkpoint found in: {path}")
        return ckpt
    return path
