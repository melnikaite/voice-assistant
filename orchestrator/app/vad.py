import logging
import os

import numpy as np
from openwakeword.vad import VAD

log = logging.getLogger(__name__)

# Defaults overridable via env vars. silence_timeout especially matters for UX:
# too short = cuts off thinking pauses; too long = sluggish.
DEFAULT_SILENCE_TIMEOUT_S = float(os.environ.get("VAD_SILENCE_TIMEOUT_S", "1.5"))
DEFAULT_MIN_SPEECH_S = float(os.environ.get("VAD_MIN_SPEECH_S", "0.3"))
DEFAULT_MAX_RECORD_S = float(os.environ.get("VAD_MAX_RECORD_S", "20"))
DEFAULT_SPEECH_THRESHOLD = float(os.environ.get("VAD_SPEECH_THRESHOLD", "0.5"))

FRAME_SAMPLES = 480  # 30ms @ 16kHz — silero VAD's recommended frame size
FRAME_BYTES = FRAME_SAMPLES * 2
SAMPLE_RATE = 16000


class VadEndpointer:
    """
    Server-side endpointing: feed Int16 PCM continuously, returns True once we
    have observed `min_speech` of speech followed by `silence_timeout` of silence.
    """

    def __init__(
        self,
        speech_threshold: float = DEFAULT_SPEECH_THRESHOLD,
        min_speech_s: float = DEFAULT_MIN_SPEECH_S,
        silence_timeout_s: float = DEFAULT_SILENCE_TIMEOUT_S,
        max_record_s: float = DEFAULT_MAX_RECORD_S,
    ):
        self._vad = VAD()
        self._th = speech_threshold
        self._min_speech_samples = int(min_speech_s * SAMPLE_RATE)
        self._silence_samples_to_end = int(silence_timeout_s * SAMPLE_RATE)
        self._max_samples = int(max_record_s * SAMPLE_RATE)
        log.info(
            "VAD endpointer: th=%.2f, min_speech=%.2fs, silence_timeout=%.2fs, max=%.0fs",
            speech_threshold,
            min_speech_s,
            silence_timeout_s,
            max_record_s,
        )
        self._buf = bytearray()
        self._speech_samples = 0
        self._silence_samples = 0
        self._total_samples = 0
        self._strict_mode = False

    def feed(self, pcm: bytes) -> bool:
        """Returns True when endpoint condition is met (speech followed by silence,
        OR hard cap on duration). After True is returned, caller should call reset()."""
        self._buf.extend(pcm)
        self._total_samples += len(pcm) // 2
        ended = False
        # Use stricter thresholds while the assistant is speaking (TTS playback).
        # Echo from speakers can otherwise trigger a false barge-in.
        if self._strict_mode:
            speech_th = max(self._th + 0.2, 0.7)
            min_speech_samples = max(self._min_speech_samples, int(0.5 * SAMPLE_RATE))
        else:
            speech_th = self._th
            min_speech_samples = self._min_speech_samples
        while len(self._buf) >= FRAME_BYTES:
            chunk = bytes(self._buf[:FRAME_BYTES])
            del self._buf[:FRAME_BYTES]
            audio = np.frombuffer(chunk, dtype=np.int16)
            try:
                score = float(self._vad.predict(audio))
            except Exception as e:
                log.warning("vad predict failed: %s", e)
                continue
            is_speech = score >= speech_th
            if is_speech:
                self._speech_samples += FRAME_SAMPLES
                self._silence_samples = 0
            else:
                if self._speech_samples > 0:
                    self._silence_samples += FRAME_SAMPLES
                    if (
                        self._speech_samples >= min_speech_samples
                        and self._silence_samples >= self._silence_samples_to_end
                    ):
                        ended = True
                        break
        if not ended and self._total_samples >= self._max_samples:
            log.info("vad: max recording length reached")
            ended = True
        return ended

    @property
    def cumulative_speech_samples(self) -> int:
        """
        Total speech samples observed since the last reset.  Used by the
        barge-in detector in ws.py: we don't want to wait for a full
        endpointer cycle (which requires a trailing silence), just to know
        that the user *started talking* during PROCESSING.
        """
        return self._speech_samples

    def set_strict(self, strict: bool) -> None:
        """
        Strict mode raises the speech threshold and the minimum speech length
        required to commit a barge-in. Use while TTS is playing so faint echo
        from speakers doesn't trigger an interrupt.
        """
        self._strict_mode = strict

    def reset(self):
        self._buf.clear()
        self._vad.reset_states()
        self._speech_samples = 0
        self._silence_samples = 0
        self._total_samples = 0
        self._strict_mode = False
