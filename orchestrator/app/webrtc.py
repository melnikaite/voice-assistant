"""
WebRTC peer connection: replaces our raw WebSocket-PCM audio transport.

Why this exists:
  • Browsers' built-in echo cancellation (AEC3) only works when both the
    outbound speaker audio AND the inbound mic come through the WebRTC
    voice engine.  When we played TTS via Web Speech API, AEC had no
    reference and could not cancel speaker→mic feedback — forcing us to
    half-duplex mute the mic during playback.  Routing both directions
    through an RTCPeerConnection puts everything inside the engine that
    knows how to subtract one from the other.
  • Server gets PCM 48 kHz mono (after Opus decode) on the inbound side
    and feeds a custom outbound AudioStreamTrack (the assistant's voice)
    with frames it produces from Piper TTS.

This module is a thin asyncio wrapper around aiortc.  ws.py owns the
signalling (WS messages: ``webrtc_offer`` / ``webrtc_answer`` /
``webrtc_ice``) and hands an ``RtcSession`` instance the audio source +
mic-frame callback.

Audio I/O contracts:
  Outbound (server → browser, the assistant's voice):
    push_pcm(samples, sample_rate)
        ``samples`` is int16 mono numpy array at any rate the TTS uses
        (typically 22050).  We resample to 48 kHz on the way out — Opus
        in WebRTC is 48 kHz mono.

  Inbound (browser → server, user's mic):
    on_mic_pcm(samples_16k)
        Called as 20 ms frames of int16 mono at 16 kHz, the rate the
        rest of the pipeline (VAD, Whisper) expects.  We resample from
        48 kHz on the way in.
"""
from __future__ import annotations

import asyncio
import fractions
import logging
from collections.abc import Callable
from typing import Awaitable

import numpy as np
from aiortc import (
    MediaStreamTrack,
    RTCIceCandidate,
    RTCPeerConnection,
    RTCSessionDescription,
)
from aiortc.sdp import candidate_from_sdp
from av import AudioFrame
from av.audio.resampler import AudioResampler

log = logging.getLogger(__name__)

# WebRTC's standard audio rate (Opus).  Frames sent into the outbound
# track must match this rate or the encoder will reject them.
RTC_SAMPLE_RATE = 48000
# Whisper / silero VAD speak 16 kHz — we resample the inbound mic to this.
WHISPER_SAMPLE_RATE = 16000
# Frame size we emit on the outbound track.  20 ms × 48 kHz = 960 samples.
# Standard Opus frame duration; matches what aiortc expects.
RTC_SAMPLES_PER_FRAME = 960
# Same 20 ms on the inbound side at 16 kHz: 320 samples per frame.
# We re-chunk decoded mic audio into frames of this size before handing
# them to the pipeline — matches what the old binary-PCM path produced.
MIC_FRAME_SAMPLES_16K = 320


# ──────────────────────────────────────────────────────────────────────────
# Outbound track: server-generated TTS audio
# ──────────────────────────────────────────────────────────────────────────


class TtsAudioTrack(MediaStreamTrack):
    """Audio track the server publishes for the assistant's voice.

    The track is always "playing": ``recv()`` returns a frame every 20 ms.
    When we have TTS audio queued the frame contains the next 960 samples
    of it; otherwise it's silence.  Continuous output is required so the
    browser's AEC keeps converging and the Opus stream doesn't stall.

    Audio gets fed in via :meth:`push_pcm` from arbitrary sample rates
    (Piper outputs 22 050 Hz, etc).  We resample to 48 kHz internally,
    chunk to 20 ms frames, and hand them out from ``recv()``.

    Call :meth:`cancel_queued` on barge-in: drops every queued frame so
    the assistant immediately falls silent.  The next recv() returns
    silence; if the caller pushes new audio it'll start playing right
    away.
    """

    kind = "audio"

    def __init__(self) -> None:
        super().__init__()
        # Resampler that takes whatever PCM the caller hands us and
        # produces 48 kHz mono int16 frames.  A single instance keeps
        # internal state for cross-call continuity (e.g. fractional-rate
        # adapters across chunk boundaries).
        self._resampler = AudioResampler(
            format="s16", layout="mono", rate=RTC_SAMPLE_RATE
        )
        # Asyncio queue of int16 numpy arrays (size = RTC_SAMPLES_PER_FRAME).
        # recv() pops one per call.
        self._frames: asyncio.Queue[np.ndarray] = asyncio.Queue()
        # Leftover samples from a push that didn't fall on a 20 ms boundary;
        # prepended to the next push so we don't lose audio at chunk seams.
        self._tail: np.ndarray = np.zeros(0, dtype=np.int16)
        # PTS counter — Opus packets need a monotonic timestamp; we advance
        # by RTC_SAMPLES_PER_FRAME per emitted frame.
        self._pts: int = 0
        self._time_base = fractions.Fraction(1, RTC_SAMPLE_RATE)

    async def push_pcm(self, samples: np.ndarray, sample_rate: int) -> None:
        """Enqueue mono int16 ``samples`` at ``sample_rate`` for playback."""
        if samples.size == 0:
            return
        # Wrap as an AudioFrame the resampler accepts.  PyAV's resampler
        # likes (1, N) for mono and the planar s16 format identifier ``s16``.
        src = AudioFrame.from_ndarray(
            samples.reshape(1, -1), format="s16", layout="mono"
        )
        src.sample_rate = int(sample_rate)
        # resample returns a list of AudioFrames at the target rate.
        for resampled in self._resampler.resample(src):
            data = resampled.to_ndarray().reshape(-1).astype(np.int16, copy=False)
            await self._enqueue_16khz_or_48khz(data)

    async def _enqueue_16khz_or_48khz(self, data: np.ndarray) -> None:
        """Chunk a flat int16 array into 20 ms frames + push to queue."""
        if self._tail.size:
            data = np.concatenate([self._tail, data])
            self._tail = np.zeros(0, dtype=np.int16)
        # Slice into RTC_SAMPLES_PER_FRAME windows; keep the tail.
        n_full = data.size // RTC_SAMPLES_PER_FRAME
        for i in range(n_full):
            start = i * RTC_SAMPLES_PER_FRAME
            await self._frames.put(
                data[start : start + RTC_SAMPLES_PER_FRAME].copy()
            )
        leftover = data.size - n_full * RTC_SAMPLES_PER_FRAME
        if leftover:
            self._tail = data[-leftover:].copy()

    def cancel_queued(self) -> None:
        """Drop everything pending (barge-in)."""
        dropped = 0
        while not self._frames.empty():
            try:
                self._frames.get_nowait()
                dropped += 1
            except asyncio.QueueEmpty:
                break
        self._tail = np.zeros(0, dtype=np.int16)
        if dropped:
            log.info("tts track: dropped %d queued frames (barge-in)", dropped)

    def has_pending(self) -> bool:
        return not self._frames.empty() or self._tail.size > 0

    async def recv(self) -> AudioFrame:
        # Pace ourselves at 20 ms per frame so we don't busy-loop when we
        # have a backlog.  aiortc will call recv() on its schedule; we
        # additionally sleep so the queue doesn't get drained instantly.
        try:
            samples = await asyncio.wait_for(self._frames.get(), timeout=0.02)
        except asyncio.TimeoutError:
            samples = np.zeros(RTC_SAMPLES_PER_FRAME, dtype=np.int16)
        frame = AudioFrame.from_ndarray(
            samples.reshape(1, -1), format="s16", layout="mono"
        )
        frame.sample_rate = RTC_SAMPLE_RATE
        frame.pts = self._pts
        frame.time_base = self._time_base
        self._pts += RTC_SAMPLES_PER_FRAME
        return frame


# ──────────────────────────────────────────────────────────────────────────
# Inbound consumer: pull mic frames, decode/resample, feed pipeline
# ──────────────────────────────────────────────────────────────────────────


async def _consume_mic_track(
    track: MediaStreamTrack,
    on_pcm: Callable[[bytes], Awaitable[None]],
) -> None:
    """Pull frames from an incoming audio track, resample to 16 kHz mono,
    re-chunk into 20 ms (320 sample) buffers, and hand each one as raw
    bytes to ``on_pcm``.

    The 320-sample chunk size matches what our binary-WS PCM frames used
    to be, so the downstream VAD endpointer (silero @ 16 kHz, 30 ms hops)
    sees the same input shape as before.

    Runs until the track ends (peer disconnects) or an exception.
    Exceptions other than CancelledError are logged but swallowed —
    callers can't usefully recover and we want clean shutdown.
    """
    resampler = AudioResampler(
        format="s16", layout="mono", rate=WHISPER_SAMPLE_RATE
    )
    tail = np.zeros(0, dtype=np.int16)
    frames_in = 0
    try:
        while True:
            frame = await track.recv()
            frames_in += 1
            for r in resampler.resample(frame):
                data = r.to_ndarray().reshape(-1).astype(np.int16, copy=False)
                if tail.size:
                    data = np.concatenate([tail, data])
                    tail = np.zeros(0, dtype=np.int16)
                n_full = data.size // MIC_FRAME_SAMPLES_16K
                for i in range(n_full):
                    start = i * MIC_FRAME_SAMPLES_16K
                    chunk = data[start : start + MIC_FRAME_SAMPLES_16K]
                    await on_pcm(chunk.tobytes())
                leftover = data.size - n_full * MIC_FRAME_SAMPLES_16K
                if leftover:
                    tail = data[-leftover:].copy()
    except asyncio.CancelledError:
        raise
    except Exception:
        log.exception("mic track consumer crashed after %d frames", frames_in)


# ──────────────────────────────────────────────────────────────────────────
# Per-WS peer session
# ──────────────────────────────────────────────────────────────────────────


class RtcSession:
    """One WebRTC peer connection bound to one WebSocket client.

    The WS layer hands us an offer SDP, we set up the peer connection,
    publish the outbound TTS track, listen for the inbound mic track,
    and answer.  ICE candidates flow both ways via the WS — we don't
    need a separate signalling server because the WS is already there.

    Lifecycle:
        s = RtcSession(on_mic_pcm=..., on_state=...)
        answer_sdp, answer_type = await s.set_remote_offer(offer_sdp, offer_type)
        # send answer to client via WS, then exchange ICE candidates with
        # add_remote_ice(candidate_dict)
        # ...
        await s.close()
    """

    def __init__(
        self,
        *,
        on_mic_pcm: Callable[[bytes], Awaitable[None]],
        send_signal: Callable[[dict], Awaitable[None]],
        session_id: int,
    ) -> None:
        self._on_mic_pcm = on_mic_pcm
        self._send_signal = send_signal
        self._session_id = session_id

        self._pc = RTCPeerConnection()
        self._tts_track = TtsAudioTrack()
        # Add the outbound track BEFORE answering — its m= line must be in
        # the SDP we return, otherwise the browser can't route remote
        # audio to its <audio> element.
        self._pc.addTrack(self._tts_track)
        self._mic_task: asyncio.Task | None = None

        @self._pc.on("track")
        def _on_track(track: MediaStreamTrack) -> None:
            log.info(
                "rtc[%d]: inbound track kind=%s id=%s",
                self._session_id, track.kind, track.id,
            )
            if track.kind == "audio":
                # Spawn a consumer task; cancelled on pc close.
                self._mic_task = asyncio.create_task(
                    _consume_mic_track(track, self._on_mic_pcm)
                )

                @track.on("ended")
                async def _on_ended() -> None:
                    log.info("rtc[%d]: mic track ended", self._session_id)
                    if self._mic_task and not self._mic_task.done():
                        self._mic_task.cancel()

        @self._pc.on("iceconnectionstatechange")
        async def _on_ice_change() -> None:
            log.info(
                "rtc[%d]: ICE state %s",
                self._session_id, self._pc.iceConnectionState,
            )

        @self._pc.on("connectionstatechange")
        async def _on_conn_change() -> None:
            log.info(
                "rtc[%d]: connection state %s",
                self._session_id, self._pc.connectionState,
            )

    # ---- TTS playback API used by the WS / pipeline layer ----

    async def push_tts_pcm(self, samples: np.ndarray, sample_rate: int) -> None:
        await self._tts_track.push_pcm(samples, sample_rate)

    def cancel_tts(self) -> None:
        self._tts_track.cancel_queued()

    def tts_busy(self) -> bool:
        return self._tts_track.has_pending()

    # ---- Signalling ----

    async def set_remote_offer(self, sdp: str, sdp_type: str) -> tuple[str, str]:
        """Apply the browser's offer and return our answer (SDP, type)."""
        await self._pc.setRemoteDescription(
            RTCSessionDescription(sdp=sdp, type=sdp_type)
        )
        answer = await self._pc.createAnswer()
        await self._pc.setLocalDescription(answer)
        return self._pc.localDescription.sdp, self._pc.localDescription.type

    async def add_remote_ice(self, candidate: dict) -> None:
        """Apply a remote ICE candidate from the browser.

        The browser sends ``{candidate: "...", sdpMid: "...",
        sdpMLineIndex: N}``; aiortc wants an ``RTCIceCandidate`` parsed
        from the SDP fragment via the helper exported from aiortc.sdp.
        Empty / null candidate (end-of-candidates signal) is ignored —
        aiortc handles that internally.
        """
        cand_str = candidate.get("candidate") or ""
        if not cand_str:
            return
        try:
            ice = candidate_from_sdp(cand_str)
        except Exception:
            log.warning(
                "rtc[%d]: failed to parse remote ICE: %r",
                self._session_id, cand_str,
            )
            return
        ice.sdpMid = candidate.get("sdpMid")
        ice.sdpMLineIndex = candidate.get("sdpMLineIndex")
        try:
            await self._pc.addIceCandidate(ice)
        except Exception:
            log.exception(
                "rtc[%d]: addIceCandidate failed for %r",
                self._session_id, cand_str,
            )

    async def close(self) -> None:
        if self._mic_task and not self._mic_task.done():
            self._mic_task.cancel()
        try:
            await self._pc.close()
        except Exception:
            log.exception("rtc[%d]: error closing PC", self._session_id)
