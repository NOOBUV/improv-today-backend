"""Local TTS server for Clara. Host-side only — deliberately not part of the API image.

Qwen3-TTS (emotion-conditioned, MLX) is the voice; Kokoro stays loaded as the fallback
for when Qwen3 fails or runs long.
"""

import io
import logging
import queue
import threading
import time
from pathlib import Path

import numpy as np
import soundfile as sf
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response, StreamingResponse
from kokoro_onnx import Kokoro
from mlx_audio.tts.utils import load_model
from pydantic import BaseModel

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("clara.voice")

MODELS = Path(__file__).parent / "models"
KOKORO_VOICE = "af_heart"
QWEN_REPO = "mlx-community/Qwen3-TTS-12Hz-0.6B-CustomVoice-bf16"
QWEN_SPEAKER = "vivian"  # undocumented, but the only stable young-female identity on English

# One instruct string per Clara emotion (app/schemas/clara.py::EmotionType). Wording matters
# for more than tone: "playful, teasing, confident" made every sassy line ramble past the
# timeout, the phrasing below lands the same character in ~6.5s.
INSTRUCT = {
    "calm": "calm and warm",
    "happy": "very happy and excited",
    "sad": "quiet, subdued, a little down",
    "stressed": "tense and hurried",
    "sassy": "sassy and sarcastic, playful",
}
DEFAULT_EMOTION = "calm"

# Low-energy lines occasionally ramble. 12Hz codec => ~12 audio tokens per second, so this
# caps a single sentence at ~20s — several times the longest thing Clara actually writes.
MAX_TOKENS = 12 * 20
# Measured worst case for a real reply is ~9.4s (a 20-word sentence read sad, which is
# genuinely slow speech rather than a ramble). Anything past this is better served fast and
# flat by Kokoro than beautifully a quarter-minute late.
# Only the whole-WAV endpoint pays this: on the streaming path the client is already hearing
# audio by then, so a long sentence is fine — MAX_TOKENS is what stops a ramble there.
SYNTH_TIMEOUT_S = 10.0

# Streaming: audio seconds per chunk. Smaller = earlier first chunk (the decoder only runs
# once the tokens for a whole chunk exist) at the cost of more decoder calls.
STREAM_INTERVAL_S = 1.0
# Deadline for the FIRST chunk, measured from when the generation actually owns MLX — not from
# request arrival, or a prefetched sentence queued behind its predecessor would fail it every
# time. Nothing audible has been committed yet at this point, so Kokoro can still take over.
FIRST_CHUNK_TIMEOUT_S = 4.0
# Ceiling on waiting for the lock itself, so a wedged generation can't pin a request forever.
LOCK_WAIT_S = 30.0

kokoro = Kokoro(str(MODELS / "kokoro-v1.0.onnx"), str(MODELS / "voices-v1.0.bin"))


def _load_qwen():
    """Load + run one generation, so the first real request isn't paying for warm-up."""
    model = load_model(QWEN_REPO)
    for _ in model.generate(
        text="Warming up.", voice=QWEN_SPEAKER, instruct=INSTRUCT[DEFAULT_EMOTION], stream=True
    ):
        pass
    return model


t0 = time.monotonic()
try:
    qwen = _load_qwen()
    qwen_error = None
    log.info("Qwen3 (%s/%s) warm in %.1fs", QWEN_REPO, QWEN_SPEAKER, time.monotonic() - t0)
except Exception as exc:  # missing weights, no metal, OOM — Kokoro still serves
    qwen, qwen_error = None, f"{type(exc).__name__}: {exc}"
    log.warning("Qwen3 unavailable, serving Kokoro only: %s", qwen_error)

# ponytail: one model, one caller. The frontend prefetches the next sentence while the current
# one plays, so this IS contended now — a queued request waits here and starts the moment MLX
# frees up. Kokoro gets its own lock: it must stay servable while an abandoned Qwen3 generation
# is still winding down.
synth_lock = threading.Lock()
kokoro_lock = threading.Lock()

app = FastAPI(title="Clara Voice")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_methods=["POST", "GET"],
    allow_headers=["*"],
    expose_headers=["X-Sample-Rate", "X-Engine"],  # the streamed PCM is undecodable without them
)


class SpeechRequest(BaseModel):
    input: str
    voice: str = KOKORO_VOICE  # Kokoro speaker; ignored by Qwen3, which is pinned to vivian
    speed: float = 1.0
    emotion: str = DEFAULT_EMOTION


def _wav(samples, sample_rate: int) -> bytes:
    buf = io.BytesIO()
    sf.write(buf, samples, sample_rate, format="WAV")
    return buf.getvalue()


def _qwen_speak(text: str, emotion: str) -> bytes:
    """Raises on anything, including overrunning SYNTH_TIMEOUT_S, so the caller can fall back.

    ponytail: the deadline is checked per streamed chunk, not enforced by a killable thread —
    MAX_TOKENS is what bounds the loop, this just stops us waiting out a slow one.
    """
    deadline = time.monotonic() + SYNTH_TIMEOUT_S
    chunks, sample_rate = [], 24000
    for result in qwen.generate(
        text=text,
        voice=QWEN_SPEAKER,
        instruct=INSTRUCT.get(emotion, INSTRUCT[DEFAULT_EMOTION]),
        max_tokens=MAX_TOKENS,
        stream=True,
    ):
        sample_rate = result.sample_rate
        chunks.append(np.asarray(result.audio, dtype=np.float32).reshape(-1))
        if time.monotonic() > deadline:
            raise TimeoutError(f"exceeded {SYNTH_TIMEOUT_S}s")
    return _wav(np.concatenate(chunks), sample_rate)


def _pcm16(audio) -> bytes:
    """Float samples in [-1, 1] to the raw little-endian int16 the browser decodes."""
    samples = np.asarray(audio, dtype=np.float32).reshape(-1)
    return (np.clip(samples, -1.0, 1.0) * 32767).astype(np.int16).tobytes()


_LOCKED = object()  # "MLX is ours now" — starts the first-chunk clock


def _qwen_worker(text: str, emotion: str, out: queue.Queue, cancel: threading.Event) -> None:
    """Generate in a thread so the request can give up on a slow start and serve Kokoro.

    Puts _LOCKED, then (sample_rate, pcm) per chunk, then None. An exception goes in the queue
    instead — the consumer decides whether that means fallback or a truncated stream.
    """
    with synth_lock:
        out.put(_LOCKED)
        try:
            for result in qwen.generate(
                text=text,
                voice=QWEN_SPEAKER,
                instruct=INSTRUCT.get(emotion, INSTRUCT[DEFAULT_EMOTION]),
                max_tokens=MAX_TOKENS,
                stream=True,
                streaming_interval=STREAM_INTERVAL_S,
            ):
                if cancel.is_set():  # client left, or we already fell back to Kokoro
                    break
                out.put((result.sample_rate, _pcm16(result.audio)))
        except Exception as exc:
            out.put(exc)
        out.put(None)


def _kokoro_response(req: "SpeechRequest") -> Response:
    with kokoro_lock:
        samples, sample_rate = kokoro.create(
            req.input, voice=req.voice, speed=req.speed, lang="en-us"
        )
    return Response(_wav(samples, sample_rate), media_type="audio/wav")


@app.get("/health")
def health() -> dict[str, str]:
    return {
        "status": "ok",
        "engine": "qwen3" if qwen else "kokoro",
        "qwen3": "ready" if qwen else f"unavailable ({qwen_error})",
        "qwen3_speaker": QWEN_SPEAKER,
        "kokoro": "ready",
        "voice": KOKORO_VOICE,
        "streaming": "ready" if qwen else "unavailable (needs qwen3)",
        "streaming_endpoint": "/v1/audio/speech/stream",
    }


@app.post("/v1/audio/speech/stream")
def speech_stream(req: SpeechRequest):
    """Audio as it's generated: raw PCM s16le mono, rate in X-Sample-Rate.

    Deliberately not WAV — a header written before the length is known would be a lie the
    client has to un-believe. If Qwen3 can't produce its first chunk in time the response is
    instead a complete Kokoro WAV, so callers must branch on Content-Type.
    """
    if qwen is None:
        return _kokoro_response(req)

    q: queue.Queue = queue.Queue()
    cancel = threading.Event()
    threading.Thread(
        target=_qwen_worker, args=(req.input, req.emotion, q, cancel), daemon=True
    ).start()

    started = time.monotonic()
    try:
        q.get(timeout=LOCK_WAIT_S)  # _LOCKED
        first = q.get(timeout=FIRST_CHUNK_TIMEOUT_S)
        if not isinstance(first, tuple):
            raise first if isinstance(first, Exception) else RuntimeError("no audio generated")
    except Exception as exc:
        cancel.set()  # the worker stops at its next chunk boundary and frees MLX
        log.warning(
            "Qwen3 stream had no first chunk after %.2fs (%s: %s) — falling back to Kokoro",
            time.monotonic() - started, type(exc).__name__, exc,
        )
        return _kokoro_response(req)

    sample_rate, first_pcm = first
    log.info(
        "qwen3-stream/%s first chunk %.2fs %r",
        req.emotion, time.monotonic() - started, req.input[:60],
    )

    def body():
        try:
            yield first_pcm
            while True:
                item = q.get()
                if item is None:
                    return
                if isinstance(item, Exception):
                    log.warning("Qwen3 stream broke mid-flight: %s", item)
                    return
                yield item[1]
        finally:
            cancel.set()  # a client that hangs up mid-sentence shouldn't hold MLX

    return StreamingResponse(
        body(),
        media_type="application/octet-stream",
        headers={"X-Sample-Rate": str(sample_rate), "X-Engine": "qwen3"},
    )


@app.post("/v1/audio/speech")
def speech(req: SpeechRequest) -> Response:
    """Whole WAV, one response. Kept for callers that can't stream; /stream is the fast path."""
    if qwen is not None:
        started = time.monotonic()
        try:
            with synth_lock:
                wav = _qwen_speak(req.input, req.emotion)
            log.info("qwen3/%s %.2fs %r", req.emotion, time.monotonic() - started, req.input[:60])
            return Response(wav, media_type="audio/wav")
        except Exception as exc:
            log.warning(
                "Qwen3 failed after %.2fs (%s: %s) — falling back to Kokoro",
                time.monotonic() - started, type(exc).__name__, exc,
            )

    return _kokoro_response(req)
