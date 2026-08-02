"""Local Kokoro TTS server for Clara. Host-side only — deliberately not part of the API image."""

import io
from pathlib import Path

import soundfile as sf
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from kokoro_onnx import Kokoro
from pydantic import BaseModel

DEFAULT_VOICE = "af_heart"
MODELS = Path(__file__).parent / "models"

kokoro = Kokoro(str(MODELS / "kokoro-v1.0.onnx"), str(MODELS / "voices-v1.0.bin"))

app = FastAPI(title="Clara Voice")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_methods=["POST", "GET"],
    allow_headers=["*"],
)


class SpeechRequest(BaseModel):
    input: str
    voice: str = DEFAULT_VOICE
    speed: float = 1.0


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "voice": DEFAULT_VOICE}


@app.post("/v1/audio/speech")
def speech(req: SpeechRequest) -> Response:
    # ponytail: synchronous — one browser, one sentence at a time. Add a thread pool if
    # concurrent callers ever show up.
    samples, sample_rate = kokoro.create(req.input, voice=req.voice, speed=req.speed, lang="en-us")
    buf = io.BytesIO()
    sf.write(buf, samples, sample_rate, format="WAV")
    return Response(buf.getvalue(), media_type="audio/wav")
