# Clara voice server (Kokoro TTS)

Local, host-side text-to-speech so Clara has a real voice instead of the browser's
robot. Runs OUTSIDE docker on purpose — the models are ~350MB and have no business
in the prod API image.

## Setup (once)

```
./setup.sh
```

Makes a python 3.12 venv (onnxruntime has no wheels for the host's 3.14), installs
`voice/requirements.txt`, and downloads the two Kokoro model files into `voice/models/`
(gitignored).

## Run

```
cd voice && .venv/bin/uvicorn app:app --port 8880
```

## API

- `GET /health` → `{"status":"ok","voice":"af_heart"}`
- `POST /v1/audio/speech` → WAV bytes. OpenAI-shaped body: `{"input": "...", "voice": "af_heart", "speed": 1.0}`

Default voice is `af_heart` (`DEFAULT_VOICE` in `app.py`). Kokoro ships ~50 more —
`af_bella`, `af_nicole`, `bf_emma`, etc.

## How the frontend uses it

`BrowserSpeechService.speakChunk()` (frontend `src/lib/speech.ts`) POSTs each sentence
here and plays the WAV. On any failure — server down, non-200, fetch error — it falls
back to `window.speechSynthesis` silently. Nothing breaks if this server isn't running;
Clara just sounds worse. CORS allows `http://localhost:3000` only.

## Notes

- ponytail: one process, synchronous inference, no queue, no cache. ~1.7s to synthesize
  3s of speech on an M2 (faster than realtime), and there is exactly one browser talking
  to it. Add a thread pool / audio cache when that stops being true.
- Parakeet STT can live in this same app later — same venv, add a `/v1/audio/transcriptions`
  route. That's the reason this is a standalone FastAPI app and not a script.
