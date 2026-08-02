# Clara voice server (Qwen3-TTS, Kokoro fallback)

Local, host-side text-to-speech so Clara has a real voice instead of the browser's
robot. Runs OUTSIDE docker on purpose — the models are GBs and have no business
in the prod API image.

Two engines, both loaded:

- **Qwen3-TTS** (`mlx-community/Qwen3-TTS-12Hz-0.6B-CustomVoice-bf16`, speaker `vivian`)
  is the voice. It takes an `instruct` string, which is how Clara's emotion reaches the
  audio instead of stopping at the transcript.
- **Kokoro** is the fallback. Any Qwen3 exception, or a synthesis that overruns
  `SYNTH_TIMEOUT_S`, logs a warning and gets answered by Kokoro instead.

## Setup (once)

```
./setup.sh
```

Makes a python 3.12 venv (onnxruntime has no wheels for the host's 3.14), installs
`voice/requirements.txt`, and downloads the two Kokoro model files into `voice/models/`
(gitignored). The ~2.3GB of Qwen3 weights download themselves into the default
Hugging Face cache (`~/.cache/huggingface`) the first time the server starts.

## Run

```
cd voice && .venv/bin/uvicorn app:app --port 8880
```

Startup loads Qwen3 and runs one throwaway generation so the first real request isn't
paying warm-up cost: ~10s once the weights are on disk, minutes on the first-ever run.

## API

- `GET /health` → `{"status":"ok","engine":"qwen3","qwen3":"ready","qwen3_speaker":"vivian","kokoro":"ready","voice":"af_heart"}`
  (`engine` is whichever is tried first; `qwen3` reads `unavailable (...)` if it failed to load)
- `POST /v1/audio/speech` → WAV bytes. OpenAI-shaped body plus an emotion:
  `{"input": "...", "emotion": "calm", "voice": "af_heart", "speed": 1.0}`

`emotion` is one of Clara's `EmotionType` values — `calm` (default), `happy`, `sad`,
`stressed`, `sassy` — mapped to a Qwen3 `instruct` string by `INSTRUCT` in `app.py`.
Anything unrecognised falls back to `calm` rather than erroring. `voice` and `speed`
apply to the Kokoro path only; Qwen3 is pinned to `vivian`.

Warm, on an M2: ~1.8s to first audio, roughly realtime overall.

## How the frontend uses it

`BrowserSpeechService.speakChunk()` (frontend `src/lib/speech.ts`) POSTs each sentence
here and plays the WAV. The turn's emotion comes off the conversation stream's
`context_ready` event — which lands before the first chunk, so even the opening sentence
is in character. On any failure — server down, non-200, fetch error — it falls back to
`window.speechSynthesis` silently. Nothing breaks if this server isn't running; Clara
just sounds worse. CORS allows `http://localhost:3000` only.

## Notes

- ponytail: one process, synchronous inference, no queue, no cache, one lock around the
  model. There is exactly one browser talking to it and it speaks a sentence at a time.
  Add a thread pool / audio cache when that stops being true.
- Qwen3 occasionally rambles on low-energy lines, so generation is capped at `MAX_TOKENS`
  (~20s of audio at the 12Hz codec rate) and the whole call at `SYNTH_TIMEOUT_S`.
- Parakeet STT can live in this same app later — same venv, add a `/v1/audio/transcriptions`
  route. That's the reason this is a standalone FastAPI app and not a script.
