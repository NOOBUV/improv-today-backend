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

- `GET /health` → adds `streaming` and `streaming_endpoint` to the engine readiness above
  (`engine` is whichever is tried first; `qwen3` reads `unavailable (...)` if it failed to load)
- `POST /v1/audio/speech/stream` → **the fast path.** Raw PCM `s16le` mono, one chunk per
  ~`STREAM_INTERVAL_S` of audio, sample rate in the `X-Sample-Rate` header,
  `Content-Type: application/octet-stream`. Not WAV: a header written before the length is
  known is a lie the client then has to un-believe.
- `POST /v1/audio/speech` → the whole WAV in one response. Kept for callers that can't stream.

Both take the same OpenAI-shaped body plus an emotion:
`{"input": "...", "emotion": "calm", "voice": "af_heart", "speed": 1.0}`

`emotion` is one of Clara's `EmotionType` values — `calm` (default), `happy`, `sad`,
`stressed`, `sassy` — mapped to a Qwen3 `instruct` string by `INSTRUCT` in `app.py`.
Anything unrecognised falls back to `calm` rather than erroring. `voice` and `speed`
apply to the Kokoro path only; Qwen3 is pinned to `vivian`.

Warm, on an M2: first chunk in ~0.9-1.1s whatever the sentence length, then generation runs
a little faster than playback. The whole-WAV endpoint instead costs the full synthesis before
a single byte — 3.5s for a short line, 8.8s for a long one.

### Fallback on the streaming path

If Qwen3 hasn't produced its **first** chunk within `FIRST_CHUNK_TIMEOUT_S`, or dies before it,
the response is a complete Kokoro WAV instead — so callers must branch on `Content-Type`.
Nothing audible has been committed at that point, so the swap is invisible. Once chunks are
flowing the stream is allowed to run to `MAX_TOKENS`; `SYNTH_TIMEOUT_S` applies to the
whole-WAV endpoint only. The deadline starts when the generation actually owns the model, not
when the request arrived — otherwise a prefetched sentence queued behind its predecessor would
fail it every time.

## How the frontend uses it

`BrowserSpeechService` (frontend `src/lib/speech.ts`) POSTs each sentence to the streaming
endpoint and schedules the PCM chunks back-to-back through Web Audio, so Clara starts talking
while the rest is still being generated. It reads one sentence ahead: when sentence N starts
playing, N+1's request is already in flight and waiting on this server's lock. The turn's
emotion comes off the conversation stream's `context_ready` event — which lands before the
first chunk, so even the opening sentence is in character. On any failure — server down,
non-200, fetch error — it falls back to `window.speechSynthesis` silently. Nothing breaks if
this server isn't running; Clara just sounds worse. CORS allows `http://localhost:3000` only,
and must expose `X-Sample-Rate` or the browser can't decode what it's given.

## Notes

- ponytail: one process, one lock around the model, no cache. The frontend's read-ahead means
  a second request now genuinely waits at that lock — it starts the instant the current
  sentence finishes generating, which is what you want from one GPU. Kokoro has its own lock so
  it stays servable while an abandoned Qwen3 generation winds down.
- Qwen3 occasionally rambles on low-energy lines, so generation is capped at `MAX_TOKENS`
  (~20s of audio at the 12Hz codec rate) and the whole call at `SYNTH_TIMEOUT_S`.
- Parakeet STT can live in this same app later — same venv, add a `/v1/audio/transcriptions`
  route. That's the reason this is a standalone FastAPI app and not a script.
