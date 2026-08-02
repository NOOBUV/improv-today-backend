#!/usr/bin/env bash
# One-time setup for the local Kokoro voice server. Safe to re-run.
set -euo pipefail
cd "$(dirname "$0")"

REL=https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0

uv venv --python 3.12 .venv          # onnxruntime has no wheels for the host's python 3.14
uv pip install --python .venv -r requirements.txt

mkdir -p models
[ -f models/kokoro-v1.0.onnx ] || curl -L -o models/kokoro-v1.0.onnx "$REL/kokoro-v1.0.onnx"
[ -f models/voices-v1.0.bin ]  || curl -L -o models/voices-v1.0.bin  "$REL/voices-v1.0.bin"

echo "Done. Run: .venv/bin/uvicorn app:app --port 8880"
