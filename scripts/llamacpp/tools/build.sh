#!/usr/bin/env bash
# Compile dump_activations against the llama.cpp checkout in the llmfinetune repo.
# Nothing is written into that repo; the binary lands next to this script.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LLAMA_CPP="${LLAMA_CPP:-/media/oneiroid/sub/workspace/llmfinetune/vendor/llama.cpp}"
LLAMA_BUILD="${LLAMA_BUILD:-$LLAMA_CPP/build-cuda}"

if [ ! -d "$LLAMA_BUILD/bin" ]; then
  echo "no llama.cpp build at $LLAMA_BUILD -- set LLAMA_BUILD" >&2
  exit 1
fi

for src in dump_activations steer_generate; do
  g++ -O2 -std=c++17 \
    -I"$LLAMA_CPP/include" \
    -I"$LLAMA_CPP/ggml/include" \
    "$HERE/$src.cpp" \
    -L"$LLAMA_BUILD/bin" -lllama -lggml -lggml-base \
    -Wl,-rpath,"$LLAMA_BUILD/bin" \
    -o "$HERE/$src"
  echo "built $HERE/$src"
done
