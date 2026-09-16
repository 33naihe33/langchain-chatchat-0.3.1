#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODEL_FILE="${MINICPM5_MODEL_FILE:-${PROJECT_ROOT}/models/minicpm5-2b/MiniCPM5-2B-Q4_K_M.gguf}"
CPU_THREADS="${CPU_THREADS:-$(sysctl -n hw.physicalcpu 2>/dev/null || getconf _NPROCESSORS_ONLN)}"
LLAMA_SERVER="${LLAMA_SERVER:-${PROJECT_ROOT}/runtime/llama.cpp/llama-server}"

exec "${LLAMA_SERVER}" -m "${MODEL_FILE}" -ngl 0 -c 8192 -t "${CPU_THREADS}" \
  --host 127.0.0.1 --port 8080 --alias MiniCPM5-2B --jinja \
  --temp 1.0 --top-p 0.95 --min-p 0.0
