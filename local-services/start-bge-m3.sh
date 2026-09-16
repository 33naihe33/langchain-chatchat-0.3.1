#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BGE_M3_ENV="${BGE_M3_ENV:-bge-m3}"
export BGE_M3_MODEL_PATH="${BGE_M3_MODEL_PATH:-${PROJECT_ROOT}/models/bge-m3}"

exec conda run --no-capture-output -n "${BGE_M3_ENV}" python -m uvicorn bge_m3_service:create_app \
  --factory --app-dir "${PROJECT_ROOT}/local-services" \
  --host 127.0.0.1 --port 8081 --workers 1
