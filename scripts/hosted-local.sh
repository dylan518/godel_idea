#!/usr/bin/env bash
# Run API + worker against local Homebrew Postgres/Redis (no Docker).
# Prereqs: brew services postgresql@16 redis; database evosci_hosted + role evo/evo;
#          python -m venv .venv && .venv/bin/pip install -e ".[hosted]"
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
if [[ ! -d .venv ]]; then
  echo "Missing .venv. Run: python3.11 -m venv .venv && .venv/bin/pip install -e \".[hosted]\"" >&2
  exit 1
fi
ENV_FILE="${ENV_FILE:-.env.hosted}"
if [[ ! -f "$ENV_FILE" ]]; then
  echo "Missing $ENV_FILE (use scripts/hosted-up.sh once to create it, or copy .env.hosted.example)." >&2
  exit 1
fi
set -a
# shellcheck disable=SC1090
source "$ENV_FILE"
set +a
export HOSTED_DATABASE_URL="${HOSTED_DATABASE_URL:-postgresql+asyncpg://evo:evo@127.0.0.1:5432/evosci_hosted}"
export HOSTED_REDIS_URL="${HOSTED_REDIS_URL:-redis://127.0.0.1:6379/0}"
export HOSTED_WEB_DIR="${HOSTED_WEB_DIR:-$ROOT/EvoScientist/hosted/web}"
export EVOSCIENTIST_HOSTED_DISABLE_MCP="${EVOSCIENTIST_HOSTED_DISABLE_MCP:-1}"
PORT="${PORT:-8765}"

.venv/bin/python -m uvicorn EvoScientist.hosted.api:app --host 127.0.0.1 --port "$PORT" &
UV_PID=$!
.venv/bin/python -m arq EvoScientist.hosted.worker.WorkerSettings &
ARQ_PID=$!
cleanup() { kill "$UV_PID" "$ARQ_PID" 2>/dev/null || true; }
trap cleanup EXIT INT TERM
wait
