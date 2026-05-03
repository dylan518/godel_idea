#!/usr/bin/env bash
# Mint an invite token (run while stack is up). Args are passed to hosted-admin.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
ENV_FILE="${ENV_FILE:-.env.hosted}"
if [[ ! -f "$ENV_FILE" ]]; then
  echo "Missing ${ENV_FILE}. Run scripts/hosted-up.sh first." >&2
  exit 1
fi
if ! command -v docker >/dev/null 2>&1; then
  echo "docker not found." >&2
  exit 1
fi
exec docker compose --env-file "$ENV_FILE" -f docker-compose.hosted.yml run --rm api hosted-admin create-invite "$@"
