#!/usr/bin/env bash
# Start Postgres + Redis + API + worker + static UI for local testing.
# Requires Docker with Compose v2. UI: http://localhost:8765/ui/
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

ENV_FILE="${ENV_FILE:-.env.hosted}"
if [[ ! -f "$ENV_FILE" ]]; then
  KEY="$(python3 -c "import secrets, base64; print(base64.urlsafe_b64encode(secrets.token_bytes(32)).decode())")"
  printf 'HOSTED_ENCRYPTION_KEY=%s\n' "$KEY" >"$ENV_FILE"
  echo "Created ${ENV_FILE} with a new HOSTED_ENCRYPTION_KEY."
fi

if ! command -v docker >/dev/null 2>&1; then
  MAC_DOCKER_CLI="/Applications/Docker.app/Contents/Resources/bin"
  if [[ -x "$MAC_DOCKER_CLI/docker" ]]; then
    export PATH="$MAC_DOCKER_CLI:$PATH"
  fi
fi
if ! command -v docker >/dev/null 2>&1; then
  echo "docker not found. Install Docker Desktop (or Colima + docker CLI), then re-run:" >&2
  echo "  $0 $*" >&2
  exit 1
fi

exec docker compose --env-file "$ENV_FILE" -f docker-compose.hosted.yml up --build "$@"
