#!/usr/bin/env bash
# One-shot Fly deploy for EvoScientist hosted (Redis app + Postgres + API/worker).
# Prerequisite: fly auth login
#
# Usage (from repo root):
#   ./scripts/fly-bootstrap.sh
#
# Optional env (defaults match fly.toml / fly.redis.toml):
#   FLY_MAIN_APP   default: parsed from fly.toml
#   FLY_REDIS_APP  default: parsed from fly.redis.toml
#   FLY_PG_NAME    default: evoscientist-db
#   FLY_REGION     default: iad
#   HOSTED_ENCRYPTION_KEY  if unset, a new Fernet key is generated and printed once

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

export PATH="${HOME}/.fly/bin:${PATH}"

if ! command -v fly >/dev/null 2>&1; then
  echo "Fly CLI not found. Install: curl -L https://fly.io/install.sh | sh"
  exit 1
fi

if ! fly auth whoami >/dev/null 2>&1; then
  echo "Not logged in. Run:  fly auth login"
  exit 1
fi

parse_toml_app() {
  local f="$1"
  grep '^app[[:space:]]*=' "$f" | head -1 | sed -E 's/^app[[:space:]]*=[[:space:]]*["'\'']?([^"'\'']+)["'\'']?.*/\1/'
}

MAIN_APP="${FLY_MAIN_APP:-$(parse_toml_app "$ROOT/fly.toml")}"
REDIS_APP="${FLY_REDIS_APP:-$(parse_toml_app "$ROOT/fly.redis.toml")}"
PG_NAME="${FLY_PG_NAME:-evoscientist-db}"
REGION="${FLY_REGION:-iad}"

echo "Using apps: main=$MAIN_APP  redis=$REDIS_APP  postgres_cluster=$PG_NAME  region=$REGION"

ensure_app() {
  local name="$1"
  set +e
  out=$(fly apps create "$name" 2>&1)
  code=$?
  set -e
  if [[ "$code" -eq 0 ]]; then
    echo "Created Fly app: $name"
    return 0
  fi
  if echo "$out" | grep -qiE 'already|taken|exists'; then
    echo "Fly app already exists: $name"
    return 0
  fi
  echo "$out" >&2
  exit "$code"
}

echo "==> Redis (Fly app running redis:7-alpine)"
ensure_app "$REDIS_APP"
fly deploy --config fly.redis.toml --remote-only --yes

REDIS_URL="redis://${REDIS_APP}.internal:6379/0"
echo "    Internal Redis URL: $REDIS_URL"

echo "==> Postgres (Fly Postgres)"
if echo "$(fly postgres list 2>/dev/null)" | grep -qF "$PG_NAME"; then
  echo "Postgres cluster already listed: $PG_NAME"
else
  set +e
  pc_out=$(fly postgres create \
    --name "$PG_NAME" \
    --region "$REGION" \
    --vm-size shared-cpu-1x \
    --volume-size 1 \
    --initial-cluster-size 1 2>&1)
  pc_code=$?
  set -e
  if [[ "$pc_code" -eq 0 ]]; then
    echo "Created Postgres cluster: $PG_NAME"
  elif echo "$pc_out" | grep -qiE 'already|taken|exists'; then
    echo "Postgres cluster already present: $PG_NAME"
  else
    echo "$pc_out" >&2
    exit "$pc_code"
  fi
fi

echo "==> Main app (API + worker)"
ensure_app "$MAIN_APP"

if [[ -z "${HOSTED_ENCRYPTION_KEY:-}" ]]; then
  if command -v openssl >/dev/null 2>&1; then
    HOSTED_ENCRYPTION_KEY="$(openssl rand -base64 32)"
  else
    HOSTED_ENCRYPTION_KEY="$(python3 -c "from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())")"
  fi
  echo ""
  echo "------------------------------------------------------------------"
  echo "SAVE THIS KEY (used to encrypt queued job credentials in Postgres):"
  echo "HOSTED_ENCRYPTION_KEY=$HOSTED_ENCRYPTION_KEY"
  echo "------------------------------------------------------------------"
  echo ""
fi

fly secrets set \
  HOSTED_ENCRYPTION_KEY="$HOSTED_ENCRYPTION_KEY" \
  HOSTED_REDIS_URL="$REDIS_URL" \
  -a "$MAIN_APP"

echo "==> Attach Postgres to main app (sets DATABASE_URL)"
fly postgres attach "$PG_NAME" --app "$MAIN_APP" --yes || {
  echo "Attach skipped or already configured (check: fly secrets list -a $MAIN_APP | grep DATABASE_URL)."
}

echo "==> Deploy API + worker"
fly deploy --remote-only --yes

echo ""
echo "Done."
echo "  URL:  https://${MAIN_APP}.fly.dev"
echo "  Health:  curl -s https://${MAIN_APP}.fly.dev/health"
echo ""
echo "Mint an invite (interactive shell on a machine):"
echo "  fly ssh console -a ${MAIN_APP}"
echo "  hosted-admin create-invite --uses 50"
echo ""
