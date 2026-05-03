# Fly.io (one vendor: Fly only)

You get **`https://<app>.fly.dev`** after deploy. Postgres and Redis are created from the Fly CLI (Redis uses the public `redis:7-alpine` image as a **second Fly app** on the same account—no separate vendor).

## Fast path (after CLI is installed)

```bash
cd /path/to/EvoScientist   # repo root (where fly.toml lives)
fly auth login
./scripts/fly-bootstrap.sh
```

The script creates Redis + Postgres + main app (if missing), sets secrets, attaches the DB, and deploys API + worker (`--remote-only` builds on Fly).

## 0. Install CLI and log in

```bash
brew install flyctl   # or: curl -L https://fly.io/install.sh | sh
fly auth login
```

## 1. Redis (internal)

Pick a **globally unique** app name or edit `fly.redis.toml` (`app = ...`).

```bash
fly apps create evoscientist-redis
fly deploy --config fly.redis.toml
```

Internal URL (use in step 4):

`redis://evoscientist-redis.internal:6379/0`

(Replace `evoscientist-redis` with your Redis app name.)

## 2. Postgres

```bash
fly postgres create --name evoscientist-db --region iad --vm-size shared-cpu-1x --volume-size 1 --initial-cluster-size 1
```

Smallest options are fine for a few beta users.

## 3. Main app name

Edit **`fly.toml`**: set `app = "your-unique-name"` (must be available on Fly).

```bash
fly apps create evoscientist-hosted   # must match fly.toml app name
```

## 4. Secrets

Generate a Fernet key:

```bash
python3 -c "from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())"
```

Set secrets on the **main** app (not Redis):

```bash
fly secrets set \
  HOSTED_ENCRYPTION_KEY='paste-fernet-key-here' \
  HOSTED_REDIS_URL='redis://evoscientist-redis.internal:6379/0' \
  -a evoscientist-hosted
```

## 5. Attach Postgres (sets `DATABASE_URL` on the app)

```bash
fly postgres attach --app evoscientist-hosted evoscientist-db
```

The API reads `DATABASE_URL` from Fly and normalizes it for asyncpg.

## 6. Deploy API + worker

From the repo root (where `fly.toml` and `Dockerfile.hosted` live):

```bash
fly deploy
```

## 7. Mint an invite

```bash
fly ssh console -a evoscientist-hosted
```

Inside the machine:

```bash
hosted-admin create-invite --uses 50
```

(`pip install` puts `hosted-admin` in `/usr/local/bin` in `Dockerfile.hosted`.)

## 8. Smoke test

```bash
curl -s https://evoscientist-hosted.fly.dev/health
```

## URLs

- API base: `https://<app-name>.fly.dev`
- `POST /v1/jobs` — `invite_token`, `anthropic_api_key`, `prompt`
- `GET /v1/jobs/{job_id}`

## Names taken?

Change `app` in `fly.toml` / `fly.redis.toml` and recreate apps, then update `HOSTED_REDIS_URL` host to match the Redis app name.
