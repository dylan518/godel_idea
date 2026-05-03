# API keys for EvoScientist (CLI and hosted)

## What the research agent actually uses (typical “paper-style” run)

| Capability | Env var | Required? | Notes |
|------------|---------|-----------|--------|
| **Claude (main model)** | `ANTHROPIC_API_KEY` | **Yes** | Same as normal EvoScientist; powers the agent and sub-agents. |
| **Web search (Tavily tool)** | `TAVILY_API_KEY` | **Recommended** | Without it, `tavily_search` is not registered — much weaker literature / current web grounding. |
| **OpenAlex** | — | **No key** | Public HTTP API; no API key in `.env.example`. |
| **MCP servers** | varies | **Optional** | Extra tools if configured under `~/.config/evoscientist/mcp.json`. Hosted Docker sets `EVOSCIENTIST_HOSTED_DISABLE_MCP=1` by default — no MCP in that mode. |

Other keys in **`.env.example`** (OpenAI, Google, OpenRouter, Ollama, etc.) only matter if you switch **provider / model** to those backends — they are **not** required for the default Anthropic + Tavily setup.

## What is already listed in `.env.example` (repo root)

Open that file for the full list. The important lines for a default Anthropic workflow are:

- `ANTHROPIC_API_KEY`
- `TAVILY_API_KEY`

Plus any optional providers you personally use.

## Hosted Docker on your laptop

Put keys into **`.env.hosted`** (see **`.env.hosted.example`**) so **api** and **worker** containers load them via `env_file`.

Default **server-key** setup uses **Gemini** (`HOSTED_DEFAULT_LLM_MODEL=gemini-3-flash`, `HOSTED_DEFAULT_LLM_PROVIDER=google-genai`): set **`GOOGLE_API_KEY`**. To use Anthropic instead, set **`HOSTED_DEFAULT_LLM_PROVIDER=anthropic`** and **`ANTHROPIC_API_KEY`**.

**`TAVILY_API_KEY`** is still recommended for web-grounded runs.

With **`HOSTED_USE_SERVER_KEYS=true`**, you do **not** paste keys into the web UI — the worker reads them from the environment.

After changing DB schema (e.g. new columns), if jobs fail on insert, reset the DB volume once:

`docker compose --env-file .env.hosted -f docker-compose.hosted.yml down -v`
