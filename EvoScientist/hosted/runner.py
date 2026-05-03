"""Run one agent turn in an isolated workspace (BYOK or server env keys)."""

from __future__ import annotations

import os

from langchain_core.messages import HumanMessage

from EvoScientist import create_cli_agent, get_effective_config
from EvoScientist.config import apply_config_to_env
from EvoScientist.cli._constants import build_metadata
from EvoScientist.hosted.settings import get_settings
from EvoScientist.paths import ensure_dirs, set_workspace_root
from EvoScientist.stream.events import stream_agent_events


def _require_server_llm_key(provider: str) -> None:
    """Ensure the API key expected for *provider* is present in the environment."""
    p = (provider or "").strip().lower()
    if p == "google-genai":
        if not os.environ.get("GOOGLE_API_KEY", "").strip():
            raise RuntimeError(
                "GOOGLE_API_KEY is not set in the worker environment. "
                "Add it to .env / secrets and set HOSTED_USE_SERVER_KEYS=true."
            )
        return
    if p == "anthropic":
        if not os.environ.get("ANTHROPIC_API_KEY", "").strip():
            raise RuntimeError(
                "ANTHROPIC_API_KEY is not set in the worker environment. "
                "Add it to .env and set HOSTED_USE_SERVER_KEYS=true for api/worker."
            )
        return
    raise RuntimeError(
        f"Unsupported HOSTED_DEFAULT_LLM_PROVIDER={provider!r} for hosted server keys."
    )


async def run_hosted_agent(
    *,
    workspace_root: str,
    prompt: str,
    thread_id: str,
    anthropic_api_key: str | None = None,
    tavily_api_key: str | None = None,
    use_server_env_keys: bool = False,
) -> str:
    """Execute a single user prompt; returns final assistant text.

    * **BYOK**: pass ``anthropic_api_key`` (and optional ``tavily_api_key``).
    * **Server env** (``use_server_env_keys=True``): uses keys from the process
      environment per ``HOSTED_DEFAULT_LLM_PROVIDER`` (e.g. ``GOOGLE_API_KEY`` for
      ``google-genai`` / Gemini, or ``ANTHROPIC_API_KEY`` for Anthropic) plus optional
      ``TAVILY_API_KEY``.
    """
    os.environ.setdefault("EVOSCIENTIST_HOSTED_DISABLE_MCP", "1")

    if use_server_env_keys:
        settings = get_settings()
        prov = settings.default_llm_provider
        _require_server_llm_key(prov)
        cfg = get_effective_config(
            {
                "auto_approve": True,
                "enable_ask_user": False,
                "model": settings.default_llm_model,
                "provider": prov,
            }
        )
        apply_config_to_env(cfg)
        set_workspace_root(workspace_root)
        ensure_dirs()
        agent = create_cli_agent(
            workspace_dir=workspace_root,
            config=cfg,
            anthropic_api_key=None,
        )
    else:
        if not anthropic_api_key or not str(anthropic_api_key).strip():
            raise RuntimeError("BYOK job missing anthropic_api_key")
        os.environ.pop("ANTHROPIC_API_KEY", None)
        os.environ.pop("ANTHROPIC_BASE_URL", None)
        if tavily_api_key:
            os.environ["TAVILY_API_KEY"] = tavily_api_key
        else:
            os.environ.pop("TAVILY_API_KEY", None)

        cfg = get_effective_config({"auto_approve": True, "enable_ask_user": False})
        apply_config_to_env(cfg)
        os.environ.pop("ANTHROPIC_API_KEY", None)
        os.environ.pop("ANTHROPIC_BASE_URL", None)

        set_workspace_root(workspace_root)
        ensure_dirs()

        agent = create_cli_agent(
            workspace_dir=workspace_root,
            config=cfg,
            anthropic_api_key=anthropic_api_key,
        )

    meta = build_metadata(workspace_root, cfg.model)
    result = ""
    async for ev in stream_agent_events(
        agent,
        HumanMessage(content=prompt),
        thread_id,
        metadata=meta,
    ):
        if ev.get("type") == "done":
            result = ev.get("response") or ev.get("content") or ""
        elif ev.get("type") == "error":
            raise RuntimeError(ev.get("message", "agent error"))

    return result
