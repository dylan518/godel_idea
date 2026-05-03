"""Environment-backed settings for the hosted stack."""

from functools import lru_cache

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


def _normalize_database_url(url: str) -> str:
    """Fly Postgres uses ``postgres://``; SQLAlchemy async needs ``postgresql+asyncpg://``."""
    u = url.strip()
    if u.startswith("postgres://"):
        return "postgresql+asyncpg://" + u[len("postgres://") :]
    if u.startswith("postgresql://") and "+asyncpg" not in u:
        return "postgresql+asyncpg://" + u[len("postgresql://") :]
    return u


class HostedSettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="HOSTED_",
        env_file=".env",
        extra="ignore",
    )

    database_url: str = Field(
        ...,
        description="Async SQLAlchemy URL, e.g. postgresql+asyncpg://user:pass@host:5432/db",
    )
    redis_url: str = Field(default="redis://localhost:6379/0", description="Redis for Arq")
    encryption_key: str = Field(
        ...,
        description="Fernet key (urlsafe base64 32-byte). Generate: python -c \"from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())\"",
    )
    api_host: str = "0.0.0.0"
    api_port: int = 8080
    job_timeout_seconds: int = 3600
    max_prompt_chars: int = 500_000
    queue_name: str = "hosted:arq"
    cors_origins: str = Field(
        default="*",
        description="Comma-separated origins or * (no credentials with *)",
    )
    work_root: str = Field(
        default="/tmp/evosci-hosted",
        description="Per-job workspace parent directory on workers",
    )
    open_mode: bool = Field(
        default=False,
        description="If true, skip invite tokens (dev/local only — never expose publicly).",
    )
    use_server_keys: bool = Field(
        default=False,
        description="If true, LLM/search keys come from server env (see default_llm_provider), not the request body.",
    )
    default_llm_model: str = Field(
        default="gemini-3-flash",
        description="Short model name for hosted server-key jobs (registry in llm/models.py).",
    )
    default_llm_provider: str = Field(
        default="google-genai",
        description="LLM provider for server-key jobs (e.g. google-genai, anthropic).",
    )
    allow_ui_restart: bool = Field(
        default=False,
        description="If true, allow POST /admin/restart from the web UI (dev/local only).",
    )

    @field_validator("open_mode", mode="before")
    @classmethod
    def _coerce_open_mode(cls, v: object) -> bool:
        if isinstance(v, bool):
            return v
        if v is None:
            return False
        return str(v).strip().lower() in ("1", "true", "yes", "on")

    @field_validator("use_server_keys", mode="before")
    @classmethod
    def _coerce_use_server_keys_field(cls, v: object) -> bool:
        if isinstance(v, bool):
            return v
        if v is None:
            return False
        return str(v).strip().lower() in ("1", "true", "yes", "on")

    @field_validator("allow_ui_restart", mode="before")
    @classmethod
    def _coerce_allow_ui_restart_field(cls, v: object) -> bool:
        if isinstance(v, bool):
            return v
        if v is None:
            return False
        return str(v).strip().lower() in ("1", "true", "yes", "on")


@lru_cache
def get_settings() -> HostedSettings:
    import os

    # Fly ``fly postgres attach`` sets DATABASE_URL; we accept that or HOSTED_DATABASE_URL.
    raw = os.environ.get("HOSTED_DATABASE_URL") or os.environ.get("DATABASE_URL")
    if raw:
        norm = _normalize_database_url(raw)
        if os.environ.get("HOSTED_DATABASE_URL") != norm:
            os.environ["HOSTED_DATABASE_URL"] = norm
    return HostedSettings()
