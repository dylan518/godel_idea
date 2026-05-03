"""Pydantic API models."""

from datetime import datetime
from typing import Self
from uuid import UUID

from pydantic import BaseModel, Field, field_validator, model_validator


class CreateJobRequest(BaseModel):
    invite_token: str | None = Field(
        default=None,
        max_length=128,
        description="Omitted or empty when HOSTED_OPEN_MODE is enabled.",
    )
    anthropic_api_key: str | None = Field(
        default=None,
        max_length=512,
        description="Omitted when HOSTED_USE_SERVER_KEYS is enabled (keys from server env).",
    )
    prompt: str = Field(..., min_length=1)
    tavily_api_key: str | None = Field(default=None, max_length=256)

    @field_validator("invite_token", mode="before")
    @classmethod
    def _strip_invite(cls, v: object) -> str | None:
        if v is None:
            return None
        s = str(v).strip()
        return s if s else None

    @field_validator("anthropic_api_key", "tavily_api_key", mode="before")
    @classmethod
    def _strip_optional_keys(cls, v: object) -> str | None:
        if v is None:
            return None
        s = str(v).strip()
        return s if s else None

    @field_validator("prompt")
    @classmethod
    def _cap_prompt(cls, v: str) -> str:
        from .settings import get_settings

        max_c = get_settings().max_prompt_chars
        if len(v) > max_c:
            raise ValueError(f"prompt exceeds {max_c} characters")
        return v

    @model_validator(mode="after")
    def _validate_invite_and_llm(self) -> Self:
        from .settings import get_settings

        s = get_settings()
        if not s.open_mode:
            if not self.invite_token or len(self.invite_token) < 8:
                raise ValueError(
                    "invite_token is required (min 8 characters) when open mode is off"
                )
        if not s.use_server_keys:
            if not self.anthropic_api_key or len(self.anthropic_api_key) < 10:
                raise ValueError(
                    "anthropic_api_key is required in the request unless "
                    "HOSTED_USE_SERVER_KEYS is enabled and ANTHROPIC_API_KEY is set on the server"
                )
        return self


class CreateJobResponse(BaseModel):
    job_id: UUID


class JobStatusResponse(BaseModel):
    job_id: UUID
    status: str
    result_text: str | None = None
    error_message: str | None = None
    created_at: datetime | None = None
    updated_at: datetime | None = None
