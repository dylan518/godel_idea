"""FastAPI HTTP surface for hosted jobs."""

from __future__ import annotations

import logging
import os
import threading
from datetime import UTC, datetime
from pathlib import Path
from uuid import UUID

from arq import create_pool
from arq.connections import RedisSettings
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy import select

from EvoScientist.hosted.crypto_util import encrypt_secret
from EvoScientist.hosted.db import init_db, session_scope
from EvoScientist.hosted.orm_models import HostedJob, InviteToken
from EvoScientist.hosted.schemas import (
    CreateJobRequest,
    CreateJobResponse,
    JobStatusResponse,
)
from EvoScientist.hosted.settings import HostedSettings, get_settings

logger = logging.getLogger(__name__)


async def _check_db() -> bool:
    try:
        async with session_scope() as session:
            await session.execute(select(InviteToken).limit(1))
        return True
    except Exception:
        logger.exception("db health check failed")
        return False


async def _check_redis(settings: HostedSettings) -> bool:
    try:
        pool = await create_pool(RedisSettings.from_dsn(settings.redis_url))
        await pool.close()
        return True
    except Exception:
        logger.exception("redis health check failed")
        return False


def _cors_config(settings: HostedSettings) -> tuple[list[str], bool]:
    raw = settings.cors_origins.strip()
    if raw == "*":
        return ["*"], False
    parts = [x.strip() for x in raw.split(",") if x.strip()]
    return (parts if parts else ["*"]), False if not parts else True


def create_app() -> FastAPI:
    settings = get_settings()
    app = FastAPI(title="EvoScientist Hosted API", version="0.1.0")

    allow_origins, allow_credentials = _cors_config(settings)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=allow_origins,
        allow_credentials=allow_credentials,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.on_event("startup")
    async def _startup() -> None:
        await init_db()
        app.state.arq = await create_pool(
            RedisSettings.from_dsn(settings.redis_url),
        )

    @app.on_event("shutdown")
    async def _shutdown() -> None:
        await app.state.arq.close()

    @app.get("/health")
    async def health() -> dict:
        db_ok = await _check_db()
        redis_ok = await _check_redis(settings)
        ok = db_ok and redis_ok
        mcp_off = os.environ.get("EVOSCIENTIST_HOSTED_DISABLE_MCP", "").strip().lower() in (
            "1",
            "true",
            "yes",
        )
        return {
            "ok": ok,
            "database": db_ok,
            "redis": redis_ok,
            "open_mode": settings.open_mode,
            "use_server_keys": settings.use_server_keys,
            "default_llm_model": settings.default_llm_model,
            "default_llm_provider": settings.default_llm_provider,
            "mcp_disabled": mcp_off,
            "can_restart": settings.allow_ui_restart,
        }

    @app.post("/admin/restart")
    async def restart_server() -> dict:
        if not settings.allow_ui_restart:
            raise HTTPException(status_code=403, detail="ui restart is disabled")

        def _exit_soon() -> None:
            import time

            time.sleep(0.25)
            os._exit(0)

        threading.Thread(target=_exit_soon, daemon=True).start()
        return {"ok": True, "message": "restart requested"}

    @app.post("/v1/jobs", response_model=CreateJobResponse)
    async def create_job(body: CreateJobRequest) -> CreateJobResponse:
        async with session_scope() as session:
            invite_id_val = None
            if not settings.open_mode:
                now = datetime.now(UTC)
                assert body.invite_token is not None
                stmt = (
                    select(InviteToken)
                    .where(InviteToken.token == body.invite_token)
                    .with_for_update()
                )
                inv = (await session.execute(stmt)).scalar_one_or_none()
                if inv is None:
                    raise HTTPException(status_code=403, detail="invalid invite token")
                if inv.expires_at is not None and inv.expires_at < now:
                    raise HTTPException(status_code=403, detail="invite token expired")
                if inv.uses_remaining < 1:
                    raise HTTPException(status_code=403, detail="invite uses exhausted")

                inv.uses_remaining -= 1
                invite_id_val = inv.id

            if settings.use_server_keys:
                job = HostedJob(
                    invite_id=invite_id_val,
                    use_server_keys=True,
                    status="queued",
                    prompt=body.prompt,
                    anthropic_key_encrypted=None,
                    tavily_key_encrypted=None,
                )
            else:
                assert body.anthropic_api_key is not None
                job = HostedJob(
                    invite_id=invite_id_val,
                    use_server_keys=False,
                    status="queued",
                    prompt=body.prompt,
                    anthropic_key_encrypted=encrypt_secret(body.anthropic_api_key),
                    tavily_key_encrypted=(
                        encrypt_secret(body.tavily_api_key)
                        if body.tavily_api_key
                        else None
                    ),
                )
            session.add(job)
            await session.flush()
            jid = job.id
            await session.commit()

        await app.state.arq.enqueue_job("run_hosted_job", str(jid))
        return CreateJobResponse(job_id=jid)

    @app.get("/v1/jobs/{job_id}", response_model=JobStatusResponse)
    async def get_job(job_id: UUID) -> JobStatusResponse:
        async with session_scope() as session:
            job = await session.get(HostedJob, job_id)
            if job is None:
                raise HTTPException(status_code=404, detail="job not found")
            return JobStatusResponse(
                job_id=job.id,
                status=job.status,
                result_text=job.result_text,
                error_message=job.error_message,
                created_at=job.created_at,
                updated_at=job.updated_at,
            )

    web_dir = os.environ.get("HOSTED_WEB_DIR", "").strip()
    if web_dir:
        p = Path(web_dir)
        if p.is_dir():
            from fastapi.staticfiles import StaticFiles

            app.mount(
                "/ui",
                StaticFiles(directory=str(p.resolve()), html=True),
                name="hosted_web",
            )

    return app


app = create_app()
