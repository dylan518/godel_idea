"""Arq worker: load job, run agent, persist result."""

from __future__ import annotations

import asyncio
import logging
import os
import shutil
import uuid
from pathlib import Path

from sqlalchemy import select

from EvoScientist.hosted.crypto_util import decrypt_secret
from EvoScientist.hosted.db import session_scope
from EvoScientist.hosted.orm_models import HostedJob
from EvoScientist.hosted.runner import run_hosted_agent
from EvoScientist.hosted.settings import get_settings

logger = logging.getLogger(__name__)


async def run_hosted_job(ctx: dict, job_id: str) -> None:
    """Process a queued job (BYOK credentials were stored encrypted)."""
    settings = get_settings()
    jid = uuid.UUID(job_id)
    workspace: str | None = None
    anthropic_key: str | None = None
    tavily_key: str | None = None
    prompt = ""

    try:
        async with session_scope() as session:
            res = await session.execute(
                select(HostedJob).where(HostedJob.id == jid).with_for_update()
            )
            job = res.scalar_one_or_none()
            if job is None:
                logger.error("hosted job %s not found", job_id)
                return
            if job.status != "queued":
                logger.warning("hosted job %s skip status=%s", job_id, job.status)
                return
            use_srv = bool(job.use_server_keys)
            if use_srv:
                prompt = job.prompt
                job.status = "running"
                await session.commit()
            else:
                if not job.anthropic_key_encrypted:
                    job.status = "failed"
                    job.error_message = "missing anthropic credentials"
                    await session.commit()
                    return
                anthropic_key = decrypt_secret(job.anthropic_key_encrypted)
                job.anthropic_key_encrypted = None
                if job.tavily_key_encrypted:
                    tavily_key = decrypt_secret(job.tavily_key_encrypted)
                    job.tavily_key_encrypted = None
                prompt = job.prompt
                job.status = "running"
                await session.commit()

        work_root = get_settings().work_root
        workspace = str(Path(work_root) / job_id)
        Path(workspace).mkdir(parents=True, exist_ok=True)

        thread_id = str(uuid.uuid4())
        if use_srv:
            result = await asyncio.wait_for(
                run_hosted_agent(
                    workspace_root=workspace,
                    prompt=prompt,
                    thread_id=thread_id,
                    use_server_env_keys=True,
                ),
                timeout=settings.job_timeout_seconds,
            )
        else:
            result = await asyncio.wait_for(
                run_hosted_agent(
                    workspace_root=workspace,
                    prompt=prompt,
                    thread_id=thread_id,
                    anthropic_api_key=anthropic_key,
                    tavily_api_key=tavily_key,
                    use_server_env_keys=False,
                ),
                timeout=settings.job_timeout_seconds,
            )

        async with session_scope() as session:
            job2 = await session.get(HostedJob, jid)
            if job2:
                job2.status = "completed"
                job2.result_text = result
                await session.commit()
    except Exception as e:
        logger.exception("hosted job %s failed", job_id)
        err = str(e)[:8000]
        try:
            async with session_scope() as session:
                job2 = await session.get(HostedJob, jid)
                if job2 and job2.status not in ("completed",):
                    job2.status = "failed"
                    job2.error_message = err
                    await session.commit()
        except Exception:
            logger.exception("could not persist failure for job %s", job_id)
    finally:
        anthropic_key = None
        tavily_key = None
        if workspace and Path(workspace).exists():
            shutil.rmtree(workspace, ignore_errors=True)
