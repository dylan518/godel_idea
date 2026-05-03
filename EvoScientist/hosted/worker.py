"""Arq worker process — run: ``hosted-worker`` or ``arq EvoScientist.hosted.worker.WorkerSettings``."""

from arq.connections import RedisSettings

from EvoScientist.hosted.settings import get_settings
from EvoScientist.hosted.tasks import run_hosted_job

_s = get_settings()


class WorkerSettings:
    functions = [run_hosted_job]
    redis_settings = RedisSettings.from_dsn(_s.redis_url)
    max_tries = 2
    max_jobs = 1
    job_timeout = _s.job_timeout_seconds
