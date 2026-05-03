"""Console entrypoints for API and worker."""

from __future__ import annotations

import os


def run_api() -> None:
    import uvicorn

    from EvoScientist.hosted.settings import get_settings

    s = get_settings()
    port = int(os.environ.get("PORT", str(s.api_port)))
    uvicorn.run(
        "EvoScientist.hosted.api:app",
        host=s.api_host,
        port=port,
        factory=False,
    )


def run_worker() -> None:
    from arq.worker import run_worker

    from EvoScientist.hosted.worker import WorkerSettings

    run_worker(WorkerSettings)
