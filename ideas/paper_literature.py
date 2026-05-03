"""Literature L for the paper RA pipeline: Semantic Scholar (arXiv:2603.08127 §4.5)."""

from __future__ import annotations

import hashlib
import json
import time
import urllib.parse
import urllib.request
from datetime import datetime, timedelta, timezone
from pathlib import Path

import sys

sys.path.insert(0, str(Path(__file__).parent))
import log as _log

logger = _log.setup("paper_lit")

CACHE_DIR = Path(__file__).parent / "results" / "semantic_scholar_cache"
CACHE_TTL_DAYS = 7
REQUEST_DELAY = 3.0  # S2 public API: be conservative
_API = "https://api.semanticscholar.org/graph/v1/paper/search"


def _cache_path(topic: str, n: int) -> Path:
    key = hashlib.md5(f"s2:{topic}:{n}".encode()).hexdigest()[:12]
    return CACHE_DIR / f"{key}.json"


def _is_fresh(path: Path) -> bool:
    if not path.exists():
        return False
    mtime = datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc)
    return datetime.now(timezone.utc) - mtime < timedelta(days=CACHE_TTL_DAYS)


def fetch_semantic_scholar_papers(topic: str, n: int = 5) -> list[dict]:
    """Return [{title, abstract, year, citations, url}, ...]."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache = _cache_path(topic, n)
    if _is_fresh(cache):
        with open(cache) as f:
            return json.load(f)

    params = urllib.parse.urlencode(
        {
            "query": topic,
            "limit": n,
            "fields": "title,abstract,year,citationCount,url",
        }
    )
    url = f"{_API}?{params}"
    headers = {"User-Agent": "godel-paper-replication/1.0"}
    try:
        time.sleep(REQUEST_DELAY)
        req = urllib.request.Request(url, headers=headers)
        with urllib.request.urlopen(req, timeout=30) as resp:
            data = json.loads(resp.read())
        papers = []
        for p in data.get("data") or []:
            ab = (p.get("abstract") or "")[:800]
            if not ab and not p.get("title"):
                continue
            papers.append(
                {
                    "title": p.get("title") or "",
                    "year": p.get("year"),
                    "citations": p.get("citationCount", 0),
                    "abstract": ab,
                    "tldr": "",
                    "url": p.get("url") or "",
                }
            )
            if len(papers) >= n:
                break
        logger.info("Semantic Scholar: %d papers for '%s'", len(papers), topic[:50])
        with open(cache, "w") as f:
            json.dump(papers, f, indent=2)
        return papers
    except Exception as e:
        logger.warning("Semantic Scholar failed for '%s': %s", topic[:40], e)
        return []


def get_semantic_scholar_context(topic: str, n: int = 5) -> str:
    from retrieval import format_context

    return format_context(fetch_semantic_scholar_papers(topic, n))
