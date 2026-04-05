"""SOTA paper retrieval via Semantic Scholar API.

Fetches recent highly-cited papers for a research topic and formats them
as context for idea generators. Results are cached to disk for 7 days.

No API key required for basic usage (100 req/5min rate limit).
"""

import hashlib
import json
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

import sys
sys.path.insert(0, str(Path(__file__).parent))
import log as _log

logger = _log.setup("retrieval")

CACHE_DIR = Path(__file__).parent / "results" / "retrieval_cache"
CACHE_TTL_DAYS = 7
API_BASE = "https://api.semanticscholar.org/graph/v1"
FIELDS = "title,abstract,year,tldr,externalIds,citationCount"
N_PAPERS = 5
REQUEST_DELAY = 3.0  # seconds before each request (rate limit: ~20 req/min free tier)
_request_lock = None  # threading.Lock, lazily initialised


def _get_lock():
    global _request_lock
    if _request_lock is None:
        import threading
        _request_lock = threading.Lock()
    return _request_lock


def _cache_path(topic: str, n: int) -> Path:
    key = hashlib.md5(f"{topic}:{n}".encode()).hexdigest()[:12]
    return CACHE_DIR / f"{key}.json"


def _is_fresh(path: Path) -> bool:
    if not path.exists():
        return False
    mtime = datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc)
    return datetime.now(timezone.utc) - mtime < timedelta(days=CACHE_TTL_DAYS)


def fetch_papers(topic: str, n: int = N_PAPERS) -> list[dict]:
    """Fetch top-N papers for topic from Semantic Scholar. Uses disk cache.

    Returns list of {title, abstract, year, tldr, url, citations}.
    Falls back to empty list on any error so callers degrade gracefully.
    """
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache = _cache_path(topic, n)

    if _is_fresh(cache):
        logger.debug("Cache hit for topic '%s'", topic[:40])
        with open(cache) as f:
            return json.load(f)

    try:
        import urllib.request
        import urllib.parse

        # Serialize API calls across threads to respect rate limit
        with _get_lock():
            time.sleep(REQUEST_DELAY)

        params = urllib.parse.urlencode({
            "query": topic,
            "fields": FIELDS,
            "limit": n * 2,        # fetch extra, filter down
            "sort": "relevance",
        })
        url = f"{API_BASE}/paper/search?{params}"
        req = urllib.request.Request(url, headers={"User-Agent": "godel-loop/1.0"})

        time.sleep(REQUEST_DELAY)
        with urllib.request.urlopen(req, timeout=15) as resp:
            data = json.loads(resp.read())

        papers = []
        for p in data.get("data", [])[:n * 2]:
            if not p.get("abstract") and not (p.get("tldr") or {}).get("text"):
                continue  # skip if no content
            tldr = (p.get("tldr") or {}).get("text") or ""
            abstract = p.get("abstract") or ""
            papers.append({
                "title": p.get("title", ""),
                "year": p.get("year"),
                "citations": p.get("citationCount", 0),
                "tldr": tldr,
                "abstract": abstract[:600],   # cap to keep context manageable
                "url": f"https://www.semanticscholar.org/paper/{p.get('paperId', '')}",
            })
            if len(papers) >= n:
                break

        logger.info("Fetched %d papers for '%s'", len(papers), topic[:40])
        with open(cache, "w") as f:
            json.dump(papers, f, indent=2)
        return papers

    except Exception as e:
        logger.warning("Paper retrieval failed for '%s': %s — using no context", topic[:40], e)
        return []


def format_context(papers: list[dict]) -> str:
    """Format fetched papers as a compact LLM context string."""
    if not papers:
        return ""

    lines = ["## Recent related work (for context — your idea must go beyond these)\n"]
    for i, p in enumerate(papers, 1):
        year = f" ({p['year']})" if p.get("year") else ""
        summary = p["tldr"] if p.get("tldr") else p["abstract"]
        lines.append(f"{i}. **{p['title']}**{year}")
        if summary:
            lines.append(f"   {summary.strip()}")
        lines.append("")

    return "\n".join(lines)


def get_topic_context(topic: str, n: int = N_PAPERS) -> str:
    """High-level helper: fetch papers + format as context string.

    Returns empty string on failure (callers should handle gracefully).
    """
    papers = fetch_papers(topic, n)
    return format_context(papers)
