"""SOTA paper retrieval via OpenAlex API.

Fetches recent highly-cited papers for a research topic and formats them
as context for idea generators. Results are cached to disk for 7 days.

Uses OPENALEX_API_KEY env var if set (higher rate limits).
Free tier (email polite pool): ~10 req/sec — much better than Semantic Scholar.
"""

import hashlib
import json
import os
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent))
import log as _log

logger = _log.setup("retrieval")

CACHE_DIR = Path(__file__).parent / "results" / "retrieval_cache"
CACHE_TTL_DAYS = 7
API_BASE = "https://api.openalex.org"
N_PAPERS = 5
# OpenAlex polite pool: ~10 req/sec — use a small delay to be safe
REQUEST_DELAY = 0.15   # seconds between requests
_request_lock = None   # threading.Lock, lazily initialised


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


def _reconstruct_abstract(inverted_index: dict) -> str:
    """Reconstruct abstract text from OpenAlex inverted-index format.

    OpenAlex stores abstracts as {word: [position, ...], ...}.
    """
    if not inverted_index:
        return ""
    position_word: dict[int, str] = {}
    for word, positions in inverted_index.items():
        for pos in positions:
            position_word[pos] = word
    if not position_word:
        return ""
    return " ".join(position_word[i] for i in sorted(position_word))


def fetch_papers(topic: str, n: int = N_PAPERS) -> list[dict]:
    """Fetch top-N papers for topic from OpenAlex. Uses disk cache.

    Returns list of {title, abstract, year, citations, url}.
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

        api_key = os.environ.get("OPENALEX_API_KEY", "")

        params: dict = {
            "search": topic,
            # OpenAlex uses ML + citation networks for relevance ranking (not boolean).
            # Default sort for search queries is relevance — don't override it.
            "per_page": n * 2,
            "select": "title,abstract_inverted_index,publication_year,cited_by_count,doi,id",
            # has_abstract:true — only return papers where OpenAlex has indexed the abstract
            # (~57% of all works; filtering here avoids wasting the per_page budget).
            # from_publication_date:2019 — prefer recent SOTA over off-topic classics.
            "filter": "from_publication_date:2019-01-01,has_abstract:true",
        }
        if api_key:
            params["api_key"] = api_key
        else:
            # Polite pool: add a contact email so OpenAlex can reach us if needed
            params["mailto"] = "research@example.com"

        url = f"{API_BASE}/works?" + urllib.parse.urlencode(params)
        headers = {"User-Agent": "godel-loop/1.0", "Accept": "application/json"}

        # Serialize API calls across threads to respect rate limit
        with _get_lock():
            time.sleep(REQUEST_DELAY)
            req = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(req, timeout=20) as resp:
                data = json.loads(resp.read())

        papers = []
        for w in data.get("results", []):
            abstract = _reconstruct_abstract(w.get("abstract_inverted_index") or {})
            if not abstract:
                continue
            doi = w.get("doi") or ""
            oa_id = w.get("id") or ""
            papers.append({
                "title": w.get("title") or "",
                "year": w.get("publication_year"),
                "citations": w.get("cited_by_count", 0),
                "abstract": abstract[:600],
                "tldr": "",   # OpenAlex doesn't have TLDRs; keep key for format_context compat
                "url": doi if doi else oa_id,
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
        summary = p.get("tldr") or p.get("abstract") or ""
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
