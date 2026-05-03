"""Ideation memory M_I retrieval K_I — arXiv:2603.08127 Eq. (1).

Paper: embedding-based cosine retrieval with top-k_I items (§4.5: k_I=2), embeddings
via ``mxbai-embed-large`` through Ollama.

If Ollama is unreachable or the store is empty, returns an empty string (cold-start),
which matches a new EvoScientist deployment with no distilled memories yet.
"""

from __future__ import annotations

import json
import math
import os
import urllib.request
from pathlib import Path

import sys

sys.path.insert(0, str(Path(__file__).parent))
import log as _log

logger = _log.setup("ideation_mem")

DEFAULT_STORE = Path(__file__).parent / "results" / "ideation_memory.json"


def _ollama_embed(text: str, model: str, base: str) -> list[float] | None:
    url = f"{base}/api/embeddings"
    body = json.dumps({"model": model, "prompt": text}).encode()
    req = urllib.request.Request(
        url, data=body, headers={"Content-Type": "application/json"}, method="POST"
    )
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            data = json.loads(resp.read())
        emb = data.get("embedding")
        return emb if isinstance(emb, list) else None
    except Exception as e:
        logger.debug("Ollama embed failed: %s", e)
        return None


def _cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)


def load_store(path: Path | None = None) -> list[dict]:
    p = path or DEFAULT_STORE
    if not p.exists():
        return []
    try:
        with open(p) as f:
            data = json.load(f)
        return data if isinstance(data, list) else data.get("items", [])
    except Exception:
        return []


def retrieve_ideation_context(
    goal: str,
    k: int,
    *,
    store_path: Path | None = None,
    embedding_model: str | None = None,
    ollama_base: str | None = None,
) -> str:
    """Return formatted K_I block for prompts, or "" if unavailable."""
    from paper_config import OLLAMA_EMBED_BASE, PAPER_EMBEDDING_MODEL

    items = load_store(store_path)
    if not items or k <= 0:
        return ""

    model = embedding_model or PAPER_EMBEDDING_MODEL
    base = ollama_base or OLLAMA_EMBED_BASE

    q_emb = _ollama_embed(goal, model, base)
    if not q_emb:
        logger.info("Ideation memory: no embedding for goal — skipping retrieval")
        return ""

    scored = []
    for it in items:
        text = it.get("text") or it.get("summary") or ""
        if not text:
            continue
        cached = it.get("embedding")
        if isinstance(cached, list) and len(cached) == len(q_emb):
            e = [float(x) for x in cached]
        else:
            e = _ollama_embed(text[:8000], model, base)
            if not e:
                continue
        scored.append((text, _cosine(q_emb, e)))

    scored.sort(key=lambda x: x[1], reverse=True)
    top = scored[:k]
    if not top:
        return ""

    lines = ["## Retrieved ideation memory (direction insights from prior tasks)\n"]
    for i, (text, sim) in enumerate(top, 1):
        lines.append(f"{i}. (similarity≈{sim:.3f}) {text.strip()}\n")
    return "\n".join(lines)
