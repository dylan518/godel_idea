"""IdeaTreeSearch-style candidate generation for arXiv:2603.08127 §3.3.

The PDF specifies a tree-structured *propose → review → refine* search producing up to
N_I candidates {(I_i, rev_i)}, grounded in literature L and memory context K_I.

The public EvoScientist repo does not include the original RA source; this module
implements that *interface* as three batched LLM steps (propose, review, refine-to-leaves),
then truncates to ``N_I_MAX``. It is closer to the paper than a fixed L1/L2/L3 taxonomy.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from systems.base import call_llm

import log as _log

logger = _log.setup("paper_tree")


def _parse_json(raw: str) -> dict:
    raw = re.sub(r"^```(?:json)?\s*\n?", "", raw.strip())
    raw = re.sub(r"\n?```\s*$", "", raw)
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        m = re.search(r"\{[\s\S]*\}", raw)
        if m:
            return json.loads(m.group(0))
        raise


PROPOSE_PROMPT = """You are the Researcher Agent (idea generation) in a scientific AI system.

Research goal G:
{goal}

Retrieved literature L (titles and abstracts — ideas must build on but not duplicate):
{literature}

Retrieved ideation memory K_I (feasible / failed directions from past tasks — may be empty):
{memory}

Task: propose diverse candidate research idea *drafts* (method sketch + rough experiment plan).
Return ONLY valid JSON:
{{"drafts": [{{"id": "d1", "text": "..."}}, ...]}}

Requirements:
- At least 5 drafts, at most 8 drafts.
- Each draft: 120–220 words, concrete and testable.
- Cover distinct angles (techniques, problem formulations, or metrics).
"""


REVIEW_PROMPT = """You review candidate research drafts for the same goal.

Research goal G:
{goal}

Literature context (abbreviated):
{literature}

Drafts:
{drafts_block}

For EACH draft, give concise review feedback (novelty risks, feasibility gaps, missing baselines).
Return ONLY valid JSON:
{{"reviews": [{{"id": "d1", "review": "..."}}, ...]}}
Use the same ids as the drafts."""


REFINE_PROMPT = """You refine research idea drafts using their reviews (propose–review–refine).

Research goal G:
{goal}

Literature context:
{literature}

Drafts with reviews:
{pairs_block}

Produce a final candidate set of DISTINCT research directions suitable for pairwise tournament.
Each leaf: short title + detailed description (method + experiment plan).
Return ONLY valid JSON:
{{"leaves": [{{"id": "l1", "title": "...", "description": "..."}}, ...]}}

Produce at least {min_leaves} and at most {max_leaves} leaves. Prioritize diversity and clarity."""


def build_paper_idea_candidates(
    goal: str,
    literature: str,
    memory: str,
    client,
    model: str,
    *,
    n_i_max: int,
    temperature: float = 0.75,
) -> list[dict]:
    """Return list of {{id, title, description}} with len ≤ n_i_max."""

    lit = literature.strip() or "(none — rely on general knowledge; still be concrete.)"
    mem = memory.strip() or "(none)"

    raw_p = call_llm(
        PROPOSE_PROMPT.format(goal=goal, literature=lit, memory=mem),
        model,
        client,
        temperature=temperature,
        max_tokens=4096,
    )
    try:
        drafts = _parse_json(raw_p).get("drafts") or []
    except Exception as e:
        logger.warning("Propose step failed: %s", e)
        drafts = []

    if not drafts:
        return [
            {
                "id": "fallback",
                "title": goal[:120],
                "description": f"Novel empirical study on: {goal}",
            }
        ]

    drafts_block = "\n\n".join(
        f"[{d.get('id', '?')}]\n{d.get('text', '')}" for d in drafts
    )
    raw_r = call_llm(
        REVIEW_PROMPT.format(goal=goal, literature=lit[:2500], drafts_block=drafts_block),
        model,
        client,
        temperature=0.35,
        max_tokens=4096,
    )
    try:
        reviews = {r["id"]: r.get("review", "") for r in (_parse_json(raw_r).get("reviews") or [])}
    except Exception as e:
        logger.warning("Review step failed: %s", e)
        reviews = {}

    pairs = []
    for d in drafts:
        did = d.get("id", "?")
        pairs.append(
            f"=== {did} ===\nDRAFT:\n{d.get('text', '')}\n\nREVIEW:\n{reviews.get(did, '(no review)')}\n"
        )
    pairs_block = "\n".join(pairs)

    min_leaves = min(len(drafts), max(5, n_i_max // 2))
    raw_f = call_llm(
        REFINE_PROMPT.format(
            goal=goal,
            literature=lit[:3000],
            pairs_block=pairs_block,
            min_leaves=min_leaves,
            max_leaves=n_i_max,
        ),
        model,
        client,
        temperature=0.45,
        max_tokens=8192,
    )
    try:
        leaves = _parse_json(raw_f).get("leaves") or []
    except Exception as e:
        logger.warning("Refine step failed: %s — using draft texts as leaves", e)
        leaves = []

    if not leaves:
        leaves = [
            {
                "id": d.get("id", f"L{i}"),
                "title": (d.get("text") or "")[:80] or "Idea",
                "description": d.get("text") or "",
            }
            for i, d in enumerate(drafts[:n_i_max])
        ]

    out = []
    for leaf in leaves[:n_i_max]:
        out.append(
            {
                "id": str(leaf.get("id", f"L{len(out)}")),
                "title": leaf.get("title") or "Untitled",
                "description": leaf.get("description") or leaf.get("text") or "",
            }
        )
    logger.info("Paper idea tree: %d leaves (cap %d)", len(out), n_i_max)
    return out
