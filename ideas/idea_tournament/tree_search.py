"""IdeaTreeSearch: 3-level tree-structured idea generation.

Implements Phase 1 of the EvoScientist idea-tournament skill.
Reference: ideas/idea-tournament/SKILL.md, references/tree-search-protocol.md

Tree structure:
  L0: Seed (the research topic)
  L1: 3 Technique variants — distinct paradigms
  L2: 2 Domain adaptations per L1 = 6 nodes
  L3: 2 Formulation variants per L2 = 12 leaf ideas

Each level is one batched LLM call. Total: 4 LLM calls for tree generation.
A review/refine pass deduplicates and sharpens the leaves.

Edit targets in prompts.py:
  TREE_L1_PROMPT, TREE_L2_PROMPT, TREE_L3_PROMPT, TREE_REVIEW_PROMPT
"""

import json
import re
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _parse_json(raw: str) -> dict:
    """Parse JSON from LLM response, stripping markdown fences."""
    raw = re.sub(r"^```(?:json)?\s*\n?", "", raw.strip())
    raw = re.sub(r"\n?```\s*$", "", raw)
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        # Try to find a JSON object anywhere in the response
        m = re.search(r"\{[\s\S]*\}", raw)
        if m:
            return json.loads(m.group(0))
        raise


def build_idea_tree(topic: str, sota_context: str, client, model: str,
                    temperature: float = 0.8) -> list[dict]:
    """Run IdeaTreeSearch. Returns list of refined leaf idea dicts.

    Each leaf: {"id": str, "title": str, "description": str}

    Gracefully degrades at each level — if L3 fails, uses L2 nodes as leaves;
    if L2 fails, uses L1 nodes. Always returns at least 1 candidate.
    """
    from systems.base import call_llm
    import idea_tournament.prompts as P
    import log as _log
    logger = _log.setup("tree_search")

    # ── L1: Technique variants ────────────────────────────────────────────────
    l1_nodes = []
    try:
        raw = call_llm(
            P.TREE_L1_PROMPT.format(topic=topic, sota_context=sota_context),
            model, client, temperature=temperature, max_tokens=1024,
        )
        l1_nodes = _parse_json(raw).get("techniques", [])
        logger.debug("L1: %d technique variants", len(l1_nodes))
    except Exception as e:
        logger.warning("L1 failed: %s — using single fallback branch", e)
        l1_nodes = [{"id": "T1", "name": "Direct approach",
                     "description": f"A novel approach to {topic}."}]

    techniques_str = "\n".join(
        f"{t['id']}: {t['name']} — {t['description']}" for t in l1_nodes
    )

    # ── L2: Domain adaptations ────────────────────────────────────────────────
    l2_nodes = []
    try:
        raw = call_llm(
            P.TREE_L2_PROMPT.format(topic=topic, techniques_str=techniques_str),
            model, client, temperature=temperature, max_tokens=1024,
        )
        l2_nodes = _parse_json(raw).get("domains", [])
        logger.debug("L2: %d domain nodes", len(l2_nodes))
    except Exception as e:
        logger.warning("L2 failed: %s — will use L1 nodes as leaves", e)

    domains_str = (
        "\n".join(f"{d['id']} [{d['domain']}]: {d['description']}" for d in l2_nodes)
        if l2_nodes else techniques_str
    )

    # ── L3: Formulation variants (leaf ideas) ─────────────────────────────────
    l3_leaves = []
    try:
        raw = call_llm(
            P.TREE_L3_PROMPT.format(topic=topic, domains_str=domains_str),
            model, client, temperature=temperature, max_tokens=2048,
        )
        l3_leaves = _parse_json(raw).get("leaves", [])
        logger.debug("L3: %d leaf candidates", len(l3_leaves))
    except Exception as e:
        logger.warning("L3 failed: %s — falling back to L2 as leaves", e)

    # Fallback cascade
    if not l3_leaves and l2_nodes:
        l3_leaves = [
            {"id": d["id"] + "-F1", "parent": d["id"],
             "title": d["domain"],
             "description": d["description"]}
            for d in l2_nodes
        ]
    if not l3_leaves:
        l3_leaves = [
            {"id": t["id"] + "-D1-F1", "parent": t["id"],
             "title": t["name"],
             "description": t["description"]}
            for t in l1_nodes
        ]
    if not l3_leaves:
        l3_leaves = [{"id": "T1-D1-F1", "title": topic,
                      "description": f"Novel approach to {topic}."}]

    # ── Review + refine: dedup and sharpen ────────────────────────────────────
    candidates_str = "\n".join(
        f"{leaf['id']}: {leaf.get('title', '?')} — {leaf.get('description', '')}"
        for leaf in l3_leaves
    )
    try:
        raw = call_llm(
            P.TREE_REVIEW_PROMPT.format(
                n=len(l3_leaves), topic=topic,
                sota_context=sota_context,
                candidates_str=candidates_str,
            ),
            model, client, temperature=0.3, max_tokens=2048,
        )
        refined = _parse_json(raw).get("refined", [])
        if refined:
            l3_leaves = refined
            logger.debug("After review: %d candidates (was %d)", len(l3_leaves),
                         len(candidates_str.splitlines()))
    except Exception as e:
        logger.warning("Review step failed: %s — using unrefined leaves", e)

    logger.info("IdeaTreeSearch complete: %d candidates for '%s'",
                len(l3_leaves), topic[:40])
    return l3_leaves
