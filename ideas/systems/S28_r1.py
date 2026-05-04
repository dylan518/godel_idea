"""S28_r1: SOTA-grounded L3 tree construction + falsifiable expansion.

Changes vs S_paper (champion):
  1. Tree construction: SOTA context now flows into L3 (leaf) generation.
     Each L3 leaf must name a specific prior-work limitation (sota_gap) and
     state a falsifiable hypothesis (method X > baseline Y on dataset D by metric Z).
     This grounds the leaf during construction rather than retrofitting at expansion time.
  2. Review pass: preserves sota_gap / hypothesis and fills them in if missing.
  3. Expansion: _expand_leaf uses sota_gap + hypothesis from the leaf as a structured
     anchor. BACKGROUND must name the prior-work failure; APPROACH must open with the
     working hypothesis and the mechanism by which it overcomes that failure;
     EXPERIMENT must name a dataset, a baseline, and a primary metric.

LLM calls: same budget as champion — ~4 (tree) + tournament pairs + n (expand).
Tournament is imported unchanged from idea_tournament.tournament.
"""

import json
import os
import re
import sys
import threading

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Tournament is unchanged — import directly.
from idea_tournament.tournament import run_tournament_ranked

from systems.base import (
    IdeaGenerator,
    DEFAULT_MODEL,
    IDEA_FORMAT,
    _parse_batch_ideas,
    call_llm,
)


# ── SOTA retrieval (unchanged from champion) ──────────────────────────────────

def _get_sota_context(topic: str) -> str:
    try:
        from retrieval import get_topic_context
        return get_topic_context(topic, n=5) or ""
    except Exception:
        return ""


# ── JSON parsing helper ───────────────────────────────────────────────────────

def _parse_json_safe(raw: str) -> dict:
    """Parse JSON from LLM response, stripping markdown fences."""
    raw = re.sub(r"^```(?:json)?\s*\n?", "", raw.strip())
    raw = re.sub(r"\n?```\s*$", "", raw)
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        m = re.search(r"\{[\s\S]*\}", raw)
        if m:
            return json.loads(m.group(0))
        raise


# ── Prompt templates (use __PLACEHOLDER__ + .replace() — never .format()) ────
# Curly braces inside the JSON examples are literal, not format fields.

_L1_PROMPT = """\
You are generating a structured idea tree for a research topic.

Topic (Level 0 seed): __TOPIC__

SOTA Context:
__SOTA__

## Task: Generate Level 1 — Technique Variants

Generate exactly 3 fundamentally different technical approaches to this topic.
Each technique must represent a distinct paradigm — not variations of the same approach.

For each, write a 2-sentence description:
  Sentence 1: What the technique does.
  Sentence 2: The key hypothesis — why this approach might work better than alternatives,
              referencing a specific limitation visible in the SOTA context above if possible.

Respond with ONLY this JSON:
{
  "techniques": [
    {"id": "T1", "name": "<short name>", "description": "<2 sentences>"},
    {"id": "T2", "name": "<short name>", "description": "<2 sentences>"},
    {"id": "T3", "name": "<short name>", "description": "<2 sentences>"}
  ]
}"""

_L2_PROMPT = """\
You are expanding a research idea tree to Level 2 (Domain adaptations).

Topic: __TOPIC__

Level 1 Techniques:
__TECHNIQUES__

## Task: Generate Level 2 — Domain Adaptations

For each technique, generate 2 application domains where the technique faces
fundamentally different constraints — each domain should create distinct research challenges.
Good variation: edge devices vs batch servers (different bottlenecks).
Bad variation: text classification vs sentiment analysis (same technical challenge).

Respond with ONLY this JSON:
{
  "domains": [
    {"id": "T1-D1", "parent": "T1", "domain": "<domain name>", "description": "<2 sentences: how technique adapts to this domain>"},
    {"id": "T1-D2", "parent": "T1", "domain": "<domain name>", "description": "<2 sentences>"},
    {"id": "T2-D1", "parent": "T2", "domain": "<domain name>", "description": "<2 sentences>"},
    {"id": "T2-D2", "parent": "T2", "domain": "<domain name>", "description": "<2 sentences>"},
    {"id": "T3-D1", "parent": "T3", "domain": "<domain name>", "description": "<2 sentences>"},
    {"id": "T3-D2", "parent": "T3", "domain": "<domain name>", "description": "<2 sentences>"}
  ]
}"""

# KEY CHANGE: L3 now receives SOTA context and must produce sota_gap + hypothesis per leaf.
_L3_SOTA_PROMPT = """\
You are expanding a research idea tree to Level 3 — leaf ideas grounded in specific SOTA gaps.

Topic: __TOPIC__

SOTA Context (use this to find specific limitations in existing work):
__SOTA__

Level 2 Domain nodes:
__DOMAINS__

## Task: Generate Level 3 — SOTA-Grounded Leaf Ideas

For each domain node, generate 2 leaf ideas. Each leaf MUST provide:
  sota_gap — Name the specific prior approach or paper that fails, and state exactly
             what it cannot do. Be specific: "Method X fails to handle Y because Z",
             not "prior work has limitations in this area".
  hypothesis — State a falsifiable prediction: "We predict [proposed method] will
               [outperform / reduce / improve] [named baseline] by [metric] on [dataset]
               under [condition]". This must be a testable claim, not a hope.
  description — 3 sentences: (1) what the idea does to address the sota_gap,
                (2) why existing methods fail at this specific sub-problem,
                (3) how to validate (name at least one dataset and one metric).

Target: up to 21 leaves where possible (at least 12). Make each specific enough to
implement immediately — no vague directions.

Respond with ONLY this JSON:
{
  "leaves": [
    {
      "id": "T1-D1-F1",
      "parent": "T1-D1",
      "title": "<one-line title>",
      "sota_gap": "<named prior-work failure: what approach fails and why>",
      "hypothesis": "<falsifiable prediction: method X will Y baseline Z on dataset D by metric M>",
      "description": "<3 sentences: addresses gap / why existing fails / how to validate>"
    }
  ]
}"""

# Review pass: preserves and enriches sota_gap / hypothesis fields.
_REVIEW_PROMPT = """\
Review and refine these __N__ research idea candidates for topic: __TOPIC__

SOTA Context:
__SOTA__

Candidates:
__CANDIDATES__

## Task
For each candidate:
1. Sharpen vague language — make every claim specific and attributable to a named method.
2. If sota_gap is missing or vague (e.g. "prior work is limited"), rewrite it to name
   the specific technique or paper that fails and explain exactly what it cannot do.
3. If hypothesis is missing or untestable, rewrite it as a falsifiable prediction:
   "Method X will outperform baseline Y on dataset D by metric Z under condition C."
4. Merge near-duplicates (same technique + domain + formulation); keep the sharper one.
5. Prune to at most 21 leaves.

Respond with ONLY this JSON (may have fewer entries after merging):
{
  "refined": [
    {
      "id": "<id>",
      "title": "<title>",
      "sota_gap": "<named prior-work limitation>",
      "hypothesis": "<falsifiable claim>",
      "description": "<refined 3-sentence description>"
    }
  ]
}"""


# ── SOTA-grounded tree builder (inlined, replaces build_idea_tree) ─────────────

def _build_sota_grounded_tree(
    topic: str, sota_context: str, client, model: str, temperature: float = 0.8
) -> list[dict]:
    """IdeaTreeSearch with SOTA flowing into L3 generation.

    Each returned leaf dict carries:
      id, title, description  (same as champion — tournament reads these)
      sota_gap                (named prior-work failure — new)
      hypothesis              (falsifiable prediction — new)

    Gracefully degrades at each level.
    """
    import log as _log
    logger = _log.setup("tree_search_s28")

    sota_clip = (sota_context or "").strip()[:4000]

    # ── L1: Technique variants (same logic as champion; SOTA visible) ─────────
    l1_nodes = []
    try:
        l1_prompt = (
            _L1_PROMPT
            .replace("__TOPIC__", topic)
            .replace("__SOTA__", sota_clip or "(none)")
        )
        raw = call_llm(l1_prompt, model, client, temperature=temperature, max_tokens=1024)
        l1_nodes = _parse_json_safe(raw).get("techniques", [])
        logger.debug("L1: %d technique variants", len(l1_nodes))
    except Exception as e:
        logger.warning("L1 failed: %s — using single fallback branch", e)
        l1_nodes = [{"id": "T1", "name": "Direct approach",
                     "description": f"A novel approach to {topic}."}]

    techniques_str = "\n".join(
        f"{t['id']}: {t['name']} — {t['description']}" for t in l1_nodes
    )

    # ── L2: Domain adaptations (same logic as champion; no SOTA needed here) ──
    l2_nodes = []
    try:
        l2_prompt = (
            _L2_PROMPT
            .replace("__TOPIC__", topic)
            .replace("__TECHNIQUES__", techniques_str)
        )
        raw = call_llm(l2_prompt, model, client, temperature=temperature, max_tokens=1024)
        l2_nodes = _parse_json_safe(raw).get("domains", [])
        logger.debug("L2: %d domain nodes", len(l2_nodes))
    except Exception as e:
        logger.warning("L2 failed: %s — will use L1 nodes as leaves", e)

    domains_str = (
        "\n".join(f"{d['id']} [{d['domain']}]: {d['description']}" for d in l2_nodes)
        if l2_nodes else techniques_str
    )

    # ── L3: SOTA-grounded leaf formulations (KEY CHANGE) ─────────────────────
    l3_leaves = []
    try:
        l3_prompt = (
            _L3_SOTA_PROMPT
            .replace("__TOPIC__", topic)
            .replace("__SOTA__", sota_clip or "(none)")
            .replace("__DOMAINS__", domains_str)
        )
        raw = call_llm(l3_prompt, model, client, temperature=temperature, max_tokens=2048)
        l3_leaves = _parse_json_safe(raw).get("leaves", [])
        logger.debug("L3: %d leaf candidates (SOTA-grounded)", len(l3_leaves))
    except Exception as e:
        logger.warning("L3 SOTA-grounded failed: %s — falling back to plain leaves", e)

    # Fallback cascade (graceful degradation — leaves may lack sota_gap/hypothesis)
    if not l3_leaves and l2_nodes:
        l3_leaves = [
            {"id": d["id"] + "-F1", "parent": d["id"],
             "title": d["domain"], "description": d["description"]}
            for d in l2_nodes
        ]
    if not l3_leaves:
        l3_leaves = [
            {"id": t["id"] + "-D1-F1", "parent": t["id"],
             "title": t["name"], "description": t["description"]}
            for t in l1_nodes
        ]
    if not l3_leaves:
        l3_leaves = [{"id": "T1-D1-F1", "title": topic,
                      "description": f"Novel approach to {topic}."}]

    # ── Review + refine: preserves / enriches sota_gap and hypothesis ─────────
    candidates_str = "\n".join(
        f"{leaf['id']}: {leaf.get('title', '?')} — {leaf.get('description', '')}"
        for leaf in l3_leaves
    )
    try:
        review_prompt = (
            _REVIEW_PROMPT
            .replace("__N__", str(len(l3_leaves)))
            .replace("__TOPIC__", topic)
            .replace("__SOTA__", sota_clip or "(none)")
            .replace("__CANDIDATES__", candidates_str)
        )
        raw = call_llm(review_prompt, model, client, temperature=0.3, max_tokens=2048)
        refined = _parse_json_safe(raw).get("refined", [])
        if refined:
            # Carry over sota_gap / hypothesis from original leaves if review omitted them
            orig_by_id = {leaf.get("id"): leaf for leaf in l3_leaves}
            for r in refined:
                orig = orig_by_id.get(r.get("id"), {})
                if not r.get("sota_gap") and orig.get("sota_gap"):
                    r["sota_gap"] = orig["sota_gap"]
                if not r.get("hypothesis") and orig.get("hypothesis"):
                    r["hypothesis"] = orig["hypothesis"]
            l3_leaves = refined[:21]
            logger.debug("After review: %d refined candidates", len(l3_leaves))
    except Exception as e:
        logger.warning("Review step failed: %s — using unrefined leaves", e)

    if len(l3_leaves) > 21:
        l3_leaves = l3_leaves[:21]

    logger.info(
        "SOTA-grounded tree complete: %d candidates for '%s'", len(l3_leaves), topic[:40]
    )
    return l3_leaves


# ── Thread-local pipeline cache (one tree + tournament per topic per worker) ──

_tls = threading.local()


def _tls_pipeline(topic: str, client, model: str, temperature: float):
    if getattr(_tls, "topic", None) != topic:
        sota = _get_sota_context(topic)
        leaves = _build_sota_grounded_tree(topic, sota, client, model, temperature=temperature)
        ranked = run_tournament_ranked(topic, leaves, client, model)
        _tls.topic = topic
        _tls.sota = sota
        _tls.ranked = ranked
        _tls.slot = 0
    return _tls.sota, _tls.ranked


# ── Expansion: structured prompt that enforces MOTIVATION + HYPOTHESIS ────────

def _expand_leaf(
    topic: str,
    leaf: dict,
    sota_context: str,
    model: str,
    client,
    temperature: float,
    variant_index: int = 0,
) -> str:
    title = leaf.get("title") or "Research direction"
    desc = leaf.get("description") or ""
    sota_gap = (leaf.get("sota_gap") or "").strip()
    hypothesis = (leaf.get("hypothesis") or "").strip()

    ctx = f"\n\nSOTA Context:\n{sota_context}\n" if sota_context else ""

    # Structured grounding block — only included if the leaf carries these fields
    grounding = ""
    if sota_gap or hypothesis:
        parts = ["Prior-work analysis from leaf selection:"]
        if sota_gap:
            parts.append(f"  GAP: {sota_gap}")
        if hypothesis:
            parts.append(f"  WORKING HYPOTHESIS: {hypothesis}")
        grounding = "\n\n" + "\n".join(parts) + "\n"

    variant_note = ""
    if variant_index > 0:
        variant_note = (
            f"\n\n(Benchmark variant #{variant_index + 1} from the same rank list — "
            "sharpen a different experimental emphasis or metric choice while keeping "
            "the core mechanistic insight.)\n"
        )

    # Structured output contract for each IDEA_FORMAT section
    section_contract = (
        "Write a complete research idea in the format below.\n"
        "Section-level requirements:\n"
        "  BACKGROUND — Must name the specific prior method or system that fails "
        "and state exactly what it cannot do (not a generic gap statement).\n"
        "  APPROACH — Must open with the working hypothesis mechanism: explain how "
        "the proposed method overcomes the named failure, not just what it does.\n"
        "  EXPERIMENT — Must name at least one concrete dataset, one named baseline, "
        "and one primary quantitative metric.\n"
        "  NOVELTY — Must contrast against the named prior work, not against 'existing approaches'.\n"
        "Be direct. No hedging language.\n\n"
    )

    prompt = (
        f"Research topic: {topic}{ctx}{grounding}{variant_note}\n"
        "The following direction was selected after SOTA-grounded idea search "
        "and pairwise Elo ranking among candidate leaves:\n\n"
        f"TITLE: {title}\n"
        f"SUMMARY: {desc}\n\n"
        + section_contract
        + IDEA_FORMAT
    )
    try:
        return call_llm(prompt, model, client, temperature, max_tokens=2048)
    except Exception:
        # Minimal fallback: return the leaf summary so generate_idea never returns empty
        return f"IDEA: {title}\n\nBACKGROUND: {desc}\n\nAPPROACH: (generation failed)\n\nEXPERIMENT: N/A\n\nNOVELTY: N/A"


# ── Generator class ───────────────────────────────────────────────────────────

class S28_r1Generator(IdeaGenerator):
    VERSION = "S28_r1"
    DESCRIPTION = (
        "IdeaTreeSearch with SOTA-grounded L3 leaves (sota_gap + hypothesis per leaf) "
        "+ Swiss Elo tournament + structured expansion requiring named prior-work failure "
        "and falsifiable hypothesis in each idea."
    )

    def get_prompt(self, topic: str) -> str:
        return f"Generate a novel research idea about: {topic}"

    def generate_idea(
        self,
        topic: str,
        client,
        model: str = DEFAULT_MODEL,
        temperature: float = 0.8,
    ) -> str:
        try:
            sota, ranked = _tls_pipeline(topic, client, model, temperature)
        except Exception:
            sota, ranked = "", []

        if not ranked:
            try:
                result = call_llm(
                    self.get_prompt(topic) + "\n\n" + IDEA_FORMAT,
                    model, client, temperature,
                )
                return result if result.strip() else f"IDEA: Novel approach to {topic}"
            except Exception:
                return f"IDEA: Novel approach to {topic}"

        i = _tls.slot
        _tls.slot = i + 1
        leaf = ranked[i % len(ranked)]
        variant = i // len(ranked)
        return _expand_leaf(topic, leaf, sota, model, client, temperature, variant_index=variant)

    def generate_batch(
        self,
        topic: str,
        client,
        model: str = DEFAULT_MODEL,
        n: int = 5,
        temperature: float = 0.9,
    ) -> list[str]:
        """One SOTA-grounded tree + one ranked tournament; top-n ranked leaves expanded."""
        sota = _get_sota_context(topic)
        leaves = _build_sota_grounded_tree(topic, sota, client, model, temperature=temperature)
        try:
            ranked = run_tournament_ranked(topic, leaves, client, model)
        except Exception:
            ranked = leaves

        if not ranked:
            try:
                raw = call_llm(
                    self.get_prompt(topic)
                    + f"\n\nProduce exactly {n} distinct ideas separated by a line "
                    "containing only ---\n\n"
                    + IDEA_FORMAT,
                    model, client, temperature, max_tokens=n * 800,
                )
                return _parse_batch_ideas(raw, n)
            except Exception:
                return [f"IDEA: Novel approach to {topic}"] * n

        texts = []
        for i in range(n):
            leaf = ranked[i % len(ranked)]
            variant = i // len(ranked)
            texts.append(
                _expand_leaf(topic, leaf, sota, model, client, temperature, variant_index=variant)
            )
        return texts


GENERATOR = S28_r1Generator()
