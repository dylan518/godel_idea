import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)).replace("/systems", ""))

from systems.base import IdeaGenerator, DEFAULT_MODEL, call_llm, IDEA_FORMAT


EXPAND_WINNER_PROMPT_V2 = """\
Research topic: {topic}

Tournament winner (selected from {n_candidates} candidates via Elo tournament):
{winner_title}
{winner_description}

SOTA context (your idea must go meaningfully beyond this existing work):
{sota_context}

Write a full, detailed research idea based on this winning direction.

## MANDATORY EXPERIMENTAL SPECIFICATION CHECKLIST
Your response MUST explicitly include ALL of the following — do not omit any:

1. **Named Datasets** (list at least 2-3 specific, real benchmark datasets by name,
   e.g., "ImageNet-1K", "SQuAD 2.0", "GLUE/SuperGLUE", "COCO", "Penn Treebank").
   Do NOT say "standard benchmarks" or "relevant datasets" — name them explicitly.

2. **Named Baselines** (list at least 3 specific prior methods/systems by name,
   e.g., "BERT-base", "GPT-4", "ResNet-50", "DPO", "LoRA", "AdamW").
   Do NOT say "existing methods" or "prior work" — name them explicitly.

3. **Quantitative Metrics** (specify at least 2-3 exact evaluation metrics with
   expected improvement ranges, e.g., "achieve >2% absolute improvement on F1",
   "reduce latency by 30% vs baseline", "perplexity below 20 on WikiText-103").
   Do NOT say "improve performance" — state specific metrics and target thresholds.

4. **Experimental Protocol** (describe the experimental setup: training compute budget,
   hardware assumptions, number of runs/seeds, ablation studies planned).

## Additional Requirements
- State the specific open problem being addressed (not a generic gap)
- Describe the proposed method concretely — include key technical components
- Explain novelty relative to the SOTA context above: what prior work CANNOT do that this CAN
- Be direct. No hedging, no vague claims. Every claim must be falsifiable.

{idea_format}"""


# ── Modified L3 prompt with mandatory problem significance sentence ────────────
TREE_L3_PROMPT_V2 = """\
You are expanding a research idea tree to Level 3 — the actual leaf research ideas.

Topic: {topic}

Level 2 Domain nodes:
{domains_str}

## Task: Generate Level 3 — Formulation Variants (Leaf Ideas)

For each domain node, generate 2 specific problem formulations. A formulation
pins down: inputs, outputs, constraints, and evaluation criteria.
Good variation: latency-constrained vs memory-constrained (different optimization targets)
Bad variation: minimize latency vs maximize throughput on same single-device setup (equivalent)

Target: 12 leaf ideas total (2 per domain). These are the candidates that will
compete in the Elo tournament. Make each one specific enough to act on immediately.

Each leaf idea description MUST contain exactly 4 sentences in this order:
  Sentence 1: What the method does — describe the concrete technical approach.
  Sentence 2: The specific open scientific problem or bottleneck in existing work that this addresses — name the exact failure mode or limitation (e.g., "Current methods fail because X when Y") and explain why solving it matters scientifically or practically.
  Sentence 3: Why this is novel vs existing work — what prior work CANNOT do that this CAN.
  Sentence 4: How to validate — name specific datasets, metrics, and what result would confirm success.

Respond with ONLY this JSON:
{{
  "leaves": [
    {{
      "id": "T1-D1-F1",
      "parent": "T1-D1",
      "title": "<one-line title>",
      "description": "<4 sentences as specified above>"
    }},
    ... (12 total, 2 per domain node)
  ]
}}"""


# ── Modified review prompt that also enforces problem significance ─────────────
TREE_REVIEW_PROMPT_V2 = """\
Review and refine these {n} research idea candidates for topic: {topic}

{sota_context}

Candidates:
{candidates_str}

## Task
For each candidate:
1. Remove vague language ("might", "could potentially", "may help") — make claims specific
2. Ensure the novelty claim explicitly states what prior work cannot do that this can
3. Ensure each description has a clear, concrete problem significance statement: name the specific bottleneck or failure mode in existing work being addressed and why it matters scientifically or practically. If missing, add it as the second sentence.
4. If two candidates are near-duplicates (same technique + domain + formulation), merge
   them — keep the sharper description, discard the weaker

Respond with ONLY this JSON (may have fewer entries after merging):
{{
  "refined": [
    {{"id": "<id>", "title": "<title>", "description": "<refined 4-sentence description>"}},
    ...
  ]
}}"""


def _parse_json_local(raw: str) -> dict:
    import json
    import re
    raw = re.sub(r"^```(?:json)?\s*\n?", "", raw.strip())
    raw = re.sub(r"\n?```\s*$", "", raw)
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        m = re.search(r"\{[\s\S]*\}", raw)
        if m:
            return json.loads(m.group(0))
        raise


def _build_idea_tree_v2(topic: str, sota_context: str, client, model: str,
                         temperature: float = 0.8) -> list:
    """Modified IdeaTreeSearch with problem-significance-enforcing L3 prompt."""
    import idea_tournament.prompts as P
    import log as _log
    logger = _log.setup("tree_search_v2")

    # ── L1: Technique variants ────────────────────────────────────────────────
    l1_nodes = []
    try:
        raw = call_llm(
            P.TREE_L1_PROMPT.format(topic=topic, sota_context=sota_context),
            model, client, temperature=temperature, max_tokens=1024,
        )
        l1_nodes = _parse_json_local(raw).get("techniques", [])
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
        l2_nodes = _parse_json_local(raw).get("domains", [])
        logger.debug("L2: %d domain nodes", len(l2_nodes))
    except Exception as e:
        logger.warning("L2 failed: %s — will use L1 nodes as leaves", e)

    domains_str = (
        "\n".join(f"{d['id']} [{d['domain']}]: {d['description']}" for d in l2_nodes)
        if l2_nodes else techniques_str
    )

    # ── L3: Formulation variants (leaf ideas) — uses modified prompt ──────────
    l3_leaves = []
    try:
        raw = call_llm(
            TREE_L3_PROMPT_V2.format(topic=topic, domains_str=domains_str),
            model, client, temperature=temperature, max_tokens=2048,
        )
        l3_leaves = _parse_json_local(raw).get("leaves", [])
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

    # ── Review + refine: uses modified review prompt ───────────────────────────
    candidates_str = "\n".join(
        f"{leaf['id']}: {leaf.get('title', '?')} — {leaf.get('description', '')}"
        for leaf in l3_leaves
    )
    try:
        raw = call_llm(
            TREE_REVIEW_PROMPT_V2.format(
                n=len(l3_leaves), topic=topic,
                sota_context=sota_context,
                candidates_str=candidates_str,
            ),
            model, client, temperature=0.3, max_tokens=2048,
        )
        refined = _parse_json_local(raw).get("refined", [])
        if refined:
            l3_leaves = refined
            logger.debug("After review: %d candidates", len(l3_leaves))
    except Exception as e:
        logger.warning("Review step failed: %s — using unrefined leaves", e)

    logger.info("IdeaTreeSearch v2 complete: %d candidates for '%s'",
                len(l3_leaves), topic[:40])
    return l3_leaves


class S5Generator(IdeaGenerator):
    VERSION = "S5"
    DESCRIPTION = (
        "EvoScientist IdeaTreeSearch + Elo Tournament with enforced concrete evaluation specs "
        "and mandatory problem significance in leaf generation. "
        "3-level tree generates ~12 leaf candidates grounded in SOTA papers, "
        "then 4-round Swiss Elo tournament selects the best. "
        "L3 prompt requires 4-sentence structure: method, problem bottleneck/significance, "
        "novelty, and validation. Final expansion mandates named datasets, baselines, "
        "and quantitative metrics with target thresholds."
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
        import idea_tournament.tournament as tourn

        # ── 1. SOTA retrieval ─────────────────────────────────────────────────
        sota_context = ""
        try:
            from retrieval import get_topic_context
            sota_context = get_topic_context(topic, n=5)
        except Exception:
            pass
        sota_block = sota_context or "(No SOTA context — generate from first principles.)"

        # ── 2. IdeaTreeSearch → leaf candidates (with problem significance) ───
        leaves = _build_idea_tree_v2(topic, sota_block, client, model, temperature)

        # ── 3. Elo tournament → best candidate ────────────────────────────────
        winner = tourn.run_tournament(topic, leaves, client, model)

        # ── 4. Expand winner into full IDEA_FORMAT (with concrete eval specs) ──
        expand_prompt = EXPAND_WINNER_PROMPT_V2.format(
            topic=topic,
            n_candidates=len(leaves),
            winner_title=winner.get("title", topic),
            winner_description=winner.get("description", ""),
            sota_context=sota_block,
            idea_format=IDEA_FORMAT,
        )
        return call_llm(expand_prompt, model, client, temperature)


GENERATOR = S5Generator()
