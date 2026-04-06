import sys
import os
import json
import re
import random

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)).replace("/systems", ""))

from systems.base import IdeaGenerator, DEFAULT_MODEL, call_llm, IDEA_FORMAT


# ── Modified L3 prompt with explicit evaluation fields ────────────────────────
TREE_L3_PROMPT_MODIFIED = """\
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

CRITICAL: Every leaf idea MUST include all of the following concrete details:
  - Specific named datasets (e.g., "ImageNet", "SQuAD 2.0", "MS-COCO")
  - Specific named baselines (e.g., "BERT-base", "ResNet-50", "GPT-2")
  - Specific metrics (e.g., "top-1 accuracy", "F1 score", "BLEU-4", "latency in ms")

Respond with ONLY this JSON:
{{
  "leaves": [
    {{
      "id": "T1-D1-F1",
      "parent": "T1-D1",
      "title": "<one-line title>",
      "datasets": "<comma-separated list of 1-3 specific benchmark datasets>",
      "baselines": "<comma-separated list of 2-3 specific baseline methods>",
      "metrics": "<comma-separated list of 2-3 specific evaluation metrics>",
      "description": "<3 sentences: (1) what it does and how, (2) why it's novel vs existing work citing specific prior methods it surpasses, (3) validation plan: which datasets vs which baselines on which metrics>"
    }},
    ... (12 total, 2 per domain node)
  ]
}}"""


def _parse_json(raw: str) -> dict:
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


def _build_idea_tree_modified(topic: str, sota_context: str, client, model: str,
                               temperature: float = 0.8) -> list:
    """Run IdeaTreeSearch with modified L3 prompt. Returns list of refined leaf idea dicts."""
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

    # ── L3: Formulation variants (leaf ideas) — using MODIFIED prompt ─────────
    l3_leaves = []
    try:
        raw = call_llm(
            TREE_L3_PROMPT_MODIFIED.format(topic=topic, domains_str=domains_str),
            model, client, temperature=temperature, max_tokens=3000,
        )
        raw_leaves = _parse_json(raw).get("leaves", [])
        # Enrich description with explicit evaluation fields if present
        for leaf in raw_leaves:
            datasets = leaf.get("datasets", "")
            baselines = leaf.get("baselines", "")
            metrics = leaf.get("metrics", "")
            desc = leaf.get("description", "")
            # Append structured eval info to description if not already embedded
            if datasets or baselines or metrics:
                eval_suffix = ""
                if datasets:
                    eval_suffix += f" Datasets: {datasets}."
                if baselines:
                    eval_suffix += f" Baselines: {baselines}."
                if metrics:
                    eval_suffix += f" Metrics: {metrics}."
                # Only append if the description doesn't already contain these
                if eval_suffix.strip() and eval_suffix.strip() not in desc:
                    leaf["description"] = desc.rstrip() + eval_suffix
            l3_leaves.append(leaf)
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
            logger.debug("After review: %d candidates", len(l3_leaves))
    except Exception as e:
        logger.warning("Review step failed: %s — using unrefined leaves", e)

    logger.info("IdeaTreeSearch complete: %d candidates for '%s'",
                len(l3_leaves), topic[:40])
    return l3_leaves


def _compare_pair(topic: str, idea_a: dict, idea_b: dict, client, model: str) -> str:
    """Compare two leaf ideas. Returns 'A', 'B', or 'tie'."""
    import idea_tournament.prompts as P

    text_a = f"{idea_a.get('title', '')}\n{idea_a.get('description', '')}"
    text_b = f"{idea_b.get('title', '')}\n{idea_b.get('description', '')}"

    flipped = random.random() < 0.5
    pa, pb = (text_b, text_a) if flipped else (text_a, text_b)

    try:
        raw = call_llm(
            P.TOURNAMENT_JUDGE_PROMPT.format(topic=topic, idea_a=pa, idea_b=pb),
            model, client, temperature=0.1, max_tokens=300,
        )
        verdict = _parse_json(raw)
        winner = verdict.get("winner", "tie")
    except Exception:
        winner = "tie"

    if flipped:
        if winner == "A":
            winner = "B"
        elif winner == "B":
            winner = "A"
    return winner


def _run_tournament(topic: str, leaves: list, client, model: str) -> dict:
    """Swiss-system Elo tournament. Returns the winning leaf idea dict."""
    import log as _log
    logger = _log.setup("tournament")

    ELO_K = 32
    ELO_START = 1500.0
    N_ROUNDS_LARGE = 4
    N_ROUNDS_SMALL = 3

    if not leaves:
        return {}
    if len(leaves) == 1:
        return leaves[0]

    n = len(leaves)
    rounds = N_ROUNDS_LARGE if n >= 10 else N_ROUNDS_SMALL
    ratings = {i: ELO_START for i in range(n)}
    matchups: set = set()
    prev_top3 = None

    for rnd in range(rounds):
        ranked = sorted(ratings, key=lambda i: ratings[i], reverse=True)
        pairs, paired = [], set()
        for a in ranked:
            if a in paired:
                continue
            for b in ranked:
                if b == a or b in paired:
                    continue
                key = (min(a, b), max(a, b))
                if key not in matchups:
                    pairs.append((a, b))
                    paired.add(a)
                    paired.add(b)
                    matchups.add(key)
                    break

        if not pairs:
            break

        for a, b in pairs:
            winner = _compare_pair(topic, leaves[a], leaves[b], client, model)
            ea = 1.0 / (1.0 + 10 ** ((ratings[b] - ratings[a]) / 400))
            score_a = 1.0 if winner == "A" else (0.0 if winner == "B" else 0.5)
            ratings[a] += ELO_K * (score_a - ea)
            ratings[b] += ELO_K * ((1 - score_a) - (1 - ea))
            logger.debug("r%d: %s vs %s → %s (%.0f vs %.0f)",
                         rnd, leaves[a].get("id", a), leaves[b].get("id", b),
                         winner, ratings[a], ratings[b])

        top3 = tuple(sorted(ratings, key=lambda i: ratings[i], reverse=True)[:3])
        if top3 == prev_top3:
            logger.debug("Top-3 stable after round %d — stopping early", rnd + 1)
            break
        prev_top3 = top3

    best = max(ratings, key=lambda i: ratings[i])
    logger.info("Tournament: %s wins (Elo=%.0f) from %d candidates for '%s'",
                leaves[best].get("id", "?"), ratings[best], n, topic[:40])
    return leaves[best]


class S4_r1Generator(IdeaGenerator):
    VERSION = "S4_r1"
    DESCRIPTION = (
        "EvoScientist IdeaTreeSearch + Elo Tournament with enforced evaluation details. "
        "Modified L3 prompt requires each leaf to specify concrete datasets, baselines, "
        "and metrics as mandatory structured fields, front-loading experimental scaffolding "
        "into candidate generation before the tournament. "
        "3-level tree generates ~12 leaf candidates, 4-round Swiss Elo tournament selects best."
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
        import idea_tournament.prompts as P

        # ── 1. SOTA retrieval ─────────────────────────────────────────────────
        sota_context = ""
        try:
            from retrieval import get_topic_context
            sota_context = get_topic_context(topic, n=5)
        except Exception:
            pass
        sota_block = sota_context or "(No SOTA context — generate from first principles.)"

        # ── 2. IdeaTreeSearch → leaf candidates (modified L3 prompt) ──────────
        leaves = _build_idea_tree_modified(topic, sota_block, client, model, temperature)

        # ── 3. Elo tournament → best candidate ────────────────────────────────
        winner = _run_tournament(topic, leaves, client, model)

        # ── 4. Expand winner into full IDEA_FORMAT ────────────────────────────
        expand_prompt = P.EXPAND_WINNER_PROMPT.format(
            topic=topic,
            n_candidates=len(leaves),
            winner_title=winner.get("title", topic),
            winner_description=winner.get("description", ""),
            sota_context=sota_block,
            idea_format=IDEA_FORMAT,
        )
        return call_llm(expand_prompt, model, client, temperature)


GENERATOR = S4_r1Generator()
