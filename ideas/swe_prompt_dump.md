# SWE Agent — Live Prompt Dump

Champion: S12 | Next: S15
Code bundle: 28,603 chars (723 lines)
SWE context: 2,871 chars
ANALYZE_PROMPT total: 39,203 chars
PROPOSE_EDIT_PROMPT total: 36,587 chars

---

## PROMPT 1: ANALYZE_PROMPT

> Sent first. Model outputs 2-3 sentence failure diagnosis.
> **Size: 39,203 chars**

You are improving a research idea generator. Analyze what is fundamentally limiting
idea quality and identify the single most impactful architectural change to make next.

FRAMING:

- The code below is the CHAMPION — it is currently winning. It is the starting point.
- "Losing verdicts" show where the PREVIOUS candidate (B) failed vs the champion (A).
- Your job: identify what STRUCTURAL change would produce genuinely better ideas.

## Champion codebase (S12)

The REAL logic lives in idea_tournament/ — target that for improvements.

### FILE: systems/S12.py

```python
"""S12_r3: Hypothesis-First Adversarial Loop.

Replaces the top-down tree/tournament with a FALSIFIABLE-HYPOTHESIS-FIRST loop:

1. HYPOTHESIS GENERATION: One LLM call produces 5 sharp, testable scientific
   hypotheses about the topic (e.g., "scaling laws break under data-constrained
   regimes because X"). Starting from *claimed truths about the world* rather
   than technique names.

2. ADVERSARIAL ATTACK (parallel, 5 calls): Each hypothesis gets independently
   attacked by an adversarial critic that probes assumption violations, dataset
   biases, theoretical gaps, and practical limitations. Each attack also forces
   a refined/alternative variant.

3. HYPOTHESIS SELECTION (1 call): Aggregate attack+revision pairs and pick the
   strongest surviving hypothesis — the one whose refined version is most
   concrete, novel, and falsifiable.

4. IDEA CONSTRUCTION (1 call): Design the full experimental idea around
   proving/disproving the surviving hypothesis. Forces concrete datasets,
   baselines, and metrics because the falsification target is explicit.

5. MULTI-PERSPECTIVE CRITIQUE (4 calls): Experimentalist, theorist, skeptic
   critique + synthesis.

6. FINAL REVISION (1 call): Incorporate critique.

LLM calls: 1 (hyp) + 5 (attacks) + 1 (select) + 1 (construct) + 4 (critique) + 1 (revise) = ~13
"""

import sys
import os
import concurrent.futures

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from systems.base import IdeaGenerator, DEFAULT_MODEL, call_llm, IDEA_FORMAT


class S12Generator(IdeaGenerator):
    VERSION = "S12"
    DESCRIPTION = (
        "Hypothesis-first adversarial loop: generate falsifiable hypotheses, "
        "attack each in parallel, select strongest survivor, build experimental "
        "idea around falsification, then multi-perspective critique."
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
        # ── Step 0: retrieve SOTA context ───────────────────────────────────
        try:
            import os as _os
            sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
            from retrieval import get_topic_context
            sota_context = get_topic_context(topic, n=5)
        except Exception:
            sota_context = ""

        context_block = f"\n\n{sota_context}\n" if sota_context else ""

        # ── Step 1: generate 5 falsifiable hypotheses ───────────────────────
        hyp_prompt = (
            f"Research topic: {topic}{context_block}\n"
            "Generate exactly 5 sharp, falsifiable scientific hypotheses about this topic.\n"
            "Each hypothesis must:\n"
            "- Make a specific claim about the world (not just 'we can improve X')\n"
            "- Be testable: name what experiment would prove it false\n"
            "- Be non-obvious: it should NOT be directly supported by the related work above\n"
            "- Be concise: 1-2 sentences max\n\n"
            "Format each as:\n"
            "H1: <hypothesis statement>\n"
            "H2: <hypothesis statement>\n"
            "H3: <hypothesis statement>\n"
            "H4: <hypothesis statement>\n"
            "H5: <hypothesis statement>"
        )
        try:
            hyp_raw = call_llm(hyp_prompt, model, client, temperature)
        except Exception as e:
            hyp_raw = f"H1: Standard approaches to {topic} fail because of distribution shift."

        # Parse hypotheses
        hypotheses = []
        for line in hyp_raw.strip().split("\n"):
            line = line.strip()
            if line and (line.startswith("H") and ":" in line[:4]):
                hyp_text = line.split(":", 1)[1].strip()
                if hyp_text:
                    hypotheses.append(hyp_text)
        if not hypotheses:
            hypotheses = [hyp_raw.strip()]
        hypotheses = hypotheses[:5]  # cap at 5

        # ── Step 2: parallel adversarial attacks on each hypothesis ─────────
        def attack_hypothesis(hyp: str) -> str:
            attack_prompt = (
                f"Research topic: {topic}\n\n"
                f"Hypothesis: {hyp}\n\n"
                "You are an adversarial critic. Attack this hypothesis on EACH of these dimensions:\n"
                "1. Assumption violation: What unstated assumption does this rely on? Give a concrete counterexample.\n"
                "2. Dataset bias: What dataset artifact could make this appear true without actually being true?\n"
                "3. Theoretical gap: What known result from the literature contradicts or undermines this?\n"
                "4. Practical limitation: What makes this hypothesis untestable or too expensive to test?\n\n"
                "Then, given these attacks, write a REVISED hypothesis that survives them:\n"
                "Revised: <1-2 sentence refined hypothesis that addresses the above attacks>"
            )
            try:
                return call_llm(attack_prompt, model, client, temperature=0.6)
            except Exception as e:
                return f"Revised: {hyp} (attack failed: {e})"

        attacks = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            futures = [executor.submit(attack_hypothesis, h) for h in hypotheses]
            for f in futures:
                try:
                    attacks.append(f.result(timeout=120))
                except Exception as e:
                    attacks.append(f"Revised: (timeout) {e}")

        # ── Step 3: select the strongest surviving hypothesis ────────────────
        pairs_str = ""
        for i, (hyp, attack) in enumerate(zip(hypotheses, attacks)):
            pairs_str += f"\n--- Hypothesis {i+1} ---\nOriginal: {hyp}\nCritique+Revision:\n{attack}\n"

        select_prompt = (
            f"Research topic: {topic}\n\n"
            f"Below are {len(hypotheses)} original hypotheses, each attacked by an adversarial critic "
            f"and refined into a stronger revised form:\n{pairs_str}\n"
            "Select the ONE hypothesis (by number) whose REVISED form is:\n"
            "- Most concrete and specific (names mechanisms, not just outcomes)\n"
            "- Most falsifiable (clearest path to a disproof experiment)\n"
            "- Most novel (least covered by standard literature)\n\n"
            "Respond with:\n"
            "SELECTED: <number 1-5>\n"
            "REVISED HYPOTHESIS: <copy the revised hypothesis text exactly>\n"
            "REASONING: <1-2 sentences on why this is the strongest>"
        )
        try:
            selection_raw = call_llm(select_prompt, model, client, temperature=0.3)
        except Exception as e:
            selection_raw = f"SELECTED: 1\nREVISED HYPOTHESIS: {hypotheses[0]}\nREASONING: fallback"

        # Extract the selected revised hypothesis
        selected_hyp = hypotheses[0]  # fallback
        for line in selection_raw.strip().split("\n"):
            if line.strip().upper().startswith("REVISED HYPOTHESIS:"):
                candidate = line.split(":", 1)[1].strip()
                if candidate:
                    selected_hyp = candidate
                    break

        # ── Step 4: construct experimental idea around the hypothesis ────────
        context_reminder = (
            f"\nExisting work to differentiate from:\n{sota_context}\n"
            if sota_context else ""
        )
        construct_prompt = (
            f"Research topic: {topic}\n"
            f"{context_reminder}\n"
            f"Core hypothesis to test: {selected_hyp}\n\n"
            "Design a concrete research experiment to PROVE OR DISPROVE this hypothesis.\n"
            "The experiment must:\n"
            "- Name specific datasets you will use (not just 'standard benchmarks')\n"
            "- Name specific baselines you will compare against\n"
            "- Define quantitative success metrics (what number proves it, what number disproves it)\n"
            "- Describe the key technical method in enough detail to implement\n"
            "- Explain precisely what result would falsify the hypothesis\n\n"
            "Write 3-4 paragraphs. Be direct and specific."
        )
        try:
            draft = call_llm(construct_prompt, model, client, temperature)
        except Exception as e:
            draft = f"Research idea for '{topic}' based on hypothesis: {selected_hyp}"

        # ── Step 5: multi-perspective critique ───────────────────────────────
        exp_prompt = (
            f"You are a hard-nosed experimentalist reviewing a research proposal about '{topic}'.\n\n"
            f"Hypothesis being tested: {selected_hyp}\n\n"
            f"Proposed experiment:\n{draft}\n\n"
            "Give 2-3 sharp criticisms focusing purely on experimental feasibility: "
            "Can these experiments actually be run? Are the measurements well-defined? "
            "What controls are missing? What will fail in practice?"
        )
        try:
            critique_exp = call_llm(exp_prompt, model, client, temperature=0.5)
        except Exception:
            critique_exp = "No experimental critique available."

        theory_prompt = (
            f"You are a rigorous theorist reviewing a research proposal about '{topic}'.\n\n"
            f"Hypothesis being tested: {selected_hyp}\n\n"
            f"Proposed experiment:\n{draft}\n\n"
            "Give 2-3 sharp criticisms focusing purely on theoretical grounding: "
            "Is the novelty claim justified? Does it overlap with known results? "
            "Are the underlying assumptions stated and defensible?"
        )
        try:
            critique_theory = call_llm(theory_prompt, model, client, temperature=0.5)
        except Exception:
            critique_theory = "No theoretical critique available."

        skeptic_prompt = (
            f"You are a skeptical reviewer who has seen many overhyped proposals about '{topic}'.\n\n"
            f"Hypothesis being tested: {selected_hyp}\n\n"
            f"Proposed experiment:\n{draft}\n\n"
            "Give 2-3 sharp criticisms focusing on: why this probably won't work, "
            "what the likely negative result is, and whether the scientific payoff "
            "justifies the effort even if it succeeds."
        )
        try:
            critique_skeptic = call_llm(skeptic_prompt, model, client, temperature=0.5)
        except Exception:
            critique_skeptic = "No skeptic critique available."

        synthesis_prompt = (
            f"Three reviewers critiqued a research idea about '{topic}' testing the hypothesis:\n"
            f"'{selected_hyp}'\n\n"
            f"Experimentalist:\n{critique_exp}\n\n"
            f"Theorist:\n{critique_theory}\n\n"
            f"Skeptic:\n{critique_skeptic}\n\n"
            "Synthesize these into the 3 most important actionable improvements "
            "the author must make. Be concise and prioritized."
        )
        try:
            synthesis = call_llm(synthesis_prompt, model, client, temperature=0.3)
        except Exception:
            synthesis = "Improve specificity, add baselines, clarify falsification criteria."

        # ── Step 6: final revision ────────────────────────────────────────────
        revise_prompt = (
            f"Research topic: {topic}\n\n"
            f"Core hypothesis: {selected_hyp}\n\n"
            f"Experimental design:\n{draft}\n\n"
            f"Key improvements required:\n{synthesis}\n"
            f"{context_reminder}\n"
            "Write the final, improved version of the research idea. "
            "Ensure the hypothesis is clearly stated, the falsification experiment is concrete, "
            "and all datasets/baselines/metrics are named explicitly."
            + IDEA_FORMAT
        )
        try:
            return call_llm(revise_prompt, model, client, temperature)
        except Exception:
            return draft if draft else f"Research idea about {topic}: {selected_hyp}"


GENERATOR = S12Generator()
```

### FILE: idea_tournament/prompts.py

```python
"""All LLM prompts for IdeaTreeSearch and Elo tournament.

These are the primary edit targets for the SWE agent. Changing a prompt here
changes how every S_sota idea is generated — no other files need touching.

Reference: ideas/idea-tournament/SKILL.md and references/tree-search-protocol.md
"""

# ── Phase 1: Tree-Structured Idea Generation ─────────────────────────────────

TREE_L1_PROMPT = """\
You are generating a structured idea tree for a research topic.

Topic (Level 0 seed): {topic}

{sota_context}

## Task: Generate Level 1 — Technique Variants

Generate exactly 3 fundamentally different technical approaches to this topic.
Each technique must represent a distinct paradigm — not variations of the same approach.
Good variation: pruning vs quantization vs distillation (different principles)
Bad variation: structured pruning vs unstructured pruning (same approach, different params)

For each, write a 2-sentence description:
  Sentence 1: What the technique does.
  Sentence 2: The key hypothesis — why this approach might work better than alternatives.

Respond with ONLY this JSON:
{{
  "techniques": [
    {{"id": "T1", "name": "<short name>", "description": "<2 sentences>"}},
    {{"id": "T2", "name": "<short name>", "description": "<2 sentences>"}},
    {{"id": "T3", "name": "<short name>", "description": "<2 sentences>"}}
  ]
}}"""

TREE_L2_PROMPT = """\
You are expanding a research idea tree to Level 2 (Domain adaptations).

Topic: {topic}

Level 1 Techniques:
{techniques_str}

## Task: Generate Level 2 — Domain Adaptations

For each technique, generate 2 application domains where the technique faces
fundamentally different constraints — each domain should create distinct research challenges.
Good variation: edge devices vs batch servers (different bottlenecks)
Bad variation: text classification vs sentiment analysis (same technical challenge)

Respond with ONLY this JSON:
{{
  "domains": [
    {{"id": "T1-D1", "parent": "T1", "domain": "<domain name>", "description": "<2 sentences: how technique adapts to this domain>"}},
    {{"id": "T1-D2", "parent": "T1", "domain": "<domain name>", "description": "<2 sentences>"}},
    {{"id": "T2-D1", "parent": "T2", "domain": "<domain name>", "description": "<2 sentences>"}},
    {{"id": "T2-D2", "parent": "T2", "domain": "<domain name>", "description": "<2 sentences>"}},
    {{"id": "T3-D1", "parent": "T3", "domain": "<domain name>", "description": "<2 sentences>"}},
    {{"id": "T3-D2", "parent": "T3", "domain": "<domain name>", "description": "<2 sentences>"}}
  ]
}}"""

TREE_L3_PROMPT = """\
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

Respond with ONLY this JSON:
{{
  "leaves": [
    {{
      "id": "T1-D1-F1",
      "parent": "T1-D1",
      "title": "<one-line title>",
      "description": "<3 sentences: (1) what it does, (2) why it's novel vs existing work, (3) how to validate>"
    }},
    ... (12 total, 2 per domain node)
  ]
}}"""

TREE_REVIEW_PROMPT = """\
Review and refine these {n} research idea candidates for topic: {topic}

{sota_context}

Candidates:
{candidates_str}

## Task
For each candidate:
1. Remove vague language ("might", "could potentially", "may help") — make claims specific
2. Ensure the novelty claim explicitly states what prior work cannot do that this can
3. If two candidates are near-duplicates (same technique + domain + formulation), merge
   them — keep the sharper description, discard the weaker

Respond with ONLY this JSON (may have fewer entries after merging):
{{
  "refined": [
    {{"id": "<id>", "title": "<title>", "description": "<refined 3-sentence description>"}},
    ...
  ]
}}"""

# ── Phase 2: Elo Tournament ───────────────────────────────────────────────────
# 4 dimensions from SKILL.md: Novelty, Feasibility, Relevance, Clarity (equal weight)

TOURNAMENT_JUDGE_PROMPT = """\
Compare two research ideas for the topic: {topic}

IDEA A:
{idea_a}

IDEA B:
{idea_b}

Score each idea on these four dimensions (1-10 scale):
1. Novelty     — How different is this from existing published work?
2. Feasibility — Can this be implemented and validated within reasonable resources?
3. Relevance   — Does this address an important open problem in the field?
4. Clarity     — Is the idea well-defined enough to start working on immediately?

Scoring guide:
  9-10: Exceptional  |  7-8: Strong  |  5-6: Adequate  |  3-4: Weak  |  1-2: Poor

Declare A winner if sum(A) > sum(B), B winner if sum(B) > sum(A), tie only if equal.

Respond ONLY with this JSON — no other text:
{{
  "scores_a": {{"novelty": <1-10>, "feasibility": <1-10>, "relevance": <1-10>, "clarity": <1-10>}},
  "scores_b": {{"novelty": <1-10>, "feasibility": <1-10>, "relevance": <1-10>, "clarity": <1-10>}},
  "winner": "<A|B|tie>"
}}"""

# ── Phase 3: Final idea expansion ─────────────────────────────────────────────

EXPAND_WINNER_PROMPT = """\
Research topic: {topic}

Tournament winner (selected from {n_candidates} candidates via Elo tournament):
{winner_title}
{winner_description}

SOTA context (your idea must go meaningfully beyond this existing work):
{sota_context}

Write a full, detailed research idea based on this winning direction.
Requirements:
- State the specific open problem being addressed (not a generic gap)
- Describe the proposed method concretely — include key technical components
- Specify evaluation: datasets, baselines, metrics
- Explain novelty relative to the SOTA context above
- Be direct. No hedging, no vague claims.

{idea_format}"""
```

### FILE: idea_tournament/tree_search.py

```python
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
```

### FILE: idea_tournament/tournament.py

```python
"""Elo tournament for leaf idea ranking.

Implements Phase 2 of the EvoScientist idea-tournament skill.
Reference: ideas/idea-tournament/SKILL.md, references/elo-ranking-guide.md

Algorithm:
  - Swiss-system pairing (avoid rematches, pair similar Elo)
  - K=32, starting Elo=1500
  - 4 dimensions: Novelty, Feasibility, Relevance, Clarity (equal weight)
  - Rounds: 4 for ≥10 candidates, 3 for fewer

Edit targets:
  - ELO_K, ELO_START: rating system parameters
  - N_ROUNDS_LARGE, N_ROUNDS_SMALL: number of tournament rounds
  - TOURNAMENT_JUDGE_PROMPT in prompts.py: scoring criteria
"""

import json
import random
import re
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# ── Tournament parameters (edit these to tune) ────────────────────────────────
ELO_K = 32          # Rating change per match (higher = more volatile)
ELO_START = 1500.0  # Starting Elo for all candidates
N_ROUNDS_LARGE = 4  # Rounds for ≥10 candidates (paper recommends 4-5)
N_ROUNDS_SMALL = 3  # Rounds for <10 candidates


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


def _compare_pair(topic: str, idea_a: dict, idea_b: dict,
                  client, model: str) -> str:
    """Compare two leaf ideas. Returns 'A', 'B', or 'tie'.

    Randomizes A/B position to prevent positional bias.
    """
    import idea_tournament.prompts as P
    from systems.base import call_llm

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


def run_tournament(topic: str, leaves: list[dict], client, model: str) -> dict:
    """Swiss-system Elo tournament. Returns the winning leaf idea dict.

    Uses N_ROUNDS_LARGE or N_ROUNDS_SMALL depending on candidate count.
    Stops early if top-3 rankings stabilize (same as previous round).
    """
    import log as _log
    logger = _log.setup("tournament")

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
        # Swiss pairing: sort by Elo, pair adjacent unmatched
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

        # Early stop if top-3 stable
        top3 = tuple(sorted(ratings, key=lambda i: ratings[i], reverse=True)[:3])
        if top3 == prev_top3:
            logger.debug("Top-3 stable after round %d — stopping early", rnd + 1)
            break
        prev_top3 = top3

    best = max(ratings, key=lambda i: ratings[i])
    logger.info("Tournament: %s wins (Elo=%.0f) from %d candidates for '%s'",
                leaves[best].get("id", "?"), ratings[best], n, topic[:40])
    return leaves[best]
```

## Why the previous candidate lost

Previous candidate win rate: 54.0% (75 pairs judged)

Topic: Emergent communication in multi-agent systems
  Judge: Idea A presents a more novel and biologically grounded perspective by explicitly linking energy constraints to emergent language properties, with stronger scientific motivation rooted in neuroscience. While both ideas are well-articulated with clear experimental designs, Idea A's integration of spike-level energy costs and its focus on generalization through biological constraints offers deeper insights into why emergent languages become robust. Idea B, though solid, addresses a more incremental question about modality diversity that has been partially explored in prior work, and its experimental setup, while rigorous, is somewhat more straightforward in execution.

Topic: Emergent communication in multi-agent systems
  Judge: Idea A achieves higher scores across most dimensions through superior experimental clarity (explicit falsification criteria, rigorous statistical design with 50+ runs), stronger scientific usefulness (addresses real-world deployment challenges with validated wireless noise models), and comparable novelty (realistic temporally-correlated noise is more novel than calibrated noise). While Idea B is slightly more feasible and offers interesting meta-communication insights, it relies on less grounded noise models and measures compositionality on linguistic datasets that may not reflect the core multi-agent coordination problem. Idea A's total score (32) exceeds Idea B's (29).

Topic: Emergent communication in multi-agent systems
  Judge: Idea A presents a more focused and theoretically grounded contribution by systematically investigating how dynamic, empirically-motivated noise shapes emergent communication protocols with explicit architectural inductive biases (noise-aware VAE, communication bottlenecks). It has clearer falsifiable hypotheses, more rigorous communication-centric metrics, and stronger novelty in coupling temporal noise dynamics with robustness mechanisms. Idea B, while addressing a real gap in sensor heterogeneity, is less novel (heterogeneous information is well-studied in MARL) and its experimental design conflates sensor diversity effects with encoder architecture differences, making causal claims harder to establish. Idea A scores 32/40 vs Idea B's 27/40.

Topic: Scaling laws for Large Language Models
  Judge: Idea A presents a more focused and novel investigation into an underexplored architectural factor (depth) with tight experimental controls and clear falsification criteria, achieving 30/40 points. Idea B addresses a broader but less novel question about non-monotonic scaling that lacks strong theoretical motivation and faces significant feasibility challenges at the 100B scale, scoring 24/40 points. While both are scientifically motivated, Idea A's methodological rigor and targeted novelty make it the stronger proposal.

Topic: Scaling laws for Large Language Models
  Judge: Idea A scores 29 total points versus Idea B's 25. While Idea B addresses a more ambitious and novel question about trillion-parameter scaling, it is severely hampered by infeasibility—training models at that scale requires computational resources beyond most research institutions. Idea A, though investigating a more incremental phenomenon (phase transitions in the 10M-10B range), is methodologically rigorous, clearly specified, and realistically achievable, making it more likely to produce reliable scientific insights that advance the field.

Topic: Mechanistic interpretability of transformer models
  Judge: Idea A scores higher across all dimensions with a total of 32 vs. 27 for Idea B. Idea A presents a more rigorous and novel approach to a fundamental question about semantic localization in attention heads, with exceptionally clear experimental design, precise success/failure criteria, and strong multi-method validation. Idea B, while addressing an important problem, relies on more established techniques (sparsity regularization) and introduces a less clearly operationalized identifiability metric, making it somewhat incremental despite its multilingual scope.

## Context: pipeline, judge preferences, experiment history

### 1. Current Pipeline

Current champion: S12
Pipeline: IdeaTreeSearch (L1→L2→L3, ~12 candidates) → Elo tournament → Expansion

  • build_idea_tree: using idea_tournament/tree_search.py
  • run_tournament: using idea_tournament/tournament.py
  • generate_idea: ~9 direct call_llm() calls + tree/tournament calls

Editable modules (primary targets for improvement):
  • idea_tournament/prompts.py (169 lines)
  • idea_tournament/tree_search.py (145 lines)
  • idea_tournament/tournament.py (139 lines)

### 2. Accumulated Judge Preferences

When the CHAMPION wins, judges say things like:

> Idea A presents a genuinely novel conceptual framework combining Bayesian inference with generative modeling to address the fundamental limitation of static structure prediction, directly tackling pro
> Idea A presents a more novel and biologically grounded perspective by explicitly linking energy constraints to emergent language properties, with stronger scientific motivation rooted in neuroscience.
> Idea A achieves higher scores across most dimensions through superior experimental clarity (explicit falsification criteria, rigorous statistical design with 50+ runs), stronger scientific usefulness 
> Idea A presents a more focused and theoretically grounded contribution by systematically investigating how dynamic, empirically-motivated noise shapes emergent communication protocols with explicit ar

When the CANDIDATE wins (good — what to aim for):

> Idea B scores higher overall (33 vs 26) with superior experimental clarity, rigorous falsifiable hypotheses, and well-defined evaluation metrics on curated benchmarks. While Idea A addresses important
> Idea B addresses a more fundamental and widespread problem (batch effects in multimodal integration) with a more novel approach (integrating batch correction into the contrastive learning objective ra
> Idea A scores higher overall (32 vs 27) with stronger novelty in its explicit multimodal cross-attention architecture and more rigorous experimental design with clearly defined unseen cell types. Idea
> Idea A scores higher overall (31 vs 29) with superior novelty in its rigorous multi-omic harmonization and GRN-specific application, greater scientific usefulness by addressing a more fundamental regu

### 3. Experiment Log

S12 vs S12: 89% — ✓ ACCEPTED
  Changed: Tournament-selected strategy (11 candidates evaluated):
Approach: Multi-Agent Roleplay Generation
Target: Hypothesis-Fir; Tournament-selected strategy
  Judge when this won: Idea A scores 29/40 while Idea B scores 26/40. Idea A presents a well-scoped, clearly executable study with strong experimental design and high feasib

S14 vs S12: 54% — ✗ REJECTED
  Judge when this won: Idea B addresses a more fundamental and widespread problem (batch effects in multimodal integration) with a more novel approach (integrating batch cor

## Edit history this session

### S12 (from S_sota) → full eval 89% (ACCEPTED)

  ✓ WORKED  (67%): Tournament-selected strategy (11 candidates evaluated):
Approach: Multi-Agent Roleplay Generation
Target: Hypothesis-Fir
  ✓ WORKED  (89%): Tournament-selected strategy (11 candidates evaluated):
Approach: Constraint-Driven Bottom-Up Generation
Target: Feasibi
  ✓ WORKED  (67%): Tournament-selected strategy (10 candidates evaluated):
Approach: Hypothesis-First Adversarial Loop
Target: Adversarial 

### S13 (from S12)

  ✓ WORKED  (56%): PREVIOUS ATTEMPT FAILED (mini-eval: 33.3%, needed >52%).

What was tried:
Tournament-selected strategy (11 candidates ev
  ✗ FAILED  (33%): Tournament-selected strategy (11 candidates evaluated):
Approach: Cross-Domain Analogy Mining
Target: Iterative Cross-Do
  ✗ FAILED  (44%): Tournament-selected strategy (9 candidates evaluated):
Approach: Hypothesis-First Backward Design
Target: Constraint-Fir

### S14 (from S12) → full eval 54% (REJECTED)

  ✗ FAILED  (0%): Tournament-selected strategy (9 candidates evaluated):
Approach: Multi-Agent Collaborative Debate
Target: Parallel Speci

### S15 (from S12)

  ✗ FAILED  (22%): Tournament-selected strategy (12 candidates evaluated):
Approach: Multi-Agent Persona Ensemble
Target: Cross-Domain Anal
  ✗ FAILED  (44%): PREVIOUS ATTEMPT FAILED (mini-eval: 22.2%, needed >52%).

What was tried:
Tournament-selected strategy (12 candidates ev
  ✗ FAILED  (22%): PREVIOUS ATTEMPT FAILED (mini-eval: 44.4%, needed >52%).

What was tried:
PREVIOUS ATTEMPT FAILED (mini-eval: 22.2%, nee

## Your task

Think ARCHITECTURALLY. The current pipeline is: SOTA retrieval → tree search → Elo tournament → expansion.
Ask: is this the right structure at all? What fundamentally different approach could produce better ideas?

Consider radical alternatives (pick the most promising given the failure pattern):

- Multi-agent debate: multiple LLM agents generate independently, then critique each other's ideas
- Cross-domain isomorphism: find analogous solved problems in other fields, transfer the solution structure
- Adversarial generation: one agent proposes, another aggressively attacks assumptions, iterate
- Constraint inversion: start from what is impossible today, work backwards to what would make it possible
- Hypothesis-first: generate a falsifiable hypothesis first, then design the idea around proving it
- Persona diversity: use radically different expert personas (skeptic, practitioner, theorist, outsider)
- Diverge-then-converge: maximize idea diversity at generation, use multiple judges to filter
- Staged refinement: generate rough seeds, then deep-dive the best one with many more LLM calls

Do NOT repeat anything from the experiment log above.
Do NOT suggest minor prompt tweaks — aim for structural changes to how ideas are generated.

Output: 2-3 sentences identifying the core limitation and the structural change direction.

---

## PROMPT 2: PROPOSE_EDIT_PROMPT

> Sent after improvement tournament picks a strategy. Model outputs full Python file.
> **Size: 36,587 chars**

You are a software engineer implementing a new research idea generation strategy.
The tournament below selected a fundamentally different approach to generating ideas —
implement it fully and faithfully.

## Full generator codebase (S12)

### FILE: systems/S12.py

```python
"""S12_r3: Hypothesis-First Adversarial Loop.

Replaces the top-down tree/tournament with a FALSIFIABLE-HYPOTHESIS-FIRST loop:

1. HYPOTHESIS GENERATION: One LLM call produces 5 sharp, testable scientific
   hypotheses about the topic (e.g., "scaling laws break under data-constrained
   regimes because X"). Starting from *claimed truths about the world* rather
   than technique names.

2. ADVERSARIAL ATTACK (parallel, 5 calls): Each hypothesis gets independently
   attacked by an adversarial critic that probes assumption violations, dataset
   biases, theoretical gaps, and practical limitations. Each attack also forces
   a refined/alternative variant.

3. HYPOTHESIS SELECTION (1 call): Aggregate attack+revision pairs and pick the
   strongest surviving hypothesis — the one whose refined version is most
   concrete, novel, and falsifiable.

4. IDEA CONSTRUCTION (1 call): Design the full experimental idea around
   proving/disproving the surviving hypothesis. Forces concrete datasets,
   baselines, and metrics because the falsification target is explicit.

5. MULTI-PERSPECTIVE CRITIQUE (4 calls): Experimentalist, theorist, skeptic
   critique + synthesis.

6. FINAL REVISION (1 call): Incorporate critique.

LLM calls: 1 (hyp) + 5 (attacks) + 1 (select) + 1 (construct) + 4 (critique) + 1 (revise) = ~13
"""

import sys
import os
import concurrent.futures

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from systems.base import IdeaGenerator, DEFAULT_MODEL, call_llm, IDEA_FORMAT


class S12Generator(IdeaGenerator):
    VERSION = "S12"
    DESCRIPTION = (
        "Hypothesis-first adversarial loop: generate falsifiable hypotheses, "
        "attack each in parallel, select strongest survivor, build experimental "
        "idea around falsification, then multi-perspective critique."
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
        # ── Step 0: retrieve SOTA context ───────────────────────────────────
        try:
            import os as _os
            sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
            from retrieval import get_topic_context
            sota_context = get_topic_context(topic, n=5)
        except Exception:
            sota_context = ""

        context_block = f"\n\n{sota_context}\n" if sota_context else ""

        # ── Step 1: generate 5 falsifiable hypotheses ───────────────────────
        hyp_prompt = (
            f"Research topic: {topic}{context_block}\n"
            "Generate exactly 5 sharp, falsifiable scientific hypotheses about this topic.\n"
            "Each hypothesis must:\n"
            "- Make a specific claim about the world (not just 'we can improve X')\n"
            "- Be testable: name what experiment would prove it false\n"
            "- Be non-obvious: it should NOT be directly supported by the related work above\n"
            "- Be concise: 1-2 sentences max\n\n"
            "Format each as:\n"
            "H1: <hypothesis statement>\n"
            "H2: <hypothesis statement>\n"
            "H3: <hypothesis statement>\n"
            "H4: <hypothesis statement>\n"
            "H5: <hypothesis statement>"
        )
        try:
            hyp_raw = call_llm(hyp_prompt, model, client, temperature)
        except Exception as e:
            hyp_raw = f"H1: Standard approaches to {topic} fail because of distribution shift."

        # Parse hypotheses
        hypotheses = []
        for line in hyp_raw.strip().split("\n"):
            line = line.strip()
            if line and (line.startswith("H") and ":" in line[:4]):
                hyp_text = line.split(":", 1)[1].strip()
                if hyp_text:
                    hypotheses.append(hyp_text)
        if not hypotheses:
            hypotheses = [hyp_raw.strip()]
        hypotheses = hypotheses[:5]  # cap at 5

        # ── Step 2: parallel adversarial attacks on each hypothesis ─────────
        def attack_hypothesis(hyp: str) -> str:
            attack_prompt = (
                f"Research topic: {topic}\n\n"
                f"Hypothesis: {hyp}\n\n"
                "You are an adversarial critic. Attack this hypothesis on EACH of these dimensions:\n"
                "1. Assumption violation: What unstated assumption does this rely on? Give a concrete counterexample.\n"
                "2. Dataset bias: What dataset artifact could make this appear true without actually being true?\n"
                "3. Theoretical gap: What known result from the literature contradicts or undermines this?\n"
                "4. Practical limitation: What makes this hypothesis untestable or too expensive to test?\n\n"
                "Then, given these attacks, write a REVISED hypothesis that survives them:\n"
                "Revised: <1-2 sentence refined hypothesis that addresses the above attacks>"
            )
            try:
                return call_llm(attack_prompt, model, client, temperature=0.6)
            except Exception as e:
                return f"Revised: {hyp} (attack failed: {e})"

        attacks = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            futures = [executor.submit(attack_hypothesis, h) for h in hypotheses]
            for f in futures:
                try:
                    attacks.append(f.result(timeout=120))
                except Exception as e:
                    attacks.append(f"Revised: (timeout) {e}")

        # ── Step 3: select the strongest surviving hypothesis ────────────────
        pairs_str = ""
        for i, (hyp, attack) in enumerate(zip(hypotheses, attacks)):
            pairs_str += f"\n--- Hypothesis {i+1} ---\nOriginal: {hyp}\nCritique+Revision:\n{attack}\n"

        select_prompt = (
            f"Research topic: {topic}\n\n"
            f"Below are {len(hypotheses)} original hypotheses, each attacked by an adversarial critic "
            f"and refined into a stronger revised form:\n{pairs_str}\n"
            "Select the ONE hypothesis (by number) whose REVISED form is:\n"
            "- Most concrete and specific (names mechanisms, not just outcomes)\n"
            "- Most falsifiable (clearest path to a disproof experiment)\n"
            "- Most novel (least covered by standard literature)\n\n"
            "Respond with:\n"
            "SELECTED: <number 1-5>\n"
            "REVISED HYPOTHESIS: <copy the revised hypothesis text exactly>\n"
            "REASONING: <1-2 sentences on why this is the strongest>"
        )
        try:
            selection_raw = call_llm(select_prompt, model, client, temperature=0.3)
        except Exception as e:
            selection_raw = f"SELECTED: 1\nREVISED HYPOTHESIS: {hypotheses[0]}\nREASONING: fallback"

        # Extract the selected revised hypothesis
        selected_hyp = hypotheses[0]  # fallback
        for line in selection_raw.strip().split("\n"):
            if line.strip().upper().startswith("REVISED HYPOTHESIS:"):
                candidate = line.split(":", 1)[1].strip()
                if candidate:
                    selected_hyp = candidate
                    break

        # ── Step 4: construct experimental idea around the hypothesis ────────
        context_reminder = (
            f"\nExisting work to differentiate from:\n{sota_context}\n"
            if sota_context else ""
        )
        construct_prompt = (
            f"Research topic: {topic}\n"
            f"{context_reminder}\n"
            f"Core hypothesis to test: {selected_hyp}\n\n"
            "Design a concrete research experiment to PROVE OR DISPROVE this hypothesis.\n"
            "The experiment must:\n"
            "- Name specific datasets you will use (not just 'standard benchmarks')\n"
            "- Name specific baselines you will compare against\n"
            "- Define quantitative success metrics (what number proves it, what number disproves it)\n"
            "- Describe the key technical method in enough detail to implement\n"
            "- Explain precisely what result would falsify the hypothesis\n\n"
            "Write 3-4 paragraphs. Be direct and specific."
        )
        try:
            draft = call_llm(construct_prompt, model, client, temperature)
        except Exception as e:
            draft = f"Research idea for '{topic}' based on hypothesis: {selected_hyp}"

        # ── Step 5: multi-perspective critique ───────────────────────────────
        exp_prompt = (
            f"You are a hard-nosed experimentalist reviewing a research proposal about '{topic}'.\n\n"
            f"Hypothesis being tested: {selected_hyp}\n\n"
            f"Proposed experiment:\n{draft}\n\n"
            "Give 2-3 sharp criticisms focusing purely on experimental feasibility: "
            "Can these experiments actually be run? Are the measurements well-defined? "
            "What controls are missing? What will fail in practice?"
        )
        try:
            critique_exp = call_llm(exp_prompt, model, client, temperature=0.5)
        except Exception:
            critique_exp = "No experimental critique available."

        theory_prompt = (
            f"You are a rigorous theorist reviewing a research proposal about '{topic}'.\n\n"
            f"Hypothesis being tested: {selected_hyp}\n\n"
            f"Proposed experiment:\n{draft}\n\n"
            "Give 2-3 sharp criticisms focusing purely on theoretical grounding: "
            "Is the novelty claim justified? Does it overlap with known results? "
            "Are the underlying assumptions stated and defensible?"
        )
        try:
            critique_theory = call_llm(theory_prompt, model, client, temperature=0.5)
        except Exception:
            critique_theory = "No theoretical critique available."

        skeptic_prompt = (
            f"You are a skeptical reviewer who has seen many overhyped proposals about '{topic}'.\n\n"
            f"Hypothesis being tested: {selected_hyp}\n\n"
            f"Proposed experiment:\n{draft}\n\n"
            "Give 2-3 sharp criticisms focusing on: why this probably won't work, "
            "what the likely negative result is, and whether the scientific payoff "
            "justifies the effort even if it succeeds."
        )
        try:
            critique_skeptic = call_llm(skeptic_prompt, model, client, temperature=0.5)
        except Exception:
            critique_skeptic = "No skeptic critique available."

        synthesis_prompt = (
            f"Three reviewers critiqued a research idea about '{topic}' testing the hypothesis:\n"
            f"'{selected_hyp}'\n\n"
            f"Experimentalist:\n{critique_exp}\n\n"
            f"Theorist:\n{critique_theory}\n\n"
            f"Skeptic:\n{critique_skeptic}\n\n"
            "Synthesize these into the 3 most important actionable improvements "
            "the author must make. Be concise and prioritized."
        )
        try:
            synthesis = call_llm(synthesis_prompt, model, client, temperature=0.3)
        except Exception:
            synthesis = "Improve specificity, add baselines, clarify falsification criteria."

        # ── Step 6: final revision ────────────────────────────────────────────
        revise_prompt = (
            f"Research topic: {topic}\n\n"
            f"Core hypothesis: {selected_hyp}\n\n"
            f"Experimental design:\n{draft}\n\n"
            f"Key improvements required:\n{synthesis}\n"
            f"{context_reminder}\n"
            "Write the final, improved version of the research idea. "
            "Ensure the hypothesis is clearly stated, the falsification experiment is concrete, "
            "and all datasets/baselines/metrics are named explicitly."
            + IDEA_FORMAT
        )
        try:
            return call_llm(revise_prompt, model, client, temperature)
        except Exception:
            return draft if draft else f"Research idea about {topic}: {selected_hyp}"


GENERATOR = S12Generator()
```

### FILE: idea_tournament/prompts.py

```python
"""All LLM prompts for IdeaTreeSearch and Elo tournament.

These are the primary edit targets for the SWE agent. Changing a prompt here
changes how every S_sota idea is generated — no other files need touching.

Reference: ideas/idea-tournament/SKILL.md and references/tree-search-protocol.md
"""

# ── Phase 1: Tree-Structured Idea Generation ─────────────────────────────────

TREE_L1_PROMPT = """\
You are generating a structured idea tree for a research topic.

Topic (Level 0 seed): {topic}

{sota_context}

## Task: Generate Level 1 — Technique Variants

Generate exactly 3 fundamentally different technical approaches to this topic.
Each technique must represent a distinct paradigm — not variations of the same approach.
Good variation: pruning vs quantization vs distillation (different principles)
Bad variation: structured pruning vs unstructured pruning (same approach, different params)

For each, write a 2-sentence description:
  Sentence 1: What the technique does.
  Sentence 2: The key hypothesis — why this approach might work better than alternatives.

Respond with ONLY this JSON:
{{
  "techniques": [
    {{"id": "T1", "name": "<short name>", "description": "<2 sentences>"}},
    {{"id": "T2", "name": "<short name>", "description": "<2 sentences>"}},
    {{"id": "T3", "name": "<short name>", "description": "<2 sentences>"}}
  ]
}}"""

TREE_L2_PROMPT = """\
You are expanding a research idea tree to Level 2 (Domain adaptations).

Topic: {topic}

Level 1 Techniques:
{techniques_str}

## Task: Generate Level 2 — Domain Adaptations

For each technique, generate 2 application domains where the technique faces
fundamentally different constraints — each domain should create distinct research challenges.
Good variation: edge devices vs batch servers (different bottlenecks)
Bad variation: text classification vs sentiment analysis (same technical challenge)

Respond with ONLY this JSON:
{{
  "domains": [
    {{"id": "T1-D1", "parent": "T1", "domain": "<domain name>", "description": "<2 sentences: how technique adapts to this domain>"}},
    {{"id": "T1-D2", "parent": "T1", "domain": "<domain name>", "description": "<2 sentences>"}},
    {{"id": "T2-D1", "parent": "T2", "domain": "<domain name>", "description": "<2 sentences>"}},
    {{"id": "T2-D2", "parent": "T2", "domain": "<domain name>", "description": "<2 sentences>"}},
    {{"id": "T3-D1", "parent": "T3", "domain": "<domain name>", "description": "<2 sentences>"}},
    {{"id": "T3-D2", "parent": "T3", "domain": "<domain name>", "description": "<2 sentences>"}}
  ]
}}"""

TREE_L3_PROMPT = """\
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

Respond with ONLY this JSON:
{{
  "leaves": [
    {{
      "id": "T1-D1-F1",
      "parent": "T1-D1",
      "title": "<one-line title>",
      "description": "<3 sentences: (1) what it does, (2) why it's novel vs existing work, (3) how to validate>"
    }},
    ... (12 total, 2 per domain node)
  ]
}}"""

TREE_REVIEW_PROMPT = """\
Review and refine these {n} research idea candidates for topic: {topic}

{sota_context}

Candidates:
{candidates_str}

## Task
For each candidate:
1. Remove vague language ("might", "could potentially", "may help") — make claims specific
2. Ensure the novelty claim explicitly states what prior work cannot do that this can
3. If two candidates are near-duplicates (same technique + domain + formulation), merge
   them — keep the sharper description, discard the weaker

Respond with ONLY this JSON (may have fewer entries after merging):
{{
  "refined": [
    {{"id": "<id>", "title": "<title>", "description": "<refined 3-sentence description>"}},
    ...
  ]
}}"""

# ── Phase 2: Elo Tournament ───────────────────────────────────────────────────
# 4 dimensions from SKILL.md: Novelty, Feasibility, Relevance, Clarity (equal weight)

TOURNAMENT_JUDGE_PROMPT = """\
Compare two research ideas for the topic: {topic}

IDEA A:
{idea_a}

IDEA B:
{idea_b}

Score each idea on these four dimensions (1-10 scale):
1. Novelty     — How different is this from existing published work?
2. Feasibility — Can this be implemented and validated within reasonable resources?
3. Relevance   — Does this address an important open problem in the field?
4. Clarity     — Is the idea well-defined enough to start working on immediately?

Scoring guide:
  9-10: Exceptional  |  7-8: Strong  |  5-6: Adequate  |  3-4: Weak  |  1-2: Poor

Declare A winner if sum(A) > sum(B), B winner if sum(B) > sum(A), tie only if equal.

Respond ONLY with this JSON — no other text:
{{
  "scores_a": {{"novelty": <1-10>, "feasibility": <1-10>, "relevance": <1-10>, "clarity": <1-10>}},
  "scores_b": {{"novelty": <1-10>, "feasibility": <1-10>, "relevance": <1-10>, "clarity": <1-10>}},
  "winner": "<A|B|tie>"
}}"""

# ── Phase 3: Final idea expansion ─────────────────────────────────────────────

EXPAND_WINNER_PROMPT = """\
Research topic: {topic}

Tournament winner (selected from {n_candidates} candidates via Elo tournament):
{winner_title}
{winner_description}

SOTA context (your idea must go meaningfully beyond this existing work):
{sota_context}

Write a full, detailed research idea based on this winning direction.
Requirements:
- State the specific open problem being addressed (not a generic gap)
- Describe the proposed method concretely — include key technical components
- Specify evaluation: datasets, baselines, metrics
- Explain novelty relative to the SOTA context above
- Be direct. No hedging, no vague claims.

{idea_format}"""
```

### FILE: idea_tournament/tree_search.py

```python
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
```

### FILE: idea_tournament/tournament.py

```python
"""Elo tournament for leaf idea ranking.

Implements Phase 2 of the EvoScientist idea-tournament skill.
Reference: ideas/idea-tournament/SKILL.md, references/elo-ranking-guide.md

Algorithm:
  - Swiss-system pairing (avoid rematches, pair similar Elo)
  - K=32, starting Elo=1500
  - 4 dimensions: Novelty, Feasibility, Relevance, Clarity (equal weight)
  - Rounds: 4 for ≥10 candidates, 3 for fewer

Edit targets:
  - ELO_K, ELO_START: rating system parameters
  - N_ROUNDS_LARGE, N_ROUNDS_SMALL: number of tournament rounds
  - TOURNAMENT_JUDGE_PROMPT in prompts.py: scoring criteria
"""

import json
import random
import re
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# ── Tournament parameters (edit these to tune) ────────────────────────────────
ELO_K = 32          # Rating change per match (higher = more volatile)
ELO_START = 1500.0  # Starting Elo for all candidates
N_ROUNDS_LARGE = 4  # Rounds for ≥10 candidates (paper recommends 4-5)
N_ROUNDS_SMALL = 3  # Rounds for <10 candidates


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


def _compare_pair(topic: str, idea_a: dict, idea_b: dict,
                  client, model: str) -> str:
    """Compare two leaf ideas. Returns 'A', 'B', or 'tie'.

    Randomizes A/B position to prevent positional bias.
    """
    import idea_tournament.prompts as P
    from systems.base import call_llm

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


def run_tournament(topic: str, leaves: list[dict], client, model: str) -> dict:
    """Swiss-system Elo tournament. Returns the winning leaf idea dict.

    Uses N_ROUNDS_LARGE or N_ROUNDS_SMALL depending on candidate count.
    Stops early if top-3 rankings stabilize (same as previous round).
    """
    import log as _log
    logger = _log.setup("tournament")

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
        # Swiss pairing: sort by Elo, pair adjacent unmatched
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

        # Early stop if top-3 stable
        top3 = tuple(sorted(ratings, key=lambda i: ratings[i], reverse=True)[:3])
        if top3 == prev_top3:
            logger.debug("Top-3 stable after round %d — stopping early", rnd + 1)
            break
        prev_top3 = top3

    best = max(ratings, key=lambda i: ratings[i])
    logger.info("Tournament: %s wins (Elo=%.0f) from %d candidates for '%s'",
                leaves[best].get("id", "?"), ratings[best], n, topic[:40])
    return leaves[best]
```

## Failure analysis

[output of ANALYZE_PROMPT — 2-3 sentence failure diagnosis]

## Tournament-selected generation strategy

Selected by IdeaTreeSearch + Elo from 11 candidates:

[tournament-selected strategy description would appear here]

## Context: pipeline, judge preferences, experiment history

### 1. Current Pipeline

Current champion: S12
Pipeline: IdeaTreeSearch (L1→L2→L3, ~12 candidates) → Elo tournament → Expansion

  • build_idea_tree: using idea_tournament/tree_search.py
  • run_tournament: using idea_tournament/tournament.py
  • generate_idea: ~9 direct call_llm() calls + tree/tournament calls

Editable modules (primary targets for improvement):
  • idea_tournament/prompts.py (169 lines)
  • idea_tournament/tree_search.py (145 lines)
  • idea_tournament/tournament.py (139 lines)

### 2. Accumulated Judge Preferences

When the CHAMPION wins, judges say things like:

> Idea A presents a genuinely novel conceptual framework combining Bayesian inference with generative modeling to address the fundamental limitation of static structure prediction, directly tackling pro
> Idea A presents a more novel and biologically grounded perspective by explicitly linking energy constraints to emergent language properties, with stronger scientific motivation rooted in neuroscience.
> Idea A achieves higher scores across most dimensions through superior experimental clarity (explicit falsification criteria, rigorous statistical design with 50+ runs), stronger scientific usefulness 
> Idea A presents a more focused and theoretically grounded contribution by systematically investigating how dynamic, empirically-motivated noise shapes emergent communication protocols with explicit ar

When the CANDIDATE wins (good — what to aim for):

> Idea B scores higher overall (33 vs 26) with superior experimental clarity, rigorous falsifiable hypotheses, and well-defined evaluation metrics on curated benchmarks. While Idea A addresses important
> Idea B addresses a more fundamental and widespread problem (batch effects in multimodal integration) with a more novel approach (integrating batch correction into the contrastive learning objective ra
> Idea A scores higher overall (32 vs 27) with stronger novelty in its explicit multimodal cross-attention architecture and more rigorous experimental design with clearly defined unseen cell types. Idea
> Idea A scores higher overall (31 vs 29) with superior novelty in its rigorous multi-omic harmonization and GRN-specific application, greater scientific usefulness by addressing a more fundamental regu

### 3. Experiment Log

S12 vs S12: 89% — ✓ ACCEPTED
  Changed: Tournament-selected strategy (11 candidates evaluated):
Approach: Multi-Agent Roleplay Generation
Target: Hypothesis-Fir; Tournament-selected strategy
  Judge when this won: Idea A scores 29/40 while Idea B scores 26/40. Idea A presents a well-scoped, clearly executable study with strong experimental design and high feasib

S14 vs S12: 54% — ✗ REJECTED
  Judge when this won: Idea B addresses a more fundamental and widespread problem (batch effects in multimodal integration) with a more novel approach (integrating batch cor

## Edit history this session (DO NOT repeat these)

### S12 (from S_sota) → full eval 89% (ACCEPTED)

  ✓ WORKED  (67%): Tournament-selected strategy (11 candidates evaluated):
Approach: Multi-Agent Roleplay Generation
Target: Hypothesis-Fir
  ✓ WORKED  (89%): Tournament-selected strategy (11 candidates evaluated):
Approach: Constraint-Driven Bottom-Up Generation
Target: Feasibi
  ✓ WORKED  (67%): Tournament-selected strategy (10 candidates evaluated):
Approach: Hypothesis-First Adversarial Loop
Target: Adversarial 

### S13 (from S12)

  ✓ WORKED  (56%): PREVIOUS ATTEMPT FAILED (mini-eval: 33.3%, needed >52%).

What was tried:
Tournament-selected strategy (11 candidates ev
  ✗ FAILED  (33%): Tournament-selected strategy (11 candidates evaluated):
Approach: Cross-Domain Analogy Mining
Target: Iterative Cross-Do
  ✗ FAILED  (44%): Tournament-selected strategy (9 candidates evaluated):
Approach: Hypothesis-First Backward Design
Target: Constraint-Fir

### S14 (from S12) → full eval 54% (REJECTED)

  ✗ FAILED  (0%): Tournament-selected strategy (9 candidates evaluated):
Approach: Multi-Agent Collaborative Debate
Target: Parallel Speci

### S15 (from S12)

  ✗ FAILED  (22%): Tournament-selected strategy (12 candidates evaluated):
Approach: Multi-Agent Persona Ensemble
Target: Cross-Domain Anal
  ✗ FAILED  (44%): PREVIOUS ATTEMPT FAILED (mini-eval: 22.2%, needed >52%).

What was tried:
Tournament-selected strategy (12 candidates ev
  ✗ FAILED  (22%): PREVIOUS ATTEMPT FAILED (mini-eval: 44.4%, needed >52%).

What was tried:
PREVIOUS ATTEMPT FAILED (mini-eval: 22.2%, nee

## Your task

Implement the tournament-selected strategy as a NEW generate_idea() method.
This is a structural change — you may completely replace the current tree search,
tournament, or expansion logic with the new approach.

Implementation guidelines:

- If the strategy involves multiple agents: implement multiple distinct call_llm() calls
with different system roles (proposer, critic, devil's advocate, synthesizer, etc.)
- If the strategy involves isomorphism/analogy: add a step that maps the topic to an
analogous domain and transfers its solution structure
- If the strategy involves adversarial generation: implement explicit attack/defense rounds
- If the strategy involves persona diversity: define each persona as a distinct prompt
and have each generate independently before synthesis
- Budget ~10-15 LLM calls per idea (same as champion) — use them differently, not fewer

CRITICAL robustness requirements (failure to follow these causes runtime crashes):

- Every call_llm() call MUST be wrapped in try/except — never let an API call crash the function
- Never call len(), enumerate(), or index into a variable that might be None or empty
→ always check: `if not results: results = [fallback_value]`
- Every intermediate result (list of candidates, parsed JSON, etc.) must have a fallback:
if parsing fails or returns empty, use a sensible default and continue
- The generate_idea() method must ALWAYS return a non-empty string, even on full failure
→ end with: `return final_idea or call_llm(fallback_prompt, model, client, temperature)`

The result must still produce output in IDEA_FORMAT. But the path to get there can be
completely different from the champion's tree-search → tournament → expand pipeline.

Then write the output file following the format below.

## Output format — CRITICAL isolation rule

Your output is a SINGLE complete Python file: systems/S15.py

The champion S_sota.py imports from idea_tournament/ at runtime.
Your candidate will be evaluated HEAD-TO-HEAD against S_sota.
For a fair comparison, you MUST inline any code you are modifying:

- Changing a prompt?  → Define the new prompt string as a local variable in
generate_idea(), do NOT use the idea_tournament.prompts version.
- Changing tree_search or tournament logic?  → Copy + modify the relevant
functions inline inside generate_idea() or as module-level helpers.
- NOT changing something?  → You may still import it from idea_tournament/.

If you import from idea_tournament/ for code you modified, both systems will
run the SAME code and the comparison will be meaningless.

The file must:

1. Import: from systems.base import IdeaGenerator, DEFAULT_MODEL, call_llm, IDEA_FORMAT
2. Define: class S15Generator(IdeaGenerator) with VERSION = "S15"
3. Implement: generate_idea(self, topic, client, model, temperature) with all logic inline
4. End with: GENERATOR = S15Generator()

Respond with ONLY the complete Python file — no markdown fences, no preamble.