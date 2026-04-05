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
