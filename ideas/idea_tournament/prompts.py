"""Programmatic prompts for IdeaTreeSearch + Elo (benchmark harness).

**Canonical specs** live under repo-root ``skills/idea-tournament`` and
``skills/research-ideation`` (same Markdown Claude Code / the agent reads).
This file keeps JSON-shaped task instructions and **appends** those skill documents
so SWE / Claude Code edits can target either the snippets here or the ``skills/*.md`` files.

Edit targets:
  • ``skills/idea-tournament/references/*.md`` — rubrics & protocols (primary)
  • This file — JSON templates and truncation limits
"""

from __future__ import annotations

import sys
from pathlib import Path

_root = Path(__file__).resolve().parent.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

from canonical_skills import load_skill_document  # noqa: E402


def _clip(text: str, max_chars: int) -> str:
    text = (text or "").strip()
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 30] + "\n\n… [truncated for context limit]\n"


# Loaded once at import — same content the interactive skill uses.
_SK_TREE_FULL = load_skill_document("idea-tournament/references/tree-search-protocol.md")
_SK_TREE_LONG = _clip(_SK_TREE_FULL, 9000)
_SK_TREE_SHORT = _clip(_SK_TREE_FULL, 4000)
_SK_ELO = _clip(
    load_skill_document("idea-tournament/references/elo-ranking-guide.md"), 8000
)
_SK_RI_LIT = _clip(
    load_skill_document("research-ideation/references/literature-tree.md"), 4000
)
_SK_PROPOSAL = _clip(
    load_skill_document("idea-tournament/references/proposal-extension.md"), 6000
)

_MARK_TREE = (
    "\n\n---\n### Canonical: `skills/idea-tournament/references/tree-search-protocol.md`\n"
)
_MARK_RI = (
    "\n\n---\n### Canonical: `skills/research-ideation/references/literature-tree.md`\n"
)
_MARK_ELO = (
    "\n\n---\n### Canonical: `skills/idea-tournament/references/elo-ranking-guide.md`\n"
)
_MARK_PROP = (
    "\n\n---\n### Canonical: `skills/idea-tournament/references/proposal-extension.md`\n"
)

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

Target: up to 21 leaf ideas total where possible (paper N_I); at least 12 if structure allows.
These are the candidates that will compete in the Elo tournament. Make each one specific enough to act on immediately.

Respond with ONLY this JSON:
{{
  "leaves": [
    {{
      "id": "T1-D1-F1",
      "parent": "T1-D1",
      "title": "<one-line title>",
      "description": "<3 sentences: (1) what it does, (2) why it's novel vs existing work, (3) how to validate>"
    }},
    ... (2 per domain node; respect N_I ≤ 21)
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
4. Prune to at most 21 leaves (paper N_I cap).

Respond with ONLY this JSON (may have fewer entries after merging):
{{
  "refined": [
    {{"id": "<id>", "title": "<title>", "description": "<refined 3-sentence description>"}},
    ...
  ]
}}"""

# ── Phase 2: Elo Tournament ───────────────────────────────────────────────────

TOURNAMENT_JUDGE_CORE = """\
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


def build_tree_l1_prompt(topic: str, sota_context: str) -> str:
    core = TREE_L1_PROMPT.format(
        topic=topic, sota_context=sota_context or "(none)"
    )
    ri = _SK_RI_LIT or "(File missing: skills/research-ideation/references/literature-tree.md)"
    tree = _SK_TREE_LONG or "(File missing: skills/idea-tournament/references/tree-search-protocol.md)"
    return core + _MARK_RI + ri + _MARK_TREE + tree


def build_tree_l2_prompt(topic: str, techniques_str: str) -> str:
    core = TREE_L2_PROMPT.format(topic=topic, techniques_str=techniques_str)
    tree = _SK_TREE_SHORT or ""
    return core + _MARK_TREE + tree if tree else core


def build_tree_l3_prompt(topic: str, domains_str: str) -> str:
    core = TREE_L3_PROMPT.format(topic=topic, domains_str=domains_str)
    tree = _SK_TREE_SHORT or ""
    return core + _MARK_TREE + tree if tree else core


def build_tree_review_prompt(
    n: int, topic: str, sota_context: str, candidates_str: str
) -> str:
    core = TREE_REVIEW_PROMPT.format(
        n=n,
        topic=topic,
        sota_context=sota_context or "(none)",
        candidates_str=candidates_str,
    )
    tree = _SK_TREE_SHORT or ""
    return core + _MARK_TREE + tree if tree else core


def format_tournament_judge_prompt(topic: str, idea_a: str, idea_b: str) -> str:
    core = TOURNAMENT_JUDGE_CORE.format(
        topic=topic, idea_a=idea_a, idea_b=idea_b
    )
    elo = _SK_ELO or ""
    return core + _MARK_ELO + elo if elo else core


def format_expand_winner_prompt(
    topic: str,
    n_candidates: int,
    winner_title: str,
    winner_description: str,
    sota_context: str,
    idea_format: str,
) -> str:
    core = EXPAND_WINNER_PROMPT.format(
        topic=topic,
        n_candidates=n_candidates,
        winner_title=winner_title,
        winner_description=winner_description,
        sota_context=sota_context or "(none)",
        idea_format=idea_format,
    )
    prop = _SK_PROPOSAL or ""
    return core + _MARK_PROP + prop if prop else core


# Back-compat: old code used TOURNAMENT_JUDGE_PROMPT.format(...) without skill appendix.
TOURNAMENT_JUDGE_PROMPT = TOURNAMENT_JUDGE_CORE
