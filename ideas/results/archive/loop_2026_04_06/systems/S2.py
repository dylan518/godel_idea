"""S_sota: EvoScientist IdeaTreeSearch + Elo Tournament.

This is a thin wrapper. The actual algorithms live in editable modules:

  ideas/idea-tournament/prompts.py     — ALL LLM prompts (primary edit target)
  ideas/idea-tournament/tree_search.py — IdeaTreeSearch algorithm (L0→L1→L2→L3)
  ideas/idea-tournament/tournament.py  — Elo tournament (Swiss-system, 4 dims)

The SWE agent should edit those files directly to improve idea generation.
Do NOT put logic in this file — keep it as a thin orchestrator.

Pipeline per idea (~14 LLM calls):
  1. SOTA retrieval (Semantic Scholar, 5 papers, cached 7 days)
  2. IdeaTreeSearch → ~12 leaf candidates (4 LLM calls)
  3. Elo tournament → best candidate (4-8 LLM calls)
  4. Expand winner → full idea in IDEA_FORMAT (1 LLM call)
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)).replace("/systems", ""))

from systems.base import IdeaGenerator, DEFAULT_MODEL, call_llm, IDEA_FORMAT


class S2Generator(IdeaGenerator):
    VERSION = "S2"
    DESCRIPTION = (
        "EvoScientist IdeaTreeSearch + Elo Tournament. "
        "3-level tree (Technique→Domain→Formulation) generates ~12 leaf candidates "
        "grounded in SOTA papers, then 4-round Swiss Elo tournament selects the best "
        "(Novelty, Feasibility, Relevance, Clarity). "
        "Edit ideas/idea-tournament/prompts.py to improve generation."
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
        # Import here so edits to these modules take effect without restarting
        import idea_tournament.tree_search as tree
        import idea_tournament.tournament as tourn
        import idea_tournament.prompts as P

        # ── 1. SOTA retrieval ─────────────────────────────────────────────────
        sota_context = ""
        try:
            from retrieval import get_topic_context
            sota_context = get_topic_context(topic, n=5)
        except Exception:
            pass
        sota_block = sota_context or "(No SOTA context — generate from first principles.)"

        # ── 2. IdeaTreeSearch → leaf candidates ───────────────────────────────
        leaves = tree.build_idea_tree(topic, sota_block, client, model, temperature)

        # ── 3. Elo tournament → best candidate ────────────────────────────────
        winner = tourn.run_tournament(topic, leaves, client, model)

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


GENERATOR = S2Generator()
