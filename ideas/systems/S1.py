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


class S1_r6Generator(IdeaGenerator):
    VERSION = "S1"
    DESCRIPTION = (
        "EvoScientist IdeaTreeSearch + Elo Tournament. "
        "3-level tree (Technique→Domain→Formulation) generates ~12 leaf candidates "
        "grounded in SOTA papers, then 4-round Swiss Elo tournament selects the best "
        "(Novelty, Feasibility, Relevance, Clarity). "
        "Expansion step uses a structured prompt that anchors on named concrete methods "
        "and a testable hypothesis while maintaining narrative coherence. "
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

        # ── 4. Assemble winner context ─────────────────────────────────────────
        winner_title = winner.get("title", topic)
        winner_description = winner.get("description", "")
        winner_mechanism = winner.get("mechanism", winner.get("method", ""))
        winner_novelty = winner.get("novelty", winner.get("novelty_score", ""))
        winner_feasibility = winner.get("feasibility", winner.get("feasibility_score", ""))
        winner_rationale = winner.get("rationale", winner.get("win_rationale", ""))
        winner_technique = winner.get("technique", "")
        winner_domain = winner.get("domain", "")

        winner_parts = [f"Title: {winner_title}"]
        if winner_description:
            winner_parts.append(f"Description: {winner_description}")
        if winner_mechanism:
            winner_parts.append(f"Core Mechanism: {winner_mechanism}")
        if winner_technique:
            winner_parts.append(f"Technique: {winner_technique}")
        if winner_domain:
            winner_parts.append(f"Domain: {winner_domain}")
        if winner_novelty:
            winner_parts.append(f"Novelty: {winner_novelty}")
        if winner_feasibility:
            winner_parts.append(f"Feasibility: {winner_feasibility}")
        if winner_rationale:
            winner_parts.append(f"Why It Won: {winner_rationale}")

        winner_details = "\n".join(winner_parts)

        # ── 5. Expand winner into full IDEA_FORMAT ────────────────────────────
        # Structured prompt that anchors on named concrete methods and a testable
        # hypothesis while maintaining narrative coherence — the balance that
        # produces high-scoring ideas consistently.
        expand_prompt = f"""You are a creative research scientist writing a compelling research proposal.

A tournament among {len(leaves)} candidate ideas has selected the following winner based on Novelty, Feasibility, Relevance, and Clarity. Develop this into a full research proposal that is both exciting and technically credible.

RESEARCH TOPIC: {topic}

WINNING IDEA:
{winner_details}

RELEVANT SOTA:
{sota_block}

Write a research proposal following this structure:

1. MOTIVATION (2-3 sentences): Why does this problem matter and why is now the right time?

2. CORE INSIGHT: What is the key observation or innovation that makes this idea work? State it clearly in 1-2 sentences.

3. TECHNICAL APPROACH: Describe the method with concrete specificity — name actual algorithms, model architectures, loss functions, or data structures you would use. At least 2-3 specific technical components must be named.

4. HYPOTHESIS & EXPERIMENTAL TEST: State a falsifiable hypothesis. Describe one concrete experiment (dataset, baseline, metric) that would confirm or refute it.

5. LIMITATIONS: Acknowledge 1-2 honest limitations or failure modes.

The proposal should tell a coherent story: gap → insight → method → test. Be specific where it adds credibility; avoid vague generalities.

{IDEA_FORMAT}"""

        return call_llm(expand_prompt, model, client, temperature)


GENERATOR = S1_r6Generator()
