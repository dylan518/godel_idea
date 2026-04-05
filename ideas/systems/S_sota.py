"""S_sota: SOTA-grounded + tournament + multi-perspective critique.

This is the new strong baseline that replaces S0 as the starting point
for the self-improvement loop. It combines three components:

1. SOTA RETRIEVAL: Fetches 5 recent Semantic Scholar papers on the topic
   before generating anything. The generator sees actual recent work and
   must produce ideas that go beyond it — grounding novelty claims.

2. MULTI-PERSPECTIVE CRITIQUE (from S5, current champion): draft → three
   reviewer personas (experimentalist, theorist, skeptic) → synthesise →
   revise. Catches orthogonal failure modes.

3. INTRA-TOPIC TOURNAMENT: Generates 3 independent candidates using the
   SOTA context, runs a 2-round Elo tournament to select the best one,
   then applies the S5-style critique to that winner. Only the tournament
   winner advances to cross-system comparison — raising quality before
   the "finals".

This is the EvoScientist-inspired baseline: tournament + SOTA context
+ multi-perspective critique, all in one pipeline.

LLM calls per idea: 3 (candidates) + tournament (2-4 comparisons) + 5 (critique) = ~12
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from systems.base import IdeaGenerator, DEFAULT_MODEL, call_llm, IDEA_FORMAT


class S_sotaGenerator(IdeaGenerator):
    VERSION = "S_sota"
    DESCRIPTION = (
        "SOTA-retrieval + intra-topic Elo tournament + multi-perspective critique. "
        "Generates 3 candidates grounded in recent papers, tournaments them, "
        "then applies experimentalist/theorist/skeptic critique to the winner."
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
            sota_context = ""  # degrade gracefully — context is helpful but not required

        context_block = (
            f"\n\n{sota_context}\n" if sota_context else ""
        )

        # ── Step 1: generate 3 independent candidates ───────────────────────
        candidate_prompt_template = (
            f"Research topic: {topic}{context_block}\n"
            "Generate a novel research idea that makes a concrete contribution "
            "beyond the related work above (if any). The idea must:\n"
            "- Address a specific open problem not fully solved by existing work\n"
            "- Propose a concrete, testable methodology\n"
            "- Be achievable with current or near-future resources\n\n"
            "Write 2-3 paragraphs. Be specific. Avoid restating what the papers above already do."
        )

        candidates = []
        for i in range(3):
            temp = temperature + (i * 0.05)   # slight temperature variation for diversity
            try:
                c = call_llm(candidate_prompt_template, model, client, temp)
                candidates.append(c)
            except Exception as e:
                candidates.append(f"ERROR: {e}")

        # Filter out errors; fall back to first if all fail
        valid = [c for c in candidates if not c.startswith("ERROR")]
        if not valid:
            valid = candidates[:1]

        # ── Step 2: Elo tournament to pick best candidate ────────────────────
        if len(valid) > 1:
            try:
                from tournament import run_tournament
                draft = run_tournament(topic, valid, client, model, rounds=2)
            except Exception:
                draft = valid[0]   # degrade gracefully
        else:
            draft = valid[0]

        # ── Step 3: multi-perspective critique (S5 approach) ─────────────────
        # Experimentalist
        exp_prompt = (
            f"You are a hard-nosed experimentalist reviewing a research proposal about '{topic}'.\n\n"
            f"Idea:\n{draft}\n\n"
            "Give 2-3 sharp criticisms focusing purely on experimental feasibility: "
            "Can these experiments actually be run? Are the measurements well-defined? "
            "What controls are missing? What will fail in practice?"
        )
        critique_exp = call_llm(exp_prompt, model, client, temperature=0.5)

        # Theorist
        theory_prompt = (
            f"You are a rigorous theorist reviewing a research proposal about '{topic}'.\n\n"
            f"Idea:\n{draft}\n\n"
            "Give 2-3 sharp criticisms focusing purely on theoretical grounding: "
            "Is the novelty claim justified? Does it overlap with known results? "
            "Are the underlying assumptions stated and defensible?"
        )
        critique_theory = call_llm(theory_prompt, model, client, temperature=0.5)

        # Skeptic
        skeptic_prompt = (
            f"You are a skeptical reviewer who has seen many overhyped proposals about '{topic}'.\n\n"
            f"Idea:\n{draft}\n\n"
            "Give 2-3 sharp criticisms focusing on: why this probably won't work, "
            "what the likely negative result is, and whether the scientific payoff "
            "justifies the effort even if it succeeds."
        )
        critique_skeptic = call_llm(skeptic_prompt, model, client, temperature=0.5)

        # Synthesise critiques
        synthesis_prompt = (
            f"Three reviewers critiqued a research idea about '{topic}'.\n\n"
            f"Experimentalist:\n{critique_exp}\n\n"
            f"Theorist:\n{critique_theory}\n\n"
            f"Skeptic:\n{critique_skeptic}\n\n"
            "Synthesize these into the 3 most important actionable improvements "
            "the author must make. Be concise and prioritized."
        )
        synthesis = call_llm(synthesis_prompt, model, client, temperature=0.3)

        # Final revision
        context_reminder = (
            f"\nNote: your idea must be clearly differentiated from the related work:\n{sota_context}\n"
            if sota_context else ""
        )
        revise_prompt = (
            f"Research topic: {topic}\n\n"
            f"Original idea:\n{draft}\n\n"
            f"Key improvements required:\n{synthesis}\n"
            f"{context_reminder}\n"
            "Write the final, improved version of the research idea. "
            "Be specific, concrete, and rigorous. Clearly state what is novel."
            + IDEA_FORMAT
        )
        return call_llm(revise_prompt, model, client, temperature)


GENERATOR = S_sotaGenerator()
