"""S7_r2: SOTA-grounded + tournament + multi-perspective critique with two-stage revision.

Key change from S_sota: Instead of a single revision that tries to satisfy all three
reviewer personas simultaneously (causing defensive, hedge-everything outputs), we use
a TWO-STAGE revision approach:

Stage 1: "Strengthen first" — take the best critique insights and use them to make
the idea MORE ambitious and specific, not more defensive. The model is explicitly
told to double down on the core thesis, not hedge it.

Stage 2: "Polish for clarity" — a lightweight final pass that sharpens the writing
and ensures the central claim is prominent, without re-opening the critique loop.

This separates the "what to improve" (synthesis) from the "how to present it"
(bold, committed thesis), avoiding the watered-down defensive output pattern.

LLM calls per idea: 3 (candidates) + tournament (2-4 comparisons) + 5 (critique) + 2 (revision stages) = ~14
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from systems.base import IdeaGenerator, DEFAULT_MODEL, call_llm, IDEA_FORMAT


class S7R2Generator(IdeaGenerator):
    VERSION = "S7_r2"
    DESCRIPTION = (
        "SOTA-retrieval + intra-topic Elo tournament + multi-perspective critique "
        "with two-stage revision: strengthen-first then polish. Avoids the "
        "defensive hedge-everything output caused by single-pass multi-critic revision."
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
            temp = temperature + (i * 0.05)
            try:
                c = call_llm(candidate_prompt_template, model, client, temp)
                candidates.append(c)
            except Exception as e:
                candidates.append(f"ERROR: {e}")

        valid = [c for c in candidates if not c.startswith("ERROR")]
        if not valid:
            valid = candidates[:1]

        # ── Step 2: Elo tournament to pick best candidate ────────────────────
        if len(valid) > 1:
            try:
                from tournament import run_tournament
                draft = run_tournament(topic, valid, client, model, rounds=2)
            except Exception:
                draft = valid[0]
        else:
            draft = valid[0]

        # ── Step 3: multi-perspective critique ────────────────────────────────
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

        # Synthesise — extract the single most important fixable flaw
        synthesis_prompt = (
            f"Three reviewers critiqued a research idea about '{topic}'.\n\n"
            f"Experimentalist:\n{critique_exp}\n\n"
            f"Theorist:\n{critique_theory}\n\n"
            f"Skeptic:\n{critique_skeptic}\n\n"
            "Identify the ONE most important weakness that, if fixed, would make "
            "this idea significantly stronger. Describe concretely what the fix is. "
            "Do NOT list multiple issues — focus only on the single highest-leverage improvement."
        )
        synthesis = call_llm(synthesis_prompt, model, client, temperature=0.3)

        # ── Step 4a: Strengthen-first revision ───────────────────────────────
        # Explicitly told to double down on the core thesis, not hedge
        context_reminder = (
            f"\nRelated work to differentiate from:\n{sota_context}\n"
            if sota_context else ""
        )
        strengthen_prompt = (
            f"Research topic: {topic}\n\n"
            f"Current idea:\n{draft}\n\n"
            f"The single most important improvement to make:\n{synthesis}\n"
            f"{context_reminder}\n"
            "Rewrite the idea incorporating this improvement. Your goal is to make "
            "the central thesis BOLDER and MORE SPECIFIC, not more cautious. "
            "Commit fully to the core claim. Do not add hedges or caveats — "
            "instead, make the methodology more concrete and the novelty claim sharper. "
            "Write 2-3 focused paragraphs with a clear, committed central thesis."
        )
        strengthened = call_llm(strengthen_prompt, model, client, temperature=temperature)

        # ── Step 4b: Polish for clarity ───────────────────────────────────────
        # Lightweight pass t1o ensure the central claim is front and center
        polish_prompt = (
            f"Research topic: {topic}\n\n"
            f"Research idea to polish:\n{strengthened}\n\n"
            "Write the final version of this research idea. Your only job is to ensure:\n"
            "1. The central novel claim appears clearly in the first 2 sentences\n"
            "2. The methodology is described concretely (not vaguely)\n"
            "3. The writing is crisp and confident — no unnecessary hedging\n"
            "Do not add new content or change the core idea. Just sharpen the presentation."
            + IDEA_FORMAT
        )
        return call_llm(polish_prompt, model, client, temperature=0.4)


GENERATOR = S7R2Generator()
