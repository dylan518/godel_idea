"""S5: Multi-perspective critique.

Improvement over S1: instead of a single self-critique, solicits three
distinct reviewer perspectives (experimentalist, theorist, skeptic),
then synthesizes them into a unified critique before revising.
Captures orthogonal failure modes that a single-voice critique misses.
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from systems.base import IdeaGenerator, DEFAULT_MODEL, call_llm, IDEA_FORMAT


class S5Generator(IdeaGenerator):
    VERSION = "S5"
    DESCRIPTION = (
        "Multi-perspective critique: draft → three reviewer perspectives "
        "(experimentalist, theorist, skeptic) → synthesized critique → revision."
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
        # Step 1: draft
        draft_prompt = (
            f"Generate a novel and interesting research idea in the field of: {topic}\n\n"
            "The idea should be original, address a real open problem, and suggest "
            "a concrete experimental approach. Write 2-3 paragraphs."
        )
        draft = call_llm(draft_prompt, model, client, temperature)

        # Step 2a: experimentalist critique
        exp_prompt = (
            f"You are a hard-nosed experimentalist reviewing a research proposal about '{topic}'.\n\n"
            f"Idea:\n{draft}\n\n"
            "Give 2-3 sharp criticisms focusing purely on experimental feasibility: "
            "Can these experiments actually be run? Are the measurements well-defined? "
            "What controls are missing? What will fail in practice?"
        )
        critique_exp = call_llm(exp_prompt, model, client, temperature=0.5)

        # Step 2b: theorist critique
        theory_prompt = (
            f"You are a rigorous theorist reviewing a research proposal about '{topic}'.\n\n"
            f"Idea:\n{draft}\n\n"
            "Give 2-3 sharp criticisms focusing purely on theoretical grounding: "
            "Is the novelty claim justified? Does it overlap with known results? "
            "Are the underlying assumptions stated and defensible?"
        )
        critique_theory = call_llm(theory_prompt, model, client, temperature=0.5)

        # Step 2c: skeptic critique
        skeptic_prompt = (
            f"You are a skeptical reviewer who has seen many overhyped proposals about '{topic}'.\n\n"
            f"Idea:\n{draft}\n\n"
            "Give 2-3 sharp criticisms focusing on: why this probably won't work, "
            "what the likely negative result is, and whether the scientific payoff "
            "justifies the effort even if it succeeds."
        )
        critique_skeptic = call_llm(skeptic_prompt, model, client, temperature=0.5)

        # Step 3: synthesize critiques
        synthesis_prompt = (
            f"Three reviewers critiqued a research idea about '{topic}'.\n\n"
            f"Experimentalist:\n{critique_exp}\n\n"
            f"Theorist:\n{critique_theory}\n\n"
            f"Skeptic:\n{critique_skeptic}\n\n"
            "Synthesize these into the 3 most important actionable improvements "
            "the author must make. Be concise and prioritized."
        )
        synthesis = call_llm(synthesis_prompt, model, client, temperature=0.3)

        # Step 4: revise
        revise_prompt = (
            f"Research topic: {topic}\n\n"
            f"Original idea:\n{draft}\n\n"
            f"Key improvements required:\n{synthesis}\n\n"
            "Write an improved version of the research idea that directly addresses "
            "all required improvements. Be specific, concrete, and rigorous."
            + IDEA_FORMAT
        )
        return call_llm(revise_prompt, model, client, temperature)


GENERATOR = S5Generator()
