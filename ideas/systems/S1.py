"""S1: Self-critique and revision.

Improvement over S0: generate a draft idea, then explicitly critique its weaknesses,
then produce a revised idea that addresses those weaknesses.
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from systems.base import IdeaGenerator, DEFAULT_MODEL, call_llm, IDEA_FORMAT


class S1Generator(IdeaGenerator):
    VERSION = "S1"
    DESCRIPTION = (
        "Self-critique and revision: draft an idea, identify its weaknesses, "
        "then revise into a stronger version."
    )

    def get_prompt(self, topic: str) -> str:
        # Used only as fallback; generate_idea overrides with multi-step
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

        # Step 2: critique
        critique_prompt = (
            f"Here is a research idea about '{topic}':\n\n{draft}\n\n"
            "Identify the 3 most significant weaknesses of this idea. Be specific and harsh. "
            "Focus on: lack of novelty, vague methodology, feasibility issues, or missed opportunities."
        )
        critique = call_llm(critique_prompt, model, client, temperature=0.5)

        # Step 3: revise
        revise_prompt = (
            f"Research topic: {topic}\n\n"
            f"Original idea:\n{draft}\n\n"
            f"Critique:\n{critique}\n\n"
            "Now write an improved version of the research idea that directly addresses "
            "every weakness identified in the critique. Be specific and concrete."
            + IDEA_FORMAT
        )
        return call_llm(revise_prompt, model, client, temperature)


GENERATOR = S1Generator()
