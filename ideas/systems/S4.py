"""S4: Two-round self-critique.

Improvement over S1: run the critique-and-revise cycle twice.
First round addresses major structural weaknesses; second round
sharpens novelty and experimental clarity on the improved draft.
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from systems.base import IdeaGenerator, DEFAULT_MODEL, call_llm, IDEA_FORMAT


class S4Generator(IdeaGenerator):
    VERSION = "S4"
    DESCRIPTION = (
        "Two-round self-critique: draft → critique → revise → critique → final revision. "
        "First round fixes structural weaknesses; second round sharpens novelty and clarity."
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

        # Step 2: first critique — structural weaknesses
        critique1_prompt = (
            f"Here is a research idea about '{topic}':\n\n{draft}\n\n"
            "Identify the 3 most significant weaknesses of this idea. Be specific and harsh. "
            "Focus on: lack of novelty, vague methodology, feasibility issues, or missed opportunities."
        )
        critique1 = call_llm(critique1_prompt, model, client, temperature=0.5)

        # Step 3: first revision
        revise1_prompt = (
            f"Research topic: {topic}\n\n"
            f"Original idea:\n{draft}\n\n"
            f"Critique:\n{critique1}\n\n"
            "Write an improved version of the research idea that directly addresses "
            "every weakness identified in the critique. Be specific and concrete."
        )
        draft2 = call_llm(revise1_prompt, model, client, temperature)

        # Step 4: second critique — novelty and experimental sharpness
        critique2_prompt = (
            f"Here is a revised research idea about '{topic}':\n\n{draft2}\n\n"
            "Now critique specifically for: (1) Is the novelty claim truly differentiated from "
            "existing work? Name specific papers or methods it might overlap with. "
            "(2) Are the experimental predictions concrete and falsifiable? "
            "(3) What is the single biggest remaining weakness?"
        )
        critique2 = call_llm(critique2_prompt, model, client, temperature=0.5)

        # Step 5: final revision
        final_prompt = (
            f"Research topic: {topic}\n\n"
            f"Revised idea:\n{draft2}\n\n"
            f"Second critique:\n{critique2}\n\n"
            "Write the final, polished version of this research idea. Address all remaining "
            "concerns. Be maximally specific about what is novel, what the experiment tests, "
            "and why it is feasible."
            + IDEA_FORMAT
        )
        return call_llm(final_prompt, model, client, temperature)


GENERATOR = S4Generator()
