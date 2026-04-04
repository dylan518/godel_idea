"""S2: Multi-candidate tournament.

Improvement over S0: generate 3 independent ideas, then ask the model to
synthesize the strongest elements of each into a single superior idea.
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from systems.base import IdeaGenerator, DEFAULT_MODEL, call_llm, IDEA_FORMAT


class S2Generator(IdeaGenerator):
    VERSION = "S2"
    DESCRIPTION = (
        "Multi-candidate tournament: generate 3 independent ideas, "
        "then synthesize the best elements of each into one superior idea."
    )

    def get_prompt(self, topic: str) -> str:
        return f"Generate a novel research idea about: {topic}"

    def generate_idea(
        self,
        topic: str,
        client,
        model: str = DEFAULT_MODEL,
        temperature: float = 0.9,
    ) -> str:
        # Step 1: generate 3 candidates independently (higher temp for diversity)
        candidates = []
        for i in range(3):
            prompt = (
                f"Generate a novel research idea in the field of: {topic}\n\n"
                f"This is attempt {i+1} of 3 — be creative and approach it from "
                f"a different angle than you might have otherwise. "
                "Focus on a specific open problem and suggest a concrete experimental approach. "
                "2-3 paragraphs."
            )
            candidates.append(call_llm(prompt, model, client, temperature))

        # Step 2: synthesize the best elements
        synthesis_prompt = (
            f"Topic: {topic}\n\n"
            "Here are 3 independently generated research ideas:\n\n"
            f"IDEA 1:\n{candidates[0]}\n\n"
            f"IDEA 2:\n{candidates[1]}\n\n"
            f"IDEA 3:\n{candidates[2]}\n\n"
            "Synthesize these into a single, stronger research idea that:\n"
            "- Takes the most novel angle from any of the three\n"
            "- Uses the clearest experimental methodology\n"
            "- Combines complementary insights where possible\n"
            "- Eliminates redundancy and weak elements\n\n"
            "Do not mention that this is a synthesis."
            + IDEA_FORMAT
        )
        return call_llm(synthesis_prompt, model, client, temperature=0.5)


GENERATOR = S2Generator()
