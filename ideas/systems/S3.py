"""S3: Structured decomposition.

Improvement over S0: before generating, decompose the topic into its core
unsolved subproblems, pick the most tractable one, then generate an idea
explicitly structured as Hypothesis / Method / Expected Result / Novelty claim.
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from systems.base import IdeaGenerator, DEFAULT_MODEL, call_llm, IDEA_FORMAT


class S3Generator(IdeaGenerator):
    VERSION = "S3"
    DESCRIPTION = (
        "Structured decomposition: identify core unsolved subproblems, "
        "pick the most tractable, then output a structured Hypothesis/Method/Result idea."
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
        # Step 1: decompose into open subproblems
        decompose_prompt = (
            f"List the 5 most important UNSOLVED subproblems in: {topic}\n\n"
            "For each, write one sentence: what specifically is unknown or blocked, "
            "and why it matters. Be precise — avoid vague statements like 'more research is needed'."
        )
        subproblems = call_llm(decompose_prompt, model, client, temperature=0.5)

        # Step 2: generate a structured idea targeting the best subproblem
        idea_prompt = (
            f"Topic: {topic}\n\n"
            f"Key unsolved subproblems:\n{subproblems}\n\n"
            "Pick the subproblem where a single well-designed experiment could make "
            "the most decisive progress. Generate a research idea in this exact format:\n\n"
            "HYPOTHESIS: [One sentence — what you predict and why]\n\n"
            "METHOD: [2-3 sentences — specific experimental design, datasets, or techniques]\n\n"
            "EXPECTED RESULT: [What a positive result would look like and how you'd measure it]\n\n"
            "Be concrete. Avoid hedging. Write as if proposing to a grant committee."
            + IDEA_FORMAT
        )
        return call_llm(idea_prompt, model, client, temperature)


GENERATOR = S3Generator()
