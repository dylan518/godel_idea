"""H0: Strong-model baseline.

Same direct-prompting strategy as S0, but forces gpt-4.1 (full model)
regardless of the model arg passed by the runner. Used to measure whether
a self-improvement loop on a cheap model (gpt-4.1-mini + S1/S4/S5) can
match or beat a stronger model's raw single-shot output.
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from systems.base import IdeaGenerator, call_llm, IDEA_FORMAT

STRONG_MODEL = "gpt-4.1"


class H0Generator(IdeaGenerator):
    VERSION = "H0"
    DESCRIPTION = (
        "Strong-model baseline: direct prompting with gpt-4.1 (full model). "
        "Tests whether cheap-model self-critique can match a stronger model's raw output."
    )

    def get_prompt(self, topic: str) -> str:
        return (
            f"Generate a novel and interesting research idea in the field of: {topic}\n\n"
            "The idea should be original, address a real open problem, and suggest "
            "a concrete experimental approach with clear testable predictions."
            + IDEA_FORMAT
        )

    def generate_idea(
        self,
        topic: str,
        client,
        model: str = STRONG_MODEL,  # always use strong model
        temperature: float = 0.8,
    ) -> str:
        # Always use the strong model, ignore whatever the runner passes
        from systems.base import make_client
        strong_client = make_client(STRONG_MODEL)
        return call_llm(self.get_prompt(topic), STRONG_MODEL, strong_client, temperature)


GENERATOR = H0Generator()
