"""S0: Baseline direct prompting.

Straightforward prompt asking for a novel research idea on the given topic.
No special reasoning structure or multi-step refinement.
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from systems.base import IdeaGenerator, DEFAULT_MODEL, IDEA_FORMAT


class S0Generator(IdeaGenerator):
    VERSION = "S0"
    DESCRIPTION = "Baseline: direct single-call prompting with no special structure."

    def get_prompt(self, topic: str) -> str:
        return (
            f"Generate a novel and interesting research idea in the field of: {topic}\n\n"
            "The idea should be:\n"
            "- Original and not already widely studied\n"
            "- Scientifically meaningful and address a real open problem\n"
            "- Feasible to investigate with current or near-future methods\n"
            "- Specific enough to suggest a concrete experimental approach\n"
            + IDEA_FORMAT
        )


GENERATOR = S0Generator()
