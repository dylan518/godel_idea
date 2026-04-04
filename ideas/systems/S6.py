"""S6: Constraint-first + rubric-aware design.

Improvement over S5: instead of drafting freely and then critiquing,
this approach works *backward from measurability*:

1. CONSTRAINT ELICITATION: Identify the most concrete, measurable quantities
   and experimental handles available for the topic — what CAN be measured,
   what CAN be controlled, what tools exist.

2. GAP IDENTIFICATION: Reason explicitly about what adjacent literature has
   addressed and, crucially, what it has NOT addressed — finding the white space.

3. IDEA CONSTRUCTION: Build the research idea forward from the constraints and
   gap, ensuring every claim is grounded in something measurable.

4. RUBRIC-AWARE REVISION: Separately optimize the idea on each scoring dimension
   (novelty, scientific usefulness, experimental clarity, feasibility) with a
   targeted pass, then integrate into a final coherent proposal.

This approach avoids the failure mode of vague ambitious ideas by anchoring
creativity to what is actually measurable and feasible from the start.
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from systems.base import IdeaGenerator, DEFAULT_MODEL, call_llm, IDEA_FORMAT


class S6Generator(IdeaGenerator):
    VERSION = "S6"
    DESCRIPTION = (
        "Constraint-first + rubric-aware: elicit measurable constraints → "
        "identify literature gaps → construct idea from constraints up → "
        "rubric-aware revision pass on all four scoring dimensions."
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

        # Step 1: Elicit concrete experimental constraints — what is measurable/controllable
        constraints_prompt = (
            f"Research area: {topic}\n\n"
            "Before generating any research idea, carefully enumerate the CONCRETE "
            "EXPERIMENTAL RESOURCES AND CONSTRAINTS available in this area:\n\n"
            "1. What specific quantities can be directly measured with current instruments?\n"
            "2. What variables can be experimentally controlled or manipulated?\n"
            "3. What datasets, model systems, or experimental platforms already exist?\n"
            "4. What are the realistic timescales and cost envelopes for experiments?\n"
            "5. What computational or theoretical tools are mature enough to use reliably?\n\n"
            "Be specific and concrete. Avoid vague generalities. "
            "This is a resource audit, not a research proposal."
        )
        constraints = call_llm(constraints_prompt, model, client, temperature=0.4)

        # Step 2: Identify literature gaps — what adjacent work has NOT addressed
        gap_prompt = (
            f"Research area: {topic}\n\n"
            f"Available experimental resources and constraints:\n{constraints}\n\n"
            "Now reason carefully about the LANDSCAPE OF EXISTING WORK in this area:\n\n"
            "1. What are the 3-4 most established results or approaches in this field?\n"
            "2. What SPECIFIC questions do these approaches leave unanswered?\n"
            "3. What assumptions do existing methods make that have never been tested?\n"
            "4. What combinations of ideas from adjacent fields have NOT been tried?\n"
            "5. What is the single most important gap that, if addressed, would "
            "   unlock significant downstream progress?\n\n"
            "Focus on gaps that are addressable given the constraints identified above. "
            "Identify the most promising white space."
        )
        gaps = call_llm(gap_prompt, model, client, temperature=0.5)

        # Step 3: Construct the research idea bottom-up from constraints and gaps
        construct_prompt = (
            f"Research area: {topic}\n\n"
            f"Concrete experimental constraints and resources:\n{constraints}\n\n"
            f"Key literature gaps identified:\n{gaps}\n\n"
            "Now design a research idea that:\n"
            "- Directly targets the most important gap identified above\n"
            "- Is BUILT AROUND the concrete measurable quantities listed\n"
            "- Has a specific, testable central hypothesis\n"
            "- Proposes a concrete experimental or computational protocol\n"
            "- Makes at least one falsifiable prediction\n\n"
            "Write 2-3 paragraphs. Every claim should be grounded in something "
            "measurable or verifiable. Do not propose anything that cannot be "
            "tested with the resources identified."
        )
        draft = call_llm(construct_prompt, model, client, temperature)

        # Step 4a: Novelty pass — sharpen the novel angle
        novelty_prompt = (
            f"Research area: {topic}\n\n"
            f"Draft research idea:\n{draft}\n\n"
            "NOVELTY REVIEW: Assess and sharpen the novelty of this idea.\n"
            "- Is the central insight genuinely new, or a minor variation of known work?\n"
            "- What is the single most novel claim, and can it be made more precise?\n"
            "- Suggest one concrete change that would make the novelty undeniable.\n\n"
            "Provide a revised statement of the core novel contribution (2-4 sentences)."
        )
        novelty_fix = call_llm(novelty_prompt, model, client, temperature=0.5)

        # Step 4b: Feasibility pass — ensure it can actually be done
        feasibility_prompt = (
            f"Research area: {topic}\n\n"
            f"Draft research idea:\n{draft}\n\n"
            f"Available constraints/resources:\n{constraints}\n\n"
            "FEASIBILITY REVIEW: Identify any steps that are NOT feasible with "
            "current or near-future resources, and propose concrete replacements "
            "that preserve the scientific goal but are actually achievable.\n"
            "Output: a short list of feasibility fixes (if any), or confirm feasibility."
        )
        feasibility_fix = call_llm(feasibility_prompt, model, client, temperature=0.4)

        # Step 4c: Experimental clarity pass — make methodology crisp
        clarity_prompt = (
            f"Research area: {topic}\n\n"
            f"Draft research idea:\n{draft}\n\n"
            "EXPERIMENTAL CLARITY REVIEW: The idea must have a crystal-clear methodology.\n"
            "- What is the exact experimental protocol, step by step?\n"
            "- What is the primary measurable outcome?\n"
            "- What controls are needed?\n"
            "- What result would confirm the hypothesis? What would falsify it?\n\n"
            "Output: a concise experimental design specification (3-6 bullet points) "
            "that makes the methodology unambiguous."
        )
        clarity_fix = call_llm(clarity_prompt, model, client, temperature=0.4)

        # Step 5: Integrate all improvements into a final polished proposal
        final_prompt = (
            f"Research area: {topic}\n\n"
            f"Original draft:\n{draft}\n\n"
            f"Sharpened novelty contribution:\n{novelty_fix}\n\n"
            f"Feasibility fixes:\n{feasibility_fix}\n\n"
            f"Experimental design specification:\n{clarity_fix}\n\n"
            "Write the FINAL research idea integrating all of the above improvements. "
            "The idea must be:\n"
            "- Clearly novel with a precisely stated new contribution\n"
            "- Addressing a real open problem of scientific importance\n"
            "- Backed by a concrete, step-by-step experimental methodology\n"
            "- Achievable with current or near-future resources\n\n"
            "Be specific, rigorous, and concrete throughout. "
            "Do not include vague aspirational language."
            + IDEA_FORMAT
        )
        return call_llm(final_prompt, model, client, temperature)


GENERATOR = S6Generator()
