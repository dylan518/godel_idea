"""S12_r3: Hypothesis-First Adversarial Loop.

Replaces the top-down tree/tournament with a FALSIFIABLE-HYPOTHESIS-FIRST loop:

1. HYPOTHESIS GENERATION: One LLM call produces 5 sharp, testable scientific
   hypotheses about the topic (e.g., "scaling laws break under data-constrained
   regimes because X"). Starting from *claimed truths about the world* rather
   than technique names.

2. ADVERSARIAL ATTACK (parallel, 5 calls): Each hypothesis gets independently
   attacked by an adversarial critic that probes assumption violations, dataset
   biases, theoretical gaps, and practical limitations. Each attack also forces
   a refined/alternative variant.

3. HYPOTHESIS SELECTION (1 call): Aggregate attack+revision pairs and pick the
   strongest surviving hypothesis — the one whose refined version is most
   concrete, novel, and falsifiable.

4. IDEA CONSTRUCTION (1 call): Design the full experimental idea around
   proving/disproving the surviving hypothesis. Forces concrete datasets,
   baselines, and metrics because the falsification target is explicit.

5. MULTI-PERSPECTIVE CRITIQUE (4 calls): Experimentalist, theorist, skeptic
   critique + synthesis.

6. FINAL REVISION (1 call): Incorporate critique.

LLM calls: 1 (hyp) + 5 (attacks) + 1 (select) + 1 (construct) + 4 (critique) + 1 (revise) = ~13
"""

import sys
import os
import concurrent.futures

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from systems.base import IdeaGenerator, DEFAULT_MODEL, call_llm, IDEA_FORMAT


class S12Generator(IdeaGenerator):
    VERSION = "S12"
    DESCRIPTION = (
        "Hypothesis-first adversarial loop: generate falsifiable hypotheses, "
        "attack each in parallel, select strongest survivor, build experimental "
        "idea around falsification, then multi-perspective critique."
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

        context_block = f"\n\n{sota_context}\n" if sota_context else ""

        # ── Step 1: generate 5 falsifiable hypotheses ───────────────────────
        hyp_prompt = (
            f"Research topic: {topic}{context_block}\n"
            "Generate exactly 5 sharp, falsifiable scientific hypotheses about this topic.\n"
            "Each hypothesis must:\n"
            "- Make a specific claim about the world (not just 'we can improve X')\n"
            "- Be testable: name what experiment would prove it false\n"
            "- Be non-obvious: it should NOT be directly supported by the related work above\n"
            "- Be concise: 1-2 sentences max\n\n"
            "Format each as:\n"
            "H1: <hypothesis statement>\n"
            "H2: <hypothesis statement>\n"
            "H3: <hypothesis statement>\n"
            "H4: <hypothesis statement>\n"
            "H5: <hypothesis statement>"
        )
        try:
            hyp_raw = call_llm(hyp_prompt, model, client, temperature)
        except Exception as e:
            hyp_raw = f"H1: Standard approaches to {topic} fail because of distribution shift."

        # Parse hypotheses
        hypotheses = []
        for line in hyp_raw.strip().split("\n"):
            line = line.strip()
            if line and (line.startswith("H") and ":" in line[:4]):
                hyp_text = line.split(":", 1)[1].strip()
                if hyp_text:
                    hypotheses.append(hyp_text)
        if not hypotheses:
            hypotheses = [hyp_raw.strip()]
        hypotheses = hypotheses[:5]  # cap at 5

        # ── Step 2: parallel adversarial attacks on each hypothesis ─────────
        def attack_hypothesis(hyp: str) -> str:
            attack_prompt = (
                f"Research topic: {topic}\n\n"
                f"Hypothesis: {hyp}\n\n"
                "You are an adversarial critic. Attack this hypothesis on EACH of these dimensions:\n"
                "1. Assumption violation: What unstated assumption does this rely on? Give a concrete counterexample.\n"
                "2. Dataset bias: What dataset artifact could make this appear true without actually being true?\n"
                "3. Theoretical gap: What known result from the literature contradicts or undermines this?\n"
                "4. Practical limitation: What makes this hypothesis untestable or too expensive to test?\n\n"
                "Then, given these attacks, write a REVISED hypothesis that survives them:\n"
                "Revised: <1-2 sentence refined hypothesis that addresses the above attacks>"
            )
            try:
                return call_llm(attack_prompt, model, client, temperature=0.6)
            except Exception as e:
                return f"Revised: {hyp} (attack failed: {e})"

        attacks = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            futures = [executor.submit(attack_hypothesis, h) for h in hypotheses]
            for f in futures:
                try:
                    attacks.append(f.result(timeout=120))
                except Exception as e:
                    attacks.append(f"Revised: (timeout) {e}")

        # ── Step 3: select the strongest surviving hypothesis ────────────────
        pairs_str = ""
        for i, (hyp, attack) in enumerate(zip(hypotheses, attacks)):
            pairs_str += f"\n--- Hypothesis {i+1} ---\nOriginal: {hyp}\nCritique+Revision:\n{attack}\n"

        select_prompt = (
            f"Research topic: {topic}\n\n"
            f"Below are {len(hypotheses)} original hypotheses, each attacked by an adversarial critic "
            f"and refined into a stronger revised form:\n{pairs_str}\n"
            "Select the ONE hypothesis (by number) whose REVISED form is:\n"
            "- Most concrete and specific (names mechanisms, not just outcomes)\n"
            "- Most falsifiable (clearest path to a disproof experiment)\n"
            "- Most novel (least covered by standard literature)\n\n"
            "Respond with:\n"
            "SELECTED: <number 1-5>\n"
            "REVISED HYPOTHESIS: <copy the revised hypothesis text exactly>\n"
            "REASONING: <1-2 sentences on why this is the strongest>"
        )
        try:
            selection_raw = call_llm(select_prompt, model, client, temperature=0.3)
        except Exception as e:
            selection_raw = f"SELECTED: 1\nREVISED HYPOTHESIS: {hypotheses[0]}\nREASONING: fallback"

        # Extract the selected revised hypothesis
        selected_hyp = hypotheses[0]  # fallback
        for line in selection_raw.strip().split("\n"):
            if line.strip().upper().startswith("REVISED HYPOTHESIS:"):
                candidate = line.split(":", 1)[1].strip()
                if candidate:
                    selected_hyp = candidate
                    break

        # ── Step 4: construct experimental idea around the hypothesis ────────
        context_reminder = (
            f"\nExisting work to differentiate from:\n{sota_context}\n"
            if sota_context else ""
        )
        construct_prompt = (
            f"Research topic: {topic}\n"
            f"{context_reminder}\n"
            f"Core hypothesis to test: {selected_hyp}\n\n"
            "Design a concrete research experiment to PROVE OR DISPROVE this hypothesis.\n"
            "The experiment must:\n"
            "- Name specific datasets you will use (not just 'standard benchmarks')\n"
            "- Name specific baselines you will compare against\n"
            "- Define quantitative success metrics (what number proves it, what number disproves it)\n"
            "- Describe the key technical method in enough detail to implement\n"
            "- Explain precisely what result would falsify the hypothesis\n\n"
            "Write 3-4 paragraphs. Be direct and specific."
        )
        try:
            draft = call_llm(construct_prompt, model, client, temperature)
        except Exception as e:
            draft = f"Research idea for '{topic}' based on hypothesis: {selected_hyp}"

        # ── Step 5: multi-perspective critique ───────────────────────────────
        exp_prompt = (
            f"You are a hard-nosed experimentalist reviewing a research proposal about '{topic}'.\n\n"
            f"Hypothesis being tested: {selected_hyp}\n\n"
            f"Proposed experiment:\n{draft}\n\n"
            "Give 2-3 sharp criticisms focusing purely on experimental feasibility: "
            "Can these experiments actually be run? Are the measurements well-defined? "
            "What controls are missing? What will fail in practice?"
        )
        try:
            critique_exp = call_llm(exp_prompt, model, client, temperature=0.5)
        except Exception:
            critique_exp = "No experimental critique available."

        theory_prompt = (
            f"You are a rigorous theorist reviewing a research proposal about '{topic}'.\n\n"
            f"Hypothesis being tested: {selected_hyp}\n\n"
            f"Proposed experiment:\n{draft}\n\n"
            "Give 2-3 sharp criticisms focusing purely on theoretical grounding: "
            "Is the novelty claim justified? Does it overlap with known results? "
            "Are the underlying assumptions stated and defensible?"
        )
        try:
            critique_theory = call_llm(theory_prompt, model, client, temperature=0.5)
        except Exception:
            critique_theory = "No theoretical critique available."

        skeptic_prompt = (
            f"You are a skeptical reviewer who has seen many overhyped proposals about '{topic}'.\n\n"
            f"Hypothesis being tested: {selected_hyp}\n\n"
            f"Proposed experiment:\n{draft}\n\n"
            "Give 2-3 sharp criticisms focusing on: why this probably won't work, "
            "what the likely negative result is, and whether the scientific payoff "
            "justifies the effort even if it succeeds."
        )
        try:
            critique_skeptic = call_llm(skeptic_prompt, model, client, temperature=0.5)
        except Exception:
            critique_skeptic = "No skeptic critique available."

        synthesis_prompt = (
            f"Three reviewers critiqued a research idea about '{topic}' testing the hypothesis:\n"
            f"'{selected_hyp}'\n\n"
            f"Experimentalist:\n{critique_exp}\n\n"
            f"Theorist:\n{critique_theory}\n\n"
            f"Skeptic:\n{critique_skeptic}\n\n"
            "Synthesize these into the 3 most important actionable improvements "
            "the author must make. Be concise and prioritized."
        )
        try:
            synthesis = call_llm(synthesis_prompt, model, client, temperature=0.3)
        except Exception:
            synthesis = "Improve specificity, add baselines, clarify falsification criteria."

        # ── Step 6: final revision ────────────────────────────────────────────
        revise_prompt = (
            f"Research topic: {topic}\n\n"
            f"Core hypothesis: {selected_hyp}\n\n"
            f"Experimental design:\n{draft}\n\n"
            f"Key improvements required:\n{synthesis}\n"
            f"{context_reminder}\n"
            "Write the final, improved version of the research idea. "
            "Ensure the hypothesis is clearly stated, the falsification experiment is concrete, "
            "and all datasets/baselines/metrics are named explicitly."
            + IDEA_FORMAT
        )
        try:
            return call_llm(revise_prompt, model, client, temperature)
        except Exception:
            return draft if draft else f"Research idea about {topic}: {selected_hyp}"


GENERATOR = S12Generator()
