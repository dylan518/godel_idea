import sys
import os
import concurrent.futures

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from systems.base import IdeaGenerator, DEFAULT_MODEL, call_llm, IDEA_FORMAT


class S23_r2Generator(IdeaGenerator):
    VERSION = "S23_r2"
    DESCRIPTION = (
        "Tractability-first hypothesis loop: generate concrete, testable hypotheses "
        "grounded in specific datasets and baselines from the start, attack each, "
        "select the most experimentally feasible survivor, build a focused experiment."
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
            from retrieval import get_topic_context
            sota_context = get_topic_context(topic, n=5)
        except Exception:
            sota_context = ""

        context_block = f"\n\n{sota_context}\n" if sota_context else ""

        # ── Step 1: generate 5 tractable, falsifiable hypotheses ─────────────
        # Key change: hypotheses must name specific datasets/tools/baselines
        # and explicitly state what existing method they beat and by how much.
        hyp_prompt = (
            f"Research topic: {topic}{context_block}\n"
            "Generate exactly 5 concrete, testable scientific hypotheses about this topic.\n\n"
            "CRITICAL REQUIREMENTS for each hypothesis:\n"
            "1. Name a SPECIFIC existing method or baseline (e.g., 'BERT-base', 'Adam optimizer', 'ResNet-50')\n"
            "2. Name a SPECIFIC dataset where you will test this (e.g., 'SQuAD 2.0', 'ImageNet-1K', 'GLUE benchmark')\n"
            "3. State a SPECIFIC measurable claim (e.g., 'improves F1 by >3 points', 'reduces latency by 20%')\n"
            "4. The core mechanism must be implementable in under 200 lines of Python\n\n"
            "Bad example: 'Attention mechanisms can be improved by incorporating graph structure'\n"
            "Good example: 'Replacing standard self-attention with a sparse local window of size 64 in BERT-base "
            "improves SQuAD 2.0 F1 by >2 points while reducing inference time by >15% on a single V100'\n\n"
            "Format:\n"
            "H1: <specific hypothesis with named baseline, dataset, and metric>\n"
            "H2: <specific hypothesis with named baseline, dataset, and metric>\n"
            "H3: <specific hypothesis with named baseline, dataset, and metric>\n"
            "H4: <specific hypothesis with named baseline, dataset, and metric>\n"
            "H5: <specific hypothesis with named baseline, dataset, and metric>"
        )
        try:
            hyp_raw = call_llm(hyp_prompt, model, client, temperature)
        except Exception:
            hyp_raw = f"H1: A simple modification to standard {topic} baselines improves performance on common benchmarks."

        # Parse hypotheses
        hypotheses = []
        for line in hyp_raw.strip().split("\n"):
            line = line.strip()
            if line and len(line) > 4 and ":" in line[:4]:
                prefix = line[:4]
                if prefix[0] == "H" and prefix[1:3].rstrip(":").isdigit():
                    hyp_text = line.split(":", 1)[1].strip()
                    if hyp_text:
                        hypotheses.append(hyp_text)
        if not hypotheses:
            # fallback: split by newline and take non-empty lines
            for line in hyp_raw.strip().split("\n"):
                line = line.strip()
                if len(line) > 20:
                    hypotheses.append(line)
        hypotheses = [h for h in hypotheses if len(h) > 10][:5]
        if not hypotheses:
            hypotheses = [f"A targeted modification to standard {topic} methods improves benchmark performance."]

        # ── Step 2: parallel feasibility-focused attacks ─────────────────────
        # Key change: attacks focus on EXPERIMENTAL feasibility, not just theory
        def attack_hypothesis(hyp: str) -> str:
            attack_prompt = (
                f"Research topic: {topic}\n\n"
                f"Hypothesis: {hyp}\n\n"
                "You are a practical research critic. Attack this hypothesis on these dimensions:\n"
                "1. DATASET AVAILABILITY: Is the named dataset publicly available? If not, what substitute exists?\n"
                "2. BASELINE REPRODUCIBILITY: Can the named baseline be reproduced in <1 week with public code?\n"
                "3. METRIC AMBIGUITY: Is the success metric precisely defined and standard in the field?\n"
                "4. IMPLEMENTATION COMPLEXITY: Can the proposed mechanism be implemented in <200 lines of Python?\n\n"
                "Then write a REVISED hypothesis that fixes any problems found above.\n"
                "The revised hypothesis MUST:\n"
                "- Use only publicly available datasets\n"
                "- Compare against baselines with public implementations\n"
                "- Define success with a standard metric and specific threshold\n"
                "- Describe a mechanism simple enough to implement in one afternoon\n\n"
                "Revised: <1-2 sentence revised hypothesis>"
            )
            try:
                return call_llm(attack_prompt, model, client, temperature=0.6)
            except Exception as e:
                return f"Revised: {hyp}"

        attacks = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            futures = [executor.submit(attack_hypothesis, h) for h in hypotheses]
            for f in futures:
                try:
                    attacks.append(f.result(timeout=120))
                except Exception:
                    attacks.append(f"Revised: {hypotheses[len(attacks)] if len(attacks) < len(hypotheses) else 'fallback hypothesis'}")

        # ── Step 3: select hypothesis prioritizing experimental clarity ───────
        # Key change: scoring rubric explicitly weights feasibility highest
        pairs_str = ""
        for i, (hyp, attack) in enumerate(zip(hypotheses, attacks)):
            pairs_str += f"\n--- Hypothesis {i+1} ---\nOriginal: {hyp}\nFeasibility Review + Revision:\n{attack}\n"

        select_prompt = (
            f"Research topic: {topic}\n\n"
            f"Below are {len(hypotheses)} hypotheses, each reviewed for experimental feasibility "
            f"and revised:\n{pairs_str}\n"
            "Select the ONE hypothesis (by number) whose REVISED form best satisfies ALL of:\n"
            "A) EXPERIMENTAL CLARITY (most important): Names specific public datasets and baselines, "
            "defines a precise measurable metric with a threshold\n"
            "B) IMPLEMENTATION SIMPLICITY: The core mechanism can be coded in one afternoon\n"
            "C) NOVELTY: Not a direct replication of published work\n\n"
            "DO NOT select a hypothesis just because it sounds intellectually impressive. "
            "Select the one you could actually run as an experiment starting tomorrow.\n\n"
            "Respond with:\n"
            "SELECTED: <number 1-5>\n"
            "REVISED HYPOTHESIS: <copy the revised hypothesis text exactly>\n"
            "REASONING: <1-2 sentences: specifically what makes this the most experimentally actionable>"
        )
        try:
            selection_raw = call_llm(select_prompt, model, client, temperature=0.3)
        except Exception:
            selection_raw = f"SELECTED: 1\nREVISED HYPOTHESIS: {hypotheses[0]}\nREASONING: fallback"

        # Extract selected hypothesis
        selected_hyp = hypotheses[0]  # fallback
        for line in selection_raw.strip().split("\n"):
            if line.strip().upper().startswith("REVISED HYPOTHESIS:"):
                candidate = line.split(":", 1)[1].strip()
                if candidate and len(candidate) > 10:
                    selected_hyp = candidate
                    break

        # ── Step 4: construct a focused, concrete experimental design ─────────
        # Key change: prompt forces researcher to commit to specific numbers
        # and explicitly simplify anything that would take >1 month
        context_reminder = (
            f"\nExisting work to differentiate from:\n{sota_context}\n"
            if sota_context else ""
        )
        construct_prompt = (
            f"Research topic: {topic}\n"
            f"{context_reminder}\n"
            f"Core hypothesis to test: {selected_hyp}\n\n"
            "Design a MINIMAL VIABLE EXPERIMENT to test this hypothesis.\n\n"
            "The experiment must be scoped so a single researcher can complete it in 4-6 weeks.\n\n"
            "Structure your response as follows:\n\n"
            "DATASETS: List the exact dataset(s) with version numbers. Use only publicly available data.\n\n"
            "BASELINES: List 2-3 specific methods with paper citations or GitHub links. "
            "All must have public implementations.\n\n"
            "METHOD: Describe the proposed modification in concrete steps. "
            "Each step should be implementable. No 'future work' or 'we plan to explore'.\n\n"
            "SUCCESS METRIC: State the ONE primary metric and the exact threshold that confirms the hypothesis. "
            "Example: 'F1 > 87.5 on SQuAD 2.0 dev set (vs. BERT-base baseline of 85.1)'\n\n"
            "FALSIFICATION: State exactly what result would DISPROVE the hypothesis.\n\n"
            "SIMPLIFICATION NOTE: If any component would take >4 weeks to implement, "
            "replace it with a simpler proxy that can be done in <1 week."
        )
        try:
            draft = call_llm(construct_prompt, model, client, temperature)
        except Exception:
            draft = f"Research idea for '{topic}' based on hypothesis: {selected_hyp}"

        # ── Step 5: focused critique from experimentalist perspective ─────────
        # Key change: only 2 critique perspectives (experimentalist + skeptic)
        # to reduce noise and keep focus on what matters for judges
        exp_prompt = (
            f"You are a hard-nosed experimentalist reviewing a research proposal about '{topic}'.\n\n"
            f"Hypothesis: {selected_hyp}\n\n"
            f"Proposed experiment:\n{draft}\n\n"
            "Give exactly 3 criticisms, each as a single actionable sentence:\n"
            "1. What specific dataset/baseline choice is weakest and what should replace it?\n"
            "2. What is the most likely confound that could invalidate the result?\n"
            "3. What missing ablation would be needed to isolate the proposed mechanism's contribution?"
        )
        try:
            critique_exp = call_llm(exp_prompt, model, client, temperature=0.5)
        except Exception:
            critique_exp = "Ensure datasets are public, add ablation studies, control for confounds."

        skeptic_prompt = (
            f"You are a skeptical reviewer for '{topic}'.\n\n"
            f"Hypothesis: {selected_hyp}\n\n"
            f"Proposed experiment:\n{draft}\n\n"
            "Give exactly 2 criticisms:\n"
            "1. What is the most likely reason this experiment will show NO improvement over baseline?\n"
            "2. Even if the hypothesis is confirmed, what would make reviewers say 'so what'? "
            "What stronger claim would make this publishable?"
        )
        try:
            critique_skeptic = call_llm(skeptic_prompt, model, client, temperature=0.5)
        except Exception:
            critique_skeptic = "Consider stronger baselines and clearer novelty claims."

        synthesis_prompt = (
            f"Two reviewers critiqued a research idea about '{topic}' testing:\n"
            f"'{selected_hyp}'\n\n"
            f"Experimentalist:\n{critique_exp}\n\n"
            f"Skeptic:\n{critique_skeptic}\n\n"
            "List the 3 most important fixes, ordered by impact. "
            "Each fix should be a single concrete action (e.g., 'Replace dataset X with Y', "
            "'Add ablation removing component Z', 'Strengthen claim to include W'). "
            "Do NOT suggest making the method more complex."
        )
        try:
            synthesis = call_llm(synthesis_prompt, model, client, temperature=0.3)
        except Exception:
            synthesis = "1. Use standard public benchmarks. 2. Add ablation study. 3. Clarify novelty over closest baseline."

        # ── Step 6: final revision ────────────────────────────────────────────
        revise_prompt = (
            f"Research topic: {topic}\n\n"
            f"Core hypothesis: {selected_hyp}\n\n"
            f"Experimental design:\n{draft}\n\n"
            f"Required improvements:\n{synthesis}\n"
            f"{context_reminder}\n"
            "Write the final research idea. Requirements:\n"
            "- State the hypothesis clearly in the first paragraph\n"
            "- Name all datasets, baselines, and metrics explicitly\n"
            "- Describe the method concretely enough to implement\n"
            "- State the falsification criterion explicitly\n"
            "- Keep the scope to what one researcher can do in 4-6 weeks\n"
            "- Do NOT add components that were not in the original design unless they directly address a listed improvement\n"
            + IDEA_FORMAT
        )
        try:
            result = call_llm(revise_prompt, model, client, temperature)
            if result and len(result) > 50:
                return result
        except Exception:
            pass

        # Fallback: return draft if revision fails
        if draft and len(draft) > 50:
            return draft
        return f"Research idea about {topic}: {selected_hyp}"


GENERATOR = S23_r2Generator()
