import sys
import os
import concurrent.futures

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from systems.base import IdeaGenerator, DEFAULT_MODEL, call_llm, IDEA_FORMAT


class S23_r1Generator(IdeaGenerator):
    VERSION = "S23_r1"
    DESCRIPTION = (
        "Hypothesis-first adversarial loop with operationalizability-gated generation: "
        "hypotheses are seeded with concrete experimental constraints upfront, attacked "
        "in parallel, selected on falsifiability+concreteness, then constructed with "
        "an experiment-first template that prevents re-elaboration during write-up."
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
        # ── Step 0: retrieve SOTA context ────────────────────────────────────
        try:
            from retrieval import get_topic_context
            sota_context = get_topic_context(topic, n=5)
        except Exception:
            sota_context = ""

        context_block = f"\n\n{sota_context}\n" if sota_context else ""

        # ── Step 1: generate 5 operationalizable hypotheses ──────────────────
        # Key fix vs S12: prompt explicitly requires dataset + falsification number
        # at generation time, not just "testable in principle"
        hyp_prompt = (
            "Research topic: " + topic + context_block + "\n"
            "Generate exactly 5 sharp, operationalizable scientific hypotheses.\n"
            "Each hypothesis MUST embed its own falsification criteria:\n"
            "- Name ONE specific publicly available dataset or benchmark\n"
            "- State ONE specific measurable outcome (a number, threshold, or direction)\n"
            "- Claim something non-obvious that the existing work above does NOT already show\n"
            "- Be 1-2 sentences max\n\n"
            "Bad example (too vague): 'Attention mechanisms fail under distribution shift.'\n"
            "Good example: 'On the WILDS-iWildCam benchmark, models trained with standard "
            "ERM will show >15% accuracy drop vs in-distribution when evaluated on the "
            "test-domain split, and this gap cannot be closed by scaling model size alone.'\n\n"
            "Format:\n"
            "H1: <hypothesis with named dataset and falsification number>\n"
            "H2: <hypothesis with named dataset and falsification number>\n"
            "H3: <hypothesis with named dataset and falsification number>\n"
            "H4: <hypothesis with named dataset and falsification number>\n"
            "H5: <hypothesis with named dataset and falsification number>"
        )
        try:
            hyp_raw = call_llm(hyp_prompt, model, client, temperature)
        except Exception:
            hyp_raw = "H1: On standard benchmarks, baseline approaches to " + topic + " fail to generalize."

        # Parse hypotheses
        hypotheses = []
        for line in (hyp_raw or "").strip().split("\n"):
            line = line.strip()
            if line and len(line) > 4 and line[0] == "H" and ":" in line[:4]:
                hyp_text = line.split(":", 1)[1].strip()
                if hyp_text:
                    hypotheses.append(hyp_text)
        if not hypotheses:
            hypotheses = [(hyp_raw or "").strip() or ("Novel approach to " + topic)]
        hypotheses = hypotheses[:5]

        # ── Step 2: parallel adversarial attacks ─────────────────────────────
        # Fix: attack prompt also checks whether the revised hypothesis RETAINS
        # the concrete dataset + number, preventing complexity-as-armor
        def attack_hypothesis(hyp: str) -> str:
            attack_prompt = (
                "Research topic: " + topic + "\n\n"
                "Hypothesis: " + hyp + "\n\n"
                "You are an adversarial critic. Attack this hypothesis on each dimension:\n"
                "1. Assumption violation: What unstated assumption does this rely on? "
                "Give a concrete counterexample.\n"
                "2. Dataset artifact: Could the named dataset make this appear true due to "
                "benchmark-specific quirks rather than a general phenomenon?\n"
                "3. Threshold arbitrariness: Is the specific number/threshold stated actually "
                "meaningful, or is it cherry-picked to look impressive?\n"
                "4. Scope creep: Does this hypothesis actually require a complex mechanism to "
                "test, or can it be tested with a simpler direct comparison?\n\n"
                "Then write a REVISED hypothesis that:\n"
                "- Survives the above attacks\n"
                "- STILL names a specific dataset and a specific falsification number\n"
                "- Is SIMPLER to test than the original (fewer moving parts)\n\n"
                "Revised: <1-2 sentence refined hypothesis, dataset named, number stated>"
            )
            try:
                return call_llm(attack_prompt, model, client, temperature=0.6)
            except Exception as e:
                return "Revised: " + hyp + " (attack failed)"

        attacks = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            futures = [executor.submit(attack_hypothesis, h) for h in hypotheses]
            for f in futures:
                try:
                    attacks.append(f.result(timeout=120))
                except Exception:
                    attacks.append("Revised: (timeout)")

        # Pad attacks if needed
        while len(attacks) < len(hypotheses):
            attacks.append("Revised: (missing)")

        # ── Step 3: select strongest surviving hypothesis ─────────────────────
        # Fix: selection criteria explicitly prioritize simplicity of test over
        # theoretical sophistication; scoring is on concreteness, not novelty-rhetoric
        pairs_str = ""
        for i, (hyp, attack) in enumerate(zip(hypotheses, attacks)):
            pairs_str += "\n--- Hypothesis " + str(i + 1) + " ---\nOriginal: " + hyp + "\nCritique+Revision:\n" + attack + "\n"

        select_prompt = (
            "Research topic: " + topic + "\n\n"
            "Below are " + str(len(hypotheses)) + " hypotheses, each attacked and revised:\n"
            + pairs_str + "\n"
            "Select the ONE hypothesis whose REVISED form is best on these criteria "
            "(in priority order):\n"
            "1. TESTABILITY: Does the revised hypothesis name a specific real dataset "
            "AND a specific measurable outcome (a number or direction)? This is the "
            "most important criterion — vague revised hypotheses should be ranked last.\n"
            "2. SIMPLICITY: Can this be tested with a straightforward comparison "
            "(train model A, train model B, measure metric X on dataset Y)? "
            "Hypotheses requiring novel architectures or multi-stage pipelines to TEST "
            "(not to implement the method) rank lower.\n"
            "3. NOVELTY: Is the claim non-obvious given the existing literature?\n\n"
            "Respond with:\n"
            "SELECTED: <number 1-5>\n"
            "REVISED HYPOTHESIS: <copy the revised hypothesis text exactly>\n"
            "DATASET: <the specific dataset named in the revised hypothesis>\n"
            "FALSIFICATION CRITERION: <the specific number or measurable outcome>\n"
            "REASONING: <1-2 sentences>"
        )
        try:
            selection_raw = call_llm(select_prompt, model, client, temperature=0.3)
        except Exception:
            selection_raw = "SELECTED: 1\nREVISED HYPOTHESIS: " + (hypotheses[0] if hypotheses else topic) + "\nDATASET: standard benchmark\nFALSIFICATION CRITERION: measurable improvement\nREASONING: fallback"

        # Extract components from selection
        selected_hyp = hypotheses[0] if hypotheses else ("Novel approach to " + topic)
        selected_dataset = "standard benchmark"
        selected_criterion = "measurable improvement"

        for line in (selection_raw or "").strip().split("\n"):
            line_stripped = line.strip()
            upper = line_stripped.upper()
            if upper.startswith("REVISED HYPOTHESIS:"):
                candidate = line_stripped.split(":", 1)[1].strip()
                if candidate:
                    selected_hyp = candidate
            elif upper.startswith("DATASET:"):
                candidate = line_stripped.split(":", 1)[1].strip()
                if candidate:
                    selected_dataset = candidate
            elif upper.startswith("FALSIFICATION CRITERION:"):
                candidate = line_stripped.split(":", 1)[1].strip()
                if candidate:
                    selected_criterion = candidate

        # ── Step 4: experiment-first idea construction ────────────────────────
        # Key fix vs S12: construction prompt is structured as an EXPERIMENT TEMPLATE
        # not a free-form "design a research idea" call. This prevents re-elaboration.
        context_reminder = (
            "\nExisting work to differentiate from:\n" + sota_context + "\n"
            if sota_context else ""
        )
        construct_prompt = (
            "Research topic: " + topic + "\n"
            + context_reminder + "\n"
            "Hypothesis to test: " + selected_hyp + "\n"
            "Target dataset: " + selected_dataset + "\n"
            "Falsification criterion: " + selected_criterion + "\n\n"
            "Write a research experiment plan with EXACTLY these four sections:\n\n"
            "SETUP: In 2-3 sentences, describe the experimental setup. "
            "Name the specific dataset split you will use, the exact metric you will measure, "
            "and the threshold that constitutes a positive vs negative result.\n\n"
            "METHOD: In 2-3 sentences, describe the proposed technical approach. "
            "Focus on what is DIFFERENT from the baselines — the one key change. "
            "Do not invent a complex new architecture; describe a targeted modification.\n\n"
            "BASELINES: List exactly 3 baselines by name (e.g., 'ERM', 'DRO', 'DANN'). "
            "For each, one sentence on why it is the right comparison.\n\n"
            "EXPECTED RESULT: One sentence stating what specific number you expect to see "
            "if the hypothesis is TRUE, and what you expect if it is FALSE.\n\n"
            "Be direct. No hedging. Every claim must reference the named dataset."
        )
        try:
            draft = call_llm(construct_prompt, model, client, temperature)
        except Exception:
            draft = "Research experiment for '" + topic + "' testing: " + selected_hyp

        # ── Step 5: multi-perspective critique ───────────────────────────────
        exp_prompt = (
            "You are a hard-nosed experimentalist reviewing a research proposal about '" + topic + "'.\n\n"
            "Hypothesis: " + selected_hyp + "\n\n"
            "Experiment plan:\n" + (draft or "") + "\n\n"
            "Give exactly 3 criticisms. Focus only on: Are the baselines the right ones? "
            "Is the dataset split appropriate for this claim? Are the metrics well-defined? "
            "What controls are missing? Keep each criticism to 1-2 sentences."
        )
        try:
            critique_exp = call_llm(exp_prompt, model, client, temperature=0.5)
        except Exception:
            critique_exp = "Ensure baselines are appropriate and metrics are well-defined."

        theory_prompt = (
            "You are a rigorous theorist reviewing a research proposal about '" + topic + "'.\n\n"
            "Hypothesis: " + selected_hyp + "\n\n"
            "Experiment plan:\n" + (draft or "") + "\n\n"
            "Give exactly 3 criticisms. Focus only on: Does the novelty claim hold up? "
            "Is there existing work that already shows this? Are the assumptions defensible? "
            "Keep each criticism to 1-2 sentences."
        )
        try:
            critique_theory = call_llm(theory_prompt, model, client, temperature=0.5)
        except Exception:
            critique_theory = "Verify novelty claims against existing literature."

        skeptic_prompt = (
            "You are a skeptical reviewer who has seen many overhyped proposals about '" + topic + "'.\n\n"
            "Hypothesis: " + selected_hyp + "\n\n"
            "Experiment plan:\n" + (draft or "") + "\n\n"
            "Give exactly 3 criticisms. Focus on: Why will this likely fail? "
            "What is the most probable negative result? Is the payoff worth it if it works? "
            "Keep each criticism to 1-2 sentences."
        )
        try:
            critique_skeptic = call_llm(skeptic_prompt, model, client, temperature=0.5)
        except Exception:
            critique_skeptic = "Consider whether the experimental payoff justifies the effort."

        synthesis_prompt = (
            "Three reviewers critiqued a research proposal about '" + topic + "'.\n\n"
            "Experimentalist:\n" + (critique_exp or "") + "\n\n"
            "Theorist:\n" + (critique_theory or "") + "\n\n"
            "Skeptic:\n" + (critique_skeptic or "") + "\n\n"
            "List the 3 most important concrete improvements the author must make. "
            "Each improvement should be actionable (e.g., 'replace baseline X with Y', "
            "'add ablation removing component Z', 'clarify what counts as success on metric M'). "
            "Be concise."
        )
        try:
            synthesis = call_llm(synthesis_prompt, model, client, temperature=0.3)
        except Exception:
            synthesis = "Improve baseline selection, clarify success metrics, add ablations."

        # ── Step 6: final revision ────────────────────────────────────────────
        # Fix vs S12: final revision prompt explicitly prohibits adding new mechanisms
        # and requires that dataset + falsification criterion remain explicit
        revise_prompt = (
            "Research topic: " + topic + "\n\n"
            "Core hypothesis: " + selected_hyp + "\n"
            "Dataset: " + selected_dataset + "\n"
            "Falsification criterion: " + selected_criterion + "\n\n"
            "Experiment plan:\n" + (draft or "") + "\n\n"
            "Required improvements:\n" + (synthesis or "") + "\n"
            + context_reminder + "\n"
            "Write the final version of this research idea. Rules:\n"
            "- Do NOT introduce new technical components that were not in the experiment plan above\n"
            "- The hypothesis must appear verbatim or nearly verbatim in the final text\n"
            "- The dataset (" + selected_dataset + ") and falsification criterion "
            "(" + selected_criterion + ") must be explicitly stated\n"
            "- Improvements should sharpen and clarify, not expand scope\n"
            "- All baselines must be named explicitly\n"
            + IDEA_FORMAT
        )
        try:
            result = call_llm(revise_prompt, model, client, temperature)
            if result and result.strip():
                return result
        except Exception:
            pass

        # Fallback: return draft if revision fails
        if draft and draft.strip():
            return draft
        return "Research idea about " + topic + ": " + selected_hyp


GENERATOR = S23_r1Generator()
