import sys
import os
import concurrent.futures

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from systems.base import IdeaGenerator, DEFAULT_MODEL, call_llm, IDEA_FORMAT


class S21_r2Generator(IdeaGenerator):
    VERSION = "S21_r2"
    DESCRIPTION = (
        "Hypothesis-first adversarial loop with grounded operationalization: "
        "hypotheses are anchored to specific datasets, baselines, and measurable "
        "thresholds DURING the adversarial loop (not retrofitted post-hoc), "
        "ensuring experimental concreteness survives into the final idea."
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

        # ── Step 1: generate 5 grounded, falsifiable hypotheses ───────────────
        # Key improvement: hypotheses are generated WITH specific experimental
        # anchors (dataset, baseline, metric, threshold) embedded at generation
        # time — not added later. This prevents the adversarial loop from
        # stripping experimental concreteness in favor of theoretical elegance.
        hyp_prompt = (
            f"Research topic: {topic}{context_block}\n"
            "Your task: generate exactly 5 falsifiable scientific hypotheses, each "
            "grounded in a specific experimental context.\n\n"
            "PROCESS (follow this for each hypothesis):\n"
            "Step A — Identify a specific failure mode of current methods:\n"
            "  Ask: 'Where exactly do existing approaches break down, and what is the structural reason?'\n"
            "  Do NOT start from a technique name. Start from an observation about why something fails.\n\n"
            "Step B — Derive a causal mechanism:\n"
            "  Ask: 'What is the underlying cause of this failure?' Name the mechanism explicitly.\n\n"
            "Step C — Ground it experimentally RIGHT NOW:\n"
            "  Name: (1) a specific real dataset this can be tested on, "
            "(2) a specific named baseline method, (3) the primary metric, "
            "(4) a concrete threshold that would confirm vs. refute.\n\n"
            "Each hypothesis MUST follow this template exactly:\n"
            "[Specific phenomenon X] causes [predicted measurable effect Y] because [theoretical mechanism Z]; "
            "testable on [specific dataset D] against [specific baseline B] using [metric M], "
            "confirmed if [threshold T], falsified if [opposite condition].\n\n"
            "Requirements:\n"
            "- X must be a specific, named property of existing systems\n"
            "- Y must be a quantifiable outcome with a direction (e.g., '>5% drop in F1')\n"
            "- Z must be a theoretical reason, not a restatement of X or Y\n"
            "- D must be a real, named dataset appropriate for this topic\n"
            "- B must be a specific named method (e.g., 'BERT-base', 'XGBoost', 'DPO')\n"
            "- M must be a standard metric for this domain\n"
            "- T must be a specific numeric threshold or directional comparison\n\n"
            "Format:\n"
            "H1: <hypothesis following the template>\n"
            "H2: <hypothesis following the template>\n"
            "H3: <hypothesis following the template>\n"
            "H4: <hypothesis following the template>\n"
            "H5: <hypothesis following the template>"
        )
        try:
            hyp_raw = call_llm(hyp_prompt, model, client, temperature)
        except Exception:
            hyp_raw = (
                f"H1: Static optimization objectives in {topic} cause >10% degraded performance "
                f"under distribution shift because they minimize expected loss over the training "
                f"distribution rather than worst-case subgroup loss; testable on WILDS benchmark "
                f"against ERM baseline using worst-group accuracy, confirmed if DRO training "
                f"improves worst-group accuracy by >5%, falsified if no improvement is observed."
            )

        # Parse hypotheses
        hypotheses = []
        for line in (hyp_raw or "").strip().split("\n"):
            line = line.strip()
            if line and len(line) > 3 and line[0] == "H" and ":" in line[:4]:
                hyp_text = line.split(":", 1)[1].strip()
                if hyp_text:
                    hypotheses.append(hyp_text)
        if not hypotheses:
            hypotheses = [hyp_raw.strip()] if hyp_raw and hyp_raw.strip() else [
                f"Existing approaches to {topic} fail under distribution shift because "
                f"they optimize average-case loss; testable on WILDS against ERM using "
                f"worst-group accuracy, confirmed if >5% improvement, falsified if no gain."
            ]
        hypotheses = hypotheses[:5]

        # ── Step 2: parallel adversarial attacks that PRESERVE experimental anchors ──
        # Key fix: the attack prompt explicitly instructs the critic to KEEP or IMPROVE
        # the experimental anchors (D, B, M, T), never remove them. This prevents the
        # adversarial loop from trading concreteness for theoretical elegance.
        def attack_hypothesis(hyp: str) -> str:
            attack_prompt = (
                f"Research topic: {topic}\n\n"
                f"Hypothesis: {hyp}\n\n"
                "You are an adversarial critic. Attack this hypothesis on EACH dimension:\n"
                "1. Mechanism validity: Is the theoretical mechanism Z actually the cause, "
                "or is there a confound? Give a concrete alternative explanation.\n"
                "2. Dataset appropriateness: Is dataset D the right testbed for isolating "
                "mechanism Z, or does it conflate multiple factors? Name a better dataset if so.\n"
                "3. Baseline adequacy: Can baseline B actually exploit mechanism Z with "
                "minor adaptation? If yes, name a baseline that genuinely cannot.\n"
                "4. Threshold calibration: Is threshold T achievable and meaningful? "
                "Is it too easy (trivially met) or too hard (unreachable)? Propose a better value.\n"
                "5. Scope: Under what conditions does mechanism Z NOT apply? "
                "Name a specific regime where the hypothesis is clearly false.\n\n"
                "Then write a REVISED hypothesis that:\n"
                "- Preserves the causal structure (X causes Y because Z)\n"
                "- KEEPS all experimental anchors (D, B, M, T) — improve them, do NOT remove them\n"
                "- Narrows the scope to where the mechanism genuinely holds\n"
                "- Makes Z more theoretically precise\n"
                "- Updates D, B, or T if your critique identified better choices\n\n"
                "CRITICAL: The revised hypothesis MUST still name a specific dataset, "
                "a specific baseline, a specific metric, and a specific threshold. "
                "Do not remove any of these elements.\n\n"
                "Revised: <revised hypothesis following the full template with D, B, M, T>"
            )
            try:
                return call_llm(attack_prompt, model, client, temperature=0.6)
            except Exception:
                return f"Revised: {hyp}"

        attacks = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            futures = [executor.submit(attack_hypothesis, h) for h in hypotheses]
            for f in futures:
                try:
                    attacks.append(f.result(timeout=120))
                except Exception:
                    attacks.append("Revised: (timeout)")

        while len(attacks) < len(hypotheses):
            attacks.append("Revised: (no attack)")

        # ── Step 3: select the strongest surviving hypothesis ──────────────────
        # Selection criteria now explicitly reward experimental grounding
        pairs_str = ""
        for i, (hyp, attack) in enumerate(zip(hypotheses, attacks)):
            pairs_str += f"\n--- Hypothesis {i+1} ---\nOriginal: {hyp}\nCritique+Revision:\n{attack}\n"

        select_prompt = (
            f"Research topic: {topic}\n\n"
            f"Below are {len(hypotheses)} hypotheses, each attacked and revised:\n{pairs_str}\n"
            "Select the ONE hypothesis whose REVISED form best satisfies ALL of:\n"
            "1. MECHANISM SPECIFICITY: Z (the theoretical reason) is precise enough to derive "
            "a quantitative prediction — names a specific mathematical property or structural constraint\n"
            "2. EXPERIMENTAL CONCRETENESS: The revised hypothesis names a specific real dataset D, "
            "a specific named baseline B, a standard metric M, and a numeric threshold T. "
            "Reject any hypothesis that lost its experimental anchors during revision.\n"
            "3. FALSIFIABILITY: The threshold T and baseline B together define a clean test — "
            "not just 'ablation study' but a specific experimental contrast\n"
            "4. NOVELTY: The mechanism Z is not directly established in standard literature\n"
            "5. FEASIBILITY: Dataset D is real and accessible; baseline B is reproducible\n\n"
            "Respond with:\n"
            "SELECTED: <number 1-5>\n"
            "REVISED HYPOTHESIS: <copy the revised hypothesis text exactly from after 'Revised:'>\n"
            "DATASET: <extract the specific dataset D named in the revised hypothesis>\n"
            "BASELINE: <extract the specific baseline B named in the revised hypothesis>\n"
            "METRIC: <extract the metric M and threshold T>\n"
            "MECHANISM: <extract just the Z component — the theoretical reason>\n"
            "REASONING: <1-2 sentences on why this hypothesis is most experimentally grounded>"
        )
        try:
            selection_raw = call_llm(select_prompt, model, client, temperature=0.3)
        except Exception:
            selection_raw = (
                f"SELECTED: 1\nREVISED HYPOTHESIS: {hypotheses[0] if hypotheses else topic}\n"
                f"DATASET: standard benchmark\nBASELINE: existing SOTA\n"
                f"METRIC: primary metric\nMECHANISM: unknown\nREASONING: fallback"
            )

        # Extract selected hypothesis and experimental anchors
        selected_hyp = hypotheses[0] if hypotheses else f"Novel approach to {topic}"
        selected_mechanism = ""
        selected_dataset = ""
        selected_baseline = ""
        selected_metric = ""

        for line in (selection_raw or "").strip().split("\n"):
            stripped = line.strip()
            upper = stripped.upper()
            if upper.startswith("REVISED HYPOTHESIS:"):
                candidate = stripped.split(":", 1)[1].strip()
                if candidate:
                    selected_hyp = candidate
            elif upper.startswith("MECHANISM:"):
                candidate = stripped.split(":", 1)[1].strip()
                if candidate:
                    selected_mechanism = candidate
            elif upper.startswith("DATASET:"):
                candidate = stripped.split(":", 1)[1].strip()
                if candidate:
                    selected_dataset = candidate
            elif upper.startswith("BASELINE:"):
                candidate = stripped.split(":", 1)[1].strip()
                if candidate:
                    selected_baseline = candidate
            elif upper.startswith("METRIC:"):
                candidate = stripped.split(":", 1)[1].strip()
                if candidate:
                    selected_metric = candidate

        # ── Step 4: construct experimental idea anchored to the grounded hypothesis ──
        # The construction prompt receives the extracted experimental anchors explicitly,
        # forcing the method design to be consistent with the already-chosen D, B, M, T.
        # This is the key structural fix: experimental anchors flow forward from hypothesis
        # to construction, not backward from construction to hypothesis.
        context_reminder = (
            f"\nExisting work to differentiate from:\n{sota_context}\n"
            if sota_context else ""
        )

        mechanism_block = (
            f"\nCore causal mechanism to operationalize: {selected_mechanism}\n"
            if selected_mechanism else ""
        )

        anchors_block = ""
        if selected_dataset or selected_baseline or selected_metric:
            anchors_block = "\nExperimental anchors (already established by hypothesis):\n"
            if selected_dataset:
                anchors_block += f"  Primary dataset: {selected_dataset}\n"
            if selected_baseline:
                anchors_block += f"  Key baseline to beat: {selected_baseline}\n"
            if selected_metric:
                anchors_block += f"  Success metric/threshold: {selected_metric}\n"
            anchors_block += (
                "These anchors are FIXED — your method design must be consistent with them. "
                "You may add additional datasets and baselines but must include these.\n"
            )

        construct_prompt = (
            f"Research topic: {topic}\n"
            f"{context_reminder}\n"
            f"Hypothesis to test: {selected_hyp}\n"
            f"{mechanism_block}\n"
            f"{anchors_block}\n"
            "Design a concrete experiment to PROVE OR DISPROVE this hypothesis.\n\n"
            "CRITICAL: Your experimental design must directly operationalize the causal mechanism "
            "(the 'because Z' part of the hypothesis). The method should work BY exploiting or "
            "demonstrating Z — not just correlate with it.\n\n"
            "Required elements:\n"
            "1. MECHANISM OPERATIONALIZATION: How does your method directly instantiate or test Z? "
            "What specific architectural choice, loss function, or algorithm embodies Z?\n"
            "2. DATASETS: Include the primary dataset above PLUS 2+ additional specific datasets. "
            "Explain why each is appropriate for testing Z specifically.\n"
            "3. BASELINES: Include the key baseline above PLUS 3+ additional specific baselines. "
            "For each, explain why it cannot exploit mechanism Z.\n"
            "4. METRICS: Use the metric/threshold above as primary. Define 2+ secondary metrics. "
            "State what result would confirm vs. refute the hypothesis.\n"
            "5. FALSIFICATION EXPERIMENT: Describe the specific ablation that would prove Z is "
            "NOT the true cause — what result would force you to abandon the hypothesis.\n\n"
            "Write 4 paragraphs. Be direct and specific. No hedging."
        )
        try:
            draft = call_llm(construct_prompt, model, client, temperature)
        except Exception:
            draft = f"Research idea for '{topic}' based on hypothesis: {selected_hyp}"

        # ── Step 5: multi-perspective critique ────────────────────────────────
        exp_prompt = (
            f"You are a hard-nosed experimentalist reviewing a research proposal about '{topic}'.\n\n"
            f"Hypothesis: {selected_hyp}\n"
            f"Causal mechanism being tested: {selected_mechanism}\n"
            f"Primary dataset: {selected_dataset or 'not specified'}\n"
            f"Key baseline: {selected_baseline or 'not specified'}\n"
            f"Success threshold: {selected_metric or 'not specified'}\n\n"
            f"Proposed experiment:\n{draft}\n\n"
            "Give 2-3 sharp criticisms focused on:\n"
            "1. Are the datasets appropriate for isolating mechanism Z, or do they conflate factors?\n"
            "2. Are the baselines truly unable to exploit Z, or could they be trivially adapted?\n"
            "3. Is the success threshold realistic and meaningful, or is it too easy/hard?\n"
            "4. Is the falsification experiment a clean test of Z vs. confounds?\n"
            "Be specific — name the exact problem, not just 'needs more baselines'."
        )
        try:
            critique_exp = call_llm(exp_prompt, model, client, temperature=0.5)
        except Exception:
            critique_exp = "No experimental critique available."

        theory_prompt = (
            f"You are a rigorous theorist reviewing a research proposal about '{topic}'.\n\n"
            f"Hypothesis: {selected_hyp}\n"
            f"Causal mechanism: {selected_mechanism}\n\n"
            f"Proposed experiment:\n{draft}\n\n"
            "Give 2-3 sharp criticisms:\n"
            "1. Is mechanism Z theoretically sound? Does it contradict known results?\n"
            "2. Is the prediction from Z actually derivable, or is the connection hand-wavy?\n"
            "3. Does the proposed method actually implement Z, or does it just correlate with it?\n"
            "4. What would a formal treatment of Z require?\n"
            "Be specific about the theoretical gap."
        )
        try:
            critique_theory = call_llm(theory_prompt, model, client, temperature=0.5)
        except Exception:
            critique_theory = "No theoretical critique available."

        skeptic_prompt = (
            f"You are a skeptical reviewer about '{topic}'.\n\n"
            f"Hypothesis: {selected_hyp}\n"
            f"Causal mechanism: {selected_mechanism}\n"
            f"Success threshold: {selected_metric or 'not specified'}\n\n"
            f"Proposed experiment:\n{draft}\n\n"
            "Give 2-3 sharp criticisms:\n"
            "1. Even if the threshold is met, does it prove Z caused Y, or just correlation?\n"
            "2. What alternative mechanism could produce the same result without Z?\n"
            "3. Is the threshold T meaningful — does beating it actually matter for the field?\n"
            "4. Is the scientific payoff worth the effort if Z turns out to be a minor factor?\n"
            "Be direct about the weakest link in the causal chain."
        )
        try:
            critique_skeptic = call_llm(skeptic_prompt, model, client, temperature=0.5)
        except Exception:
            critique_skeptic = "No skeptic critique available."

        synthesis_prompt = (
            f"Three reviewers critiqued a research idea about '{topic}'.\n"
            f"Hypothesis: '{selected_hyp}'\n"
            f"Mechanism: '{selected_mechanism}'\n"
            f"Experimental anchors: dataset={selected_dataset or 'TBD'}, "
            f"baseline={selected_baseline or 'TBD'}, metric/threshold={selected_metric or 'TBD'}\n\n"
            f"Experimentalist:\n{critique_exp}\n\n"
            f"Theorist:\n{critique_theory}\n\n"
            f"Skeptic:\n{critique_skeptic}\n\n"
            "Synthesize into the 3 most important actionable improvements. "
            "Prioritize in this order:\n"
            "1. Fixes that strengthen the causal argument (that Z is the true mechanism)\n"
            "2. Fixes that make the experimental test cleaner and more decisive\n"
            "3. Fixes that improve feasibility without sacrificing concreteness\n"
            "Do NOT suggest removing experimental anchors (dataset, baseline, threshold) — "
            "only suggest replacing them with better ones."
        )
        try:
            synthesis = call_llm(synthesis_prompt, model, client, temperature=0.3)
        except Exception:
            synthesis = "Strengthen causal argument, improve baseline selection, clarify falsification threshold."

        # ── Step 6: final revision ─────────────────────────────────────────────
        revise_prompt = (
            f"Research topic: {topic}\n\n"
            f"Core hypothesis: {selected_hyp}\n"
            f"Causal mechanism (Z): {selected_mechanism}\n"
            f"Experimental anchors: dataset={selected_dataset or 'TBD'}, "
            f"baseline={selected_baseline or 'TBD'}, metric/threshold={selected_metric or 'TBD'}\n\n"
            f"Experimental design:\n{draft}\n\n"
            f"Key improvements required:\n{synthesis}\n"
            f"{context_reminder}\n"
            "Write the final, improved research idea. Requirements:\n"
            "1. State the hypothesis clearly with X (phenomenon), Y (measurable effect), "
            "and Z (theoretical mechanism) explicit\n"
            "2. Explain how the proposed METHOD directly operationalizes mechanism Z — "
            "what specific component embodies Z\n"
            "3. Name ALL datasets explicitly (include the primary dataset above)\n"
            "4. Name ALL baselines explicitly (include the key baseline above) and explain "
            "why each cannot exploit Z\n"
            "5. State the primary metric and the specific numeric threshold that confirms/refutes\n"
            "6. Describe the falsification experiment — what specific result would prove Z wrong\n"
            "7. Explain why existing methods cannot achieve this because they lack mechanism Z\n"
            + IDEA_FORMAT
        )
        try:
            result = call_llm(revise_prompt, model, client, temperature)
            if result and result.strip():
                return result
        except Exception:
            pass

        # Final fallback
        if draft and draft.strip():
            return draft
        return f"Research idea about {topic}: {selected_hyp}"


GENERATOR = S21_r2Generator()
