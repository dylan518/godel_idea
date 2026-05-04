import sys
import os
import concurrent.futures

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from systems.base import IdeaGenerator, DEFAULT_MODEL, call_llm, IDEA_FORMAT


class S21_r3Generator(IdeaGenerator):
    VERSION = "S21_r3"
    DESCRIPTION = (
        "Hypothesis-first adversarial loop with mechanistically-grounded falsification: "
        "hypotheses are selected based on the precision of their CAUSAL MECHANISM (Z), "
        "and experimental thresholds are derived from the mechanism's theoretical predictions "
        "rather than chosen arbitrarily. The falsification criteria flow from mechanism to "
        "measurement, not from measurement to mechanism."
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

        # ── Step 1: generate 5 mechanistically-grounded hypotheses ───────────
        # Key improvement over S21_r2: hypotheses are generated with an explicit
        # CAUSAL MECHANISM (Z) that is precise enough to derive a quantitative
        # prediction. The threshold T is derived FROM Z, not chosen arbitrarily.
        # This prevents "fabricated precision" where a number is stated but has
        # no theoretical grounding.
        hyp_prompt = (
            f"Research topic: {topic}{context_block}\n"
            "Your task: generate exactly 5 falsifiable scientific hypotheses. "
            "Each hypothesis must be grounded in a SPECIFIC CAUSAL MECHANISM that "
            "predicts a measurable outcome.\n\n"
            "PROCESS (follow this for each hypothesis):\n\n"
            "Step A — Identify a specific structural failure in existing methods:\n"
            "  Ask: 'What specific architectural or algorithmic property of existing "
            "approaches causes them to fail in a particular way?'\n"
            "  Name the property explicitly (e.g., 'softmax attention normalizes across "
            "all tokens equally', 'batch normalization computes statistics over the full batch').\n\n"
            "Step B — State the causal mechanism precisely:\n"
            "  The mechanism Z must be a specific mathematical or structural reason — "
            "not a restatement of the failure. Z should be precise enough that a theorist "
            "could derive a quantitative prediction from it.\n"
            "  Good Z: 'because gradient magnitudes from rare classes are dominated by "
            "frequent-class gradients in proportion to their frequency ratio'\n"
            "  Bad Z: 'because the model doesn't handle imbalanced data well'\n\n"
            "Step C — Derive the threshold from the mechanism:\n"
            "  Given mechanism Z, what does it PREDICT quantitatively? "
            "The threshold T must follow from Z, not be chosen arbitrarily.\n"
            "  Ask: 'If Z is the true cause, what magnitude of effect should we see, "
            "and WHY that magnitude (not a different one)?'\n\n"
            "Step D — Ground it experimentally:\n"
            "  Name: (1) a specific real dataset where the failure mode is observable, "
            "(2) a specific named baseline that exhibits the failure, "
            "(3) the primary metric that directly probes Z, "
            "(4) the threshold T derived from the mechanism.\n\n"
            "Each hypothesis MUST follow this template exactly:\n"
            "[Specific structural property X] causes [predicted measurable effect Y] "
            "because [precise causal mechanism Z — specific enough to derive T]; "
            "this predicts [threshold T and WHY this magnitude follows from Z]; "
            "testable on [specific dataset D] against [specific baseline B] using [metric M], "
            "confirmed if [threshold condition], falsified if [opposite condition].\n\n"
            "Requirements:\n"
            "- X must be a named structural/algorithmic property, not a vague description\n"
            "- Z must be precise enough that a theorist could derive T from Z alone\n"
            "- T must be JUSTIFIED by Z, not arbitrarily chosen\n"
            "- D must be a real, named dataset where the failure is clearly observable\n"
            "- B must be a specific named method that exhibits property X\n"
            "- M must directly probe whether Z is operative (not a proxy metric)\n\n"
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
                f"H1: Uniform loss weighting across classes in {topic} causes >15% degraded "
                f"minority-class recall because gradient magnitudes from frequent classes "
                f"dominate in proportion to their frequency ratio, predicting that the "
                f"performance gap scales with the imbalance ratio (hence >15% for 10:1 ratio); "
                f"testable on imbalanced CIFAR-10 against standard cross-entropy using "
                f"minority-class recall, confirmed if focal loss closes gap by >15%, "
                f"falsified if gap remains <5%."
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
                f"they optimize average-case loss, predicting >10% worst-group accuracy "
                f"gap proportional to subgroup frequency imbalance; "
                f"testable on WILDS against ERM using worst-group accuracy, "
                f"confirmed if >10% improvement, falsified if no gain."
            ]
        hypotheses = hypotheses[:5]

        # ── Step 2: parallel adversarial attacks focused on mechanism validity ──
        # Key fix: the attack explicitly probes whether Z is mechanistically sound
        # and whether T actually follows from Z. This prevents selecting hypotheses
        # where the threshold is arbitrary or disconnected from the mechanism.
        def attack_hypothesis(hyp: str) -> str:
            attack_prompt = (
                f"Research topic: {topic}\n\n"
                f"Hypothesis: {hyp}\n\n"
                "You are an adversarial critic. Attack this hypothesis on EACH dimension:\n\n"
                "1. MECHANISM VALIDITY: Is Z (the causal mechanism) actually the cause, "
                "or is there a confound? Does Z make a specific, derivable prediction, "
                "or is it hand-wavy? Give a concrete alternative mechanism that would "
                "produce the same observed effect Y without Z being true.\n\n"
                "2. THRESHOLD DERIVATION: Does threshold T actually follow from mechanism Z? "
                "If I accept Z as true, does T follow mathematically or empirically? "
                "Or is T an arbitrary number that would be the same regardless of Z? "
                "If T is not derived from Z, propose a threshold that IS derived from Z.\n\n"
                "3. MEASUREMENT VALIDITY: Does metric M actually probe whether Z is operative? "
                "Could M improve without Z being the cause (i.e., does M confound Z with "
                "other factors)? Name a more direct probe of Z if so.\n\n"
                "4. DATASET APPROPRIATENESS: Does dataset D actually exhibit property X "
                "(the structural failure)? Or does D conflate multiple factors that "
                "prevent isolating Z? Name a better dataset if so.\n\n"
                "5. BASELINE ADEQUACY: Does baseline B actually exhibit property X? "
                "Could B be trivially modified to avoid X, invalidating the comparison? "
                "Name a baseline that more purely exhibits X.\n\n"
                "Then write a REVISED hypothesis that:\n"
                "- Sharpens Z so it is mechanistically precise (derivable prediction)\n"
                "- Ensures T is DERIVED from Z, not arbitrary\n"
                "- Chooses M to directly probe Z, not a proxy\n"
                "- KEEPS all experimental anchors (D, B, M, T) — improve them, do NOT remove\n"
                "- Narrows scope to where Z genuinely holds\n\n"
                "CRITICAL: The revised hypothesis MUST explain WHY threshold T follows "
                "from mechanism Z. Do not just state a number — justify it from Z.\n\n"
                "Revised: <revised hypothesis with explicit derivation of T from Z>"
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

        # ── Step 3: select hypothesis with strongest MECHANISTIC grounding ────
        # Selection now explicitly rewards: (a) Z being precise enough to derive T,
        # (b) T being theoretically motivated, (c) M directly probing Z.
        # This prevents selecting hypotheses with arbitrary thresholds.
        pairs_str = ""
        for i, (hyp, attack) in enumerate(zip(hypotheses, attacks)):
            pairs_str += f"\n--- Hypothesis {i+1} ---\nOriginal: {hyp}\nCritique+Revision:\n{attack}\n"

        select_prompt = (
            f"Research topic: {topic}\n\n"
            f"Below are {len(hypotheses)} hypotheses, each attacked and revised:\n{pairs_str}\n"
            "Select the ONE hypothesis whose REVISED form best satisfies ALL of:\n\n"
            "1. MECHANISM PRECISION: Z (the theoretical reason) is specific enough that "
            "a theorist could derive a quantitative prediction from Z alone — without "
            "additional assumptions. Z names a specific mathematical property, structural "
            "constraint, or information-theoretic principle.\n\n"
            "2. THRESHOLD DERIVATION: Threshold T is JUSTIFIED by Z — the revised hypothesis "
            "explains WHY T has that value (not a different one) based on the mechanism. "
            "REJECT any hypothesis where T appears arbitrary or could be any number.\n\n"
            "3. MEASUREMENT DIRECTNESS: Metric M directly probes whether Z is operative, "
            "not a proxy that could improve for other reasons. The experiment isolates Z.\n\n"
            "4. EXPERIMENTAL COMPLETENESS: Names specific dataset D, baseline B, metric M, "
            "threshold T — all four present and appropriate.\n\n"
            "5. NOVELTY: Mechanism Z is not directly established in standard literature.\n\n"
            "6. FEASIBILITY: Dataset D is real and accessible; baseline B is reproducible.\n\n"
            "Respond with:\n"
            "SELECTED: <number 1-5>\n"
            "REVISED HYPOTHESIS: <copy the revised hypothesis text exactly from after 'Revised:'>\n"
            "MECHANISM: <extract Z — the precise causal mechanism>\n"
            "THRESHOLD JUSTIFICATION: <why does T follow from Z specifically?>\n"
            "DATASET: <extract the specific dataset D>\n"
            "BASELINE: <extract the specific baseline B>\n"
            "METRIC: <extract the metric M and threshold T>\n"
            "REASONING: <1-2 sentences on why this hypothesis has the most mechanistically-grounded threshold>"
        )
        try:
            selection_raw = call_llm(select_prompt, model, client, temperature=0.3)
        except Exception:
            selection_raw = (
                f"SELECTED: 1\nREVISED HYPOTHESIS: {hypotheses[0] if hypotheses else topic}\n"
                f"MECHANISM: unknown\nTHRESHOLD JUSTIFICATION: not specified\n"
                f"DATASET: standard benchmark\nBASELINE: existing SOTA\n"
                f"METRIC: primary metric\nREASONING: fallback"
            )

        # Extract selected hypothesis and experimental anchors
        selected_hyp = hypotheses[0] if hypotheses else f"Novel approach to {topic}"
        selected_mechanism = ""
        selected_threshold_justification = ""
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
            elif upper.startswith("THRESHOLD JUSTIFICATION:"):
                candidate = stripped.split(":", 1)[1].strip()
                if candidate:
                    selected_threshold_justification = candidate
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

        # ── Step 4: construct experimental design FROM the mechanism ──────────
        # Key structural fix: the construct prompt explicitly requires the method
        # to OPERATIONALIZE Z (the mechanism), not just correlate with it.
        # The experimental procedure must directly manipulate or measure Z.
        # The falsification experiment must test Z vs. the best alternative mechanism.
        context_reminder = (
            f"\nExisting work to differentiate from:\n{sota_context}\n"
            if sota_context else ""
        )

        mechanism_block = (
            f"\nCore causal mechanism (Z) to operationalize: {selected_mechanism}\n"
            if selected_mechanism else ""
        )

        threshold_block = (
            f"\nWhy the threshold follows from Z: {selected_threshold_justification}\n"
            if selected_threshold_justification else ""
        )

        anchors_block = ""
        if selected_dataset or selected_baseline or selected_metric:
            anchors_block = "\nExperimental anchors (fixed by hypothesis):\n"
            if selected_dataset:
                anchors_block += f"  Primary dataset: {selected_dataset}\n"
            if selected_baseline:
                anchors_block += f"  Key baseline to beat: {selected_baseline}\n"
            if selected_metric:
                anchors_block += f"  Success metric/threshold: {selected_metric}\n"
            anchors_block += (
                "These anchors are FIXED. Your method must be consistent with them. "
                "You may add datasets and baselines but must include these.\n"
            )

        construct_prompt = (
            f"Research topic: {topic}\n"
            f"{context_reminder}\n"
            f"Hypothesis to test: {selected_hyp}\n"
            f"{mechanism_block}\n"
            f"{threshold_block}\n"
            f"{anchors_block}\n"
            "Design a concrete experiment to PROVE OR DISPROVE this hypothesis.\n\n"
            "CRITICAL REQUIREMENTS:\n\n"
            "1. MECHANISM OPERATIONALIZATION: Your method must DIRECTLY IMPLEMENT or "
            "DIRECTLY TEST mechanism Z. Do not describe a method that merely correlates "
            "with Z. Specify:\n"
            "   (a) What specific component (loss function, architecture choice, algorithm "
            "step) directly embodies Z?\n"
            "   (b) How does this component differ from baseline B specifically because "
            "of Z (not for other reasons)?\n"
            "   (c) What intermediate measurement would confirm Z is active during training/inference?\n\n"
            "2. EXPERIMENTAL PROCEDURE: Describe the exact experimental contrast:\n"
            "   (a) What is manipulated (the independent variable that isolates Z)?\n"
            "   (b) What is measured (the dependent variable that probes Z)?\n"
            "   (c) What is held constant (to prevent confounds with Z)?\n\n"
            "3. DATASETS: Include the primary dataset above PLUS 2+ additional datasets. "
            "For each, explain whether Z should be MORE or LESS operative, and what "
            "result this predicts.\n\n"
            "4. BASELINES: Include the key baseline above PLUS 3+ additional baselines. "
            "For each, explain specifically WHY it cannot exploit Z (not just 'it doesn't').\n\n"
            "5. FALSIFICATION EXPERIMENT: Describe the specific ablation that tests Z "
            "against the BEST ALTERNATIVE MECHANISM. What result would force you to "
            "conclude Z is NOT the cause, even if the primary metric improves?\n\n"
            "Write 4 focused paragraphs. Be precise about what is manipulated and measured."
        )
        try:
            draft = call_llm(construct_prompt, model, client, temperature)
        except Exception:
            draft = f"Research idea for '{topic}' based on hypothesis: {selected_hyp}"

        # ── Step 5: multi-perspective critique ────────────────────────────────
        exp_prompt = (
            f"You are a hard-nosed experimentalist reviewing a research proposal about '{topic}'.\n\n"
            f"Hypothesis: {selected_hyp}\n"
            f"Causal mechanism being tested (Z): {selected_mechanism}\n"
            f"Why threshold T follows from Z: {selected_threshold_justification or 'not specified'}\n"
            f"Primary dataset: {selected_dataset or 'not specified'}\n"
            f"Key baseline: {selected_baseline or 'not specified'}\n"
            f"Success threshold: {selected_metric or 'not specified'}\n\n"
            f"Proposed experiment:\n{draft}\n\n"
            "Give 2-3 sharp criticisms focused on:\n"
            "1. Does the experimental procedure actually isolate Z, or does it confound Z "
            "with other factors? Name the specific confound.\n"
            "2. Is the threshold T actually derived from Z, or is it arbitrary? "
            "If arbitrary, what threshold would Z actually predict?\n"
            "3. Does the falsification experiment test Z against the best alternative "
            "mechanism, or against a strawman?\n"
            "4. Could the primary metric improve without Z being operative?\n"
            "Be specific — name the exact problem."
        )
        try:
            critique_exp = call_llm(exp_prompt, model, client, temperature=0.5)
        except Exception:
            critique_exp = "No experimental critique available."

        theory_prompt = (
            f"You are a rigorous theorist reviewing a research proposal about '{topic}'.\n\n"
            f"Hypothesis: {selected_hyp}\n"
            f"Causal mechanism (Z): {selected_mechanism}\n"
            f"Threshold justification: {selected_threshold_justification or 'not specified'}\n\n"
            f"Proposed experiment:\n{draft}\n\n"
            "Give 2-3 sharp criticisms:\n"
            "1. Is mechanism Z theoretically sound? Is the quantitative prediction "
            "actually derivable from Z, or does it require additional assumptions?\n"
            "2. Does the proposed method actually IMPLEMENT Z, or does it just "
            "correlate with Z? What would a method that purely implements Z look like?\n"
            "3. What is the best theoretical alternative to Z that would produce "
            "the same experimental outcome? How would you distinguish them?\n"
            "4. Is the threshold T the right order of magnitude given Z's mechanism?\n"
            "Be specific about the theoretical gap."
        )
        try:
            critique_theory = call_llm(theory_prompt, model, client, temperature=0.5)
        except Exception:
            critique_theory = "No theoretical critique available."

        skeptic_prompt = (
            f"You are a skeptical reviewer about '{topic}'.\n\n"
            f"Hypothesis: {selected_hyp}\n"
            f"Causal mechanism (Z): {selected_mechanism}\n"
            f"Success threshold: {selected_metric or 'not specified'}\n\n"
            f"Proposed experiment:\n{draft}\n\n"
            "Give 2-3 sharp criticisms:\n"
            "1. Even if the threshold is met, does it PROVE Z caused Y? "
            "What alternative mechanism could produce the same result without Z?\n"
            "2. Is the threshold T meaningful — does beating it prove Z is operative, "
            "or could it be beaten by a method that has nothing to do with Z?\n"
            "3. Is the falsification experiment genuinely risky? Would the authors "
            "actually abandon the hypothesis if the falsification condition is met?\n"
            "Be direct about the weakest link in the causal chain."
        )
        try:
            critique_skeptic = call_llm(skeptic_prompt, model, client, temperature=0.5)
        except Exception:
            critique_skeptic = "No skeptic critique available."

        synthesis_prompt = (
            f"Three reviewers critiqued a research idea about '{topic}'.\n"
            f"Hypothesis: '{selected_hyp}'\n"
            f"Mechanism (Z): '{selected_mechanism}'\n"
            f"Threshold justification: '{selected_threshold_justification or 'not specified'}'\n"
            f"Experimental anchors: dataset={selected_dataset or 'TBD'}, "
            f"baseline={selected_baseline or 'TBD'}, metric/threshold={selected_metric or 'TBD'}\n\n"
            f"Experimentalist:\n{critique_exp}\n\n"
            f"Theorist:\n{critique_theory}\n\n"
            f"Skeptic:\n{critique_skeptic}\n\n"
            "Synthesize into the 3 most important actionable improvements. "
            "Prioritize in this order:\n"
            "1. Fixes that make Z more mechanistically precise (so T can be derived from Z)\n"
            "2. Fixes that make the experimental procedure isolate Z more cleanly\n"
            "3. Fixes that strengthen the falsification experiment against the best alternative\n"
            "Do NOT suggest removing experimental anchors — only suggest improving them. "
            "Do NOT suggest adding vague 'additional experiments' — be specific."
        )
        try:
            synthesis = call_llm(synthesis_prompt, model, client, temperature=0.3)
        except Exception:
            synthesis = "Strengthen mechanism Z precision, improve threshold derivation, sharpen falsification experiment."

        # ── Step 6: final revision ─────────────────────────────────────────────
        revise_prompt = (
            f"Research topic: {topic}\n\n"
            f"Core hypothesis: {selected_hyp}\n"
            f"Causal mechanism (Z): {selected_mechanism}\n"
            f"Why threshold T follows from Z: {selected_threshold_justification or 'derive from Z'}\n"
            f"Experimental anchors: dataset={selected_dataset or 'TBD'}, "
            f"baseline={selected_baseline or 'TBD'}, metric/threshold={selected_metric or 'TBD'}\n\n"
            f"Experimental design:\n{draft}\n\n"
            f"Key improvements required:\n{synthesis}\n"
            f"{context_reminder}\n"
            "Write the final, improved research idea. Requirements:\n\n"
            "1. STATE THE HYPOTHESIS clearly with:\n"
            "   - X: the specific structural property that causes the failure\n"
            "   - Y: the measurable effect (with direction and magnitude)\n"
            "   - Z: the precise causal mechanism (specific enough to derive T)\n"
            "   - Why T follows from Z (not arbitrary)\n\n"
            "2. EXPLAIN HOW THE METHOD IMPLEMENTS Z:\n"
            "   - What specific component (loss, architecture, algorithm) directly embodies Z?\n"
            "   - What intermediate measurement confirms Z is active?\n"
            "   - Why existing methods fail because they lack Z (not just 'they perform worse')\n\n"
            "3. EXPERIMENTAL PROCEDURE:\n"
            "   - What is manipulated (independent variable isolating Z)?\n"
            "   - What is measured (dependent variable probing Z)?\n"
            "   - What is held constant (to prevent confounds)?\n\n"
            "4. ALL DATASETS explicitly named (include primary dataset above)\n\n"
            "5. ALL BASELINES explicitly named (include key baseline above) with explanation "
            "of WHY each cannot exploit Z\n\n"
            "6. PRIMARY METRIC and the specific threshold that confirms/refutes, "
            "with explanation of why this threshold follows from Z\n\n"
            "7. FALSIFICATION EXPERIMENT: What specific result would prove Z is NOT the "
            "true cause — specifically, what would the best alternative mechanism predict "
            "differently, and how would you measure it?\n"
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


GENERATOR = S21_r3Generator()
