import sys
import os
import concurrent.futures

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from systems.base import IdeaGenerator, DEFAULT_MODEL, call_llm, IDEA_FORMAT


class S21_r5Generator(IdeaGenerator):
    VERSION = "S21_r5"
    DESCRIPTION = (
        "Hypothesis-first adversarial loop with mechanism-grounded falsification. "
        "Key improvement over S21_r4: the adversarial attack loop now requires attackers "
        "to propose a CONCRETE ALTERNATIVE MECHANISM and a discriminating experiment, "
        "AND the selection step explicitly penalizes hypotheses where the mechanism Z "
        "is not quantitatively predictive of the threshold T. The construction step "
        "is restructured to derive the method FROM the mechanism, not retrofit the mechanism "
        "onto the method. This closes the chain-of-custody gap between selection and construction."
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

        # ── Step 1: generate 5 falsifiable hypotheses ────────────────────────
        # Each hypothesis must have a mechanism Z that is quantitatively predictive:
        # Z must imply a specific numeric threshold T through a derivable argument.
        # This is the key upstream constraint that prevents vague mechanisms from
        # surviving into the construction step.
        hyp_prompt = (
            f"Research topic: {topic}{context_block}\n"
            "Generate exactly 5 falsifiable scientific hypotheses. "
            "Each hypothesis must be MECHANISTICALLY QUANTITATIVE — the causal mechanism Z "
            "must be specific enough to DERIVE the threshold T through a mathematical or "
            "structural argument. If you cannot derive T from Z, Z is too vague.\n\n"
            "REQUIREMENTS for each hypothesis:\n\n"
            "1. PROPERTY X: Name a specific structural/algorithmic property of existing methods "
            "that causes a measurable failure. Be precise: 'uniform loss weighting' not 'poor training'.\n\n"
            "2. MECHANISM Z: State the causal mechanism as a quantitative claim. Z must:\n"
            "   - Name the structural reason X causes the failure\n"
            "   - Predict a direction AND magnitude (e.g., 'gradient magnitudes scale with class "
            "frequency ratio r, so minority recall degrades by ~(r-1)/r × baseline recall')\n"
            "   - Be falsifiable independently of the downstream experiment\n\n"
            "3. THRESHOLD T: Derive T from Z. Write: 'T = [value] because Z predicts [derivation]'. "
            "T must follow from Z — not be a round number you chose independently.\n\n"
            "4. EXPERIMENTAL SKELETON:\n"
            "   - Dataset D: a real, named benchmark where property X is observable\n"
            "   - Baseline B: a specific named method that exhibits property X\n"
            "   - Metric M: the measurement that directly probes whether Z is operative\n"
            "   - Control C: a minimal ablation that isolates Z from the most obvious confound\n\n"
            "5. FALSIFICATION CONDITION: 'Falsified if [specific measurement] shows [specific "
            "result] on [specific dataset/condition]' — not 'no improvement'.\n\n"
            "Template (follow exactly):\n"
            "H[n]: Property X=[property]; Effect Y=[measurable effect]; "
            "Mechanism Z=[quantitative causal claim]; "
            "Threshold T=[value] because [derivation from Z]; "
            "Dataset D=[name], Baseline B=[name], Metric M=[name], Control C=[ablation]; "
            "Falsified if: [specific falsification condition].\n\n"
            "Format:\n"
            "H1: <hypothesis>\nH2: <hypothesis>\nH3: <hypothesis>\nH4: <hypothesis>\nH5: <hypothesis>"
        )
        try:
            hyp_raw = call_llm(hyp_prompt, model, client, temperature)
        except Exception:
            hyp_raw = (
                f"H1: Property X=uniform loss weighting in {topic}; "
                f"Effect Y=>15% minority-class recall degradation; "
                f"Mechanism Z=gradient magnitudes scale with class frequency ratio r, "
                f"so minority gradients are suppressed by factor r at 10:1 imbalance; "
                f"Threshold T=15% recall gap because r=10 predicts (r-1)/r=90% gradient suppression; "
                f"Dataset D=imbalanced CIFAR-10, Baseline B=standard cross-entropy, "
                f"Metric M=minority-class recall, Control C=reweight by 1/frequency; "
                f"Falsified if: reweighted baseline closes recall gap to <3% on same dataset."
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
                f"Property X=average-case loss optimization in {topic}; "
                f"Effect Y=>10% worst-group accuracy gap; "
                f"Mechanism Z=subgroup frequency imbalance causes gradient weighting "
                f"proportional to frequency, suppressing minority subgroup updates; "
                f"Threshold T=10% because 5:1 frequency ratio predicts 80% gradient suppression; "
                f"Dataset D=WILDS, Baseline B=ERM, Metric M=worst-group accuracy, "
                f"Control C=subsample to uniform frequency; "
                f"Falsified if: uniform subsampling closes worst-group gap to <2%."
            ]
        hypotheses = hypotheses[:5]

        # ── Step 2: parallel adversarial attacks ─────────────────────────────
        # KEY CHANGE from S21_r4: attacks now must (a) challenge whether Z is
        # quantitatively predictive of T, (b) propose a SPECIFIC ALTERNATIVE
        # MECHANISM Z' that would produce the same Y, and (c) design a
        # discriminating experiment that distinguishes Z from Z'.
        # The revision must KEEP Z's quantitative derivation of T intact or
        # REPLACE it with a better derivation — never make T vaguer.
        def attack_hypothesis(hyp: str) -> str:
            attack_prompt = (
                f"Research topic: {topic}\n\n"
                f"Hypothesis: {hyp}\n\n"
                "You are an adversarial critic. Your goal: force the hypothesis to be MORE "
                "MECHANISTICALLY PRECISE and EXPERIMENTALLY DISCRIMINATING.\n\n"
                "Attack on EACH of these dimensions:\n\n"
                "1. MECHANISM QUANTITATIVENESS: Does mechanism Z actually DERIVE threshold T, "
                "or is T a round number chosen independently? \n"
                "   → If T is not derived from Z, provide the actual derivation from Z, or "
                "show that Z does not predict T and the hypothesis is internally inconsistent.\n"
                "   → If Z is qualitative ('gradients are dominated'), make it quantitative "
                "('gradient magnitudes scale as frequency ratio r, so suppression = (r-1)/r').\n\n"
                "2. ALTERNATIVE MECHANISM: What is the BEST alternative mechanism Z' that "
                "would produce the same observed effect Y without Z being true?\n"
                "   → Z' must be specific (not 'other factors'). Name the structural reason.\n"
                "   → Describe ONE experiment that DISTINGUISHES Z from Z': what does Z predict "
                "vs. what Z' predicts in that experiment?\n\n"
                "3. CONTROL CONDITION: Is control C sufficient to isolate Z from Z'?\n"
                "   → If not, propose a better control that directly manipulates Z while "
                "holding Z' constant. Name the exact manipulation.\n\n"
                "4. FALSIFICATION SHARPNESS: Could the authors escape the falsification "
                "condition by reinterpreting results?\n"
                "   → Rewrite as: 'Falsified if [specific measurement] = [specific value] "
                "on [specific dataset/condition] under [specific experimental setup]'.\n\n"
                "Then write a REVISED hypothesis that:\n"
                "- KEEPS all experimental anchors (D, B, M, C) — improve them, never remove\n"
                "- Makes Z quantitative: Z must derive T through a stated argument\n"
                "- States T with its derivation from Z\n"
                "- Adds the discriminating experiment against Z'\n"
                "- Sharpens the falsification condition to be unambiguous\n"
                "- Does NOT add new theoretical components — only sharpen existing ones\n\n"
                "CRITICAL: A good revision has a SIMPLER Z with a CLEARER quantitative "
                "prediction, not a more elaborate theoretical framework.\n\n"
                "Revised: <revised hypothesis — mechanistically quantitative, all anchors present>"
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

        # ── Step 3: select hypothesis with strongest MECHANISM-EXPERIMENT CHAIN ──
        # KEY CHANGE from S21_r4: selection now explicitly requires that Z
        # quantitatively predicts T. A hypothesis where T cannot be derived from Z
        # is REJECTED even if it has all four anchors. This enforces the
        # chain-of-custody: Z → T → experiment → falsification.
        pairs_str = ""
        for i, (hyp, attack) in enumerate(zip(hypotheses, attacks)):
            pairs_str += f"\n--- Hypothesis {i+1} ---\nOriginal: {hyp}\nCritique+Revision:\n{attack}\n"

        select_prompt = (
            f"Research topic: {topic}\n\n"
            f"Below are {len(hypotheses)} hypotheses, each attacked and revised:\n{pairs_str}\n"
            "Select the ONE hypothesis whose REVISED form best satisfies ALL of:\n\n"
            "1. MECHANISM-THRESHOLD CHAIN (most important): Z must DERIVE T through a stated "
            "quantitative argument. If T is just a round number with no derivation from Z, "
            "REJECT the hypothesis. The chain must be: Z (structural reason) → T (numeric "
            "prediction) → M (measurement that probes Z) → falsification condition.\n\n"
            "2. EXPERIMENTAL CONCRETENESS: The revised hypothesis names a specific real "
            "dataset D, a specific named baseline B, a specific metric M, and a control "
            "condition C that isolates Z from the best alternative mechanism Z'. "
            "REJECT any hypothesis missing any of these.\n\n"
            "3. FALSIFIABILITY: The falsification condition names a specific measurement, "
            "a specific value, and a specific dataset/condition. REJECT vague conditions.\n\n"
            "4. DISCRIMINATING EXPERIMENT: There is a stated experiment that distinguishes Z "
            "from the best alternative Z'. Z and Z' must predict different outcomes.\n\n"
            "5. MECHANISM SIMPLICITY: Prefer Z that is structurally simple and directly "
            "testable over Z that is theoretically elaborate. Complexity is a liability.\n\n"
            "6. NOVELTY: Z is not directly established in standard literature.\n\n"
            "7. FEASIBILITY: D is real and accessible; B is reproducible.\n\n"
            "TIEBREAKER: The hypothesis where T is most directly and transparently derived "
            "from Z wins. A clean derivation beats a complex one.\n\n"
            "Respond with:\n"
            "SELECTED: <number 1-5>\n"
            "REVISED HYPOTHESIS: <copy the revised hypothesis text exactly from after 'Revised:'>\n"
            "MECHANISM: <extract Z — the quantitative causal mechanism, 1-2 sentences>\n"
            "THRESHOLD DERIVATION: <extract the derivation of T from Z>\n"
            "DATASET: <extract the specific dataset D>\n"
            "BASELINE: <extract the specific baseline B>\n"
            "METRIC: <extract the metric M>\n"
            "CONTROL: <extract the control condition C>\n"
            "ALTERNATIVE MECHANISM: <extract Z' — the best alternative mechanism>\n"
            "DISCRIMINATING EXPERIMENT: <extract the experiment that distinguishes Z from Z'>\n"
            "FALSIFICATION: <extract the specific falsification condition>\n"
            "REASONING: <1-2 sentences on why this hypothesis has the strongest Z→T→M chain>"
        )
        try:
            selection_raw = call_llm(select_prompt, model, client, temperature=0.3)
        except Exception:
            selection_raw = (
                f"SELECTED: 1\nREVISED HYPOTHESIS: {hypotheses[0] if hypotheses else topic}\n"
                f"MECHANISM: unknown\nTHRESHOLD DERIVATION: not specified\n"
                f"DATASET: standard benchmark\nBASELINE: existing SOTA\n"
                f"METRIC: primary metric\nCONTROL: ablation\n"
                f"ALTERNATIVE MECHANISM: not specified\nDISCRIMINATING EXPERIMENT: not specified\n"
                f"FALSIFICATION: not specified\nREASONING: fallback"
            )

        # Extract selected hypothesis and experimental anchors
        selected_hyp = hypotheses[0] if hypotheses else f"Novel approach to {topic}"
        selected_mechanism = ""
        selected_threshold_derivation = ""
        selected_dataset = ""
        selected_baseline = ""
        selected_metric = ""
        selected_control = ""
        selected_alt_mechanism = ""
        selected_discriminating_exp = ""
        selected_falsification = ""

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
            elif upper.startswith("THRESHOLD DERIVATION:"):
                candidate = stripped.split(":", 1)[1].strip()
                if candidate:
                    selected_threshold_derivation = candidate
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
            elif upper.startswith("CONTROL:"):
                candidate = stripped.split(":", 1)[1].strip()
                if candidate:
                    selected_control = candidate
            elif upper.startswith("ALTERNATIVE MECHANISM:"):
                candidate = stripped.split(":", 1)[1].strip()
                if candidate:
                    selected_alt_mechanism = candidate
            elif upper.startswith("DISCRIMINATING EXPERIMENT:"):
                candidate = stripped.split(":", 1)[1].strip()
                if candidate:
                    selected_discriminating_exp = candidate
            elif upper.startswith("FALSIFICATION:"):
                candidate = stripped.split(":", 1)[1].strip()
                if candidate:
                    selected_falsification = candidate

        # ── Step 4: construct experimental design FROM the mechanism ─────────
        # KEY CHANGE from S21_r4: the construction prompt is now organized around
        # the MECHANISM Z as the central organizing principle. The method is derived
        # FROM Z (not retrofitted onto Z). The discriminating experiment against Z'
        # is given equal prominence to the positive test. The threshold derivation
        # is explicitly anchored to Z throughout.
        context_reminder = (
            f"\nExisting work to differentiate from:\n{sota_context}\n"
            if sota_context else ""
        )

        mechanism_block = ""
        if selected_mechanism:
            mechanism_block = f"\nCausal mechanism Z: {selected_mechanism}\n"
        if selected_threshold_derivation:
            mechanism_block += f"Threshold derivation from Z: {selected_threshold_derivation}\n"
        if selected_alt_mechanism:
            mechanism_block += f"Best alternative mechanism Z': {selected_alt_mechanism}\n"
        if selected_discriminating_exp:
            mechanism_block += f"Discriminating experiment (Z vs Z'): {selected_discriminating_exp}\n"

        anchors_block = ""
        if selected_dataset or selected_baseline or selected_metric or selected_control:
            anchors_block = "\nExperimental anchors (FIXED — do not change these):\n"
            if selected_dataset:
                anchors_block += f"  Primary dataset D: {selected_dataset}\n"
            if selected_baseline:
                anchors_block += f"  Key baseline B: {selected_baseline}\n"
            if selected_metric:
                anchors_block += f"  Primary metric M: {selected_metric}\n"
            if selected_control:
                anchors_block += f"  Control condition C: {selected_control}\n"
            if selected_falsification:
                anchors_block += f"  Falsification condition: {selected_falsification}\n"
            anchors_block += (
                "These anchors are FIXED. You may add datasets and baselines but must include these.\n"
            )

        construct_prompt = (
            f"Research topic: {topic}\n"
            f"{context_reminder}\n"
            f"Hypothesis: {selected_hyp}\n"
            f"{mechanism_block}\n"
            f"{anchors_block}\n"
            "Design a concrete research idea to TEST this hypothesis. "
            "The mechanism Z is the organizing principle — derive everything from Z.\n\n"
            "PARAGRAPH 1 — THE MECHANISM AND ITS PREDICTIONS:\n"
            "State mechanism Z precisely and derive its predictions:\n"
            "  (a) What is the structural/algorithmic reason (Z) that property X causes effect Y?\n"
            "  (b) What does Z predict QUANTITATIVELY? Derive the threshold T from Z step by step.\n"
            "  (c) What does Z predict in the CONTROL condition C? What result confirms Z is active?\n"
            "  (d) What does the alternative mechanism Z' predict in the same control? "
            "Why do Z and Z' predict DIFFERENT outcomes in C?\n\n"
            "PARAGRAPH 2 — THE METHOD (derived from Z):\n"
            "Describe the proposed method as a direct implementation of Z:\n"
            "  (a) What specific component (loss function, architecture, algorithm step) "
            "directly operationalizes Z? Explain how it addresses Z, not just the symptom Y.\n"
            "  (b) How does this component differ from baseline B SPECIFICALLY BECAUSE of Z? "
            "Not 'it performs better' — the structural difference that Z implies.\n"
            "  (c) What intermediate measurement would confirm Z is active in your method "
            "(not just that performance improved)?\n"
            "  (d) Why cannot existing methods exploit Z — the specific structural reason, "
            "not 'they perform worse'.\n\n"
            "PARAGRAPH 3 — DATASETS AND BASELINES:\n"
            "Include primary dataset D and key baseline B PLUS:\n"
            "  - 2 additional datasets: for each, state whether Z should be MORE or LESS "
            "operative (based on the structural argument), and what result this predicts\n"
            "  - 3 additional baselines: for each, state specifically WHY it cannot exploit Z\n\n"
            "PARAGRAPH 4 — THE DISCRIMINATING EXPERIMENT:\n"
            "Describe the experiment that distinguishes Z from Z':\n"
            "  (a) What exactly do you manipulate in the control condition C?\n"
            "  (b) What does Z predict in this manipulation? (specific direction and magnitude)\n"
            "  (c) What does Z' predict? Why are these predictions different?\n"
            "  (d) What specific result would force you to conclude Z is NOT the cause?\n"
            "  (e) Confirm the falsification condition is measurable in this experiment.\n\n"
            "Be precise. Name specific datasets, baselines, metrics, and thresholds. "
            "Derive thresholds from Z — do not choose round numbers independently."
        )
        try:
            draft = call_llm(construct_prompt, model, client, temperature)
        except Exception:
            draft = f"Research idea for '{topic}' based on hypothesis: {selected_hyp}"

        # ── Step 5: multi-perspective critique ────────────────────────────────
        exp_prompt = (
            f"You are a hard-nosed experimentalist reviewing a research proposal about '{topic}'.\n\n"
            f"Hypothesis: {selected_hyp}\n"
            f"Mechanism Z: {selected_mechanism or 'not specified'}\n"
            f"Threshold derivation: {selected_threshold_derivation or 'not specified'}\n"
            f"Dataset: {selected_dataset or 'not specified'}\n"
            f"Baseline: {selected_baseline or 'not specified'}\n"
            f"Metric/Control: {selected_metric or 'not specified'} / {selected_control or 'not specified'}\n"
            f"Alternative mechanism Z': {selected_alt_mechanism or 'not specified'}\n"
            f"Discriminating experiment: {selected_discriminating_exp or 'not specified'}\n"
            f"Falsification condition: {selected_falsification or 'not specified'}\n\n"
            f"Proposed experiment:\n{draft}\n\n"
            "Give 2-3 sharp criticisms focused on:\n"
            "1. Does the control condition C actually distinguish Z from Z'? Could Z' also "
            "predict the same result in C? Name the specific overlap and how to close it.\n"
            "2. Is the threshold T actually derived from Z, or is it a post-hoc rationalization? "
            "Show the specific step in the derivation that is weakest.\n"
            "3. Is the intermediate measurement (confirming Z is active) actually measuring Z, "
            "or measuring the effect Y? These must be different — name the difference.\n"
            "Be specific — name the exact problem, not a general concern."
        )
        try:
            critique_exp = call_llm(exp_prompt, model, client, temperature=0.5)
        except Exception:
            critique_exp = "No experimental critique available."

        theory_prompt = (
            f"You are a rigorous theorist reviewing a research proposal about '{topic}'.\n\n"
            f"Hypothesis: {selected_hyp}\n"
            f"Mechanism Z: {selected_mechanism or 'not specified'}\n"
            f"Threshold derivation: {selected_threshold_derivation or 'not specified'}\n\n"
            f"Proposed experiment:\n{draft}\n\n"
            "Give 2-3 sharp criticisms:\n"
            "1. Is the derivation of T from Z mathematically valid? Identify the specific "
            "step where the derivation requires an unstated assumption.\n"
            "2. Does the proposed method IMPLEMENT Z, or does it correlate with Z? "
            "What would a method that purely implements Z look like — is it simpler or different?\n"
            "3. Is Z the minimal mechanism that explains Y, or does it contain unnecessary "
            "components? Strip Z to its core and state what the minimal version predicts.\n"
            "Be specific about the theoretical gap — do not suggest adding complexity."
        )
        try:
            critique_theory = call_llm(theory_prompt, model, client, temperature=0.5)
        except Exception:
            critique_theory = "No theoretical critique available."

        skeptic_prompt = (
            f"You are a skeptical reviewer about '{topic}'.\n\n"
            f"Hypothesis: {selected_hyp}\n"
            f"Alternative mechanism Z': {selected_alt_mechanism or 'not specified'}\n"
            f"Discriminating experiment: {selected_discriminating_exp or 'not specified'}\n"
            f"Falsification condition: {selected_falsification or 'not specified'}\n\n"
            f"Proposed experiment:\n{draft}\n\n"
            "Give 2-3 sharp criticisms:\n"
            "1. Is the discriminating experiment actually decisive? Could Z' predict the same "
            "result as Z in the stated experiment? Name the specific overlap.\n"
            "2. Even if the threshold T is met, does it prove Z caused Y? Name the specific "
            "confound that would produce the same result without Z being true.\n"
            "3. Is the falsification condition genuinely risky? Could the authors meet the "
            "success threshold while the falsification condition is also met?\n"
            "Be direct about the weakest link in the Z→T→M→falsification chain."
        )
        try:
            critique_skeptic = call_llm(skeptic_prompt, model, client, temperature=0.5)
        except Exception:
            critique_skeptic = "No skeptic critique available."

        synthesis_prompt = (
            f"Three reviewers critiqued a research idea about '{topic}'.\n"
            f"Hypothesis: '{selected_hyp}'\n"
            f"Mechanism Z: '{selected_mechanism or 'not specified'}'\n"
            f"Threshold derivation: '{selected_threshold_derivation or 'not specified'}'\n"
            f"Experimental anchors: dataset={selected_dataset or 'TBD'}, "
            f"baseline={selected_baseline or 'TBD'}, "
            f"metric={selected_metric or 'TBD'}, "
            f"control={selected_control or 'TBD'}\n"
            f"Alternative Z': {selected_alt_mechanism or 'TBD'}\n"
            f"Falsification: {selected_falsification or 'TBD'}\n\n"
            f"Experimentalist:\n{critique_exp}\n\n"
            f"Theorist:\n{critique_theory}\n\n"
            f"Skeptic:\n{critique_skeptic}\n\n"
            "Synthesize into the 3 most important actionable improvements. "
            "Prioritize in this order:\n"
            "1. Fixes that strengthen the Z→T derivation (make T follow more directly from Z)\n"
            "2. Fixes that make the discriminating experiment more decisive (Z vs Z' predictions "
            "more distinct)\n"
            "3. Fixes that make the falsification condition more specific and unambiguous\n"
            "Do NOT suggest removing experimental anchors. "
            "Do NOT suggest adding new theoretical components. "
            "Do NOT suggest vague 'additional experiments' — be specific about what to change."
        )
        try:
            synthesis = call_llm(synthesis_prompt, model, client, temperature=0.3)
        except Exception:
            synthesis = "Strengthen Z→T derivation, sharpen discriminating experiment, tighten falsification condition."

        # ── Step 6: final revision ─────────────────────────────────────────────
        revise_prompt = (
            f"Research topic: {topic}\n\n"
            f"Core hypothesis: {selected_hyp}\n"
            f"Causal mechanism Z: {selected_mechanism or 'derive from hypothesis'}\n"
            f"Threshold derivation from Z: {selected_threshold_derivation or 'derive from Z'}\n"
            f"Best alternative mechanism Z': {selected_alt_mechanism or 'identify from hypothesis'}\n"
            f"Discriminating experiment: {selected_discriminating_exp or 'design from Z vs Z'}\n"
            f"Experimental anchors (FIXED):\n"
            f"  Dataset D: {selected_dataset or 'TBD'}\n"
            f"  Baseline B: {selected_baseline or 'TBD'}\n"
            f"  Metric M: {selected_metric or 'TBD'}\n"
            f"  Control C: {selected_control or 'TBD'}\n"
            f"  Falsification condition: {selected_falsification or 'TBD'}\n\n"
            f"Experimental design:\n{draft}\n\n"
            f"Key improvements required:\n{synthesis}\n"
            f"{context_reminder}\n"
            "Write the final, improved research idea. Requirements:\n\n"
            "1. HYPOTHESIS: State clearly with:\n"
            "   - X: the specific structural property causing the failure\n"
            "   - Y: the measurable effect (direction and magnitude)\n"
            "   - Z: the quantitative causal mechanism (1-2 sentences, must derive T)\n"
            "   - T: the threshold with its derivation from Z (show the argument)\n"
            "   - Falsification condition: specific measurement, specific value, specific dataset\n\n"
            "2. METHOD (derived from Z):\n"
            "   - What specific component directly operationalizes Z?\n"
            "   - How does it differ from baseline B specifically because of Z?\n"
            "   - What intermediate measurement confirms Z is active (not just Y improved)?\n"
            "   - Why existing methods fail — the structural reason Z implies, not 'they perform worse'\n\n"
            "3. EXPERIMENTAL PROCEDURE:\n"
            "   - Primary test: independent variable, dependent variable, expected result if Z true\n"
            "   - Control condition C: what you manipulate to isolate Z from Z'\n"
            "   - Discriminating test: what Z predicts vs. what Z' predicts in C\n"
            "   - What is held constant to prevent confounds\n\n"
            "4. DATASETS: All datasets explicitly named (include D above). "
            "For each additional dataset, state whether Z should be more/less operative and why.\n\n"
            "5. BASELINES: All baselines explicitly named (include B above). "
            "For each, explain specifically WHY it cannot exploit Z.\n\n"
            "6. SUCCESS CRITERION: Threshold T with its derivation from Z. "
            "State the exact measurement that confirms Z is operative.\n\n"
            "7. FALSIFICATION EXPERIMENT: The specific result that proves Z is NOT the cause. "
            "State what Z predicts vs. what Z' predicts in the discriminating experiment, "
            "and what specific measurement outcome would force abandoning Z.\n"
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


GENERATOR = S21_r5Generator()
