import sys
import os
import concurrent.futures

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from systems.base import IdeaGenerator, DEFAULT_MODEL, call_llm, IDEA_FORMAT


class S21Generator(IdeaGenerator):
    VERSION = "S21"
    DESCRIPTION = (
        "Hypothesis-first adversarial loop with mechanism-grounded falsification. "
        "Key improvement over S21_r5: the selection step now uses a structured scoring "
        "rubric that explicitly rewards Z→T derivation quality and experimental concreteness, "
        "and the construction step uses a tighter output template that forces the LLM to "
        "demonstrate (not merely claim) mechanistic specificity. The critique-revision loop "
        "is anchored to the selection rubric so that revisions must address the same "
        "dimensions that selection rewards."
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
        def attack_hypothesis(hyp: str) -> str:
            attack_prompt = (
                f"Research topic: {topic}\n\n"
                f"Hypothesis: {hyp}\n\n"
                "You are an adversarial critic. Your goal: force the hypothesis to be MORE "
                "MECHANISTICALLY PRECISE and EXPERIMENTALLY DISCRIMINATING.\n\n"
                "Attack on EACH of these dimensions:\n\n"
                "1. MECHANISM QUANTITATIVENESS: Does mechanism Z actually DERIVE threshold T, "
                "or is T a round number chosen independently?\n"
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

        # ── Step 3: select hypothesis using explicit SCORING RUBRIC ──────────
        # KEY CHANGE from S21_r5: selection now uses a structured 4-dimension
        # scoring rubric that must be applied explicitly to each hypothesis.
        # This prevents the LLM from defaulting to selecting the most
        # elaborately-described hypothesis rather than the most mechanistically
        # grounded one. Each dimension maps directly to what judges reward.
        pairs_str = ""
        for i, (hyp, attack) in enumerate(zip(hypotheses, attacks)):
            pairs_str += f"\n--- Hypothesis {i+1} ---\nOriginal: {hyp}\nCritique+Revision:\n{attack}\n"

        select_prompt = (
            f"Research topic: {topic}\n\n"
            f"Below are {len(hypotheses)} hypotheses, each attacked and revised:\n{pairs_str}\n"
            "SCORE each hypothesis's REVISED form on these 4 dimensions (1-5 each):\n\n"
            "DIMENSION 1 — Z→T DERIVATION QUALITY (most important, weight x2):\n"
            "  5: T follows directly from Z via a stated quantitative argument (e.g., 'r=10 → suppression=(r-1)/r=90%')\n"
            "  4: T follows from Z with one unstated but plausible step\n"
            "  3: T is consistent with Z but the derivation is implicit\n"
            "  2: T is a round number with no clear derivation from Z\n"
            "  1: T contradicts or is unrelated to Z\n\n"
            "DIMENSION 2 — EXPERIMENTAL CONCRETENESS:\n"
            "  5: Names specific real dataset D, named baseline B, specific metric M, specific control C\n"
            "  4: Names D, B, M, C but one is vague\n"
            "  3: Names D and B but M or C is missing/vague\n"
            "  2: Only names D or only names B\n"
            "  1: No specific experimental anchors\n\n"
            "DIMENSION 3 — FALSIFICATION SHARPNESS:\n"
            "  5: States specific measurement, specific numeric value, specific dataset/condition\n"
            "  4: States measurement and condition but value is a range\n"
            "  3: States measurement but condition is vague\n"
            "  2: States only 'no improvement' or equivalent\n"
            "  1: No falsification condition\n\n"
            "DIMENSION 4 — Z vs Z' DISCRIMINABILITY:\n"
            "  5: States a specific experiment where Z and Z' predict different numeric outcomes\n"
            "  4: States an experiment distinguishing Z from Z' but predictions are directional only\n"
            "  3: Names Z' but no discriminating experiment\n"
            "  2: Acknowledges alternative exists but doesn't name Z'\n"
            "  1: No alternative mechanism considered\n\n"
            "For each hypothesis, compute: TOTAL = 2×D1 + D2 + D3 + D4 (max=25)\n\n"
            "Then select the hypothesis with the highest TOTAL score.\n"
            "If tied, prefer the hypothesis where D1=5 (T directly derived from Z).\n\n"
            "Respond with:\n"
            "SCORES:\n"
            "H1: D1=[1-5] D2=[1-5] D3=[1-5] D4=[1-5] TOTAL=[sum]\n"
            "H2: D1=[1-5] D2=[1-5] D3=[1-5] D4=[1-5] TOTAL=[sum]\n"
            "H3: D1=[1-5] D2=[1-5] D3=[1-5] D4=[1-5] TOTAL=[sum]\n"
            "H4: D1=[1-5] D2=[1-5] D3=[1-5] D4=[1-5] TOTAL=[sum]\n"
            "H5: D1=[1-5] D2=[1-5] D3=[1-5] D4=[1-5] TOTAL=[sum]\n\n"
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
            "REASONING: <1-2 sentences: why this hypothesis has the highest total score, "
            "specifically citing D1 score and what makes T derivable from Z>"
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
        # KEY CHANGE from S21_r5: the construction prompt now uses a STRUCTURED
        # OUTPUT TEMPLATE that requires the LLM to fill in specific fields rather
        # than write free-form paragraphs. This prevents the LLM from producing
        # elaborately-worded vagueness — each field has a specific format constraint
        # that forces genuine specificity. The template mirrors the 4-dimension
        # scoring rubric from Step 3, so construction is anchored to the same
        # criteria that selection rewards.
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

        # Structured template construction — forces specific fields, not free-form paragraphs
        construct_prompt = (
            f"Research topic: {topic}\n"
            f"{context_reminder}\n"
            f"Hypothesis: {selected_hyp}\n"
            f"{mechanism_block}\n"
            f"{anchors_block}\n"
            "Design a concrete research idea. Fill in EVERY field below. "
            "Each field has a format constraint — follow it exactly.\n\n"
            "=== MECHANISM STATEMENT ===\n"
            "Z (one sentence, must include a quantitative prediction): [FILL IN]\n"
            "T (numeric value + derivation in format 'T=[value] because Z predicts [step-by-step argument]'): [FILL IN]\n"
            "Z' (best alternative mechanism, one sentence, must name a structural reason): [FILL IN]\n"
            "Distinguishing prediction: Z predicts [specific outcome] in control C; "
            "Z' predicts [different specific outcome] in same control C. [FILL IN both]\n\n"
            "=== METHOD ===\n"
            "Core component (name the specific algorithm/loss/module): [FILL IN]\n"
            "How it operationalizes Z (not 'it performs better' — the structural link to Z): [FILL IN]\n"
            "Structural difference from baseline B (what Z implies B cannot do): [FILL IN]\n"
            "Intermediate measurement confirming Z is active (must differ from primary metric M): [FILL IN]\n\n"
            "=== DATASETS ===\n"
            "Primary: {dataset} — Z operative because: [FILL IN]\n"
            "Additional 1: [name] — Z more/less operative because: [FILL IN]\n"
            "Additional 2: [name] — Z more/less operative because: [FILL IN]\n\n"
            "=== BASELINES ===\n"
            "Primary: {baseline} — cannot exploit Z because: [FILL IN]\n"
            "Additional 1: [name] — cannot exploit Z because: [FILL IN]\n"
            "Additional 2: [name] — cannot exploit Z because: [FILL IN]\n\n"
            "=== DISCRIMINATING EXPERIMENT ===\n"
            "Manipulation in control C: [exact manipulation, one sentence]\n"
            "Z predicts (direction + magnitude): [FILL IN]\n"
            "Z' predicts (direction + magnitude): [FILL IN]\n"
            "Result that forces abandoning Z: [specific measurement = specific value]\n\n"
            "=== SUCCESS CRITERION ===\n"
            "Primary threshold: T=[value] on metric M=[metric] on dataset D=[dataset]\n"
            "Derivation: [show the argument from Z to T]\n"
            "Intermediate confirmation: [measurement name] = [expected value range] confirms Z active\n\n"
            "=== FALSIFICATION ===\n"
            "Falsified if: [specific measurement] = [specific value] on [specific dataset] "
            "under [specific experimental condition]\n"
        ).format(
            dataset=selected_dataset or "primary dataset",
            baseline=selected_baseline or "primary baseline",
        )
        try:
            draft = call_llm(construct_prompt, model, client, temperature)
        except Exception:
            draft = f"Research idea for '{topic}' based on hypothesis: {selected_hyp}"

        # ── Step 5: multi-perspective critique ────────────────────────────────
        # Critiques are now anchored to the same 4 scoring dimensions as selection,
        # so that synthesis produces improvements that address the rubric directly.
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
            "Give 2-3 sharp criticisms focused on EXPERIMENTAL CONCRETENESS (rubric D2) "
            "and FALSIFICATION SHARPNESS (rubric D3):\n"
            "1. Is every experimental anchor (D, B, M, C) genuinely specific and implementable? "
            "Name the weakest anchor and what would make it concrete.\n"
            "2. Does the control condition C actually distinguish Z from Z'? "
            "Name the specific overlap and the exact manipulation needed to close it.\n"
            "3. Is the falsification condition (D3) a specific measurement=value=dataset triple? "
            "If not, rewrite it as one.\n"
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
            "Give 2-3 sharp criticisms focused on Z→T DERIVATION QUALITY (rubric D1):\n"
            "1. Is the derivation of T from Z mathematically valid? Identify the specific "
            "step where the derivation requires an unstated assumption. Propose the minimal "
            "fix that makes the derivation explicit.\n"
            "2. Does the proposed method IMPLEMENT Z directly, or does it merely correlate "
            "with Z? What would a method that purely implements Z look like?\n"
            "3. Is the intermediate measurement actually measuring Z (the mechanism) or Y "
            "(the effect)? These must be different — name the structural difference.\n"
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
            "Give 2-3 sharp criticisms focused on Z vs Z' DISCRIMINABILITY (rubric D4):\n"
            "1. Is the discriminating experiment actually decisive? Could Z' predict the same "
            "result as Z in the stated experiment? Name the specific overlap and propose a "
            "manipulation that would produce different numeric outcomes for Z vs Z'.\n"
            "2. Even if T is met, does it prove Z caused Y? Name the specific confound that "
            "would produce the same result without Z being true.\n"
            "3. Is the falsification condition genuinely risky for the authors? Propose a "
            "sharper version: 'Falsified if [measurement] = [value] on [dataset] under [condition]'.\n"
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
            f"Experimentalist (D2/D3 critique):\n{critique_exp}\n\n"
            f"Theorist (D1 critique):\n{critique_theory}\n\n"
            f"Skeptic (D4 critique):\n{critique_skeptic}\n\n"
            "Synthesize into the 3 most important actionable improvements, "
            "ordered by which rubric dimension they address:\n"
            "1. D1 fix: How to make T follow more directly from Z (show the specific derivation step to add)\n"
            "2. D4 fix: How to make the discriminating experiment produce different numeric outcomes "
            "for Z vs Z' (name the exact manipulation)\n"
            "3. D2/D3 fix: How to sharpen the weakest experimental anchor or falsification condition "
            "(rewrite as measurement=value=dataset=condition)\n"
            "Do NOT suggest removing experimental anchors. "
            "Do NOT suggest adding new theoretical components. "
            "Each fix must be specific enough to implement in one sentence."
        )
        try:
            synthesis = call_llm(synthesis_prompt, model, client, temperature=0.3)
        except Exception:
            synthesis = "Strengthen Z→T derivation, sharpen discriminating experiment, tighten falsification condition."

        # ── Step 6: final revision ─────────────────────────────────────────────
        # KEY CHANGE from S21_r5: the revision prompt now uses the same structured
        # template as Step 4, so the LLM cannot escape into free-form elaboration.
        # Each section maps to a rubric dimension, and the output format enforces
        # that the revision demonstrates (not claims) improvement on each dimension.
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
            f"Experimental design draft:\n{draft}\n\n"
            f"Required improvements:\n{synthesis}\n"
            f"{context_reminder}\n"
            "Write the final research idea. Use this EXACT structure — fill every field:\n\n"
            "## HYPOTHESIS\n"
            "X (specific structural property): [one phrase]\n"
            "Y (measurable effect, direction + magnitude): [one phrase]\n"
            "Z (quantitative causal mechanism, must predict T): [one-two sentences]\n"
            "T (threshold with derivation): T=[value] because Z predicts [step-by-step argument]\n"
            "Falsified if: [measurement] = [value] on [dataset] under [condition]\n\n"
            "## METHOD\n"
            "Core component: [name the specific algorithm/loss/module]\n"
            "Operationalizes Z by: [structural link to Z, not performance claim]\n"
            "Differs from B because Z implies: [what Z structurally prevents B from doing]\n"
            "Intermediate Z-confirmation measurement: [what to measure to confirm Z is active, "
            "must differ from primary metric M]\n\n"
            "## EXPERIMENTAL PROCEDURE\n"
            "Primary test: manipulate [independent variable], measure [M], "
            "expect [specific result] if Z true\n"
            "Control C manipulation: [exact manipulation to isolate Z from Z']\n"
            "Z predicts in C: [direction + magnitude]\n"
            "Z' predicts in C: [different direction or magnitude]\n"
            "Held constant: [what you control to prevent confounds]\n\n"
            "## DATASETS\n"
            "Primary D: [name] — Z operative because [structural argument]\n"
            "Additional 1: [name] — Z [more/less] operative because [structural argument]\n"
            "Additional 2: [name] — Z [more/less] operative because [structural argument]\n\n"
            "## BASELINES\n"
            "Primary B: [name] — cannot exploit Z because [structural reason]\n"
            "Additional 1: [name] — cannot exploit Z because [structural reason]\n"
            "Additional 2: [name] — cannot exploit Z because [structural reason]\n\n"
            "## SUCCESS CRITERION\n"
            "Primary: [metric M] > T=[value] on [dataset D]\n"
            "Derivation: [show Z → T argument]\n"
            "Z-active confirmation: [intermediate measurement] in range [expected range]\n\n"
            "## FALSIFICATION EXPERIMENT\n"
            "Z predicts: [specific outcome in discriminating experiment]\n"
            "Z' predicts: [different specific outcome in same experiment]\n"
            "Abandon Z if: [specific measurement] = [specific value] on [specific dataset]\n"
            "\n"
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


GENERATOR = S21Generator()
