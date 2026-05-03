import sys
import os
import concurrent.futures

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from systems.base import IdeaGenerator, DEFAULT_MODEL, call_llm, IDEA_FORMAT


class S21_r4Generator(IdeaGenerator):
    VERSION = "S21_r4"
    DESCRIPTION = (
        "Hypothesis-first adversarial loop with mechanism-grounded falsification. "
        "Key improvement over S21_r3: the adversarial attack loop is restructured to "
        "optimize for EXPERIMENTAL FALSIFIABILITY rather than theoretical elaboration. "
        "Attackers must propose a concrete alternative experiment that would disprove the "
        "hypothesis, forcing revisions to stay experimentally grounded. The selection step "
        "rewards concreteness and falsifiability over mechanistic complexity."
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
        # Each hypothesis must be grounded in a specific mechanism AND have a
        # concrete experimental skeleton baked in from the start. This prevents
        # the adversarial loop from drifting toward theoretical elaboration —
        # the skeleton is a hard anchor from the very first step.
        hyp_prompt = (
            f"Research topic: {topic}{context_block}\n"
            "Generate exactly 5 falsifiable scientific hypotheses. "
            "Each hypothesis MUST be immediately testable — not a theoretical claim "
            "that requires further specification before an experiment can be designed.\n\n"
            "REQUIREMENTS for each hypothesis:\n\n"
            "1. NAME a specific structural/algorithmic property X of existing methods "
            "that causes a measurable failure.\n\n"
            "2. STATE the causal mechanism Z: a specific mathematical or structural reason "
            "why X causes the failure. Z must be precise enough to predict a magnitude.\n\n"
            "3. SPECIFY the experimental skeleton — all four must be named:\n"
            "   - Dataset D: a real, named benchmark where the failure is observable\n"
            "   - Baseline B: a specific named method that exhibits property X\n"
            "   - Metric M: the specific measurement that directly probes whether Z is operative\n"
            "   - Threshold T: a specific number derived from Z (with a brief justification)\n\n"
            "4. STATE the falsification condition: what specific result would prove the "
            "hypothesis WRONG (not just 'no improvement' — a specific outcome).\n\n"
            "Template (follow exactly):\n"
            "H[n]: [Property X] causes [effect Y] because [mechanism Z]; "
            "predicted effect: [magnitude and direction]; "
            "dataset: [D], baseline: [B], metric: [M], threshold: [T] because [why T follows from Z]; "
            "falsified if: [specific falsification condition].\n\n"
            "Format:\n"
            "H1: <hypothesis>\nH2: <hypothesis>\nH3: <hypothesis>\nH4: <hypothesis>\nH5: <hypothesis>"
        )
        try:
            hyp_raw = call_llm(hyp_prompt, model, client, temperature)
        except Exception:
            hyp_raw = (
                f"H1: Uniform loss weighting in {topic} causes >15% degraded minority-class "
                f"recall because frequent-class gradients dominate in proportion to frequency ratio; "
                f"predicted effect: >15% recall gap at 10:1 imbalance; "
                f"dataset: imbalanced CIFAR-10, baseline: standard cross-entropy, "
                f"metric: minority-class recall, threshold: 15% gap because 10:1 ratio predicts "
                f"this magnitude from gradient domination; falsified if focal loss closes gap by <5%."
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
                f"they optimize average-case loss; predicted: >10% worst-group accuracy gap; "
                f"dataset: WILDS, baseline: ERM, metric: worst-group accuracy, "
                f"threshold: 10% because subgroup frequency imbalance predicts this; "
                f"falsified if ERM matches worst-group accuracy within 3%."
            ]
        hypotheses = hypotheses[:5]

        # ── Step 2: parallel adversarial attacks focused on experimental validity ──
        # KEY CHANGE from S21_r3: attacks now optimize for EXPERIMENTAL FALSIFIABILITY.
        # The attacker must (a) challenge whether the named experiment actually tests Z,
        # (b) propose a CONCRETE ALTERNATIVE EXPERIMENT that would distinguish Z from
        # the best alternative mechanism, and (c) ensure the revised hypothesis keeps
        # all four experimental anchors (D, B, M, T) concrete and grounded.
        # This prevents the adversarial loop from producing theoretically elaborate but
        # experimentally vague revisions.
        def attack_hypothesis(hyp: str) -> str:
            attack_prompt = (
                f"Research topic: {topic}\n\n"
                f"Hypothesis: {hyp}\n\n"
                "You are an adversarial critic. Your goal: force the hypothesis to be MORE "
                "EXPERIMENTALLY CONCRETE and FALSIFIABLE, not more theoretically elaborate.\n\n"
                "Attack on EACH of these dimensions:\n\n"
                "1. EXPERIMENTAL VALIDITY: Does the named experiment (D, B, M, T) actually "
                "test whether mechanism Z is operative? Or does it only test whether the "
                "proposed method outperforms the baseline (which could happen for many reasons)?\n"
                "   → Propose a SPECIFIC CONTROL EXPERIMENT that isolates Z from confounds. "
                "Name the exact manipulation (what you change), the exact measurement (what "
                "you record), and what result would confirm vs. disconfirm Z.\n\n"
                "2. ALTERNATIVE MECHANISM: What is the BEST alternative mechanism Z' that "
                "would produce the same observed effect Y without Z being true?\n"
                "   → Describe a specific experiment that would DISTINGUISH Z from Z'. "
                "What result would Z predict vs. what Z' predicts?\n\n"
                "3. THRESHOLD CONCRETENESS: Is threshold T a real number derived from Z, "
                "or is it a guess? If Z is the true mechanism, what does Z actually predict "
                "quantitatively? Derive a better T from Z.\n\n"
                "4. FALSIFICATION SHARPNESS: Is the falsification condition specific enough? "
                "Could the hypothesis survive being 'falsified' by reinterpreting results?\n"
                "   → Rewrite the falsification condition as: 'falsified if [specific "
                "measurement] shows [specific result] on [specific dataset/condition]'.\n\n"
                "Then write a REVISED hypothesis that:\n"
                "- KEEPS all four experimental anchors (D, B, M, T) — improve them, never remove\n"
                "- Makes T a real number with a brief derivation from Z\n"
                "- Sharpens the falsification condition to be unambiguous\n"
                "- Adds the control experiment that isolates Z\n"
                "- Does NOT add new theoretical components — only sharpen what's there\n\n"
                "CRITICAL: The revised hypothesis must be MORE EXPERIMENTALLY CONCRETE than "
                "the original, not more theoretically elaborate. A good revision has fewer "
                "moving parts, not more.\n\n"
                "Revised: <revised hypothesis — concrete, falsifiable, all anchors present>"
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

        # ── Step 3: select hypothesis with strongest EXPERIMENTAL GROUNDING ──
        # KEY CHANGE from S21_r3: selection now explicitly rewards experimental
        # concreteness and falsifiability over mechanistic elaboration.
        # A hypothesis with a simpler Z but cleaner experimental test beats one
        # with a complex Z and a vague experiment.
        pairs_str = ""
        for i, (hyp, attack) in enumerate(zip(hypotheses, attacks)):
            pairs_str += f"\n--- Hypothesis {i+1} ---\nOriginal: {hyp}\nCritique+Revision:\n{attack}\n"

        select_prompt = (
            f"Research topic: {topic}\n\n"
            f"Below are {len(hypotheses)} hypotheses, each attacked and revised:\n{pairs_str}\n"
            "Select the ONE hypothesis whose REVISED form best satisfies ALL of:\n\n"
            "1. EXPERIMENTAL CONCRETENESS (most important): The revised hypothesis names "
            "a specific real dataset D, a specific named baseline B, a specific metric M, "
            "and a specific numeric threshold T. All four must be present and appropriate. "
            "REJECT any hypothesis missing any of these four.\n\n"
            "2. FALSIFIABILITY: The falsification condition is unambiguous — a specific "
            "measurement on a specific dataset that would force abandoning the hypothesis. "
            "REJECT any hypothesis where the falsification condition is vague or escapable.\n\n"
            "3. MECHANISM CLARITY: The causal mechanism Z is specific enough to explain "
            "WHY the threshold T has that value. Z does not need to be complex — it needs "
            "to be precise enough to make a directional prediction.\n\n"
            "4. EXPERIMENTAL ISOLATION: The experiment isolates Z from confounds. There is "
            "a clear control condition that distinguishes Z from the best alternative mechanism.\n\n"
            "5. NOVELTY: The mechanism Z is not directly established in standard literature.\n\n"
            "6. FEASIBILITY: Dataset D is real and accessible; baseline B is reproducible.\n\n"
            "TIEBREAKER: Prefer the hypothesis with FEWER moving parts — a clean, simple "
            "experiment that directly tests Z beats a complex experiment with many components.\n\n"
            "Respond with:\n"
            "SELECTED: <number 1-5>\n"
            "REVISED HYPOTHESIS: <copy the revised hypothesis text exactly from after 'Revised:'>\n"
            "MECHANISM: <extract Z — the precise causal mechanism, 1-2 sentences>\n"
            "DATASET: <extract the specific dataset D>\n"
            "BASELINE: <extract the specific baseline B>\n"
            "METRIC: <extract the metric M>\n"
            "THRESHOLD: <extract the specific threshold T and its justification>\n"
            "FALSIFICATION: <extract the specific falsification condition>\n"
            "REASONING: <1-2 sentences on why this hypothesis is most experimentally concrete and falsifiable>"
        )
        try:
            selection_raw = call_llm(select_prompt, model, client, temperature=0.3)
        except Exception:
            selection_raw = (
                f"SELECTED: 1\nREVISED HYPOTHESIS: {hypotheses[0] if hypotheses else topic}\n"
                f"MECHANISM: unknown\nDATASET: standard benchmark\nBASELINE: existing SOTA\n"
                f"METRIC: primary metric\nTHRESHOLD: not specified\n"
                f"FALSIFICATION: not specified\nREASONING: fallback"
            )

        # Extract selected hypothesis and experimental anchors
        selected_hyp = hypotheses[0] if hypotheses else f"Novel approach to {topic}"
        selected_mechanism = ""
        selected_dataset = ""
        selected_baseline = ""
        selected_metric = ""
        selected_threshold = ""
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
            elif upper.startswith("THRESHOLD:"):
                candidate = stripped.split(":", 1)[1].strip()
                if candidate:
                    selected_threshold = candidate
            elif upper.startswith("FALSIFICATION:"):
                candidate = stripped.split(":", 1)[1].strip()
                if candidate:
                    selected_falsification = candidate

        # ── Step 4: construct experimental design FROM the hypothesis ─────────
        # KEY CHANGE from S21_r3: the construction prompt is structured around
        # the EXPERIMENTAL PROCEDURE first, then the method. This prevents the
        # model from generating a method and then retrofitting an experiment.
        # The falsification experiment is given equal weight to the positive result.
        context_reminder = (
            f"\nExisting work to differentiate from:\n{sota_context}\n"
            if sota_context else ""
        )

        mechanism_block = (
            f"\nCausal mechanism (Z) to test: {selected_mechanism}\n"
            if selected_mechanism else ""
        )

        anchors_block = ""
        if selected_dataset or selected_baseline or selected_metric or selected_threshold:
            anchors_block = "\nExperimental anchors (FIXED — do not change these):\n"
            if selected_dataset:
                anchors_block += f"  Primary dataset: {selected_dataset}\n"
            if selected_baseline:
                anchors_block += f"  Key baseline: {selected_baseline}\n"
            if selected_metric:
                anchors_block += f"  Primary metric: {selected_metric}\n"
            if selected_threshold:
                anchors_block += f"  Success threshold: {selected_threshold}\n"
            if selected_falsification:
                anchors_block += f"  Falsification condition: {selected_falsification}\n"
            anchors_block += (
                "These anchors are FIXED. Your method must be consistent with them. "
                "You may add datasets and baselines but must include these.\n"
            )

        construct_prompt = (
            f"Research topic: {topic}\n"
            f"{context_reminder}\n"
            f"Hypothesis: {selected_hyp}\n"
            f"{mechanism_block}\n"
            f"{anchors_block}\n"
            "Design a concrete research idea to TEST this hypothesis. Structure as follows:\n\n"
            "PARAGRAPH 1 — THE EXPERIMENT (most important):\n"
            "Describe the exact experimental procedure:\n"
            "  (a) What is the INDEPENDENT VARIABLE — what do you manipulate to isolate Z?\n"
            "  (b) What is the DEPENDENT VARIABLE — what do you measure to probe Z?\n"
            "  (c) What is held CONSTANT to prevent confounds with Z?\n"
            "  (d) What is the CONTROL CONDITION — the experiment that distinguishes Z "
            "from the best alternative mechanism?\n"
            "Be specific: name the exact manipulation, the exact measurement, and the "
            "expected result if Z is true vs. if Z is false.\n\n"
            "PARAGRAPH 2 — THE METHOD:\n"
            "Describe the proposed method. Focus on the specific component that directly "
            "operationalizes Z — not the full system. Explain:\n"
            "  (a) What specific component (loss function, architecture choice, algorithm "
            "step) directly embodies Z?\n"
            "  (b) How does this component differ from baseline B specifically because of Z?\n"
            "  (c) Why existing methods fail — not 'they perform worse' but the specific "
            "structural reason they cannot exploit Z.\n\n"
            "PARAGRAPH 3 — DATASETS AND BASELINES:\n"
            "Include the primary dataset and key baseline above PLUS:\n"
            "  - 2 additional datasets: for each, state whether Z should be MORE or LESS "
            "operative, and what result this predicts\n"
            "  - 3 additional baselines: for each, state specifically WHY it cannot exploit Z\n\n"
            "PARAGRAPH 4 — FALSIFICATION:\n"
            "Describe the specific experiment that would DISPROVE the hypothesis:\n"
            "  (a) What is the best alternative mechanism Z' that could explain the results?\n"
            "  (b) What does Z predict vs. what Z' predicts in the control condition?\n"
            "  (c) What specific result would force you to conclude Z is NOT the cause?\n"
            "  (d) State the falsification condition from the hypothesis and confirm the "
            "experiment can actually measure it.\n\n"
            "Be precise. Name specific datasets, baselines, metrics, and thresholds. "
            "Do not add theoretical components beyond what is needed to test Z."
        )
        try:
            draft = call_llm(construct_prompt, model, client, temperature)
        except Exception:
            draft = f"Research idea for '{topic}' based on hypothesis: {selected_hyp}"

        # ── Step 5: multi-perspective critique ────────────────────────────────
        exp_prompt = (
            f"You are a hard-nosed experimentalist reviewing a research proposal about '{topic}'.\n\n"
            f"Hypothesis: {selected_hyp}\n"
            f"Mechanism (Z): {selected_mechanism or 'not specified'}\n"
            f"Dataset: {selected_dataset or 'not specified'}\n"
            f"Baseline: {selected_baseline or 'not specified'}\n"
            f"Metric/Threshold: {selected_metric or 'not specified'} / {selected_threshold or 'not specified'}\n"
            f"Falsification condition: {selected_falsification or 'not specified'}\n\n"
            f"Proposed experiment:\n{draft}\n\n"
            "Give 2-3 sharp criticisms focused on:\n"
            "1. Does the experimental procedure actually isolate Z, or does it confound Z "
            "with other factors? Name the specific confound and how to eliminate it.\n"
            "2. Is the control condition (distinguishing Z from the best alternative) "
            "actually decisive? Could both Z and Z' predict the same result in the control?\n"
            "3. Is the falsification condition specific and unambiguous? Would the authors "
            "actually abandon the hypothesis if it is met, or could they reinterpret results?\n"
            "4. Is the threshold T meaningful — does beating it prove Z is operative?\n"
            "Be specific — name the exact problem, not a general concern."
        )
        try:
            critique_exp = call_llm(exp_prompt, model, client, temperature=0.5)
        except Exception:
            critique_exp = "No experimental critique available."

        theory_prompt = (
            f"You are a rigorous theorist reviewing a research proposal about '{topic}'.\n\n"
            f"Hypothesis: {selected_hyp}\n"
            f"Mechanism (Z): {selected_mechanism or 'not specified'}\n"
            f"Threshold: {selected_threshold or 'not specified'}\n\n"
            f"Proposed experiment:\n{draft}\n\n"
            "Give 2-3 sharp criticisms:\n"
            "1. Is mechanism Z theoretically sound? Is the quantitative prediction "
            "actually derivable from Z, or does it require additional assumptions not stated?\n"
            "2. Does the proposed method actually IMPLEMENT Z, or does it correlate with Z? "
            "What would a method that purely implements Z look like — simpler or different?\n"
            "3. What is the best theoretical alternative to Z that would produce the same "
            "experimental outcome? Is the control experiment actually decisive between them?\n"
            "Be specific about the theoretical gap — do not suggest adding complexity."
        )
        try:
            critique_theory = call_llm(theory_prompt, model, client, temperature=0.5)
        except Exception:
            critique_theory = "No theoretical critique available."

        skeptic_prompt = (
            f"You are a skeptical reviewer about '{topic}'.\n\n"
            f"Hypothesis: {selected_hyp}\n"
            f"Falsification condition: {selected_falsification or 'not specified'}\n"
            f"Success threshold: {selected_threshold or 'not specified'}\n\n"
            f"Proposed experiment:\n{draft}\n\n"
            "Give 2-3 sharp criticisms:\n"
            "1. Even if the threshold is met, does it PROVE Z caused Y? Name the specific "
            "alternative mechanism that would produce the same result without Z.\n"
            "2. Is the falsification experiment genuinely risky? Is there any way the authors "
            "could meet the success threshold while the falsification condition is also met?\n"
            "3. Is the experimental design simple enough to be reproducible? Name the specific "
            "implementation detail most likely to cause irreproducibility.\n"
            "Be direct about the weakest link."
        )
        try:
            critique_skeptic = call_llm(skeptic_prompt, model, client, temperature=0.5)
        except Exception:
            critique_skeptic = "No skeptic critique available."

        synthesis_prompt = (
            f"Three reviewers critiqued a research idea about '{topic}'.\n"
            f"Hypothesis: '{selected_hyp}'\n"
            f"Mechanism (Z): '{selected_mechanism or 'not specified'}'\n"
            f"Experimental anchors: dataset={selected_dataset or 'TBD'}, "
            f"baseline={selected_baseline or 'TBD'}, "
            f"metric={selected_metric or 'TBD'}, "
            f"threshold={selected_threshold or 'TBD'}\n"
            f"Falsification: {selected_falsification or 'TBD'}\n\n"
            f"Experimentalist:\n{critique_exp}\n\n"
            f"Theorist:\n{critique_theory}\n\n"
            f"Skeptic:\n{critique_skeptic}\n\n"
            "Synthesize into the 3 most important actionable improvements. "
            "Prioritize in this order:\n"
            "1. Fixes that make the experimental procedure more directly isolate Z "
            "(cleaner control condition, fewer confounds)\n"
            "2. Fixes that make the falsification condition more specific and unambiguous\n"
            "3. Fixes that make the method more directly implement Z (simpler, not more complex)\n"
            "Do NOT suggest removing experimental anchors — only suggest improving them. "
            "Do NOT suggest adding new theoretical components. "
            "Do NOT suggest adding vague 'additional experiments' — be specific about what to change."
        )
        try:
            synthesis = call_llm(synthesis_prompt, model, client, temperature=0.3)
        except Exception:
            synthesis = "Improve experimental isolation of Z, sharpen falsification condition, simplify method to directly implement Z."

        # ── Step 6: final revision ─────────────────────────────────────────────
        revise_prompt = (
            f"Research topic: {topic}\n\n"
            f"Core hypothesis: {selected_hyp}\n"
            f"Causal mechanism (Z): {selected_mechanism or 'derive from hypothesis'}\n"
            f"Experimental anchors (FIXED):\n"
            f"  Dataset: {selected_dataset or 'TBD'}\n"
            f"  Baseline: {selected_baseline or 'TBD'}\n"
            f"  Metric: {selected_metric or 'TBD'}\n"
            f"  Threshold: {selected_threshold or 'TBD'}\n"
            f"  Falsification condition: {selected_falsification or 'TBD'}\n\n"
            f"Experimental design:\n{draft}\n\n"
            f"Key improvements required:\n{synthesis}\n"
            f"{context_reminder}\n"
            "Write the final, improved research idea. Requirements:\n\n"
            "1. HYPOTHESIS: State clearly with:\n"
            "   - X: the specific structural property causing the failure\n"
            "   - Y: the measurable effect (direction and magnitude)\n"
            "   - Z: the precise causal mechanism (1-2 sentences, specific enough to predict T)\n"
            "   - T: the threshold with a brief derivation from Z\n"
            "   - Falsification condition: specific and unambiguous\n\n"
            "2. METHOD: Describe the specific component that directly implements Z.\n"
            "   - What is the component (loss, architecture, algorithm step)?\n"
            "   - How does it differ from baseline B specifically because of Z?\n"
            "   - What intermediate measurement confirms Z is active?\n"
            "   - Why existing methods fail — the specific structural reason, not 'they perform worse'\n\n"
            "3. EXPERIMENTAL PROCEDURE:\n"
            "   - Independent variable (what you manipulate to isolate Z)\n"
            "   - Dependent variable (what you measure to probe Z)\n"
            "   - Control condition (what distinguishes Z from the best alternative mechanism)\n"
            "   - What is held constant to prevent confounds\n\n"
            "4. DATASETS: All datasets explicitly named (include primary dataset above). "
            "For each additional dataset, state whether Z should be more/less operative.\n\n"
            "5. BASELINES: All baselines explicitly named (include key baseline above). "
            "For each, explain specifically WHY it cannot exploit Z.\n\n"
            "6. SUCCESS CRITERION: The specific threshold T with justification from Z. "
            "State the exact condition that confirms the hypothesis.\n\n"
            "7. FALSIFICATION EXPERIMENT: The specific result that would prove Z is NOT "
            "the true cause. State what the best alternative mechanism predicts differently, "
            "and how you would measure it. This must be as specific as the success criterion.\n"
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


GENERATOR = S21_r4Generator()
