"""S19_r1: Grounded Competing Predictions (Arbitration Structure).

A fundamentally different approach to idea generation. The core insight from
analyzing S15 vs S18 failures:

PROBLEM WITH S15: Generates hypotheses in domain-agnostic mode (grant-proposal
altitude), then attacks them adversarially. Adversarial selection makes hypotheses
survive by being orthogonal/unfalsifiable — not by being commensurable. The
experiment is designed to DEMONSTRATE the winning hypothesis, not DISCRIMINATE
between competing explanations. Result: experimentally murky ideas.

NEW APPROACH — "Arbitration Structure":
The experiment IS the idea. Instead of "we propose X and test it," the output is
"we will resolve whether H_A or H_B is correct by measuring Y on dataset Z,
where H_A predicts Y > X% and H_B predicts Y < X%."

KEY FIXES over S15:
1. UPSTREAM GROUNDING: Anchor to specific published observations with named datasets
   and approximate effect sizes BEFORE any hypothesis is formed. No more domain-
   agnostic hypothesis generation.
2. COMMENSURABLE PAIRS: Generate H_A and H_B as a PAIR from the same anomaly — not
   selected post-hoc from independent adversaries. Both hypotheses address the same
   observation, differ on exactly one measurable dimension, make opposite quantitative
   predictions.
3. DECISION EXPERIMENT: The experiment is defined by what it must DISCRIMINATE,
   not what it must DEMONSTRATE. Quantitative thresholds are justified by published
   effect sizes, not arbitrary.

Pipeline (8 LLM calls total):
1. ANOMALY MINING (1 call): From SOTA context, extract 3 specific empirical anomalies
   with named datasets, approximate magnitudes, and current theory failures.
2. COMPETING PAIRS (1 call): For each anomaly, generate a commensurable H_A/H_B pair
   with opposite quantitative predictions and a named discriminating measurement.
3. PAIR SELECTION (1 call): Choose the best (anomaly, H_A, H_B) triple by
   discriminability, feasibility, and scientific impact.
4. DISCRIMINATING EXPERIMENT (1 call): Design the minimal experiment that definitively
   tells H_A from H_B — specific datasets, named baselines, justified thresholds.
5. PARALLEL CRITIQUE (2 calls): Experimentalist + statistician focused on whether the
   experiment genuinely discriminates the two hypotheses.
6. SYNTHESIS (1 call): Actionable improvements.
7. FINAL REVISION (1 call): Revise to IDEA_FORMAT with arbitration framing.
"""

import sys
import os
import concurrent.futures

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from systems.base import IdeaGenerator, DEFAULT_MODEL, call_llm, IDEA_FORMAT


class S19Generator(IdeaGenerator):
    VERSION = "S19"
    DESCRIPTION = (
        "Grounded competing predictions (arbitration structure): mine specific empirical "
        "anomalies from SOTA, generate commensurable hypothesis pairs from the same "
        "anomaly, design minimal discriminating experiment. The experiment IS the idea."
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
            import os as _os
            sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
            from retrieval import get_topic_context
            sota_context = get_topic_context(topic, n=6)
        except Exception:
            sota_context = ""

        context_block = f"\n\nRecent literature:\n{sota_context}\n" if sota_context else ""
        no_context_note = (
            f"\n\nNote: No specific paper context available — draw on your knowledge of "
            f"recent empirical results in {topic}.\n"
        ) if not sota_context else ""

        # ── Step 1: anomaly mining ────────────────────────────────────────────
        # Extract specific empirical anomalies BEFORE any hypothesis formation.
        # This grounds the entire pipeline in published observations with numbers.
        anomaly_prompt = (
            f"Research topic: {topic}{context_block}{no_context_note}\n"
            "Identify 3 SPECIFIC EMPIRICAL ANOMALIES in this field.\n\n"
            "An anomaly is a published empirical finding that is:\n"
            "- Surprising or counter-intuitive given the dominant theoretical account\n"
            "- Observed across multiple studies or settings (not a single lab artifact)\n"
            "- Currently unexplained or under-explained by existing models\n\n"
            "For each anomaly use EXACTLY this format:\n"
            "ANOMALY_1:\n"
            "OBSERVATION: <one-sentence description of what was found>\n"
            "DATASET: <specific benchmark or dataset where this was observed, or 'cross-study'>\n"
            "MAGNITUDE: <approximate quantitative description, e.g. '8% gap vs baseline'>\n"
            "THEORY_FAILURE: <why current theory fails to explain this, specifically>\n\n"
            "ANOMALY_2:\n"
            "OBSERVATION: ...\n"
            "DATASET: ...\n"
            "MAGNITUDE: ...\n"
            "THEORY_FAILURE: ...\n\n"
            "ANOMALY_3:\n"
            "OBSERVATION: ...\n"
            "DATASET: ...\n"
            "MAGNITUDE: ...\n"
            "THEORY_FAILURE: ...\n\n"
            "Be specific. Name actual benchmarks, model families, or empirical patterns. "
            "Do NOT invent false citations. If unsure of exact numbers, give order-of-magnitude estimates."
        )
        try:
            anomaly_raw = call_llm(anomaly_prompt, model, client, temperature=0.7,
                                   max_tokens=1200)
        except Exception:
            anomaly_raw = (
                f"ANOMALY_1:\n"
                f"OBSERVATION: {topic} methods plateau unexpectedly beyond a threshold scale.\n"
                f"DATASET: general finding across multiple benchmarks\n"
                f"MAGNITUDE: <5% improvement despite 10x compute increase\n"
                f"THEORY_FAILURE: Power-law scaling laws predict continued improvement at this scale.\n"
                f"ANOMALY_2:\n"
                f"OBSERVATION: Transfer learning from large pretrained models underperforms "
                f"task-specific training on low-data regimes.\n"
                f"DATASET: various domain-specific benchmarks\n"
                f"MAGNITUDE: 10-15% gap in low-data settings\n"
                f"THEORY_FAILURE: Pretraining should provide useful inductive biases regardless of regime.\n"
                f"ANOMALY_3:\n"
                f"OBSERVATION: Larger models show worse calibration despite better accuracy.\n"
                f"DATASET: NLP/CV benchmarks\n"
                f"MAGNITUDE: ECE degrades ~0.05-0.1 per decade of scale\n"
                f"THEORY_FAILURE: More capacity should improve uncertainty estimation."
            )

        # ── Step 2: generate commensurable competing pairs ────────────────────
        # H_A and H_B are generated AS A PAIR from the same anomaly — both address
        # the same observation, differ on exactly one measurable dimension, and make
        # OPPOSITE quantitative predictions.
        pairs_prompt = (
            f"Research topic: {topic}\n\n"
            f"Empirical anomalies:\n{anomaly_raw}\n\n"
            "For each anomaly, generate a pair of competing mechanistic explanations.\n\n"
            "CRITICAL REQUIREMENTS — the pair MUST be COMMENSURABLE:\n"
            "- Both hypotheses address the SAME observation (same anomaly)\n"
            "- They differ on EXACTLY ONE measurable quantity or causal mechanism\n"
            "- They make OPPOSITE quantitative predictions on at least one metric\n"
            "- Both are plausible given current theory\n"
            "- A SINGLE controlled experiment can discriminate them\n\n"
            "Do NOT generate hypotheses that address different mechanisms or levels of analysis. "
            "The point is that H_A and H_B must be testable against each other with one experiment.\n\n"
            "For each anomaly use EXACTLY this format:\n"
            "=== PAIR FOR ANOMALY_N ===\n"
            "H_A: <mechanistic explanation A, naming the specific mechanism> "
            "→ PREDICTS: [metric name] will be [higher/lower] by approximately [X]% "
            "compared to [specific baseline condition]\n"
            "H_B: <mechanistic explanation B, opposite causal claim to A> "
            "→ PREDICTS: [same metric] will be [opposite direction] by approximately [Y]% "
            "compared to [same baseline condition]\n"
            "DISCRIMINATOR: <the single measurement that tells H_A from H_B; "
            "must be the SAME quantity for both>\n"
            "EXPERIMENT_CLASS: <ablation|comparison|perturbation|longitudinal|transfer>\n\n"
            "Generate pairs for all 3 anomalies."
        )
        try:
            pairs_raw = call_llm(pairs_prompt, model, client, temperature=0.75,
                                 max_tokens=1200)
        except Exception:
            pairs_raw = (
                "=== PAIR FOR ANOMALY_1 ===\n"
                f"H_A: The {topic} plateau is caused by data saturation (models have seen "
                "all informative patterns) → PREDICTS: adding 10x more diverse data will "
                "improve accuracy by >12% on held-out test sets\n"
                f"H_B: The {topic} plateau is caused by model capacity limits (not data) "
                "→ PREDICTS: adding 10x more diverse data will improve accuracy by <3%, "
                "while doubling model size will improve by >10%\n"
                "DISCRIMINATOR: accuracy gain from 10x data augmentation vs. 2x model scaling\n"
                "EXPERIMENT_CLASS: ablation\n"
            )

        # ── Step 3: select the best (anomaly, H_A, H_B) triple ───────────────
        select_prompt = (
            f"Research topic: {topic}\n\n"
            f"Anomalies:\n{anomaly_raw}\n\n"
            f"Competing pairs:\n{pairs_raw}\n\n"
            "Select the SINGLE BEST (anomaly, H_A, H_B) triple to build the research idea around.\n\n"
            "Score each triple on:\n"
            "1. DISCRIMINABILITY: Does one experiment cleanly resolve H_A vs H_B? "
            "(penalize if the same result could be explained by both)\n"
            "2. FEASIBILITY: Can the discriminating experiment run with existing tools, "
            "datasets, and compute? (penalize exotic requirements)\n"
            "3. IMPACT: Would resolving H_A vs H_B change how the field operates? "
            "(penalize if the answer doesn't matter in practice)\n"
            "4. NOVELTY: Is this anomaly or dispute not yet resolved in recent papers?\n\n"
            "Respond with EXACTLY:\n"
            "SELECTED_ANOMALY: <full text of the chosen anomaly's OBSERVATION line>\n"
            "DATASET_CONTEXT: <the dataset/benchmark from that anomaly>\n"
            "SELECTED_H_A: <full H_A text including its quantitative prediction>\n"
            "SELECTED_H_B: <full H_B text including its quantitative prediction>\n"
            "SELECTED_DISCRIMINATOR: <the discriminating measurement>\n"
            "REASONING: <2-3 sentences: why this triple is most discriminable and impactful>"
        )
        try:
            selection_raw = call_llm(select_prompt, model, client, temperature=0.3)
        except Exception:
            selection_raw = ""

        # Parse the selected triple
        def _extract_field(text: str, key: str) -> str:
            for line in text.strip().split("\n"):
                if line.strip().upper().startswith(key.upper() + ":"):
                    val = line.split(":", 1)[1].strip()
                    if val:
                        return val
            return ""

        selected_anomaly = _extract_field(selection_raw, "SELECTED_ANOMALY")
        dataset_context = _extract_field(selection_raw, "DATASET_CONTEXT")
        selected_ha = _extract_field(selection_raw, "SELECTED_H_A")
        selected_hb = _extract_field(selection_raw, "SELECTED_H_B")
        selected_discriminator = _extract_field(selection_raw, "SELECTED_DISCRIMINATOR")

        # Robust fallbacks
        if not selected_anomaly:
            selected_anomaly = f"performance gap in {topic} not explained by current theory"
        if not selected_ha:
            selected_ha = f"Mechanism A (capacity-driven) explains the anomaly in {topic}"
        if not selected_hb:
            selected_hb = f"Mechanism B (data-driven) provides an alternative explanation"
        if not selected_discriminator:
            selected_discriminator = "performance delta under controlled ablation"
        if not dataset_context:
            dataset_context = "existing benchmarks in the field"

        # ── Step 4: discriminating experiment design ──────────────────────────
        # The experiment is defined by what it must DISCRIMINATE — not what it demonstrates.
        # Quantitative thresholds must be justified from published effect sizes.
        context_reminder = (
            f"\nExisting work to differentiate from:\n{sota_context}\n"
            if sota_context else ""
        )
        experiment_prompt = (
            f"Research topic: {topic}\n"
            f"{context_reminder}\n"
            f"EMPIRICAL ANOMALY TO RESOLVE:\n{selected_anomaly}\n"
            f"Observed in: {dataset_context}\n\n"
            f"COMPETING MECHANISTIC EXPLANATIONS:\n"
            f"H_A: {selected_ha}\n"
            f"H_B: {selected_hb}\n\n"
            f"KEY DISCRIMINATING MEASUREMENT: {selected_discriminator}\n\n"
            "Design the MINIMAL discriminating experiment that definitively tells H_A from H_B.\n\n"
            "You MUST specify all of the following:\n\n"
            "DATASETS: Name specific datasets with approximate train/test split sizes "
            "(e.g., 'ImageNet-1K train: 1.2M, test: 50K' not 'standard image datasets'). "
            "Justify why these datasets are appropriate for discriminating H_A from H_B.\n\n"
            "BASELINES: Name specific model architectures or published methods to compare "
            "(e.g., 'ResNet-50, ViT-B/16, and CLIP zero-shot' not 'existing methods'). "
            "Each baseline must correspond to one of the hypotheses or serve as a control.\n\n"
            "FALSIFICATION THRESHOLDS: Derive exact numerical thresholds from published "
            "effect sizes in the literature. Format: 'if [metric] > [X]%, H_A is supported; "
            "if [metric] < [Y]%, H_B is supported; if [Y]% ≤ [metric] ≤ [X]%, result is "
            "ambiguous — specify what that ambiguity means'. Do NOT use arbitrary round numbers.\n\n"
            "CONFOUND CONTROLS: Name at least one control condition that rules out an "
            "alternative explanation for either outcome.\n\n"
            "AMBIGUITY HANDLING: What result would be genuinely ambiguous and what would "
            "that ambiguity tell us about the underlying mechanism?\n\n"
            "Write 4-5 focused paragraphs. Every number must be traceable to known baselines."
        )
        try:
            experiment_design = call_llm(experiment_prompt, model, client, temperature,
                                         max_tokens=1200)
        except Exception:
            experiment_design = (
                f"Research experiment for '{topic}': controlled comparison to test whether "
                f"{selected_ha} or {selected_hb}, measured via {selected_discriminator} "
                f"on named datasets with explicit quantitative thresholds."
            )

        # ── Step 5: parallel critique ─────────────────────────────────────────
        def _critique(prompt: str) -> str:
            try:
                return call_llm(prompt, model, client, temperature=0.5)
            except Exception:
                return "No critique available."

        exp_critique_prompt = (
            f"You are a hard-nosed experimentalist reviewing a DISCRIMINATING EXPERIMENT "
            f"about '{topic}'.\n\n"
            f"The experiment must distinguish:\n"
            f"H_A: {selected_ha}\n"
            f"H_B: {selected_hb}\n\n"
            f"Via measurement: {selected_discriminator}\n\n"
            f"Proposed experiment:\n{experiment_design}\n\n"
            "Give 3 sharp criticisms focused on:\n"
            "1. DATASET ADEQUACY: Are the named datasets appropriate for discriminating "
            "H_A from H_B, or do they conflate the two mechanisms?\n"
            "2. THRESHOLD VALIDITY: Are the quantitative falsification thresholds actually "
            "justified by known effect sizes, or are they arbitrary?\n"
            "3. CONFOUND RISK: What confound could produce the H_A result even if H_B is "
            "true? What control is missing to rule this out?\n"
            "Be specific. Reference known benchmarks or published results where possible."
        )

        stats_critique_prompt = (
            f"You are a rigorous statistician reviewing a research experiment about '{topic}'.\n\n"
            f"The experiment must discriminate:\n"
            f"H_A: {selected_ha}\n"
            f"H_B: {selected_hb}\n\n"
            f"Via measurement: {selected_discriminator}\n\n"
            f"Proposed experiment:\n{experiment_design}\n\n"
            "Give 3 sharp criticisms focused on:\n"
            "1. STATISTICAL POWER: Are the proposed sample sizes sufficient to detect "
            "the stated effect size with adequate power (>0.8)? What n is actually needed?\n"
            "2. METRIC INDEPENDENCE: Could the discriminating metric be confounded by a "
            "variable that affects both H_A and H_B in the same direction?\n"
            "3. SMALLEST MEANINGFUL EFFECT: What is the minimum effect size that would be "
            "scientifically interpretable here, and does the proposed threshold capture it?"
        )

        critique_exp = "No experimental critique available."
        critique_stats = "No statistical critique available."
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
            f_exp = executor.submit(_critique, exp_critique_prompt)
            f_stats = executor.submit(_critique, stats_critique_prompt)
            try:
                critique_exp = f_exp.result(timeout=120)
            except Exception:
                pass
            try:
                critique_stats = f_stats.result(timeout=120)
            except Exception:
                pass

        # ── Step 6: synthesis ─────────────────────────────────────────────────
        synthesis_prompt = (
            f"A discriminating experiment about '{topic}' was critiqued by two reviewers.\n\n"
            f"The experiment tests: H_A ({selected_ha}) vs H_B ({selected_hb})\n"
            f"Via: {selected_discriminator}\n\n"
            f"Experimentalist critique:\n{critique_exp}\n\n"
            f"Statistician critique:\n{critique_stats}\n\n"
            "Synthesize these into EXACTLY 3 specific, actionable improvements:\n"
            "1. <improvement that strengthens dataset/baseline selection>\n"
            "2. <improvement that validates or corrects the falsification thresholds>\n"
            "3. <improvement that adds a confound control or statistical safeguard>\n\n"
            "Each improvement must be concrete enough to implement directly."
        )
        try:
            synthesis = call_llm(synthesis_prompt, model, client, temperature=0.3)
        except Exception:
            synthesis = (
                "1. Name specific datasets with documented statistics from the literature.\n"
                "2. Justify thresholds from published baseline performance numbers.\n"
                "3. Add a control condition that can rule out the confound."
            )

        # ── Step 7: final revision ────────────────────────────────────────────
        revise_prompt = (
            f"Research topic: {topic}\n\n"
            f"ANOMALY BEING RESOLVED: {selected_anomaly}\n\n"
            f"COMPETING HYPOTHESES:\n"
            f"H_A: {selected_ha}\n"
            f"H_B: {selected_hb}\n\n"
            f"DISCRIMINATING MEASUREMENT: {selected_discriminator}\n\n"
            f"EXPERIMENTAL DESIGN:\n{experiment_design}\n\n"
            f"REQUIRED IMPROVEMENTS:\n{synthesis}\n"
            f"{context_reminder}\n"
            "Write the final, improved version of this research idea.\n\n"
            "FRAMING REQUIREMENT — write as an ARBITRATION EXPERIMENT, not an advocacy pitch:\n"
            "- Lead with the specific empirical anomaly that motivates the study\n"
            "- State both competing mechanistic explanations and their quantitative predictions\n"
            "- Define the discriminating experiment in terms of what it must DECIDE, not just measure\n"
            "- Every dataset, baseline, and threshold must be named explicitly\n"
            "- The falsification condition must be unmistakable: a reader should be able to "
            "say 'if the paper reports X, then H_A wins; if it reports Y, H_B wins'\n"
            "- Avoid advocacy phrases like 'we propose' or 'our method'; prefer 'we will test "
            "whether H_A or H_B accounts for [anomaly] by measuring [discriminator]'\n"
            + IDEA_FORMAT
        )
        try:
            return call_llm(revise_prompt, model, client, temperature, max_tokens=1200)
        except Exception:
            return experiment_design if experiment_design else (
                f"Research idea for '{topic}': discriminating experiment between "
                f"{selected_ha} and {selected_hb} via {selected_discriminator}."
            )


GENERATOR = S19Generator()
