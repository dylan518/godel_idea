"""S20_r1: Benchmark-Anchored Counterfactual Ideation.

Fundamental departure from both S15 (hypothesis-first adversarial) and SFED
(skeleton-first): this pipeline mines NAMED BENCHMARK FAILURES from the SOTA
context FIRST -- concrete (dataset, baseline, failure_mode, gap) tuples where
current methods demonstrably fall short. All downstream generation is anchored
to those named failures, so experimental specificity is enforced at the ROOT of
the pipeline rather than hoped for at revision time.

Root cause addressed: prior pipelines generate a concept first and hope
concrete dataset/baseline names emerge later. This pipeline starts from what we
KNOW fails on NAMED benchmarks, then designs experiments as counterfactuals that
would fix exactly those failures.

Pipeline per idea:
  1. SOTA retrieval (OpenAlex, 5 papers)
  2. FAILURE MINING (1 call): extract 5 (dataset, baseline, failure_mode, gap)
     tuples from SOTA context; infer plausible ones if SOTA is sparse.
  3. COUNTERFACTUAL GENERATION (5 parallel calls): for each named failure,
     design the minimal methodological change that fixes it -- MODIFICATION,
     MECHANISM, PREDICTED IMPROVEMENT (quantitative), COMPARISON BASELINES,
     BOUNDARY (what it does NOT fix).
  4. FEASIBILITY + IMPACT SCORING (1 call): score each counterfactual on
     implementability / generality / novelty / impact; select top 2.
  5. CROSS-POLLINATION SYNTHESIS (1 call): combine top 2 into a unified idea
     with a shared mechanism that explains BOTH failure modes.
  6. EXPERIMENTAL OPERATIONALIZATION (1 call): expand synthesis into primary +
     ablation datasets, baseline hierarchy, primary + secondary metrics,
     quantitative success threshold, falsification criterion.
  7. MULTI-PERSPECTIVE CRITIQUE (3 parallel calls): experimentalist (can it run?),
     replication skeptic (does it generalise?), domain expert (right failure?).
  8. SPECIFICITY REVISION (1 call): final pass that explicitly requires >=2 named
     datasets, >=3 named baselines, a quantitative delta, and the anchored failure
     mode in each IDEA_FORMAT section.

LLM calls: 1 (mine) + 5 (counterfactuals) + 1 (score) + 1 (synth) + 1 (ops)
           + 3 (critique) + 1 (revise) = 13
"""

import sys
import os
import concurrent.futures

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from systems.base import IdeaGenerator, DEFAULT_MODEL, call_llm, IDEA_FORMAT


class S20_r1Generator(IdeaGenerator):
    VERSION = "S20_r1"
    DESCRIPTION = (
        "Benchmark-anchored counterfactual ideation: mine named dataset/baseline "
        "failures first, generate targeted counterfactuals in parallel, cross-pollinate "
        "top two into a unified mechanism, operationalize into a concrete experiment, "
        "multi-perspective critique, then specificity-enforcing final revision."
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
        # -- Step 0: SOTA retrieval -------------------------------------------
        try:
            from retrieval import get_topic_context
            sota_context = get_topic_context(topic, n=5) or ""
        except Exception:
            sota_context = ""

        context_block = (
            f"\n\nSOTA CONTEXT:\n{sota_context}\n" if sota_context else ""
        )

        # -- Step 1: failure mining -------------------------------------------
        failure_prompt = (
            f"Research topic: {topic}{context_block}\n"
            "You are a research failure analyst. Identify 5 concrete, named benchmark "
            "failures where current methods demonstrably fall short.\n\n"
            "For each failure specify:\n"
            "  DATASET: exact name (e.g., SQuAD 2.0, ImageNet-C, GLUE BoolQ, MT-Bench)\n"
            "  BASELINE: specific method that fails (e.g., GPT-4, ResNet-50, BERT-large)\n"
            "  FAILURE MODE: precisely HOW it fails (not just 'accuracy is low')\n"
            "  GAP: quantitative characterisation "
            "(e.g., '15% below human, degrades -8 pts under noise')\n\n"
            "Use concrete names from the SOTA context above; if unavailable, infer "
            "plausible named benchmarks from the topic.\n\n"
            "Format each entry exactly as:\n"
            "F1:\n"
            "  DATASET: <name>\n"
            "  BASELINE: <method>\n"
            "  FAILURE MODE: <description>\n"
            "  GAP: <quantification>\n"
            "F2:\n  DATASET: ...\n  BASELINE: ...\n  FAILURE MODE: ...\n  GAP: ...\n"
            "F3:\n  DATASET: ...\n  BASELINE: ...\n  FAILURE MODE: ...\n  GAP: ...\n"
            "F4:\n  DATASET: ...\n  BASELINE: ...\n  FAILURE MODE: ...\n  GAP: ...\n"
            "F5:\n  DATASET: ...\n  BASELINE: ...\n  FAILURE MODE: ...\n  GAP: ..."
        )
        try:
            failure_raw = call_llm(failure_prompt, model, client, temperature)
        except Exception:
            failure_raw = (
                "F1:\n  DATASET: Standard benchmark\n  BASELINE: SOTA model\n"
                "  FAILURE MODE: Degrades under distribution shift\n"
                "  GAP: Significant accuracy drop"
            )

        # Parse structured failures
        failures: list[dict] = []
        current: dict = {}
        for line in failure_raw.strip().split("\n"):
            stripped = line.strip()
            if (
                len(stripped) >= 2
                and stripped[0] == "F"
                and stripped[1].isdigit()
                and stripped.endswith(":")
            ):
                if current and "dataset" in current:
                    failures.append(current)
                current = {"id": stripped.rstrip(":")}
            elif ":" in stripped:
                key, _, val = stripped.partition(":")
                key = key.strip().upper()
                val = val.strip()
                if key == "DATASET":
                    current["dataset"] = val
                elif key == "BASELINE":
                    current["baseline"] = val
                elif key == "FAILURE MODE":
                    current["failure_mode"] = val
                elif key == "GAP":
                    current["gap"] = val
        if current and "dataset" in current:
            failures.append(current)

        if not failures:
            failures = [{
                "id": "F1",
                "dataset": f"{topic} standard benchmark",
                "baseline": "current SOTA",
                "failure_mode": (
                    f"performance degrades on distribution-shifted {topic} inputs"
                ),
                "gap": "notable accuracy drop relative to human performance",
            }]
        failures = failures[:5]

        # -- Step 2: parallel counterfactual generation ----------------------
        def _counterfactual(f: dict) -> str:
            ds = f.get("dataset", "unnamed dataset")
            bl = f.get("baseline", "existing method")
            fm = f.get("failure_mode", "performance degradation")
            gap = f.get("gap", "performance gap")
            prompt = (
                f"Research topic: {topic}\n\n"
                f"Specific failure to fix:\n"
                f"  Dataset:        {ds}\n"
                f"  Failing method: {bl}\n"
                f"  How it fails:   {fm}\n"
                f"  Gap:            {gap}\n\n"
                "Design a COUNTERFACTUAL EXPERIMENT: the minimal methodological change "
                "that would fix exactly this failure. Be surgical, not generic.\n\n"
                "Your answer MUST use these exact headers:\n"
                "MODIFICATION: <the exact technical change -- not 'improve X', "
                "describe the mechanism in one sentence>\n"
                f"MECHANISM: <why this directly fixes '{fm}'>\n"
                f"PREDICTED IMPROVEMENT: <quantitative estimate on {ds}, "
                "e.g., '+5-10% F1'>\n"
                "COMPARISON BASELINES: <3+ named methods for the ablation table>\n"
                "BOUNDARY: <one failure mode this does NOT fix -- bound the claim>"
            )
            try:
                return call_llm(prompt, model, client, temperature=0.7)
            except Exception:
                return (
                    f"MODIFICATION: Targeted fix for {fm}\n"
                    f"MECHANISM: Directly addresses the identified failure mode\n"
                    f"PREDICTED IMPROVEMENT: Measurable gain on {ds}\n"
                    f"COMPARISON BASELINES: {bl} and standard variants\n"
                    "BOUNDARY: Does not address orthogonal failure modes"
                )

        counterfactuals: list[str] = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as ex:
            futs = [ex.submit(_counterfactual, f) for f in failures]
            for fut in futs:
                try:
                    counterfactuals.append(fut.result(timeout=120))
                except Exception as e_:
                    counterfactuals.append(f"MODIFICATION: (timeout) {e_}")

        # -- Step 3: score + select top 2 ------------------------------------
        pairs_str = ""
        for i, (f, cf) in enumerate(zip(failures, counterfactuals)):
            pairs_str += (
                f"\n--- Counterfactual {i + 1} ---\n"
                f"Failure: {f.get('failure_mode', '')} on {f.get('dataset', '')}\n"
                f"{cf}\n"
            )

        score_prompt = (
            f"Research topic: {topic}\n\n"
            f"Score each of these {len(failures)} counterfactual experiments on 4 axes "
            "(1-5 each):\n"
            "  IMPLEMENTABILITY -- can a PhD student run this in 3 months?\n"
            "  GENERALITY -- does fixing this failure generalise beyond one dataset?\n"
            "  NOVELTY -- how different from published fixes for this failure mode?\n"
            "  IMPACT -- how important is the contribution if it works?\n\n"
            f"{pairs_str}\n"
            "Select the TOP 2 with highest total score.\n\n"
            f"Respond ONLY in this exact format (no extra text):\n"
            f"TOP1: <integer 1-{len(failures)}>\n"
            f"TOP1_SCORE: <total>/20\n"
            f"TOP2: <integer 1-{len(failures)}>\n"
            f"TOP2_SCORE: <total>/20\n"
            "REASONING: <one sentence>"
        )
        try:
            score_raw = call_llm(score_prompt, model, client, temperature=0.3)
        except Exception:
            score_raw = (
                "TOP1: 1\nTOP1_SCORE: 16/20\nTOP2: 2\nTOP2_SCORE: 14/20\n"
                "REASONING: fallback selection"
            )

        top_indices: list[int] = []
        for line in score_raw.strip().split("\n"):
            s = line.strip()
            if (
                s.startswith("TOP")
                and ":" in s
                and "SCORE" not in s
                and "REASON" not in s
            ):
                try:
                    v = int(s.split(":", 1)[1].strip()) - 1
                    if 0 <= v < len(failures) and v not in top_indices:
                        top_indices.append(v)
                except (ValueError, IndexError):
                    pass
        # Fallback: use first two distinct indices
        for i in range(len(failures)):
            if len(top_indices) >= 2:
                break
            if i not in top_indices:
                top_indices.append(i)

        top_failures = [failures[i] for i in top_indices[:2]]
        top_cfs = [counterfactuals[i] for i in top_indices[:2]]

        # -- Step 4: cross-pollination synthesis -----------------------------
        f_a = top_failures[0]
        cf_a = top_cfs[0]
        f_b = top_failures[1] if len(top_failures) > 1 else top_failures[0]
        cf_b = top_cfs[1] if len(top_cfs) > 1 else top_cfs[0]

        synth_prompt = (
            f"Research topic: {topic}\n\n"
            "Two counterfactual experiments have been selected:\n\n"
            f"COUNTERFACTUAL A (failure on {f_a.get('dataset', 'dataset A')}):\n"
            f"  Failure mode: {f_a.get('failure_mode', '')}\n"
            f"  Proposed fix:\n{cf_a}\n\n"
            f"COUNTERFACTUAL B (failure on {f_b.get('dataset', 'dataset B')}):\n"
            f"  Failure mode: {f_b.get('failure_mode', '')}\n"
            f"  Proposed fix:\n{cf_b}\n\n"
            "CROSS-POLLINATION TASK: Design a unified research idea that:\n"
            "1. Addresses BOTH failure modes through a SINGLE shared mechanism "
            "(not two separate patches -- explain what the failures have in common)\n"
            "2. Uses BOTH named datasets as benchmarks (one primary, one validation)\n"
            "3. Names the unifying principle -- the deep reason both failures occur\n"
            "4. Is more valuable than either counterfactual alone because it reveals "
            "a structural property of the problem\n\n"
            "Write 3-4 focused paragraphs. Specifics over generalities."
        )
        try:
            synthesis = call_llm(synth_prompt, model, client, temperature)
        except Exception:
            synthesis = (
                f"Unified approach to {topic} addressing failure modes on "
                f"{f_a.get('dataset', 'primary dataset')} and "
                f"{f_b.get('dataset', 'secondary dataset')}."
            )

        # -- Step 5: experimental operationalization -------------------------
        primary_ds = f_a.get("dataset", f"{topic} benchmark")
        secondary_ds = f_b.get("dataset", f"{topic} held-out set")
        primary_bl = f_a.get("baseline", "current SOTA")
        secondary_bl = f_b.get("baseline", "strong baseline")

        ops_prompt = (
            f"Research topic: {topic}\n\n"
            f"Proposed research idea:\n{synthesis}\n\n"
            "EXPERIMENTAL OPERATIONALIZATION -- turn this into a runnable experiment.\n"
            "Provide every item below with these exact headers:\n\n"
            f"PRIMARY DATASET: {primary_ds} "
            "(confirm appropriateness or name a better one)\n"
            "ABLATION DATASETS: name 2 additional datasets for robustness testing\n"
            f"PRIMARY BASELINE: {primary_bl}\n"
            "ADDITIONAL BASELINES: name >=3 comparison methods spanning "
            "naive / current SOTA / ablated variant\n"
            "PRIMARY METRIC: define exactly "
            "(e.g., 'F1 score on held-out test split', 'Top-1 accuracy on ImageNet val')\n"
            "SECONDARY METRICS: name 2 efficiency or robustness metrics\n"
            "SUCCESS THRESHOLD: what quantitative delta is a meaningful result? "
            "(e.g., '+2% absolute F1 with p<0.05 over PRIMARY BASELINE')\n"
            "FALSIFICATION CRITERION: what negative result would prove the core "
            "hypothesis wrong?\n\n"
            "Be specific -- name real models, datasets, metrics. "
            "Avoid placeholder examples."
        )
        try:
            ops = call_llm(ops_prompt, model, client, temperature=0.5)
        except Exception:
            ops = (
                f"PRIMARY DATASET: {primary_ds}\n"
                f"ABLATION DATASETS: {secondary_ds}, held-out split\n"
                f"PRIMARY BASELINE: {primary_bl}\n"
                f"ADDITIONAL BASELINES: {secondary_bl}, random baseline, ablated variant\n"
                "PRIMARY METRIC: task-appropriate accuracy\n"
                "SECONDARY METRICS: efficiency, calibration\n"
                "SUCCESS THRESHOLD: Statistically significant improvement\n"
                "FALSIFICATION CRITERION: No improvement over primary baseline"
            )

        # -- Step 6: parallel multi-perspective critique ---------------------
        def _crit_experimentalist(s: str, o: str) -> str:
            try:
                return call_llm(
                    f"You are a hard-nosed experimentalist reviewing a research "
                    f"proposal on '{topic}'.\n\n"
                    f"Proposed idea:\n{s}\n\nExperimental design:\n{o}\n\n"
                    "Give 2-3 sharp, specific criticisms on EXPERIMENTAL FEASIBILITY:"
                    "\n- Are the named datasets publicly available and suitable?"
                    "\n- Are the named baselines fairly comparable "
                    "(same compute budget, same input format)?"
                    "\n- What confound could produce a spurious positive result?"
                    "\n- What is the most likely failure mode of the experiment itself?",
                    model, client, temperature=0.5,
                )
            except Exception:
                return "Verify dataset availability and baseline comparability."

        def _crit_replication(s: str, o: str) -> str:
            try:
                return call_llm(
                    f"You are a replication skeptic reviewing a research proposal "
                    f"on '{topic}'.\n\n"
                    f"Proposed idea:\n{s}\n\nExperimental design:\n{o}\n\n"
                    "Give 2-3 sharp criticisms on GENERALIZABILITY AND REPRODUCIBILITY:"
                    "\n- Would this result replicate in a different lab with different hardware?"
                    "\n- Are the named datasets representative of real-world distribution?"
                    "\n- Could the success threshold be gamed by hyperparameter tuning?"
                    "\n- What would make this result fragile or non-robust?",
                    model, client, temperature=0.5,
                )
            except Exception:
                return "Consider reproducibility and external validity carefully."

        def _crit_domain_expert(s: str, o: str) -> str:
            try:
                return call_llm(
                    f"You are a domain expert in {topic} reviewing a research proposal.\n\n"
                    f"Proposed idea:\n{s}\n\nExperimental design:\n{o}\n\n"
                    "Give 2-3 sharp criticisms from a DOMAIN KNOWLEDGE perspective:"
                    "\n- Are these the right failure modes to fix, or are more important ones being ignored?"
                    "\n- Does the proposed mechanism match how practitioners think about this problem?"
                    "\n- Are the named baselines truly the strongest competitors?"
                    "\n- What domain knowledge appears to be missing?",
                    model, client, temperature=0.5,
                )
            except Exception:
                return "Ensure failure modes and baselines are domain-appropriate."

        crit_exp = crit_rep = crit_dom = ""
        with concurrent.futures.ThreadPoolExecutor(max_workers=3) as ex:
            fe = ex.submit(_crit_experimentalist, synthesis, ops)
            fr = ex.submit(_crit_replication, synthesis, ops)
            fd = ex.submit(_crit_domain_expert, synthesis, ops)
            try:
                crit_exp = fe.result(timeout=120)
            except Exception:
                crit_exp = "Verify dataset availability and baseline fairness."
            try:
                crit_rep = fr.result(timeout=120)
            except Exception:
                crit_rep = "Consider reproducibility and generalisation."
            try:
                crit_dom = fd.result(timeout=120)
            except Exception:
                crit_dom = "Ensure domain-appropriate failure modes and baselines."

        # -- Step 7: specificity-enforcing final revision --------------------
        anchor_summary = ("; ").join(
            f"{f.get('dataset', '?')} ({f.get('failure_mode', '?')}"
            for f in top_failures
        )

        revise_prompt = (
            f"Research topic: {topic}\n"
            f"Anchored failure modes: {anchor_summary}\n\n"
            f"Synthesis idea:\n{synthesis}\n\n"
            f"Experimental design:\n{ops}\n\n"
            "Reviewer critiques:\n"
            f"[Experimentalist] {crit_exp}\n\n"
            f"[Replication skeptic] {crit_rep}\n\n"
            f"[Domain expert] {crit_dom}\n\n"
            f"{context_block}\n"
            "Write the FINAL research idea. NON-NEGOTIABLE requirements:\n"
            "1. BACKGROUND section MUST name the specific failure mode and at least one "
            "named dataset where it occurs\n"
            "2. EXPERIMENT section MUST name >=2 datasets by name\n"
            "3. EXPERIMENT section MUST name >=3 baseline methods by name\n"
            "4. EXPERIMENT section MUST state a quantitative success threshold "
            "(e.g., '+X% on metric Y over baseline Z')\n"
            "5. NOVELTY section MUST explain why the named baselines fail to address this\n"
            "6. Address the most important critique from the three reviewers\n"
            + IDEA_FORMAT
        )
        try:
            return call_llm(revise_prompt, model, client, temperature)
        except Exception:
            return (
                synthesis
                if synthesis
                else f"Research idea about {topic}: {anchor_summary}"
            )


GENERATOR = S20_r1Generator()
