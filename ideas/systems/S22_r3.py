import sys
import os
import concurrent.futures

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from systems.base import IdeaGenerator, DEFAULT_MODEL, call_llm, IDEA_FORMAT


class S22_r3Generator(IdeaGenerator):
    VERSION = "S22_r3"
    DESCRIPTION = (
        "Hypothesis-first adversarial loop with experimental constraints injected "
        "at generation time: hypotheses are generated WITH concrete dataset/metric "
        "anchors, adversarial attacks include operational/feasibility critiques, "
        "and selection rewards experimental tractability alongside centrality."
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
        sota_context = ""
        try:
            from retrieval import get_topic_context
            sota_context = get_topic_context(topic, n=5)
        except Exception:
            sota_context = ""

        context_block = (
            "\n\nRelevant existing work:\n" + sota_context + "\n"
            if sota_context else ""
        )

        # ── Step 1: generate 5 falsifiable hypotheses WITH experimental anchors ──
        # KEY FIX: Hypotheses are generated with concrete experimental scaffolds
        # baked in from the start — not retrofitted later. Each hypothesis must
        # name the dataset that would test it and the metric that would falsify it.
        hyp_prompt = (
            "Research topic: " + topic + context_block + "\n"
            "Generate exactly 5 sharp, falsifiable scientific hypotheses about this topic.\n\n"
            "CRITICAL: Each hypothesis must be generated WITH its experimental scaffold.\n"
            "A hypothesis without a concrete test is not a hypothesis — it is speculation.\n\n"
            "Each hypothesis must:\n"
            "1. Make a specific mechanistic claim about a CENTRAL phenomenon in this topic\n"
            "   (not a peripheral implementation detail or engineering sub-problem)\n"
            "2. Name 1-2 SPECIFIC, REAL datasets that exist and could test this claim\n"
            "   (name actual benchmark datasets used in this field, not generic placeholders)\n"
            "3. Name the EXACT metric that would falsify it (e.g., 'top-1 accuracy on X', "
            "'F1 on Y', 'perplexity on Z') and a rough threshold\n"
            "4. Be non-obvious: NOT directly supported by the related work above\n\n"
            "Format EXACTLY as:\n"
            "H1: <hypothesis statement>\n"
            "H1_DATASETS: <dataset1, dataset2>\n"
            "H1_METRIC: <exact metric name and falsification threshold>\n"
            "H2: <hypothesis statement>\n"
            "H2_DATASETS: <dataset1, dataset2>\n"
            "H2_METRIC: <exact metric name and falsification threshold>\n"
            "H3: <hypothesis statement>\n"
            "H3_DATASETS: <dataset1, dataset2>\n"
            "H3_METRIC: <exact metric name and falsification threshold>\n"
            "H4: <hypothesis statement>\n"
            "H4_DATASETS: <dataset1, dataset2>\n"
            "H4_METRIC: <exact metric name and falsification threshold>\n"
            "H5: <hypothesis statement>\n"
            "H5_DATASETS: <dataset1, dataset2>\n"
            "H5_METRIC: <exact metric name and falsification threshold>"
        )
        try:
            hyp_raw = call_llm(hyp_prompt, model, client, temperature)
        except Exception:
            hyp_raw = (
                "H1: Standard approaches to " + topic + " fail under distribution shift "
                "because they overfit to dataset-specific artifacts.\n"
                "H1_DATASETS: Standard benchmark dataset\n"
                "H1_METRIC: Accuracy drop > 5% under shift"
            )

        # Parse hypotheses with their experimental scaffolds
        hypotheses = []
        hyp_datasets = {}
        hyp_metrics = {}

        current_hyp_idx = None
        for line in (hyp_raw or "").strip().split("\n"):
            line = line.strip()
            if not line:
                continue
            # Check for hypothesis line
            if len(line) > 3 and line[0] == "H" and line[1].isdigit() and ":" in line[:4]:
                parts = line.split(":", 1)
                label = parts[0].strip()
                text = parts[1].strip() if len(parts) > 1 else ""
                if "_DATASETS" in label:
                    idx = label.replace("_DATASETS", "").replace("H", "")
                    try:
                        hyp_datasets[int(idx) - 1] = text
                    except Exception:
                        pass
                elif "_METRIC" in label:
                    idx = label.replace("_METRIC", "").replace("H", "")
                    try:
                        hyp_metrics[int(idx) - 1] = text
                    except Exception:
                        pass
                else:
                    # Plain hypothesis
                    if text:
                        current_hyp_idx = len(hypotheses)
                        hypotheses.append(text)

        if not hypotheses:
            hypotheses = [(hyp_raw or "").strip() or ("Novel approach to " + topic)]
        hypotheses = hypotheses[:5]

        # ── Step 2: parallel adversarial attacks — including operational critique ──
        # KEY FIX: Adversarial attacks now include OPERATIONAL/FEASIBILITY attacks,
        # not just logical/theoretical ones. The attack critiques whether the named
        # datasets actually support the falsification claim, whether the metric
        # threshold is calibrated to known SOTA, and whether the experimental
        # design has hidden costs.
        def attack_hypothesis(args):
            i, hyp = args
            datasets_str = hyp_datasets.get(i, "unspecified datasets")
            metric_str = hyp_metrics.get(i, "unspecified metric")
            attack_prompt = (
                "Research topic: " + topic + "\n\n"
                "Hypothesis: " + hyp + "\n"
                "Proposed test datasets: " + datasets_str + "\n"
                "Proposed falsification metric: " + metric_str + "\n\n"
                "You are an adversarial critic. Attack this hypothesis on ALL of these dimensions:\n\n"
                "LOGICAL ATTACKS:\n"
                "1. Assumption violation: What unstated assumption does this rely on? "
                "Give a concrete counterexample.\n"
                "2. Theoretical gap: What known result from the literature contradicts "
                "or undermines this?\n\n"
                "OPERATIONAL ATTACKS:\n"
                "3. Dataset mismatch: Do the named datasets actually have the properties "
                "needed to test this hypothesis? Are they the right domain, scale, and "
                "annotation type? Name a specific mismatch if one exists.\n"
                "4. Metric calibration: Is the falsification threshold realistic given "
                "known SOTA on these datasets? Would the threshold be too easy (proving "
                "nothing) or impossible (unfalsifiable in practice)?\n"
                "5. Experimental cost: What hidden engineering or compute cost would make "
                "this experiment infeasible in practice? (e.g., requires inaccessible data, "
                "months of training, proprietary models)\n\n"
                "Then write a REVISED hypothesis that:\n"
                "- Survives these attacks\n"
                "- Names corrected/better datasets if the originals were mismatched\n"
                "- Has a calibrated falsification threshold\n"
                "- Is experimentally tractable within 2-3 months\n\n"
                "Revised: <1-2 sentence refined hypothesis>\n"
                "Revised_Datasets: <corrected dataset names>\n"
                "Revised_Metric: <corrected metric and threshold>"
            )
            try:
                result = call_llm(attack_prompt, model, client, temperature=0.6)
                return result or ("Revised: " + hyp + "\nRevised_Datasets: " + datasets_str + "\nRevised_Metric: " + metric_str)
            except Exception:
                return "Revised: " + hyp + "\nRevised_Datasets: " + datasets_str + "\nRevised_Metric: " + metric_str

        attacks = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            futures = [executor.submit(attack_hypothesis, (i, h)) for i, h in enumerate(hypotheses)]
            for f in futures:
                try:
                    attacks.append(f.result(timeout=120))
                except Exception:
                    attacks.append("Revised: (timeout)")

        while len(attacks) < len(hypotheses):
            attacks.append("Revised: (no attack)")

        # Parse revised datasets and metrics from attacks
        revised_datasets = {}
        revised_metrics = {}
        for i, attack_text in enumerate(attacks):
            for line in (attack_text or "").strip().split("\n"):
                ls = line.strip()
                if ls.upper().startswith("REVISED_DATASETS:"):
                    val = ls.split(":", 1)[1].strip() if ":" in ls else ""
                    if val:
                        revised_datasets[i] = val
                elif ls.upper().startswith("REVISED_METRIC:"):
                    val = ls.split(":", 1)[1].strip() if ":" in ls else ""
                    if val:
                        revised_metrics[i] = val

        # ── Step 3: select hypothesis by CENTRALITY + EXPERIMENTAL TRACTABILITY ──
        pairs_str = ""
        for i, (hyp, attack) in enumerate(zip(hypotheses, attacks)):
            orig_ds = hyp_datasets.get(i, "unspecified")
            orig_metric = hyp_metrics.get(i, "unspecified")
            rev_ds = revised_datasets.get(i, orig_ds)
            rev_metric = revised_metrics.get(i, orig_metric)
            pairs_str += (
                "\n--- Candidate " + str(i + 1) + " ---\n"
                "Original: " + hyp + "\n"
                "Original datasets: " + orig_ds + "\n"
                "Original metric: " + orig_metric + "\n"
                "Critique+Revision:\n" + attack + "\n"
                "Revised datasets: " + rev_ds + "\n"
                "Revised metric: " + rev_metric + "\n"
            )

        select_prompt = (
            "Research topic: " + topic + "\n\n"
            "Below are " + str(len(hypotheses)) + " hypotheses, each attacked and revised "
            "with operational feasibility critiques:\n"
            + pairs_str + "\n"
            "Select the ONE hypothesis (original OR revised form) that is BEST.\n\n"
            "Rank by these criteria IN ORDER OF PRIORITY:\n\n"
            "1. CENTRALITY (highest priority): Which hypothesis addresses the CORE mechanism "
            "of '" + topic + "'? A central hypothesis, if confirmed, would change how "
            "researchers think about the whole topic.\n\n"
            "2. EXPERIMENTAL TRACTABILITY: Which hypothesis has the most concrete, "
            "realistic experimental scaffold? The named datasets must actually exist and "
            "be appropriate. The metric threshold must be calibrated to known SOTA. "
            "The experiment must be runnable in 2-3 months.\n\n"
            "3. MECHANISM SPECIFICITY: Which hypothesis names a specific causal mechanism "
            "(not just 'method X works better than Y' but WHY and HOW)?\n\n"
            "4. FALSIFIABILITY: Which hypothesis has a clear 'if this is false, we will see X' "
            "structure with a specific predicted outcome?\n\n"
            "5. NOVELTY: Which is least covered by standard literature?\n\n"
            "IMPORTANT: Do NOT select a hypothesis whose experimental scaffold is vague or "
            "whose datasets are mismatched to the claim. A central hypothesis that is "
            "experimentally tractable beats a central hypothesis that cannot be tested cleanly.\n\n"
            "Respond with:\n"
            "SELECTED: <number 1-5>\n"
            "FORM: <ORIGINAL or REVISED>\n"
            "CHOSEN HYPOTHESIS: <copy the exact hypothesis text>\n"
            "CHOSEN DATASETS: <the datasets to use for this hypothesis>\n"
            "CHOSEN METRIC: <the exact metric and threshold>\n"
            "CENTRALITY ARGUMENT: <1-2 sentences: why does this address the CORE of the topic?>\n"
            "TRACTABILITY ARGUMENT: <1-2 sentences: why is this experimentally feasible?>\n"
            "FALSIFICATION CONDITION: <one sentence: what specific result would definitively disprove this?>\n"
            "REASONING: <1-2 sentences on why this beats alternatives>"
        )
        try:
            selection_raw = call_llm(select_prompt, model, client, temperature=0.3)
        except Exception:
            selection_raw = (
                "SELECTED: 1\nFORM: ORIGINAL\n"
                "CHOSEN HYPOTHESIS: " + (hypotheses[0] if hypotheses else topic) + "\n"
                "CHOSEN DATASETS: Standard benchmark\n"
                "CHOSEN METRIC: Task accuracy\n"
                "CENTRALITY ARGUMENT: Addresses core topic mechanism\n"
                "TRACTABILITY ARGUMENT: Uses standard benchmarks\n"
                "FALSIFICATION CONDITION: No improvement over baseline\n"
                "REASONING: fallback"
            )

        # Extract selected hypothesis and supporting info
        selected_hyp = hypotheses[0] if hypotheses else topic
        chosen_datasets = ""
        chosen_metric = ""
        falsification_condition = ""
        centrality_argument = ""
        tractability_argument = ""
        selected_form = "REVISED"
        selected_idx = 0

        for line in (selection_raw or "").strip().split("\n"):
            ls = line.strip()
            ls_upper = ls.upper()
            if ls_upper.startswith("SELECTED:"):
                try:
                    selected_idx = int(ls.split(":", 1)[1].strip()) - 1
                except Exception:
                    selected_idx = 0
            elif ls_upper.startswith("FORM:"):
                selected_form = ls.split(":", 1)[1].strip().upper() if ":" in ls else "REVISED"
            elif ls_upper.startswith("CHOSEN HYPOTHESIS:"):
                candidate = ls.split(":", 1)[1].strip() if ":" in ls else ""
                if candidate:
                    selected_hyp = candidate
            elif ls_upper.startswith("CHOSEN DATASETS:"):
                chosen_datasets = ls.split(":", 1)[1].strip() if ":" in ls else ""
            elif ls_upper.startswith("CHOSEN METRIC:"):
                chosen_metric = ls.split(":", 1)[1].strip() if ":" in ls else ""
            elif ls_upper.startswith("CENTRALITY ARGUMENT:"):
                centrality_argument = ls.split(":", 1)[1].strip() if ":" in ls else ""
            elif ls_upper.startswith("TRACTABILITY ARGUMENT:"):
                tractability_argument = ls.split(":", 1)[1].strip() if ":" in ls else ""
            elif ls_upper.startswith("FALSIFICATION CONDITION:"):
                falsification_condition = ls.split(":", 1)[1].strip() if ":" in ls else ""

        # If FORM is ORIGINAL, recover the original hypothesis and its scaffold
        if selected_form == "ORIGINAL":
            if 0 <= selected_idx < len(hypotheses):
                selected_hyp = hypotheses[selected_idx]
                if not chosen_datasets and selected_idx in hyp_datasets:
                    chosen_datasets = hyp_datasets[selected_idx]
                if not chosen_metric and selected_idx in hyp_metrics:
                    chosen_metric = hyp_metrics[selected_idx]
        else:
            # Use revised datasets/metrics if available
            if not chosen_datasets and selected_idx in revised_datasets:
                chosen_datasets = revised_datasets[selected_idx]
            if not chosen_metric and selected_idx in revised_metrics:
                chosen_metric = revised_metrics[selected_idx]

        # Fallback for datasets/metric
        if not chosen_datasets:
            chosen_datasets = hyp_datasets.get(0, "standard benchmark for " + topic)
        if not chosen_metric:
            chosen_metric = hyp_metrics.get(0, "primary task metric")

        # ── Step 3.5: design minimal discriminating experiment ────────────────
        # KEY FIX: Experimental design is grounded in the datasets/metric that
        # were validated through the adversarial attack, not generated fresh.
        falsif_spec_prompt = (
            "Research topic: " + topic + "\n"
            "Core hypothesis: " + selected_hyp + "\n"
            "Pre-validated datasets: " + chosen_datasets + "\n"
            "Pre-validated metric: " + chosen_metric + "\n"
            + ("Centrality: " + centrality_argument + "\n" if centrality_argument else "")
            + ("Tractability: " + tractability_argument + "\n" if tractability_argument else "")
            + ("Falsification condition: " + falsification_condition + "\n" if falsification_condition else "")
            + "\nDesign the MINIMAL discriminating experiment for this hypothesis.\n\n"
            "The datasets and metric above have already been validated as appropriate "
            "for testing this hypothesis. Use them as your starting point.\n\n"
            "Constraints:\n"
            "- Use the pre-validated datasets above (add a second only if it rules out "
            "a specific confound the first cannot)\n"
            "- Use AT MOST 3 baselines. Each must rule out a specific alternative explanation.\n"
            "- Use the pre-validated metric as primary. One secondary metric is optional.\n"
            "- Every element must be justified by its role in falsifying the hypothesis.\n\n"
            "Specify:\n"
            "DATASET 1: <name> — DISCRIMINATING PROPERTY: <why it tests this specific hypothesis>\n"
            "DATASET 2: <name or NONE> — DISCRIMINATING PROPERTY: <what confound does this rule out?>\n"
            "BASELINE 1: <name> — ALTERNATIVE RULED OUT: <what alternative explanation does this eliminate?>\n"
            "BASELINE 2: <name> — ALTERNATIVE RULED OUT: <what alternative explanation?>\n"
            "BASELINE 3: <name or NONE> — ALTERNATIVE RULED OUT: <what alternative explanation?>\n"
            "PRIMARY METRIC: <exact metric name>\n"
            "SUCCESS THRESHOLD: <specific number or range> — DERIVATION: <how calibrated to known SOTA?>\n"
            "CRITICAL CONTROL: <one control experiment that rules out the main confound>\n"
            "NEGATIVE RESULT VALUE: <what would a negative result tell us scientifically?>"
        )
        try:
            falsif_spec = call_llm(falsif_spec_prompt, model, client, temperature=0.4)
        except Exception:
            falsif_spec = (
                "DATASET 1: " + chosen_datasets + " — DISCRIMINATING PROPERTY: Pre-validated for this hypothesis\n"
                "BASELINE 1: Best published method — ALTERNATIVE RULED OUT: Current state of the art\n"
                "PRIMARY METRIC: " + chosen_metric + "\n"
                "SUCCESS THRESHOLD: Statistically significant improvement over baseline\n"
                "CRITICAL CONTROL: Ablation without key component\n"
                "NEGATIVE RESULT VALUE: Hypothesis does not hold in this setting"
            )

        # ── Step 4: construct experimental idea ───────────────────────────────
        context_reminder = (
            "\nExisting work to differentiate from:\n" + sota_context + "\n"
            if sota_context else ""
        )

        construct_prompt = (
            "Research topic: " + topic + "\n"
            + context_reminder + "\n"
            "Core hypothesis (CENTRAL to the topic — not a peripheral sub-problem): "
            + selected_hyp + "\n\n"
            "MINIMAL EXPERIMENTAL SPEC (hard constraints — these datasets and metric "
            "were validated as appropriate for this hypothesis through adversarial critique):\n"
            + (falsif_spec or "Use appropriate datasets and baselines") + "\n\n"
            "Design a complete research experiment that proves or disproves this hypothesis.\n\n"
            "SCOPE RULES (enforce strictly):\n"
            "- Use ONLY the datasets in the spec above. Do not add more.\n"
            "- Use ONLY the baselines in the spec above. Do not add more.\n"
            "- Report ONLY the primary metric as the main result.\n"
            "- The method should have the MINIMUM number of components needed to test "
            "the hypothesis. Do not add components 'to improve performance.'\n\n"
            "Requirements:\n"
            "- State the central mechanism the hypothesis claims\n"
            "- Describe the key technical method in enough detail to implement\n"
            "- Explain WHY the selected datasets are discriminating for this specific hypothesis\n"
            "- Explain the critical control experiment and why it rules out the main confound\n"
            "- State explicitly what a negative result would mean scientifically\n"
            "- Explain how the success threshold was calibrated to known SOTA\n\n"
            "Write 3-4 paragraphs. Be direct and specific. "
            "Experimental tractability is as important as novelty."
        )
        try:
            draft = call_llm(construct_prompt, model, client, temperature)
        except Exception:
            draft = "Research idea for '" + topic + "' based on hypothesis: " + selected_hyp

        # ── Step 5: multi-perspective critique ───────────────────────────────
        exp_prompt = (
            "You are a hard-nosed experimentalist reviewing a research proposal about '"
            + topic + "'.\n\n"
            "Hypothesis being tested: " + selected_hyp + "\n"
            "Datasets: " + chosen_datasets + "\n"
            "Metric: " + chosen_metric + "\n\n"
            "Proposed experiment:\n" + (draft or "") + "\n\n"
            "Give 2-3 sharp criticisms focusing purely on experimental feasibility: "
            "Do these datasets actually have the right properties to test this hypothesis? "
            "Is the metric threshold calibrated to known SOTA or arbitrary? "
            "What controls are missing? What will fail in practice? "
            "Is there a hidden engineering cost that makes this infeasible in 2-3 months?"
        )
        try:
            critique_exp = call_llm(exp_prompt, model, client, temperature=0.5)
        except Exception:
            critique_exp = "No experimental critique available."

        theory_prompt = (
            "You are a rigorous theorist reviewing a research proposal about '"
            + topic + "'.\n\n"
            "Hypothesis being tested: " + selected_hyp + "\n\n"
            "Proposed experiment:\n" + (draft or "") + "\n\n"
            "Give 2-3 sharp criticisms focusing purely on theoretical grounding: "
            "Is the novelty claim justified? Does it overlap with known results? "
            "Is the mechanism actually the cause of the predicted effect? "
            "Is this hypothesis central to the topic or does it address a peripheral sub-problem?"
        )
        try:
            critique_theory = call_llm(theory_prompt, model, client, temperature=0.5)
        except Exception:
            critique_theory = "No theoretical critique available."

        skeptic_prompt = (
            "You are a skeptical reviewer who has seen many overhyped proposals about '"
            + topic + "'.\n\n"
            "Hypothesis being tested: " + selected_hyp + "\n\n"
            "Proposed experiment:\n" + (draft or "") + "\n\n"
            "Give 2-3 sharp criticisms: why this probably won't work, "
            "what the likely negative result is, and whether the scientific payoff "
            "justifies the effort even if it succeeds. "
            "Is the discriminating experiment actually discriminating, or could "
            "confounds explain the result? Is the threshold too easy to hit?"
        )
        try:
            critique_skeptic = call_llm(skeptic_prompt, model, client, temperature=0.5)
        except Exception:
            critique_skeptic = "No skeptic critique available."

        synthesis_prompt = (
            "Three reviewers critiqued a research idea about '" + topic
            + "' testing the hypothesis:\n"
            "'" + selected_hyp + "'\n\n"
            "Experimentalist:\n" + (critique_exp or "") + "\n\n"
            "Theorist:\n" + (critique_theory or "") + "\n\n"
            "Skeptic:\n" + (critique_skeptic or "") + "\n\n"
            "Synthesize these into the 3 most important actionable improvements. "
            "Focus on: (1) experimental tractability — are the datasets and metric "
            "threshold correct and calibrated?, "
            "(2) centrality — does the hypothesis address the core of the topic?, "
            "(3) mechanism specificity and falsifiability. "
            "Be concise and prioritized."
        )
        try:
            synthesis = call_llm(synthesis_prompt, model, client, temperature=0.3)
        except Exception:
            synthesis = "Improve experimental tractability, calibrate thresholds, clarify mechanism."

        # ── Step 6: final revision ─────────────────────────────────────────────
        revise_prompt = (
            "Research topic: " + topic + "\n\n"
            "Core hypothesis (central to the topic): " + selected_hyp + "\n\n"
            "Validated experimental spec:\n" + (falsif_spec or "") + "\n\n"
            "Experimental design:\n" + (draft or "") + "\n\n"
            "Key improvements required:\n" + (synthesis or "") + "\n"
            + context_reminder + "\n"
            "Write the final, improved version of the research idea.\n"
            "Requirements:\n"
            "- Open by stating the CENTRAL mechanism or question this addresses — "
            "explain why this is a core question for '" + topic + "', not a peripheral detail\n"
            "- State the hypothesis clearly with its specific mechanistic claim\n"
            "- Name the discriminating datasets with WHY they are discriminating for "
            "THIS hypothesis (not generic prestige)\n"
            "- Name the baselines with WHAT alternative explanation each one rules out\n"
            "- State the success threshold and explain how it was calibrated to known SOTA\n"
            "- Describe the critical control experiment\n"
            "- Explain what a negative result would mean scientifically\n"
            "- Keep scope minimal: only components required by the hypothesis\n"
            "- Incorporate the reviewer improvements above\n"
            + IDEA_FORMAT
        )
        try:
            result = call_llm(revise_prompt, model, client, temperature)
            return result or draft or ("Research idea about " + topic + ": " + selected_hyp)
        except Exception:
            return draft if draft else ("Research idea about " + topic + ": " + selected_hyp)


GENERATOR = S22_r3Generator()
