"""S17_r3: Grounded Prior Injection + Constrained Revision Protocol (GPIC).

CORE INSIGHT: Fix the generation DISTRIBUTION and revision GRAMMAR, not the
selection filter. Feasibility must be a PRIOR CONSTRAINT on generation and
revision — not a post-hoc filter on output.

ROOT CAUSE of S15/S16 failures: The adversarial loop's fitness function rewards
survival under attack. "Surviving" an attack on practical limitations means the
revised hypothesis retreated to a regime where practical limits don't apply
(fault-tolerant quantum hardware, 1T+ models, etc.). This creates a structural
anti-feasibility bias that judges consistently penalise.

NEW PIPELINE (14 LLM calls):

0. SOTA RETRIEVAL: Fetch recent papers to ground the context.

1. EMPIRICAL ANCHOR MINING (1 call):
   Extract what EXISTS TODAY: named datasets, named baselines with performance
   numbers, and documented failure modes. Grounds ALL subsequent generation.

2. GROUNDED HYPOTHESIS GENERATION (1 call):
   Each of 5 hypotheses MUST reference a specific named observation from Step 1.
   The generation prior is forced toward "anomaly in X on dataset Y" patterns
   rather than "theoretically possible improvement to Z".

3. SCOPE-NARROWING ATTACK (5 parallel calls):
   Attack grammar is INVERTED: instead of "survive by solving X", the revised
   hypothesis must NARROW SCOPE to a testable subproblem that avoids X.

4. FEASIBILITY-FIRST SELECTION (1 call):
   Select for IMMEDIATE TESTABILITY first (≤ $200 compute, existing datasets),
   then specificity, then novelty as a tiebreaker. Opposite priority from S15.

5. COMPUTE-BOUNDED EXPERIMENT CONSTRUCTION (1 call):
   Hard compute ceiling baked into the prompt: ≤ 8 GPUs × 48h.

6. OVERREACH CRITIQUE (3 parallel calls):
   Critiques focus on what can be REMOVED/NARROWED, not what must be added.

7. SYNTHESIS + SCOPE-CONSTRAINED REVISION (2 calls):
   Revision is explicitly prohibited from expanding scope. Only narrow/sharpen.
"""

import sys
import os
import concurrent.futures

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from systems.base import IdeaGenerator, DEFAULT_MODEL, call_llm, IDEA_FORMAT


class S17_r3Generator(IdeaGenerator):
    VERSION = "S17_r3"
    DESCRIPTION = (
        "GPIC: Grounded Prior Injection + Constrained Revision Protocol. "
        "Anchors hypothesis generation in named empirical observations today, "
        "uses scope-narrowing attack grammar instead of survive-by-solving, "
        "selects for immediate testability first, and prohibits scope expansion "
        "during revision."
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
        # ── Stage 0: SOTA Retrieval ──────────────────────────────────────────
        try:
            import os as _os
            sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
            from retrieval import get_topic_context
            sota_context = get_topic_context(topic, n=5)
        except Exception:
            sota_context = ""

        context_block = f"\n\n{sota_context}\n" if sota_context else ""

        # ── Stage 1: EMPIRICAL ANCHOR MINING ────────────────────────────────
        # Forces grounding in what EXISTS TODAY before any hypothesis generation.
        # This changes the generation prior from "what is theoretically possible"
        # to "what is anomalous in what we already observe".
        anchor_prompt = (
            f"Research topic: {topic}{context_block}\n"
            "You are an empirical scientist. Before generating any hypothesis, "
            "extract CONCRETE FACTS about what exists in this field TODAY.\n\n"
            "List EXACTLY the following (real names and numbers only — do not invent):\n\n"
            "DATASETS: Name 3-5 specific, publicly available datasets commonly used "
            "in this field (e.g., 'CIFAR-10', 'WikiText-103', 'SuperGLUE', 'SQuAD 1.1').\n\n"
            "BASELINES: Name 3-5 specific published methods with known benchmark numbers "
            "(e.g., 'BERT achieves 88.5 F1 on SQuAD 1.1', 'ResNet-50 gets 76.1% top-1 on ImageNet').\n\n"
            "FAILURE_MODES: Name 3-5 specific, documented failure modes or limitations of "
            "current approaches (e.g., 'transformers fail on compositional generalization in "
            "the SCAN benchmark', 'LLMs hallucinate on multi-hop reasoning in HotpotQA').\n\n"
            "OPEN_CONTRADICTIONS: Name 1-2 specific findings that either contradict each "
            "other or that current theory cannot explain (e.g., 'scaling improves few-shot "
            "on MMLU but hurts calibration on TruthfulQA').\n\n"
            "Be maximally specific. Use real dataset/method names from the literature."
        )
        try:
            anchor_raw = call_llm(anchor_prompt, model, client, temperature=0.3)
        except Exception:
            anchor_raw = (
                f"DATASETS: Standard benchmarks for {topic}\n"
                "BASELINES: Existing SOTA methods\n"
                "FAILURE_MODES: Known limitations in the literature\n"
                "OPEN_CONTRADICTIONS: Unexplained observations"
            )

        # ── Stage 2: GROUNDED HYPOTHESIS GENERATION ─────────────────────────
        # Each hypothesis MUST reference a specific named observation from Stage 1.
        # This forces the generation prior away from "theoretically possible"
        # toward "anomalous in existing data".
        hyp_prompt = (
            f"Research topic: {topic}\n\n"
            f"Empirical anchors (what EXISTS TODAY in this field):\n{anchor_raw}\n\n"
            "Generate exactly 5 GROUNDED hypotheses. Each hypothesis MUST:\n"
            "1. Reference a SPECIFIC named dataset, baseline, or failure mode from the "
            "anchor list above (quote the name exactly)\n"
            "2. Identify a SPECIFIC GAP or anomaly — not 'we can improve X' but "
            "'X fails on Y because of Z' or 'X works in A but not B because of C'\n"
            "3. Be testable using EXISTING RESOURCES: no new hardware, no >100B parameter "
            "models, no fault-tolerant quantum computers — only what a grad student can run "
            "on a university cluster in one week\n"
            "4. State the FALSIFICATION CONDITION: the exact measurement and threshold "
            "that would prove this hypothesis wrong\n\n"
            "Format each as:\n"
            "H1: [Hypothesis text] | ANCHOR: [exact named dataset/baseline/failure mode] | "
            "FALSIFIED_IF: [metric name < threshold]\n"
            "H2: [same format]\n"
            "H3: [same format]\n"
            "H4: [same format]\n"
            "H5: [same format]"
        )
        try:
            hyp_raw = call_llm(hyp_prompt, model, client, temperature)
        except Exception as e:
            hyp_raw = (
                f"H1: Standard {topic} approaches show degraded performance on distribution shift "
                f"| ANCHOR: existing benchmarks | FALSIFIED_IF: accuracy drops < 2%"
            )

        # Parse hypotheses
        hypotheses = []
        for line in hyp_raw.strip().split("\n"):
            line = line.strip()
            if line and (line.startswith("H") and ":" in line[:4]):
                hyp_text = line.split(":", 1)[1].strip()
                if hyp_text:
                    hypotheses.append(hyp_text)
        if not hypotheses:
            hypotheses = [hyp_raw.strip()]
        hypotheses = hypotheses[:5]

        # ── Stage 3: SCOPE-NARROWING ATTACK ─────────────────────────────────
        # Key innovation: the revision grammar is INVERTED.
        # Old grammar: "survive the attack by solving X" → scope inflation
        # New grammar: "narrow to testable subproblem that avoids X" → scope reduction
        def scope_narrowing_attack(hyp: str) -> str:
            attack_prompt = (
                f"Research topic: {topic}\n\n"
                f"Hypothesis: {hyp}\n\n"
                "You are a critic whose job is to find where this hypothesis OVERREACHES. "
                "Your goal is to produce the MOST FOCUSED, NARROWEST version that still "
                "captures the core scientific insight.\n\n"
                "Attack it on exactly these three dimensions:\n"
                "1. RESOURCE OVERREACH: Does this require compute, hardware, or data "
                "not available to a standard academic lab with one A100 GPU?\n"
                "   → If yes: what is the MINIMUM setup the core claim actually needs?\n"
                "2. SCOPE OVERREACH: Is the claim broader than what one experiment can "
                "establish on one or two named datasets?\n"
                "   → If yes: what is the NARROWEST claim that this experiment directly tests?\n"
                "3. ASSUMPTION OVERREACH: What is the ONE most critical unstated assumption?\n"
                "   → What SUBSET of the hypothesis holds WITHOUT this assumption?\n\n"
                "Now write a NARROWED hypothesis that:\n"
                "- Is testable with ≤ $200 of compute (single GPU, ≤ 48 hours)\n"
                "- Names ONE specific existing dataset and ONE specific baseline\n"
                "- States the exact metric and numeric threshold for falsification\n"
                "- Makes ONLY the claim the evidence directly supports\n\n"
                "NARROWED: <1-2 sentence focused hypothesis>"
            )
            try:
                return call_llm(attack_prompt, model, client, temperature=0.6)
            except Exception as e:
                return f"NARROWED: {hyp}"

        narrowed_attacks = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            futures = [executor.submit(scope_narrowing_attack, h) for h in hypotheses]
            for f in futures:
                try:
                    narrowed_attacks.append(f.result(timeout=120))
                except Exception as e:
                    narrowed_attacks.append(f"NARROWED: (timeout) {hypotheses[len(narrowed_attacks)]}")

        # ── Stage 4: FEASIBILITY-FIRST SELECTION ────────────────────────────
        # Priority order is INVERTED relative to S15.
        # S15 selected for: novelty + concreteness → scope inflation
        # S17_r3 selects for: immediate testability first, novelty as tiebreaker
        pairs_str = ""
        for i, (hyp, attack) in enumerate(zip(hypotheses, narrowed_attacks)):
            pairs_str += (
                f"\n--- Hypothesis {i+1} ---\n"
                f"Original: {hyp}\n"
                f"Scope-Narrowed Form:\n{attack}\n"
            )

        select_prompt = (
            f"Research topic: {topic}\n\n"
            f"Below are {len(hypotheses)} hypotheses, each narrowed to their most "
            f"focused, testable form:\n{pairs_str}\n"
            "Select the ONE hypothesis (by number) whose NARROWED form best satisfies "
            "these criteria IN STRICT PRIORITY ORDER:\n\n"
            "PRIORITY 1 — IMMEDIATE TESTABILITY (must pass this gate first):\n"
            "  - Can this be tested in ≤ 1 week on a standard academic GPU cluster?\n"
            "  - Does it name specific, publicly available datasets?\n"
            "  - Does it compare against a published baseline with known numbers?\n\n"
            "PRIORITY 2 — SPECIFICITY:\n"
            "  - Does it name the exact measurement and numeric threshold for falsification?\n"
            "  - Is the technical method specific enough to implement?\n\n"
            "PRIORITY 3 — NOVELTY (tiebreaker only, lower priority than above):\n"
            "  - Is the gap it identifies genuinely underexplored?\n\n"
            "Respond with:\n"
            "SELECTED: <number 1-5>\n"
            "NARROWED HYPOTHESIS: <copy the NARROWED hypothesis text exactly from above>\n"
            "TESTABILITY JUSTIFICATION: <1 sentence on why this can be run with standard academic resources>"
        )
        try:
            selection_raw = call_llm(select_prompt, model, client, temperature=0.3)
        except Exception as e:
            selection_raw = (
                f"SELECTED: 1\n"
                f"NARROWED HYPOTHESIS: {hypotheses[0]}\n"
                "TESTABILITY JUSTIFICATION: fallback selection"
            )

        # Extract the selected narrowed hypothesis
        selected_hyp = hypotheses[0]  # fallback
        for line in selection_raw.strip().split("\n"):
            if line.strip().upper().startswith("NARROWED HYPOTHESIS:"):
                candidate = line.split(":", 1)[1].strip()
                if candidate:
                    selected_hyp = candidate
                    break

        # ── Stage 5: COMPUTE-BOUNDED EXPERIMENT CONSTRUCTION ────────────────
        # Hard constraints baked into the prompt prevent scope inflation
        context_reminder = (
            f"\nExisting work to differentiate from:\n{sota_context}\n"
            if sota_context else ""
        )
        construct_prompt = (
            f"Research topic: {topic}\n"
            f"{context_reminder}\n"
            f"Empirical anchors:\n{anchor_raw}\n\n"
            f"Core hypothesis to test: {selected_hyp}\n\n"
            "Design a concrete research experiment. HARD CONSTRAINTS — violating these "
            "disqualifies the proposal:\n"
            "  COMPUTE: ≤ 8 GPU × 48 hours total (standard A100 or V100)\n"
            "  DATA: ONLY publicly available, named datasets with DOIs or download links\n"
            "  BASELINES: ONLY methods with published benchmark numbers to compare against\n\n"
            "The experiment design MUST specify:\n"
            "1. DATASET(S): Exact name(s), e.g., 'WikiText-103', 'CIFAR-10-C' — no generics\n"
            "2. BASELINES: Exact method names + their published numbers on those datasets\n"
            "3. PRIMARY METRIC + THRESHOLD: The exact number that confirms vs. falsifies\n"
            "4. TECHNICAL METHOD: Implementation detail sufficient to code in 2 weeks\n"
            "5. MINIMUM POSITIVE RESULT: Most conservative claim that still validates the hypothesis\n\n"
            "Write 3-4 paragraphs. Every claim must reference something that already exists. "
            "Do NOT propose experiments requiring resources beyond the constraints above."
        )
        try:
            draft = call_llm(construct_prompt, model, client, temperature)
        except Exception as e:
            draft = f"Research idea for '{topic}' based on hypothesis: {selected_hyp}"

        # ── Stage 6: OVERREACH CRITIQUE (parallel) ──────────────────────────
        # Focus: what can be REMOVED/NARROWED, not what needs to be added.
        # This is the opposite orientation from S15's critique stage.
        scope_critiquer_prompt = (
            f"You are a hard-nosed reviewer for a top ML venue, reviewing a proposal about '{topic}'.\n\n"
            f"Hypothesis: {selected_hyp}\n\n"
            f"Proposed experiment:\n{draft}\n\n"
            "Your job: identify OVERREACH — where this proposal claims MORE than its experiment "
            "can actually establish.\n\n"
            "Give 2-3 specific criticisms in this form:\n"
            "'The claim that [X] is not established by experiment [Y] because [Z]. "
            "A more defensible claim would be: [narrower specific claim].'\n\n"
            "Focus exclusively on SCOPE and INFERENTIAL VALIDITY, not on execution details."
        )

        feasibility_critiquer_prompt = (
            f"You are a pragmatic experimentalist reviewing a proposal about '{topic}'.\n\n"
            f"Hypothesis: {selected_hyp}\n\n"
            f"Proposed experiment:\n{draft}\n\n"
            "Your job: identify HIDDEN COSTS and UNSTATED ASSUMPTIONS in the experimental design.\n\n"
            "Give 2-3 specific criticisms naming:\n"
            "- The EXACT hidden resource (GPU-hours, labeled data, model checkpoints)\n"
            "- A CHEAPER ALTERNATIVE that tests the same core hypothesis\n\n"
            "Do not suggest adding new experiments — only find ways to make this one leaner."
        )

        novelty_critiquer_prompt = (
            f"You are a thorough literature reviewer for '{topic}'.\n\n"
            f"Hypothesis: {selected_hyp}\n\n"
            f"Proposed experiment:\n{draft}\n\n"
            "Your job: identify PRIOR ART that overlaps with this proposal.\n\n"
            "Name 2-3 specific prior results and explain:\n"
            "- Paper/method name + what it already showed\n"
            "- The EXACT dimension along which this proposal must differ to be novel\n"
            "- Format: 'Prior work X already showed Y; this proposal must establish Z instead.'"
        )

        def run_critique(prompt: str) -> str:
            try:
                return call_llm(prompt, model, client, temperature=0.5)
            except Exception:
                return "No critique available."

        with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
            f_scope = executor.submit(run_critique, scope_critiquer_prompt)
            f_feas = executor.submit(run_critique, feasibility_critiquer_prompt)
            f_novel = executor.submit(run_critique, novelty_critiquer_prompt)
            try:
                critique_scope = f_scope.result(timeout=120)
            except Exception:
                critique_scope = "No scope critique available."
            try:
                critique_feas = f_feas.result(timeout=120)
            except Exception:
                critique_feas = "No feasibility critique available."
            try:
                critique_novel = f_novel.result(timeout=120)
            except Exception:
                critique_novel = "No novelty critique available."

        # ── Stage 7: SYNTHESIS ───────────────────────────────────────────────
        synthesis_prompt = (
            f"Three reviewers critiqued a research proposal about '{topic}' testing:\n"
            f"'{selected_hyp}'\n\n"
            f"Scope reviewer (overreach/inferential validity):\n{critique_scope}\n\n"
            f"Feasibility reviewer (hidden costs):\n{critique_feas}\n\n"
            f"Novelty reviewer (prior art):\n{critique_novel}\n\n"
            "Synthesize these into exactly 3 SPECIFIC, ACTIONABLE improvements.\n\n"
            "CRITICAL CONSTRAINT: Each improvement must NARROW or SHARPEN the proposal — "
            "NOT expand it. Do NOT suggest adding new experiments, new datasets, or new methods. "
            "Only suggest refinements that make the existing proposal more precise, defensible, "
            "and grounded."
        )
        try:
            synthesis = call_llm(synthesis_prompt, model, client, temperature=0.3)
        except Exception:
            synthesis = (
                "1. Narrow the primary claim to what the named experiment directly establishes.\n"
                "2. Add the exact published baseline number for comparison.\n"
                "3. Specify the precise falsification threshold."
            )

        # ── Stage 8: SCOPE-CONSTRAINED REVISION ─────────────────────────────
        # Explicit prohibition against scope expansion during revision.
        # This is the final safeguard against the adversarial-revision inflation bug.
        revise_prompt = (
            f"Research topic: {topic}\n\n"
            f"Core hypothesis: {selected_hyp}\n\n"
            f"Experimental design:\n{draft}\n\n"
            f"Required improvements:\n{synthesis}\n"
            f"{context_reminder}\n"
            "REVISION RULES (violating these invalidates the response):\n"
            "  DO NOT add new experiments, datasets, or methods not already in the design\n"
            "  DO NOT expand the scope of the hypothesis — only narrow or clarify\n"
            "  DO NOT increase compute requirements — stay within 8 GPU × 48h\n"
            "  DO name specific datasets, baselines with their published numbers, and thresholds\n"
            "  DO make the falsification condition precise: exact metric and numeric threshold\n"
            "  DO ensure the idea is testable by a PhD student with standard academic resources\n\n"
            "Write the final, improved research idea. Every claim must be anchored to "
            "named, existing datasets or published results."
            + IDEA_FORMAT
        )
        try:
            return call_llm(revise_prompt, model, client, temperature)
        except Exception:
            return draft if draft else f"Research idea about {topic}: {selected_hyp}"


GENERATOR = S17_r3Generator()
