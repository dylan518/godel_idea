"""S17_r1: Resource-Grounded Hypothesis-First Loop.

Addresses the core failure mode of S15/S16: hypotheses are generated in an
unconstrained space, producing ambitious-but-unrunnable experiments. Judges
consistently penalise this on the FEASIBILITY dimension.

ROOT CAUSE FIX: Move feasibility UPSTREAM as a generative prior — not a
post-hoc filter. Define *what is actually buildable* BEFORE any hypothesis is
generated. Every downstream step is then anchored to those real resources.

Pipeline:
1. SOTA RETRIEVAL (0 LLM calls): fetch recent papers for context
2. RESOURCE ENVELOPE (1 call): name specific publicly available datasets,
   open-source baselines, compute tier, and computable metrics for this topic
3. ENVELOPE-ANCHORED HYPOTHESIS GENERATION (1 call): generate 5 hypotheses
   that are explicitly grounded in the resource envelope from the start —
   not free-range generation followed by filtering
4. ADVERSARIAL ATTACKS (5 calls, parallel): attack each hypothesis; dimension 4
   now explicitly checks resource-envelope compliance and forces in-bounds revision
5. SELECT STRONGEST (1 call): pick the most novel, falsifiable, in-bounds hypothesis
6. GROUNDED EXPERIMENT CONSTRUCTION (1 call): build experiment naming specific
   datasets / compute / baselines drawn from the envelope
7. FEASIBILITY STRESS TEST (1 call): ML-engineer reviewer asks "can I reproduce
   this on a university cluster in 6 months?" — output concrete gaps to fix
8. NOVELTY ELEVATION (1 call): novelty reviewer asks "is this genuinely
   non-incremental?" — sharpens the claim without blowing the compute budget
9. FINAL REVISION (1 call): synthesise feasibility fixes + novelty lift

LLM calls: 1+1+5+1+1+1+1+1 = 12
"""

import sys
import os
import concurrent.futures

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from systems.base import IdeaGenerator, DEFAULT_MODEL, call_llm, IDEA_FORMAT


class S17_r1Generator(IdeaGenerator):
    VERSION = "S17_r1"
    DESCRIPTION = (
        "Resource-grounded hypothesis-first loop: define a feasibility envelope "
        "before generating hypotheses (bakes feasibility in as a generative prior), "
        "adversarial attack with envelope-compliance check, then separate novelty "
        "elevation to prevent incremental mediocrity."
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

        # ── Step 0: retrieve SOTA context ─────────────────────────────────────
        try:
            import os as _os
            sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
            from retrieval import get_topic_context
            sota_context = get_topic_context(topic, n=5)
        except Exception:
            sota_context = ""

        context_block = f"\n\n{sota_context}\n" if sota_context else ""
        context_reminder = (
            f"\nExisting work to differentiate from:\n{sota_context}\n"
            if sota_context else ""
        )

        # ── Step 1: define the resource envelope ──────────────────────────────
        # This is the structural innovation: establish what is ACTUALLY BUILDABLE
        # before any hypothesis is generated. Feasibility must shape generation,
        # not filter it retroactively.
        envelope_prompt = (
            f"Research topic: {topic}{context_block}\n"
            "You are a senior ML researcher advising a PhD student who has access to:\n"
            "  • A university GPU cluster (up to 8× A100 80 GB GPUs for 1–2 weeks)\n"
            "  • Standard cloud compute (~$5 000 total budget)\n"
            "  • Publicly available pretrained models up to 13 B parameters\n"
            "  • Open-source code and publicly released datasets\n\n"
            "Define a RESOURCE ENVELOPE — what this student can actually build and test:\n\n"
            "DATASETS: 4–6 specific, publicly available datasets for this topic. "
            "Name them precisely (e.g. 'GLUE', 'ImageNet-1k', 'WikiText-103', "
            "'CIFAR-10-C', 'SQuAD 2.0'). Avoid vague labels like 'standard benchmarks'.\n\n"
            "BASELINES: 3–5 specific methods or models with publicly available code. "
            "Name the paper, model, or repository (e.g. 'LLaMA-2-7B', 'BERT-base', "
            "'MC-Dropout from Gal & Ghahramani 2016 — github.com/yaringal/DropoutUncertaintyExps', "
            "'DPO from Rafailov et al. 2023'). Must have real, working open-source implementations.\n\n"
            "COMPUTE TIER: The maximum experiment scale that fits in the envelope. "
            "Be concrete (e.g. 'fine-tune models up to 7 B params on 4×A100 for ≤5 days; "
            "full training from scratch up to 125 M params; inference-only for larger models').\n\n"
            "METRICS: 3–4 specific, computable metrics (e.g. 'ECE on CIFAR-10-C', "
            "'perplexity on WikiText-103', 'AUROC on ImageNet-O', 'accuracy on MMLU'). "
            "Must be reproducible without proprietary tools.\n\n"
            "Be specific. Name real resources. This envelope will constrain every "
            "hypothesis generated next."
        )
        try:
            envelope_raw = call_llm(envelope_prompt, model, client, temperature=0.3,
                                    max_tokens=600)
        except Exception:
            envelope_raw = (
                "DATASETS: Publicly available task-specific benchmarks.\n"
                "BASELINES: Published open-source models in the field.\n"
                "COMPUTE TIER: Single 8×A100 node, models up to 7 B parameters.\n"
                "METRICS: Standard task accuracy and calibration metrics."
            )

        # ── Step 2: envelope-anchored hypothesis generation ───────────────────
        # Hypotheses are generated CONSTRAINED by the resource envelope from the
        # very start — not generated freely and pruned later.
        hyp_prompt = (
            f"Research topic: {topic}{context_block}\n"
            f"Resource envelope — what this lab can actually build:\n{envelope_raw}\n\n"
            "Generate exactly 5 sharp, falsifiable scientific hypotheses about this topic.\n\n"
            "HARD CONSTRAINTS — every hypothesis MUST:\n"
            "  1. Reference AT LEAST ONE specific dataset from the envelope above\n"
            "  2. Reference AT LEAST ONE specific baseline from the envelope above\n"
            "  3. Require NO MORE compute than the envelope's compute tier\n"
            "  4. Make a specific, non-obvious mechanistic claim (not 'we can improve X')\n"
            "  5. Be falsifiable: state what experimental result would disprove it\n\n"
            "These are grounded hypotheses, not wish-lists. If you cannot make a "
            "hypothesis that fits the envelope, make the MOST NOVEL hypothesis that does.\n\n"
            "H1: <hypothesis — 1–2 sentences naming envelope resources>\n"
            "H2: <hypothesis>\nH3: <hypothesis>\nH4: <hypothesis>\nH5: <hypothesis>"
        )
        try:
            hyp_raw = call_llm(hyp_prompt, model, client, temperature)
        except Exception as e:
            hyp_raw = f"H1: Standard approaches to {topic} fail under distribution shift."

        hypotheses = []
        for line in hyp_raw.strip().split("\n"):
            line = line.strip()
            if line and line.startswith("H") and ":" in line[:4]:
                hyp_text = line.split(":", 1)[1].strip()
                if hyp_text:
                    hypotheses.append(hyp_text)
        if not hypotheses:
            hypotheses = [hyp_raw.strip()]
        hypotheses = hypotheses[:5]

        # ── Step 3: parallel adversarial attacks (envelope-aware) ─────────────
        # Attack dimension 4 is now a resource-compliance check, not a generic
        # "practical limitation" probe.
        def attack_hypothesis(hyp: str) -> str:
            attack_prompt = (
                f"Research topic: {topic}\n\n"
                f"Resource envelope:\n{envelope_raw}\n\n"
                f"Hypothesis: {hyp}\n\n"
                "You are an adversarial critic. Attack this hypothesis on each dimension:\n"
                "1. Assumption violation: What unstated assumption does it rely on? "
                "Give a concrete counterexample.\n"
                "2. Dataset bias: What artifact in the named dataset could make this "
                "appear true without being true?\n"
                "3. Theoretical gap: What known result contradicts or undermines this?\n"
                "4. Resource compliance: Does this hypothesis require resources OUTSIDE "
                "the envelope (frontier models, proprietary data, >2-week compute on 8×A100)? "
                "If yes, state exactly what exceeds the envelope.\n\n"
                "Then write a REVISED hypothesis that:\n"
                "  • Survives attacks 1–3 (more specific, guards against dataset artifacts)\n"
                "  • Stays STRICTLY within the resource envelope\n"
                "  • Remains as novel and mechanistically sharp as possible\n\n"
                "Revised: <1–2 sentence refined hypothesis>"
            )
            try:
                return call_llm(attack_prompt, model, client, temperature=0.6)
            except Exception as e:
                return f"Revised: {hyp} (attack failed: {e})"

        attacks = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            futures = [executor.submit(attack_hypothesis, h) for h in hypotheses]
            for f in futures:
                try:
                    attacks.append(f.result(timeout=120))
                except Exception as e:
                    attacks.append(f"Revised: (timeout) {e}")

        # ── Step 4: select the strongest in-envelope hypothesis ───────────────
        pairs_str = ""
        for i, (hyp, attack) in enumerate(zip(hypotheses, attacks)):
            pairs_str += (
                f"\n--- Hypothesis {i+1} ---\n"
                f"Original: {hyp}\n"
                f"Critique + Revision:\n{attack}\n"
            )

        select_prompt = (
            f"Research topic: {topic}\n\n"
            f"Resource envelope:\n{envelope_raw}\n\n"
            f"Below are {len(hypotheses)} hypotheses, each attacked and revised:\n{pairs_str}\n"
            "Select the ONE hypothesis whose REVISED form best satisfies ALL of:\n"
            "  1. Within resource envelope (disqualify any that exceed it)\n"
            "  2. Most concrete — names specific datasets, baselines, mechanisms\n"
            "  3. Most falsifiable — clearest path to a disproof experiment\n"
            "  4. Most novel — least covered by standard prior work\n\n"
            "Respond with:\n"
            "SELECTED: <number 1–5>\n"
            "REVISED HYPOTHESIS: <copy the revised hypothesis text exactly>\n"
            "REASONING: <1–2 sentences on why this is strongest>"
        )
        try:
            selection_raw = call_llm(select_prompt, model, client, temperature=0.3)
        except Exception:
            selection_raw = (
                f"SELECTED: 1\n"
                f"REVISED HYPOTHESIS: {hypotheses[0]}\n"
                "REASONING: fallback"
            )

        selected_hyp = hypotheses[0]
        for line in selection_raw.strip().split("\n"):
            if line.strip().upper().startswith("REVISED HYPOTHESIS:"):
                candidate = line.split(":", 1)[1].strip()
                if candidate:
                    selected_hyp = candidate
                    break

        # ── Step 5: grounded experiment construction ──────────────────────────
        construct_prompt = (
            f"Research topic: {topic}\n"
            f"{context_reminder}\n"
            f"Resource envelope:\n{envelope_raw}\n\n"
            f"Core hypothesis to test: {selected_hyp}\n\n"
            "Design a concrete experiment to PROVE OR DISPROVE this hypothesis.\n"
            "You MUST draw resources from the envelope above — do not invent new ones.\n\n"
            "Requirements:\n"
            "  • Name the EXACT datasets (from the envelope) you will use and why\n"
            "  • Name the EXACT baselines (from the envelope) to compare against\n"
            "  • State the EXACT compute plan (within the envelope's compute tier)\n"
            "  • Define quantitative decision thresholds: what result proves the hypothesis, "
            "what result falsifies it\n"
            "  • Describe the key technical method with enough detail to implement in 6 months\n"
            "  • Identify what would go wrong if the hypothesis is false and how you'd know\n\n"
            "Write 3–4 paragraphs. Be direct and specific."
        )
        try:
            draft = call_llm(construct_prompt, model, client, temperature)
        except Exception:
            draft = f"Research idea for '{topic}' testing: {selected_hyp}"

        # ── Step 6: feasibility stress test ───────────────────────────────────
        # A hard-headed ML engineer reviewer, not a theorist — asks "can I actually
        # reproduce this on a university cluster in 6 months?"
        feasibility_prompt = (
            f"You are a senior ML engineer tasked with REPRODUCING this experiment "
            f"on a university GPU cluster in 6 months.\n\n"
            f"Stated resource envelope:\n{envelope_raw}\n\n"
            f"Proposed experiment:\n{draft}\n\n"
            "Answer these questions concisely:\n"
            "1. What in this experiment would take longer than 2 weeks to run? "
            "(Flag specific steps, not general concerns.)\n"
            "2. Are any datasets or codebases referenced NOT publicly available "
            "or likely broken/undocumented?\n"
            "3. What is the single biggest implementation blocker a PhD student "
            "would hit in the first month?\n"
            "4. State ONE concrete change that would make this clearly reproducible "
            "within the envelope. (If already reproducible, say so explicitly.)\n\n"
            "Be specific. If the experiment is already feasible, confirm it."
        )
        try:
            feasibility_critique = call_llm(feasibility_prompt, model, client,
                                            temperature=0.4)
        except Exception:
            feasibility_critique = "Experiment appears feasible within the stated resource envelope."

        # ── Step 7: novelty elevation ──────────────────────────────────────────
        # Prevents the feasibility constraint from collapsing ideas into
        # incremental variations on existing work. Explicitly sharpens the claim.
        novelty_prompt = (
            f"You are a novelty reviewer who rejects incremental papers.\n\n"
            f"Research topic: {topic}\n"
            f"Core hypothesis: {selected_hyp}\n\n"
            f"Proposed experiment:\n{draft}\n\n"
            f"Recent related work:\n{sota_context or 'Not available.'}\n\n"
            "Answer concisely:\n"
            "1. Is the core claim GENUINELY SURPRISING? Would an expert say "
            "'I didn't know that' if the experiment succeeded? (Yes/No + one sentence why.)\n"
            "2. What is the sharpest, most non-obvious VERSION of this insight? "
            "(Do NOT require frontier compute — keep it within the resource envelope.)\n"
            "3. Which existing paper does this most closely resemble, and what is the "
            "ONE crisp differentiator that makes this idea distinct from that paper?\n\n"
            "Give 2–3 sentences of targeted novelty improvement that the author should "
            "incorporate into the final version."
        )
        try:
            novelty_elevation = call_llm(novelty_prompt, model, client, temperature=0.6)
        except Exception:
            novelty_elevation = "The core hypothesis appears novel relative to existing work."

        # ── Step 8: final revision ─────────────────────────────────────────────
        revise_prompt = (
            f"Research topic: {topic}\n\n"
            f"Core hypothesis: {selected_hyp}\n\n"
            f"Experimental design:\n{draft}\n\n"
            f"Feasibility critique (address these gaps):\n{feasibility_critique}\n\n"
            f"Novelty elevation (incorporate this sharpening):\n{novelty_elevation}\n\n"
            f"{context_reminder}\n"
            "Write the FINAL research idea. Requirements:\n"
            "  • Hypothesis clearly stated and falsifiable\n"
            "  • All datasets, baselines, and metrics are named explicitly and "
            "are publicly available\n"
            "  • Compute budget fits within a university GPU cluster (≤8×A100, ≤2 weeks)\n"
            "  • The novelty claim is sharp: state what prior paper this is closest to "
            "and what is the one crisp thing this contribution adds\n"
            "  • A PhD student could start this experiment within one month\n"
            + IDEA_FORMAT
        )
        try:
            return call_llm(revise_prompt, model, client, temperature)
        except Exception:
            return draft if draft else f"Research idea about {topic}: {selected_hyp}"


GENERATOR = S17_r1Generator()
