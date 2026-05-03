"""S24_r1: Complexity-Resistant Adversarial Loop.

Root-cause fix for complexity creep in adversarial hypothesis selection.
The original adversarial loop (S12) rewards hypotheses that *survive* attacks
by becoming more hedged and multi-mechanism — the most attack-resistant
hypotheses are those with escape hatches. This produces over-specified
hypotheses that the construction step faithfully translates into multi-component
architectures, which the judge penalises on feasibility and experimental clarity.

Architecture:

1. HYPOTHESIS GENERATION: 5 single-mechanism falsifiable hypotheses.
   Each must express "X causes Y because Z" — one mechanism, one claim.

2. SIMPLICITY-PRESERVING ADVERSARIAL ATTACK (parallel, 5 calls):
   The adversary attacks COMPLEXITY AND MECHANISM-STACKING — not assumption
   gaps that get plugged with more components. Revised hypotheses must be
   SIMPLER, not more hedged.

3. HYPOTHESIS SELECTION (1 call): Prefer the SIMPLEST, most directly testable
   revised hypothesis. Explicitly penalise multi-component and conditionally-
   specified hypotheses.

4. PRE-COMMIT FALSIFICATION THRESHOLD (1 call): Before constructing the
   experiment, lock in: single-sentence mechanism claim, primary metric,
   named dataset, named baseline, and specific quantitative rejection threshold.
   Construction cannot diverge from this pre-committed target.

5. IDEA CONSTRUCTION (1 call): Build around the pre-committed threshold.
   Hard constraint: ONE method, no sub-components, no auxiliary objectives.
   Experiment must test the threshold — not showcase the method broadly.

6. MULTI-PERSPECTIVE CRITIQUE (4 calls): Experimentalist, theorist, skeptic
   critique + synthesis. Synthesis explicitly forbids "add more components".

7. FINAL REVISION (1 call): Single-mechanism, theoretical justification,
   explicit falsification criterion.

LLM calls: 1 (hyp) + 5 (attacks) + 1 (select) + 1 (threshold) + 1 (construct)
           + 3 (critique) + 1 (synthesis) + 1 (revise) = 14
"""

import sys
import os
import concurrent.futures

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from systems.base import IdeaGenerator, DEFAULT_MODEL, call_llm, IDEA_FORMAT


class S24_r1Generator(IdeaGenerator):
    VERSION = "S24_r1"
    DESCRIPTION = (
        "Complexity-resistant adversarial loop: adversary attacks mechanism-stacking "
        "not assumption gaps; selection favours simplicity over robustness; pre-committed "
        "falsification threshold locks in metric/dataset/baseline before construction; "
        "construction hard-constrained to one method."
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
            sota_context = get_topic_context(topic, n=5)
        except Exception:
            sota_context = ""

        context_block = f"\n\n{sota_context}\n" if sota_context else ""

        # ── Step 1: generate 5 single-mechanism falsifiable hypotheses ────────
        # Constraint added: every hypothesis must be ONE mechanism (X causes Y
        # because Z), preventing multi-component claims from entering the loop.
        hyp_prompt = (
            f"Research topic: {topic}{context_block}\n"
            "Generate exactly 5 sharp, falsifiable scientific hypotheses about this topic.\n\n"
            "CRITICAL CONSTRAINT: each hypothesis must express EXACTLY ONE mechanism.\n"
            "The required form is:\n"
            "  '[Specific intervention/property] improves [specific outcome] because "
            "[one named causal reason]'\n\n"
            "Requirements for each hypothesis:\n"
            "- ONE mechanism only — no compound claims, no sub-components, no 'and also'\n"
            "- Specific: name the mechanism, not just 'a better method'\n"
            "- Falsifiable: name the single experiment that would prove it false\n"
            "- Non-obvious: should not be directly supported by the related work above\n"
            "- Concise: 1-2 sentences max\n\n"
            "Format:\n"
            "H1: <hypothesis>\n"
            "H2: <hypothesis>\n"
            "H3: <hypothesis>\n"
            "H4: <hypothesis>\n"
            "H5: <hypothesis>"
        )
        try:
            hyp_raw = call_llm(hyp_prompt, model, client, temperature)
        except Exception as e:
            hyp_raw = (
                f"H1: Standard approaches to {topic} fail because of distribution shift "
                f"in the input space."
            )

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

        # ── Step 2: simplicity-preserving adversarial attack ─────────────────
        # KEY CHANGE: adversary attacks COMPLEXITY AND MECHANISM-STACKING, not
        # just assumption gaps. Revised hypotheses must be SIMPLER — the attack
        # strips away everything except the minimal load-bearing claim.
        def attack_hypothesis(hyp: str) -> str:
            attack_prompt = (
                f"Research topic: {topic}\n\n"
                f"Hypothesis: {hyp}\n\n"
                "You are a SIMPLICITY CRITIC. Your sole job is to find the minimal "
                "load-bearing claim in this hypothesis.\n\n"
                "Attack it on these dimensions:\n"
                "1. MECHANISM COUNT: Does this require more than ONE mechanism to work? "
                "Identify any hidden sub-components, auxiliary systems, or compound claims.\n"
                "2. HEDGING: Does the hypothesis use escape hatches like 'combined with', "
                "'when also using', 'jointly with', or conditional dependencies that make "
                "it untestable in a single experiment?\n"
                "3. THEORETICAL GROUNDING: Is there a clear named reason WHY this mechanism "
                "produces the claimed effect, or is it asserted by analogy?\n"
                "4. TESTABILITY: Can this be falsified by ONE controlled experiment, or does "
                "testing require building multiple interacting systems first?\n\n"
                "Then write a REVISED hypothesis that:\n"
                "- Strips to the single most load-bearing mechanism\n"
                "- Removes all compound claims, auxiliary components, and hedges\n"
                "- States the causal reason directly (X works because Y — one reason only)\n"
                "- Can be falsified by a single controlled experiment\n\n"
                "Revised: <1-2 sentence simplified hypothesis>"
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

        # ── Step 3: select the SIMPLEST, most directly testable hypothesis ────
        # KEY CHANGE: selection criterion is simplicity and directness, not
        # robustness. Explicitly penalise multi-component hypotheses.
        pairs_str = ""
        for i, (hyp, attack) in enumerate(zip(hypotheses, attacks)):
            pairs_str += (
                f"\n--- Hypothesis {i+1} ---\n"
                f"Original: {hyp}\n"
                f"Simplicity critique + Simplified:\n{attack}\n"
            )

        select_prompt = (
            f"Research topic: {topic}\n\n"
            f"Below are {len(hypotheses)} hypotheses, each attacked by a simplicity critic "
            f"and stripped to their minimal core claim:\n{pairs_str}\n"
            "Select the ONE hypothesis whose SIMPLIFIED form is:\n"
            "- SIMPLEST: expresses exactly one mechanism with no sub-components\n"
            "- MOST DIRECT: states X causes Y because Z without hedges or conditions\n"
            "- MOST TESTABLE: falsifiable by a single controlled experiment\n"
            "- MOST NOVEL: the specific causal mechanism is not already established in "
            "standard literature\n\n"
            "PENALISE hypotheses that: require building multiple interacting systems, "
            "use compound mechanisms, or are only testable after solving several sub-problems "
            "first.\n\n"
            "Respond with:\n"
            "SELECTED: <number 1-5>\n"
            "REVISED HYPOTHESIS: <copy the simplified hypothesis text exactly>\n"
            "REASONING: <1-2 sentences on why this is the simplest and most directly testable>"
        )
        try:
            selection_raw = call_llm(select_prompt, model, client, temperature=0.3)
        except Exception as e:
            selection_raw = (
                f"SELECTED: 1\nREVISED HYPOTHESIS: {hypotheses[0]}\nREASONING: fallback"
            )

        selected_hyp = hypotheses[0]  # fallback
        for line in selection_raw.strip().split("\n"):
            if line.strip().upper().startswith("REVISED HYPOTHESIS:"):
                candidate = line.split(":", 1)[1].strip()
                if candidate:
                    selected_hyp = candidate
                    break

        # ── Step 3.5: pre-commit falsification threshold ──────────────────────
        # NEW STEP: lock in the quantitative threshold BEFORE constructing the
        # experiment. This prevents the construction step from building a system
        # to showcase the method broadly rather than test the specific threshold.
        threshold_prompt = (
            f"Research topic: {topic}\n\n"
            f"Hypothesis to test: {selected_hyp}\n\n"
            "Before designing the experiment, commit to each of the following. "
            "Be specific — no placeholders.\n\n"
            "1. MECHANISM: One sentence. What X causes Y because Z. No hedges.\n"
            "2. METRIC: The single most important measurable outcome "
            "(e.g., classification accuracy on held-out set, logical error rate per shot, "
            "task success rate). Name it precisely.\n"
            "3. DATASET: The specific named dataset or benchmark for the primary test. "
            "Real names only (e.g., MiniGrid-FourRooms, IBM Quantum ibmq_manila, "
            "scRNA-seq dataset GSE..., MiniHack, Atari-100k).\n"
            "4. BASELINE: The specific named comparison method. No 'prior work' or "
            "'existing methods' — name the actual algorithm or paper.\n"
            "5. THRESHOLD: The specific quantitative change that would REJECT the hypothesis "
            "— e.g., 'if improvement over [baseline] on [dataset] is less than X%, reject'.\n\n"
            "Format exactly as:\n"
            "MECHANISM: <one sentence>\n"
            "METRIC: <metric name and measurement procedure>\n"
            "DATASET: <specific named dataset>\n"
            "BASELINE: <specific named method>\n"
            "THRESHOLD: <if [condition], reject the hypothesis>"
        )
        try:
            threshold_raw = call_llm(threshold_prompt, model, client, temperature=0.3)
        except Exception as e:
            threshold_raw = (
                f"MECHANISM: {selected_hyp}\n"
                "METRIC: primary task metric\n"
                "DATASET: standard benchmark\n"
                "BASELINE: prior SOTA\n"
                "THRESHOLD: if no improvement over baseline, reject"
            )

        # ── Step 4: construct experiment around the pre-committed threshold ───
        context_reminder = (
            f"\nExisting work to differentiate from:\n{sota_context}\n"
            if sota_context else ""
        )
        construct_prompt = (
            f"Research topic: {topic}\n"
            f"{context_reminder}\n"
            f"Core hypothesis: {selected_hyp}\n\n"
            f"Pre-committed falsification criteria (non-negotiable):\n{threshold_raw}\n\n"
            "Design a concrete research experiment to PROVE OR DISPROVE this hypothesis.\n\n"
            "HARD CONSTRAINTS — violating any of these disqualifies the proposal:\n"
            "- ONE method/mechanism only. No multi-component architectures. "
            "No auxiliary objectives, sub-modules, or interacting systems.\n"
            "- The experiment must directly test the pre-committed threshold — "
            "not showcase the method broadly.\n"
            "- Use the datasets, baselines, and metrics already committed above "
            "(you may add at most one secondary dataset for ablation, no more).\n"
            "- State explicitly what result would FALSIFY the hypothesis using the "
            "threshold already defined.\n\n"
            "WHAT TO INCLUDE:\n"
            "- The single technical method, described concretely enough to implement\n"
            "- Theoretical justification: WHY this mechanism should cause the claimed "
            "effect (cite a principle, theorem, or known result if possible)\n"
            "- Controlled experiment design: what varies, what is held constant, "
            "what confounds are ruled out\n"
            "- The specific falsification criterion matching the pre-committed threshold\n\n"
            "Write 3-4 paragraphs. Be direct and specific. "
            "Do not add extra sub-systems or secondary objectives."
        )
        try:
            draft = call_llm(construct_prompt, model, client, temperature)
        except Exception as e:
            draft = f"Research idea for '{topic}' based on hypothesis: {selected_hyp}"

        # ── Step 5: multi-perspective critique ───────────────────────────────
        exp_prompt = (
            f"You are a hard-nosed experimentalist reviewing a research proposal "
            f"about '{topic}'.\n\n"
            f"Hypothesis being tested: {selected_hyp}\n\n"
            f"Pre-committed threshold:\n{threshold_raw}\n\n"
            f"Proposed experiment:\n{draft}\n\n"
            "Give 2-3 sharp criticisms focusing on experimental feasibility: "
            "Can these experiments actually be run with available tools? "
            "Are the measurements well-defined? What controls are missing? "
            "Does the experiment actually test the pre-committed threshold, "
            "or does it drift into showcasing the method broadly?"
        )
        try:
            critique_exp = call_llm(exp_prompt, model, client, temperature=0.5)
        except Exception:
            critique_exp = "No experimental critique available."

        theory_prompt = (
            f"You are a rigorous theorist reviewing a research proposal about '{topic}'.\n\n"
            f"Hypothesis being tested: {selected_hyp}\n\n"
            f"Proposed experiment:\n{draft}\n\n"
            "Give 2-3 sharp criticisms focusing on theoretical grounding: "
            "Is the mechanism claim (X causes Y because Z) theoretically justified — "
            "is there a principle or prior result that supports WHY this should work? "
            "Does it overlap with known results? Are the underlying assumptions "
            "stated and defensible?"
        )
        try:
            critique_theory = call_llm(theory_prompt, model, client, temperature=0.5)
        except Exception:
            critique_theory = "No theoretical critique available."

        skeptic_prompt = (
            f"You are a skeptical reviewer who has seen many overhyped proposals "
            f"about '{topic}'.\n\n"
            f"Hypothesis being tested: {selected_hyp}\n\n"
            f"Proposed experiment:\n{draft}\n\n"
            "Give 2-3 sharp criticisms: why this probably won't work, what the "
            "likely negative result is, and whether the scientific payoff justifies "
            "the effort even if it succeeds."
        )
        try:
            critique_skeptic = call_llm(skeptic_prompt, model, client, temperature=0.5)
        except Exception:
            critique_skeptic = "No skeptic critique available."

        synthesis_prompt = (
            f"Three reviewers critiqued a research idea about '{topic}' testing:\n"
            f"'{selected_hyp}'\n\n"
            f"Experimentalist:\n{critique_exp}\n\n"
            f"Theorist:\n{critique_theory}\n\n"
            f"Skeptic:\n{critique_skeptic}\n\n"
            "Synthesize into the 3 most important actionable improvements. "
            "Be concise and prioritised.\n\n"
            "IMPORTANT: do NOT suggest adding more components, sub-systems, or auxiliary "
            "methods. Focus only on sharpening the single-mechanism claim, "
            "strengthening its theoretical justification, and tightening the "
            "experimental falsification criteria."
        )
        try:
            synthesis = call_llm(synthesis_prompt, model, client, temperature=0.3)
        except Exception:
            synthesis = (
                "Sharpen the mechanism claim, add theoretical justification, "
                "tighten the falsification threshold."
            )

        # ── Step 6: final revision ────────────────────────────────────────────
        revise_prompt = (
            f"Research topic: {topic}\n\n"
            f"Core hypothesis: {selected_hyp}\n\n"
            f"Pre-committed falsification criteria:\n{threshold_raw}\n\n"
            f"Experimental design:\n{draft}\n\n"
            f"Key improvements required:\n{synthesis}\n"
            f"{context_reminder}\n"
            "Write the final, improved version of the research idea.\n\n"
            "Requirements:\n"
            "- State the hypothesis as a single-mechanism claim (X causes Y because Z)\n"
            "- Provide theoretical justification for WHY the mechanism causes the effect\n"
            "- Name the exact datasets, baselines, and metrics from the pre-committed "
            "criteria\n"
            "- State the specific falsification threshold explicitly\n"
            "- ONE method only — do not add sub-components or auxiliary systems\n"
            + IDEA_FORMAT
        )
        try:
            return call_llm(revise_prompt, model, client, temperature)
        except Exception:
            return draft if draft else f"Research idea about {topic}: {selected_hyp}"


GENERATOR = S24_r1Generator()
