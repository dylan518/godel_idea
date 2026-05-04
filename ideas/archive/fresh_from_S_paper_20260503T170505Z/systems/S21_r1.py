import sys
import os
import concurrent.futures

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from systems.base import IdeaGenerator, DEFAULT_MODEL, call_llm, IDEA_FORMAT


class S21_r1Generator(IdeaGenerator):
    VERSION = "S21_r1"
    DESCRIPTION = (
        "Hypothesis-first adversarial loop with mechanistic hypothesis generation: "
        "hypotheses are constructed from theoretical observations about WHY existing "
        "methods fail, not from technique names. Each hypothesis embeds a causal "
        "mechanism at generation time, not retrofitted post-hoc."
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

        # ── Step 1: generate 5 mechanistic falsifiable hypotheses ─────────────
        # Key fix: hypotheses are derived from causal observations about failure
        # modes of existing methods, not from technique names. Each hypothesis
        # must embed X (mechanism), Y (predicted effect), and Z (theoretical reason).
        hyp_prompt = (
            f"Research topic: {topic}{context_block}\n"
            "Your task: generate exactly 5 mechanistic, falsifiable scientific hypotheses.\n\n"
            "PROCESS (follow this for each hypothesis):\n"
            "Step A — Identify a specific failure mode of current methods:\n"
            "  Ask: 'Where exactly do existing approaches break down, and what is the structural reason?'\n"
            "  Do NOT start from a technique name. Start from an observation about why something fails.\n\n"
            "Step B — Derive a causal mechanism:\n"
            "  Ask: 'What is the underlying cause of this failure?' Name the mechanism explicitly.\n\n"
            "Step C — State a testable prediction:\n"
            "  Ask: 'If the mechanism is real, what specific measurable outcome follows?'\n"
            "  Name the experiment that would FALSIFY this (i.e., what result would prove it wrong).\n\n"
            "Each hypothesis MUST follow this template exactly:\n"
            "[Specific phenomenon X] causes [predicted measurable effect Y] because [theoretical mechanism Z]; "
            "falsified if [concrete experimental contrast that would disprove it].\n\n"
            "Requirements:\n"
            "- X must be a specific, named property of existing systems (not 'current methods')\n"
            "- Y must be a quantifiable outcome (not 'worse performance')\n"
            "- Z must be a theoretical reason, not a restatement of X or Y\n"
            "- The falsification condition must name a specific experimental contrast\n"
            "- The hypothesis must NOT be directly supported by the related work above\n\n"
            "Format:\n"
            "H1: <hypothesis following the template>\n"
            "H2: <hypothesis following the template>\n"
            "H3: <hypothesis following the template>\n"
            "H4: <hypothesis following the template>\n"
            "H5: <hypothesis following the template>"
        )
        try:
            hyp_raw = call_llm(hyp_prompt, model, client, temperature)
        except Exception as e:
            hyp_raw = (
                f"H1: Static optimization objectives in {topic} cause degraded performance "
                f"under distribution shift because they minimize expected loss over the training "
                f"distribution rather than worst-case subgroup loss; falsified if a model trained "
                f"with distributionally robust objectives shows no improvement on held-out subgroups."
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
                f"they optimize average-case loss; falsified if robust training shows no benefit."
            ]
        hypotheses = hypotheses[:5]

        # ── Step 2: parallel adversarial attacks ──────────────────────────────
        def attack_hypothesis(hyp: str) -> str:
            attack_prompt = (
                f"Research topic: {topic}\n\n"
                f"Hypothesis: {hyp}\n\n"
                "You are an adversarial critic. Attack this hypothesis on EACH dimension:\n"
                "1. Mechanism validity: Is the theoretical mechanism Z actually the cause, "
                "or is there a confound? Give a concrete alternative explanation.\n"
                "2. Measurement validity: Can Y actually be measured as described? "
                "What operationalization problem makes this ambiguous?\n"
                "3. Scope: Under what conditions does the mechanism Z NOT apply? "
                "Name a specific regime where the hypothesis is clearly false.\n"
                "4. Prior work: Does existing literature already establish or refute this mechanism? "
                "Name a specific result that undermines the novelty claim.\n\n"
                "Then write a REVISED hypothesis that:\n"
                "- Preserves the causal structure (X causes Y because Z)\n"
                "- Narrows the scope to where the mechanism genuinely holds\n"
                "- Makes Z more theoretically precise\n"
                "- Makes the falsification condition more operationally specific\n\n"
                "Revised: <revised hypothesis following the X causes Y because Z; "
                "falsified if [contrast] template>"
            )
            try:
                return call_llm(attack_prompt, model, client, temperature=0.6)
            except Exception as e:
                return f"Revised: {hyp}"

        attacks = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            futures = [executor.submit(attack_hypothesis, h) for h in hypotheses]
            for f in futures:
                try:
                    attacks.append(f.result(timeout=120))
                except Exception as e:
                    attacks.append(f"Revised: (timeout)")

        # Ensure attacks list matches hypotheses length
        while len(attacks) < len(hypotheses):
            attacks.append("Revised: (no attack)")

        # ── Step 3: select the strongest surviving hypothesis ──────────────────
        pairs_str = ""
        for i, (hyp, attack) in enumerate(zip(hypotheses, attacks)):
            pairs_str += f"\n--- Hypothesis {i+1} ---\nOriginal: {hyp}\nCritique+Revision:\n{attack}\n"

        select_prompt = (
            f"Research topic: {topic}\n\n"
            f"Below are {len(hypotheses)} hypotheses, each attacked and revised:\n{pairs_str}\n"
            "Select the ONE hypothesis whose REVISED form best satisfies ALL of:\n"
            "1. MECHANISM SPECIFICITY: Z (the theoretical reason) is precise enough to derive "
            "a quantitative prediction — not just 'reduces redundancy' but names a specific "
            "mathematical property or structural constraint\n"
            "2. FALSIFIABILITY: The experimental contrast that would disprove it is concrete — "
            "names specific conditions, not just 'ablation study'\n"
            "3. NOVELTY: The mechanism Z is not directly established in standard literature\n"
            "4. SCOPE VALIDITY: The revised hypothesis correctly limits its claims to where "
            "the mechanism genuinely applies\n\n"
            "Respond with:\n"
            "SELECTED: <number 1-5>\n"
            "REVISED HYPOTHESIS: <copy the revised hypothesis text exactly from after 'Revised:'>\n"
            "MECHANISM: <extract just the Z component — the theoretical reason>\n"
            "REASONING: <1-2 sentences on why this mechanism is the most theoretically grounded>"
        )
        try:
            selection_raw = call_llm(select_prompt, model, client, temperature=0.3)
        except Exception as e:
            selection_raw = f"SELECTED: 1\nREVISED HYPOTHESIS: {hypotheses[0] if hypotheses else topic}\nMECHANISM: unknown\nREASONING: fallback"

        # Extract selected hypothesis and mechanism
        selected_hyp = hypotheses[0] if hypotheses else f"Novel approach to {topic}"
        selected_mechanism = ""
        for line in (selection_raw or "").strip().split("\n"):
            stripped = line.strip()
            if stripped.upper().startswith("REVISED HYPOTHESIS:"):
                candidate = stripped.split(":", 1)[1].strip()
                if candidate:
                    selected_hyp = candidate
            elif stripped.upper().startswith("MECHANISM:"):
                candidate = stripped.split(":", 1)[1].strip()
                if candidate:
                    selected_mechanism = candidate

        # ── Step 4: construct experimental idea around the mechanism ───────────
        # The construction prompt is anchored to the mechanism Z, not just the
        # hypothesis surface form. This forces the experimental design to operationalize
        # the causal mechanism rather than just test a vague prediction.
        context_reminder = (
            f"\nExisting work to differentiate from:\n{sota_context}\n"
            if sota_context else ""
        )

        mechanism_block = (
            f"\nCore causal mechanism to operationalize: {selected_mechanism}\n"
            if selected_mechanism else ""
        )

        construct_prompt = (
            f"Research topic: {topic}\n"
            f"{context_reminder}\n"
            f"Hypothesis to test: {selected_hyp}\n"
            f"{mechanism_block}\n"
            "Design a concrete experiment to PROVE OR DISPROVE this hypothesis.\n\n"
            "CRITICAL: Your experimental design must directly operationalize the causal mechanism "
            "(the 'because Z' part of the hypothesis). The method should work BY exploiting or "
            "demonstrating Z — not just correlate with it.\n\n"
            "Required elements:\n"
            "1. MECHANISM OPERATIONALIZATION: How does your method directly instantiate or test Z? "
            "What specific architectural choice, loss function, or algorithm embodies Z?\n"
            "2. DATASETS: Name 3+ specific datasets (not 'standard benchmarks'). "
            "Explain why each dataset is appropriate for testing Z specifically.\n"
            "3. BASELINES: Name 4+ specific baselines. For each, explain why it cannot "
            "exploit mechanism Z (this justifies why your approach should outperform it).\n"
            "4. METRICS: Define the primary metric that directly measures whether Z is operating. "
            "Define secondary metrics. State the specific threshold that would confirm vs. refute.\n"
            "5. FALSIFICATION EXPERIMENT: Describe the specific ablation or control condition "
            "that would prove the hypothesis false if Z is not the true cause.\n\n"
            "Write 4 paragraphs. Be direct and specific. No hedging."
        )
        try:
            draft = call_llm(construct_prompt, model, client, temperature)
        except Exception as e:
            draft = f"Research idea for '{topic}' based on hypothesis: {selected_hyp}"

        # ── Step 5: multi-perspective critique ────────────────────────────────
        exp_prompt = (
            f"You are a hard-nosed experimentalist reviewing a research proposal about '{topic}'.\n\n"
            f"Hypothesis: {selected_hyp}\n"
            f"Causal mechanism being tested: {selected_mechanism}\n\n"
            f"Proposed experiment:\n{draft}\n\n"
            "Give 2-3 sharp criticisms: Are the datasets appropriate for isolating mechanism Z? "
            "Are the baselines truly unable to exploit Z, or could they be adapted to? "
            "Is the falsification experiment actually a clean test of Z vs. confounds?"
        )
        try:
            critique_exp = call_llm(exp_prompt, model, client, temperature=0.5)
        except Exception:
            critique_exp = "No experimental critique available."

        theory_prompt = (
            f"You are a rigorous theorist reviewing a research proposal about '{topic}'.\n\n"
            f"Hypothesis: {selected_hyp}\n"
            f"Causal mechanism: {selected_mechanism}\n\n"
            f"Proposed experiment:\n{draft}\n\n"
            "Give 2-3 sharp criticisms: Is mechanism Z theoretically sound? "
            "Does it contradict known results? Is the prediction from Z actually derivable, "
            "or is the connection between Z and Y hand-wavy? What would a formal treatment require?"
        )
        try:
            critique_theory = call_llm(theory_prompt, model, client, temperature=0.5)
        except Exception:
            critique_theory = "No theoretical critique available."

        skeptic_prompt = (
            f"You are a skeptical reviewer about '{topic}'.\n\n"
            f"Hypothesis: {selected_hyp}\n"
            f"Causal mechanism: {selected_mechanism}\n\n"
            f"Proposed experiment:\n{draft}\n\n"
            "Give 2-3 sharp criticisms: Even if the experiment succeeds, does it prove Z caused Y, "
            "or just that they correlate? What alternative mechanism could produce the same results? "
            "Is the scientific payoff worth the effort if Z turns out to be a minor factor?"
        )
        try:
            critique_skeptic = call_llm(skeptic_prompt, model, client, temperature=0.5)
        except Exception:
            critique_skeptic = "No skeptic critique available."

        synthesis_prompt = (
            f"Three reviewers critiqued a research idea about '{topic}'.\n"
            f"Hypothesis: '{selected_hyp}'\n"
            f"Mechanism: '{selected_mechanism}'\n\n"
            f"Experimentalist:\n{critique_exp}\n\n"
            f"Theorist:\n{critique_theory}\n\n"
            f"Skeptic:\n{critique_skeptic}\n\n"
            "Synthesize into the 3 most important actionable improvements. "
            "Prioritize fixes that strengthen the causal argument (that Z is the true mechanism) "
            "over fixes that just add more experiments."
        )
        try:
            synthesis = call_llm(synthesis_prompt, model, client, temperature=0.3)
        except Exception:
            synthesis = "Strengthen causal argument, add mechanism-specific baselines, clarify falsification."

        # ── Step 6: final revision ─────────────────────────────────────────────
        revise_prompt = (
            f"Research topic: {topic}\n\n"
            f"Core hypothesis: {selected_hyp}\n"
            f"Causal mechanism (Z): {selected_mechanism}\n\n"
            f"Experimental design:\n{draft}\n\n"
            f"Key improvements required:\n{synthesis}\n"
            f"{context_reminder}\n"
            "Write the final, improved research idea. Requirements:\n"
            "1. State the hypothesis clearly with X, Y, and Z explicit\n"
            "2. Explain how the proposed METHOD directly operationalizes mechanism Z\n"
            "3. Name all datasets, baselines, and metrics explicitly\n"
            "4. Describe the falsification experiment — what result would prove the hypothesis wrong\n"
            "5. Explain why existing methods cannot achieve this because they lack mechanism Z\n"
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


GENERATOR = S21_r1Generator()
