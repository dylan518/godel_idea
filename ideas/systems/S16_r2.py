"""S16_r2: Cross-Domain Isomorphism Transfer + Adversarial Loop.

Extends S15's adversarial hypothesis loop with a cross-domain isomorphism step
BEFORE hypothesis generation. Given the research topic, the system identifies
2-3 analogous problems in different fields that have already been solved, extracts
their solution structures, and instantiates those structures back into the target
domain as seeded hypotheses. This produces ideas that are:
  - Structurally grounded (borrowed from a proven solution in another field)
  - Genuinely novel in the target domain (no one has made the transfer yet)
  - Experimentally concrete (inherited specificity from the source solution)

Pipeline:
1. SOTA RETRIEVAL (external): fetch 5 related papers
2. CROSS-DOMAIN ANALOGS (1 call): identify 2-3 isomorphic solved problems in other fields
3. ISOMORPHISM TRANSFER (1 call): extract solution structures and instantiate as hypotheses
4. ADVERSARIAL ATTACKS (5 calls, parallel): attack each transferred hypothesis
5. HYPOTHESIS SELECTION (1 call): pick strongest survivor
6. IDEA CONSTRUCTION (1 call): design full experiment around hypothesis
7. MULTI-PERSPECTIVE CRITIQUE (4 calls: exp + theory + skeptic + synthesis)
8. FINAL REVISION (1 call): incorporate critique

LLM calls: 1 + 1 + 5 + 1 + 1 + 4 + 1 = ~14
"""

import sys
import os
import concurrent.futures

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from systems.base import IdeaGenerator, DEFAULT_MODEL, call_llm, IDEA_FORMAT


class S16_r2Generator(IdeaGenerator):
    VERSION = "S16_r2"
    DESCRIPTION = (
        "Cross-domain isomorphism transfer + adversarial loop: find analogous solved "
        "problems in other fields, transfer solution structures as hypotheses, attack "
        "each in parallel, select strongest survivor, build experimental idea around "
        "falsification, then multi-perspective critique."
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
        # ── Step 0: retrieve SOTA context ───────────────────────────────────
        try:
            import os as _os
            sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
            from retrieval import get_topic_context
            sota_context = get_topic_context(topic, n=5)
        except Exception:
            sota_context = ""

        context_block = f"\n\n{sota_context}\n" if sota_context else ""

        # ── Step 1: identify cross-domain analogs ────────────────────────────
        # Find 2-3 isomorphic problems in DIFFERENT fields that have already been solved.
        # The goal: harvest proven solution structures and transfer them to the target domain.
        analog_prompt = (
            f"Research topic: {topic}{context_block}\n"
            "Identify 2-3 analogous problems in DIFFERENT scientific or engineering fields "
            "that are structurally isomorphic to this topic and have already been solved.\n\n"
            "Structural isomorphism means: the underlying mathematical or causal structure "
            "is similar even if the surface domain is completely different.\n\n"
            "Examples of good analogies:\n"
            "  - 'zero-shot generalization in NLP' ↔ 'species distribution shift in ecology' "
            "(both: model trained on one distribution must generalize to unseen environments)\n"
            "  - 'neural network compression' ↔ 'signal compression in telecommunications' "
            "(both: preserve information while reducing representation size)\n"
            "  - 'reward hacking in RL' ↔ 'Goodhart's law in economics' "
            "(both: optimization pressure on a proxy metric corrupts the true objective)\n\n"
            "For each analog, output:\n"
            "ANALOG 1:\n"
            "  Source field: <field name>\n"
            "  Analogous problem: <1 sentence — what problem was solved in that field>\n"
            "  Solution structure: <2-3 sentences — the KEY mechanism/principle that solved it, "
            "abstracted away from field-specific details>\n"
            "  Structural mapping: <1 sentence — how the source problem maps to the target topic>\n\n"
            "ANALOG 2:\n"
            "  ...\n\n"
            "ANALOG 3:\n"
            "  ..."
        )
        try:
            analogs_raw = call_llm(analog_prompt, model, client, temperature=0.9)
        except Exception as e:
            analogs_raw = f"ANALOG 1:\n  Source field: Control theory\n  Analogous problem: Robust control under model uncertainty\n  Solution structure: Decompose uncertainty into structured and unstructured parts; design controllers that are robust to the former and adaptive to the latter.\n  Structural mapping: {topic} faces similar structure-versus-noise decomposition challenges."

        # ── Step 2: transfer solution structures as hypotheses ───────────────
        transfer_prompt = (
            f"Research topic: {topic}{context_block}\n"
            f"Cross-domain analogs with proven solution structures:\n{analogs_raw}\n\n"
            "Your task: for each analog, INSTANTIATE the solution structure as a concrete, "
            "falsifiable hypothesis about the target research topic.\n\n"
            "Rules for transfer:\n"
            "  - Keep the MECHANISM from the source field, replace all domain-specific terms\n"
            "  - The hypothesis must be non-obvious: it should NOT be a simple restatement of "
            "the original problem. The novelty comes from the structural transfer.\n"
            "  - Each hypothesis must be testable: name what experiment would disprove it\n"
            "  - Add 1-2 hypotheses that combine insights from multiple analogs (optional bonus)\n\n"
            "Also generate 1-2 additional hypotheses that are NOT derived from analogs — "
            "purely field-internal hypotheses that challenge conventional wisdom about the topic.\n\n"
            "Format:\n"
            "H1: <transferred hypothesis from ANALOG 1>\n"
            "H2: <transferred hypothesis from ANALOG 2>\n"
            "H3: <transferred hypothesis from ANALOG 3 OR cross-analog synthesis>\n"
            "H4: <field-internal challenge hypothesis>\n"
            "H5: <field-internal challenge hypothesis>"
        )
        try:
            hyp_raw = call_llm(transfer_prompt, model, client, temperature=temperature)
        except Exception as e:
            # Fallback: generate hypotheses directly without transfer
            hyp_raw = f"H1: Standard approaches to {topic} fail because of distribution shift between training and deployment."

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
        hypotheses = hypotheses[:5]  # cap at 5

        # ── Step 3: parallel adversarial attacks on each hypothesis ─────────
        def attack_hypothesis(hyp: str) -> str:
            attack_prompt = (
                f"Research topic: {topic}\n\n"
                f"Hypothesis: {hyp}\n\n"
                "You are an adversarial critic. Attack this hypothesis on EACH of these dimensions:\n"
                "1. Assumption violation: What unstated assumption does this rely on? Give a concrete counterexample.\n"
                "2. Dataset bias: What dataset artifact could make this appear true without actually being true?\n"
                "3. Theoretical gap: What known result from the literature contradicts or undermines this?\n"
                "4. Practical limitation: What makes this hypothesis untestable or too expensive to test?\n\n"
                "Then, given these attacks, write a REVISED hypothesis that survives them:\n"
                "Revised: <1-2 sentence refined hypothesis that addresses the above attacks>"
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

        # ── Step 4: select the strongest surviving hypothesis ────────────────
        pairs_str = ""
        for i, (hyp, attack) in enumerate(zip(hypotheses, attacks)):
            pairs_str += f"\n--- Hypothesis {i+1} ---\nOriginal: {hyp}\nCritique+Revision:\n{attack}\n"

        select_prompt = (
            f"Research topic: {topic}\n\n"
            f"Below are {len(hypotheses)} hypotheses (some transferred from cross-domain analogs, "
            f"some field-internal), each attacked by an adversarial critic and refined:\n{pairs_str}\n"
            "Select the ONE hypothesis (by number) whose REVISED form is:\n"
            "- Most concrete and specific (names mechanisms, not just outcomes)\n"
            "- Most falsifiable (clearest path to a disproof experiment)\n"
            "- Most novel (least covered by standard literature — cross-domain transfers "
            "are especially valuable if they hold up to the attacks)\n\n"
            "Respond with:\n"
            "SELECTED: <number 1-5>\n"
            "REVISED HYPOTHESIS: <copy the revised hypothesis text exactly>\n"
            "REASONING: <1-2 sentences on why this is the strongest>"
        )
        try:
            selection_raw = call_llm(select_prompt, model, client, temperature=0.3)
        except Exception as e:
            selection_raw = f"SELECTED: 1\nREVISED HYPOTHESIS: {hypotheses[0]}\nREASONING: fallback"

        # Extract the selected revised hypothesis
        selected_hyp = hypotheses[0]  # fallback
        for line in selection_raw.strip().split("\n"):
            if line.strip().upper().startswith("REVISED HYPOTHESIS:"):
                candidate = line.split(":", 1)[1].strip()
                if candidate:
                    selected_hyp = candidate
                    break

        # ── Step 5: construct experimental idea around the hypothesis ────────
        context_reminder = (
            f"\nExisting work to differentiate from:\n{sota_context}\n"
            if sota_context else ""
        )
        construct_prompt = (
            f"Research topic: {topic}\n"
            f"{context_reminder}\n"
            f"Core hypothesis to test: {selected_hyp}\n\n"
            "This hypothesis was derived via cross-domain structural transfer — "
            "it borrows a proven solution mechanism from another field. "
            "Leverage that provenance: reference the source analogy when explaining "
            "why the mechanism should work, then ground it in the specific constraints "
            "of the target field.\n\n"
            "Design a concrete research experiment to PROVE OR DISPROVE this hypothesis.\n"
            "The experiment must:\n"
            "- Name specific datasets you will use (not just 'standard benchmarks')\n"
            "- Name specific baselines you will compare against\n"
            "- Define quantitative success metrics (what number proves it, what number disproves it)\n"
            "- Describe the key technical method in enough detail to implement\n"
            "- Explain precisely what result would falsify the hypothesis\n\n"
            "Write 3-4 paragraphs. Be direct and specific."
        )
        try:
            draft = call_llm(construct_prompt, model, client, temperature)
        except Exception as e:
            draft = f"Research idea for '{topic}' based on hypothesis: {selected_hyp}"

        # ── Step 6: multi-perspective critique ───────────────────────────────
        exp_prompt = (
            f"You are a hard-nosed experimentalist reviewing a research proposal about '{topic}'.\n\n"
            f"Hypothesis being tested: {selected_hyp}\n\n"
            f"Proposed experiment:\n{draft}\n\n"
            "Give 2-3 sharp criticisms focusing purely on experimental feasibility: "
            "Can these experiments actually be run? Are the measurements well-defined? "
            "What controls are missing? What will fail in practice?"
        )
        try:
            critique_exp = call_llm(exp_prompt, model, client, temperature=0.5)
        except Exception:
            critique_exp = "No experimental critique available."

        theory_prompt = (
            f"You are a rigorous theorist reviewing a research proposal about '{topic}'.\n\n"
            f"Hypothesis being tested: {selected_hyp}\n\n"
            f"Proposed experiment:\n{draft}\n\n"
            "Give 2-3 sharp criticisms focusing purely on theoretical grounding: "
            "Is the novelty claim justified? Does it overlap with known results? "
            "Are the underlying assumptions stated and defensible?"
        )
        try:
            critique_theory = call_llm(theory_prompt, model, client, temperature=0.5)
        except Exception:
            critique_theory = "No theoretical critique available."

        skeptic_prompt = (
            f"You are a skeptical reviewer who has seen many overhyped proposals about '{topic}'.\n\n"
            f"Hypothesis being tested: {selected_hyp}\n\n"
            f"Proposed experiment:\n{draft}\n\n"
            "Give 2-3 sharp criticisms focusing on: why this probably won't work, "
            "what the likely negative result is, and whether the scientific payoff "
            "justifies the effort even if it succeeds."
        )
        try:
            critique_skeptic = call_llm(skeptic_prompt, model, client, temperature=0.5)
        except Exception:
            critique_skeptic = "No skeptic critique available."

        synthesis_prompt = (
            f"Three reviewers critiqued a research idea about '{topic}' testing the hypothesis:\n"
            f"'{selected_hyp}'\n\n"
            f"Experimentalist:\n{critique_exp}\n\n"
            f"Theorist:\n{critique_theory}\n\n"
            f"Skeptic:\n{critique_skeptic}\n\n"
            "Synthesize these into the 3 most important actionable improvements "
            "the author must make. Be concise and prioritized."
        )
        try:
            synthesis = call_llm(synthesis_prompt, model, client, temperature=0.3)
        except Exception:
            synthesis = "Improve specificity, add baselines, clarify falsification criteria."

        # ── Step 7: final revision ────────────────────────────────────────────
        revise_prompt = (
            f"Research topic: {topic}\n\n"
            f"Core hypothesis: {selected_hyp}\n\n"
            f"Experimental design:\n{draft}\n\n"
            f"Key improvements required:\n{synthesis}\n"
            f"{context_reminder}\n"
            "Write the final, improved version of the research idea. "
            "Ensure the hypothesis is clearly stated, the falsification experiment is concrete, "
            "and all datasets/baselines/metrics are named explicitly. "
            "If the hypothesis was derived from a cross-domain analogy, briefly mention "
            "the structural parallel — this strengthens the novelty claim."
            + IDEA_FORMAT
        )
        try:
            return call_llm(revise_prompt, model, client, temperature)
        except Exception:
            return draft if draft else f"Research idea about {topic}: {selected_hyp}"


GENERATOR = S16_r2Generator()
