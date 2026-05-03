import sys
import os
import concurrent.futures

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from systems.base import IdeaGenerator, DEFAULT_MODEL, call_llm, IDEA_FORMAT


class S22Generator(IdeaGenerator):
    VERSION = "S22"
    DESCRIPTION = (
        "Hypothesis-first adversarial loop with structured hypothesis representation: "
        "hypotheses are generated in explicit MECHANISM/CONDITION/FALSIFICATION format, "
        "attacked in parallel, selected by structural strength, then built into a "
        "concrete experimental idea with multi-perspective critique."
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

        # ── Step 1: generate 5 structured falsifiable hypotheses ─────────────
        # Key change: force hypotheses into MECHANISM/CONDITION/FALSIFICATION
        # structure at generation time, not post-hoc extraction.
        hyp_prompt = (
            "Research topic: " + topic + context_block + "\n"
            "Generate exactly 5 sharp, falsifiable scientific hypotheses about this topic.\n\n"
            "CRITICAL: Each hypothesis MUST be written in this exact 3-part structure:\n"
            "  MECHANISM: <The specific causal or algorithmic process that produces the effect>\n"
            "  CONDITION: <The precise setting/regime where this mechanism operates>\n"
            "  FALSIFICATION: <The single experimental result that would disprove this>\n\n"
            "Requirements:\n"
            "- MECHANISM must name a specific process, not just an outcome\n"
            "  BAD: 'model performance degrades'\n"
            "  GOOD: 'attention entropy collapse in early layers causes gradient starvation'\n"
            "- CONDITION must bound the claim to a testable regime\n"
            "  BAD: 'in large models'\n"
            "  GOOD: 'in transformer models with >1B parameters trained on <100B tokens'\n"
            "- FALSIFICATION must be a specific measurable threshold\n"
            "  BAD: 'if the method does not improve results'\n"
            "  GOOD: 'if accuracy gap between method and baseline is <2% on MMLU'\n"
            "- Each hypothesis must be non-obvious and NOT directly supported by the related work above\n\n"
            "Format:\n"
            "H1:\n"
            "MECHANISM: <text>\n"
            "CONDITION: <text>\n"
            "FALSIFICATION: <text>\n\n"
            "H2:\n"
            "MECHANISM: <text>\n"
            "CONDITION: <text>\n"
            "FALSIFICATION: <text>\n\n"
            "H3:\n"
            "MECHANISM: <text>\n"
            "CONDITION: <text>\n"
            "FALSIFICATION: <text>\n\n"
            "H4:\n"
            "MECHANISM: <text>\n"
            "CONDITION: <text>\n"
            "FALSIFICATION: <text>\n\n"
            "H5:\n"
            "MECHANISM: <text>\n"
            "CONDITION: <text>\n"
            "FALSIFICATION: <text>"
        )
        try:
            hyp_raw = call_llm(hyp_prompt, model, client, temperature)
        except Exception as e:
            hyp_raw = (
                "H1:\n"
                "MECHANISM: Standard approaches fail due to distribution shift in the input space\n"
                "CONDITION: When training and test distributions differ by >0.3 KL divergence\n"
                "FALSIFICATION: If baseline accuracy drops <5% under distribution shift on standard benchmarks\n"
            )

        # Parse structured hypotheses
        def parse_structured_hypotheses(raw_text):
            """Parse hypotheses in MECHANISM/CONDITION/FALSIFICATION format."""
            hypotheses = []
            # Split on H1:, H2:, etc.
            import re
            blocks = re.split(r'\nH\d+:', '\n' + raw_text)
            for block in blocks:
                block = block.strip()
                if not block:
                    continue
                mech = cond = fals = ""
                for line in block.split('\n'):
                    line = line.strip()
                    if line.upper().startswith('MECHANISM:'):
                        mech = line.split(':', 1)[1].strip()
                    elif line.upper().startswith('CONDITION:'):
                        cond = line.split(':', 1)[1].strip()
                    elif line.upper().startswith('FALSIFICATION:'):
                        fals = line.split(':', 1)[1].strip()
                if mech and cond and fals:
                    hypotheses.append({
                        'mechanism': mech,
                        'condition': cond,
                        'falsification': fals,
                        'full': f"MECHANISM: {mech}\nCONDITION: {cond}\nFALSIFICATION: {fals}"
                    })
            return hypotheses

        hypotheses = parse_structured_hypotheses(hyp_raw)

        # Fallback: if parsing fails, try to extract any content
        if not hypotheses:
            fallback_mech = f"Standard approaches to {topic} fail due to a specific structural limitation"
            fallback_cond = "when evaluated on held-out test sets with distribution shift"
            fallback_fals = "if the proposed method shows <3% improvement over baselines on standard benchmarks"
            hypotheses = [{
                'mechanism': fallback_mech,
                'condition': fallback_cond,
                'falsification': fallback_fals,
                'full': f"MECHANISM: {fallback_mech}\nCONDITION: {fallback_cond}\nFALSIFICATION: {fallback_fals}"
            }]

        hypotheses = hypotheses[:5]

        # ── Step 2: parallel adversarial attacks on each structured hypothesis ─
        def attack_hypothesis(hyp_dict: dict) -> dict:
            hyp_full = hyp_dict['full']
            attack_prompt = (
                "Research topic: " + topic + "\n\n"
                "Hypothesis:\n" + hyp_full + "\n\n"
                "You are an adversarial critic. Attack this hypothesis on EACH dimension:\n"
                "1. Mechanism flaw: Is the named mechanism actually responsible for the claimed effect, "
                "or is there a confound? Give a concrete counterexample.\n"
                "2. Condition boundary: Does the condition actually bound the claim, or does the mechanism "
                "operate (or fail) outside these bounds in ways that invalidate the hypothesis?\n"
                "3. Falsification weakness: Is the falsification criterion actually decisive, or could "
                "the hypothesis be true even if this criterion is met/failed?\n"
                "4. Prior art: Does existing literature already establish or refute this mechanism?\n\n"
                "Then write a REVISED hypothesis that survives these attacks, keeping the same structure:\n"
                "REVISED MECHANISM: <text>\n"
                "REVISED CONDITION: <text>\n"
                "REVISED FALSIFICATION: <text>"
            )
            try:
                attack_text = call_llm(attack_prompt, model, client, temperature=0.6)
                # Parse revised hypothesis
                rev_mech = rev_cond = rev_fals = ""
                for line in attack_text.split('\n'):
                    line = line.strip()
                    if line.upper().startswith('REVISED MECHANISM:'):
                        rev_mech = line.split(':', 1)[1].strip()
                    elif line.upper().startswith('REVISED CONDITION:'):
                        rev_cond = line.split(':', 1)[1].strip()
                    elif line.upper().startswith('REVISED FALSIFICATION:'):
                        rev_fals = line.split(':', 1)[1].strip()

                if rev_mech and rev_cond and rev_fals:
                    revised = {
                        'mechanism': rev_mech,
                        'condition': rev_cond,
                        'falsification': rev_fals,
                        'full': f"MECHANISM: {rev_mech}\nCONDITION: {rev_cond}\nFALSIFICATION: {rev_fals}"
                    }
                else:
                    # Keep original if revision parsing fails
                    revised = hyp_dict.copy()

                return {'original': hyp_dict, 'attack': attack_text, 'revised': revised}
            except Exception as e:
                return {'original': hyp_dict, 'attack': str(e), 'revised': hyp_dict}

        attack_results = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            futures = [executor.submit(attack_hypothesis, h) for h in hypotheses]
            for f in futures:
                try:
                    attack_results.append(f.result(timeout=120))
                except Exception as e:
                    # Fallback: use first hypothesis unchanged
                    if hypotheses:
                        attack_results.append({
                            'original': hypotheses[0],
                            'attack': f"timeout: {e}",
                            'revised': hypotheses[0]
                        })

        # Ensure we have at least one result
        if not attack_results and hypotheses:
            attack_results = [{'original': hypotheses[0], 'attack': '', 'revised': hypotheses[0]}]

        # ── Step 3: select the strongest surviving hypothesis ─────────────────
        # Selection prompt emphasizes structural quality of the revised hypothesis
        pairs_str = ""
        for i, result in enumerate(attack_results):
            orig = result.get('original', {})
            rev = result.get('revised', {})
            atk = result.get('attack', '')
            pairs_str += (
                f"\n--- Hypothesis {i+1} ---\n"
                f"ORIGINAL:\n{orig.get('full', '')}\n\n"
                f"ATTACK SUMMARY:\n{atk[:400]}\n\n"
                f"REVISED:\n{rev.get('full', '')}\n\n"
            )

        select_prompt = (
            "Research topic: " + topic + "\n\n"
            "Below are " + str(len(attack_results)) + " hypotheses, each attacked and revised:\n"
            + pairs_str + "\n"
            "Select the ONE hypothesis whose REVISED form is strongest. Evaluate each revised hypothesis by:\n"
            "1. MECHANISM SPECIFICITY: Does it name a precise causal/algorithmic process (not just an outcome)?\n"
            "2. CONDITION PRECISION: Is the regime exactly bounded (specific model sizes, data scales, settings)?\n"
            "3. FALSIFICATION DECISIVENESS: Is the falsification criterion a single clean measurable threshold?\n"
            "4. NOVELTY: Is this mechanism non-obvious and not directly established by prior work?\n\n"
            "IMPORTANT: Prefer hypotheses that are specific and feasible over those that are ambitious but vague.\n\n"
            "Respond with:\n"
            "SELECTED: <number 1-" + str(len(attack_results)) + ">\n"
            "REVISED MECHANISM: <exact text from the selected revised hypothesis>\n"
            "REVISED CONDITION: <exact text from the selected revised hypothesis>\n"
            "REVISED FALSIFICATION: <exact text from the selected revised hypothesis>\n"
            "REASONING: <1-2 sentences on why this hypothesis has the strongest structure>"
        )
        try:
            selection_raw = call_llm(select_prompt, model, client, temperature=0.3)
        except Exception as e:
            selection_raw = ""

        # Extract selected structured hypothesis
        selected_hyp = attack_results[0].get('revised', attack_results[0].get('original', hypotheses[0] if hypotheses else {}))

        if selection_raw:
            sel_mech = sel_cond = sel_fals = ""
            for line in selection_raw.strip().split('\n'):
                line = line.strip()
                if line.upper().startswith('REVISED MECHANISM:'):
                    sel_mech = line.split(':', 1)[1].strip()
                elif line.upper().startswith('REVISED CONDITION:'):
                    sel_cond = line.split(':', 1)[1].strip()
                elif line.upper().startswith('REVISED FALSIFICATION:'):
                    sel_fals = line.split(':', 1)[1].strip()

            if sel_mech and sel_cond and sel_fals:
                selected_hyp = {
                    'mechanism': sel_mech,
                    'condition': sel_cond,
                    'falsification': sel_fals,
                    'full': f"MECHANISM: {sel_mech}\nCONDITION: {sel_cond}\nFALSIFICATION: {sel_fals}"
                }

        # Ensure selected_hyp is a dict with required keys
        if not isinstance(selected_hyp, dict):
            selected_hyp = {'mechanism': str(selected_hyp), 'condition': 'standard setting', 'falsification': 'if baseline matches proposed method', 'full': str(selected_hyp)}
        for key in ('mechanism', 'condition', 'falsification', 'full'):
            if key not in selected_hyp:
                selected_hyp[key] = ''

        # ── Step 4: construct experimental idea around the structured hypothesis ─
        context_reminder = (
            "\nExisting work to differentiate from:\n" + sota_context + "\n"
            if sota_context else ""
        )

        construct_prompt = (
            "Research topic: " + topic + "\n"
            + context_reminder + "\n"
            "Core hypothesis to test:\n"
            "  MECHANISM: " + selected_hyp.get('mechanism', '') + "\n"
            "  CONDITION: " + selected_hyp.get('condition', '') + "\n"
            "  FALSIFICATION: " + selected_hyp.get('falsification', '') + "\n\n"
            "Design a concrete research experiment to PROVE OR DISPROVE this hypothesis.\n\n"
            "Structure your response around the hypothesis components:\n\n"
            "MECHANISM TEST: How will you directly observe or measure the named mechanism? "
            "What instrumentation, probing, or analysis reveals whether the mechanism is active?\n\n"
            "CONDITION SETUP: What specific experimental setup instantiates the stated condition? "
            "Name exact datasets, model sizes, and training configurations.\n\n"
            "FALSIFICATION EXPERIMENT: What is the single key experiment whose result definitively "
            "proves or disproves the hypothesis? Name the baseline, the metric, and the threshold "
            "from the FALSIFICATION field above.\n\n"
            "METHOD: What technical approach tests the hypothesis? Describe the key components "
            "in enough detail to implement. Be specific about architecture choices, training "
            "procedures, or algorithmic steps.\n\n"
            "Write 3-4 paragraphs total. Be direct and specific — name actual datasets and baselines."
        )
        try:
            draft = call_llm(construct_prompt, model, client, temperature)
        except Exception as e:
            draft = (
                f"Research idea for '{topic}' testing the hypothesis that "
                f"{selected_hyp.get('mechanism', 'a specific mechanism')} "
                f"under {selected_hyp.get('condition', 'specific conditions')}. "
                f"The key falsification experiment: {selected_hyp.get('falsification', 'comparing against baselines')}."
            )

        # ── Step 5: multi-perspective critique ───────────────────────────────
        hyp_summary = (
            "Mechanism: " + selected_hyp.get('mechanism', '') + "\n"
            "Condition: " + selected_hyp.get('condition', '') + "\n"
            "Falsification: " + selected_hyp.get('falsification', '')
        )

        exp_prompt = (
            "You are a hard-nosed experimentalist reviewing a research proposal about '" + topic + "'.\n\n"
            "Hypothesis being tested:\n" + hyp_summary + "\n\n"
            "Proposed experiment:\n" + draft + "\n\n"
            "Give 2-3 sharp criticisms focusing purely on experimental feasibility:\n"
            "- Is the mechanism actually measurable with the proposed instrumentation?\n"
            "- Does the experimental setup correctly instantiate the stated condition?\n"
            "- Is the falsification criterion operationalized precisely enough to be decisive?\n"
            "- What controls are missing? What will fail in practice?"
        )
        try:
            critique_exp = call_llm(exp_prompt, model, client, temperature=0.5)
        except Exception:
            critique_exp = "No experimental critique available."

        theory_prompt = (
            "You are a rigorous theorist reviewing a research proposal about '" + topic + "'.\n\n"
            "Hypothesis being tested:\n" + hyp_summary + "\n\n"
            "Proposed experiment:\n" + draft + "\n\n"
            "Give 2-3 sharp criticisms focusing on theoretical grounding:\n"
            "- Is the named mechanism theoretically sound? Does it contradict known results?\n"
            "- Is the novelty claim justified given existing literature?\n"
            "- Are the underlying assumptions stated and defensible?\n"
            "- Does the falsification criterion actually test the mechanism, or a proxy?"
        )
        try:
            critique_theory = call_llm(theory_prompt, model, client, temperature=0.5)
        except Exception:
            critique_theory = "No theoretical critique available."

        skeptic_prompt = (
            "You are a skeptical reviewer who has seen many overhyped proposals about '" + topic + "'.\n\n"
            "Hypothesis being tested:\n" + hyp_summary + "\n\n"
            "Proposed experiment:\n" + draft + "\n\n"
            "Give 2-3 sharp criticisms:\n"
            "- Why is the stated mechanism probably not the real cause of the effect?\n"
            "- What is the most likely negative result and why?\n"
            "- Is the falsification criterion set at a threshold that makes the hypothesis unfalsifiable in practice?\n"
            "- Does the scientific payoff justify the effort even if it succeeds?"
        )
        try:
            critique_skeptic = call_llm(skeptic_prompt, model, client, temperature=0.5)
        except Exception:
            critique_skeptic = "No skeptic critique available."

        synthesis_prompt = (
            "Three reviewers critiqued a research idea about '" + topic + "' testing the hypothesis:\n"
            + hyp_summary + "\n\n"
            "Experimentalist:\n" + critique_exp + "\n\n"
            "Theorist:\n" + critique_theory + "\n\n"
            "Skeptic:\n" + critique_skeptic + "\n\n"
            "Synthesize these into the 3 most important actionable improvements "
            "the author must make. Focus on: (1) making the mechanism test more direct, "
            "(2) tightening the experimental conditions, (3) sharpening the falsification criterion. "
            "Be concise and prioritized."
        )
        try:
            synthesis = call_llm(synthesis_prompt, model, client, temperature=0.3)
        except Exception:
            synthesis = "Improve mechanism specificity, tighten experimental conditions, sharpen falsification criterion."

        # ── Step 6: final revision ────────────────────────────────────────────
        revise_prompt = (
            "Research topic: " + topic + "\n\n"
            "Core hypothesis:\n" + hyp_summary + "\n\n"
            "Experimental design:\n" + draft + "\n\n"
            "Key improvements required:\n" + synthesis + "\n"
            + context_reminder + "\n"
            "Write the final, improved version of the research idea.\n"
            "Requirements:\n"
            "- Open with a clear statement of the mechanism being tested and why it matters\n"
            "- Specify the exact experimental condition (model sizes, data scales, settings)\n"
            "- Name specific datasets, baselines, and metrics throughout\n"
            "- Include the key falsification experiment with its precise threshold\n"
            "- Explain what a positive result means and what a negative result means\n"
            "- Be direct and specific — no hedging language\n"
            + IDEA_FORMAT
        )
        try:
            final = call_llm(revise_prompt, model, client, temperature)
            if final and final.strip():
                return final
        except Exception:
            pass

        # Fallback: return draft if final revision fails
        if draft and draft.strip():
            return draft

        return (
            f"Research idea about {topic}: Testing whether "
            f"{selected_hyp.get('mechanism', 'a specific mechanism')} "
            f"under {selected_hyp.get('condition', 'specific conditions')}. "
            f"Falsification: {selected_hyp.get('falsification', 'comparison with baselines')}."
        )


GENERATOR = S22Generator()
