import sys
import os
import concurrent.futures

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from systems.base import IdeaGenerator, DEFAULT_MODEL, call_llm, IDEA_FORMAT


class S22_r2Generator(IdeaGenerator):
    VERSION = "S22_r2"
    DESCRIPTION = (
        "Hypothesis-first adversarial loop with mechanism-grounded hypothesis generation: "
        "hypotheses are generated with explicit WHY-EXISTING-METHODS-FAIL reasoning baked in, "
        "attacked in parallel, selected by structural strength and existing-method differentiation, "
        "then built into a concrete experimental idea with multi-perspective critique."
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
        # Key change from S22_r1: hypotheses must explain WHY existing methods fail
        # (the mechanism that existing methods miss), not just what a new method does.
        # This grounds the mechanism in the gap between existing and proposed work.
        hyp_prompt = (
            "Research topic: " + topic + context_block + "\n"
            "Generate exactly 5 sharp, falsifiable scientific hypotheses about this topic.\n\n"
            "CRITICAL REQUIREMENT: Each hypothesis must explain WHY EXISTING METHODS FAIL "
            "by naming the specific mechanism they miss, AND what exploiting that mechanism enables.\n\n"
            "Each hypothesis MUST use this exact 4-part structure:\n"
            "  EXISTING_FLAW: <The specific structural/algorithmic limitation of current methods>\n"
            "  MECHANISM: <The causal process that existing methods miss, which causes their failure>\n"
            "  EXPLOIT: <How a method that directly targets this mechanism would work differently>\n"
            "  FALSIFICATION: <The single measurable result that would disprove this hypothesis>\n\n"
            "Requirements:\n"
            "- EXISTING_FLAW must name a specific limitation, not just 'existing methods are suboptimal'\n"
            "  BAD: 'current methods don't generalize well'\n"
            "  GOOD: 'current methods use fixed-width attention that cannot adapt to input complexity'\n"
            "- MECHANISM must name the causal process behind the flaw\n"
            "  BAD: 'this causes poor performance'\n"
            "  GOOD: 'attention entropy collapses in early layers when sequence complexity exceeds a threshold, "
            "starving later layers of gradient signal'\n"
            "- EXPLOIT must describe a concrete technical approach that targets the mechanism\n"
            "  BAD: 'a better method would improve this'\n"
            "  GOOD: 'dynamically routing tokens to variable-depth attention stacks based on per-token entropy "
            "prevents collapse while maintaining throughput'\n"
            "- FALSIFICATION must be a specific measurable threshold\n"
            "  BAD: 'if the method does not improve results'\n"
            "  GOOD: 'if accuracy gap between method and baseline is <2% on MMLU at 1B parameter scale'\n"
            "- Each hypothesis must be non-obvious and NOT directly supported by the related work above\n\n"
            "Format:\n"
            "H1:\n"
            "EXISTING_FLAW: <text>\n"
            "MECHANISM: <text>\n"
            "EXPLOIT: <text>\n"
            "FALSIFICATION: <text>\n\n"
            "H2:\n"
            "EXISTING_FLAW: <text>\n"
            "MECHANISM: <text>\n"
            "EXPLOIT: <text>\n"
            "FALSIFICATION: <text>\n\n"
            "H3:\n"
            "EXISTING_FLAW: <text>\n"
            "MECHANISM: <text>\n"
            "EXPLOIT: <text>\n"
            "FALSIFICATION: <text>\n\n"
            "H4:\n"
            "EXISTING_FLAW: <text>\n"
            "MECHANISM: <text>\n"
            "EXPLOIT: <text>\n"
            "FALSIFICATION: <text>\n\n"
            "H5:\n"
            "EXISTING_FLAW: <text>\n"
            "MECHANISM: <text>\n"
            "EXPLOIT: <text>\n"
            "FALSIFICATION: <text>"
        )
        try:
            hyp_raw = call_llm(hyp_prompt, model, client, temperature)
        except Exception:
            hyp_raw = (
                "H1:\n"
                "EXISTING_FLAW: Standard approaches use fixed representations that cannot adapt to distribution shift\n"
                "MECHANISM: When training and test distributions differ, fixed representations fail to capture the "
                "shifted feature manifold, causing systematic prediction errors\n"
                "EXPLOIT: A method that dynamically updates representations based on test-time statistics would "
                "directly target this manifold mismatch\n"
                "FALSIFICATION: If accuracy gap between method and baseline is <3% under distribution shift on standard benchmarks\n"
            )

        # Parse structured hypotheses with 4-part format
        def parse_structured_hypotheses(raw_text):
            import re
            hypotheses = []
            blocks = re.split(r'\nH\d+:', '\n' + raw_text)
            for block in blocks:
                block = block.strip()
                if not block:
                    continue
                flaw = mech = exploit = fals = ""
                for line in block.split('\n'):
                    line = line.strip()
                    upper = line.upper()
                    if upper.startswith('EXISTING_FLAW:'):
                        flaw = line.split(':', 1)[1].strip()
                    elif upper.startswith('MECHANISM:'):
                        mech = line.split(':', 1)[1].strip()
                    elif upper.startswith('EXPLOIT:'):
                        exploit = line.split(':', 1)[1].strip()
                    elif upper.startswith('FALSIFICATION:'):
                        fals = line.split(':', 1)[1].strip()
                # Accept if we have at least mechanism and falsification
                if mech and fals:
                    if not flaw:
                        flaw = f"Existing approaches to {topic} have a structural limitation"
                    if not exploit:
                        exploit = f"A method targeting the mechanism: {mech}"
                    hypotheses.append({
                        'flaw': flaw,
                        'mechanism': mech,
                        'exploit': exploit,
                        'falsification': fals,
                        'full': (
                            f"EXISTING_FLAW: {flaw}\n"
                            f"MECHANISM: {mech}\n"
                            f"EXPLOIT: {exploit}\n"
                            f"FALSIFICATION: {fals}"
                        )
                    })
            return hypotheses

        hypotheses = parse_structured_hypotheses(hyp_raw)

        if not hypotheses:
            fallback_flaw = f"Standard approaches to {topic} use fixed representations that cannot adapt"
            fallback_mech = f"The fixed representation fails to capture distribution-specific structure"
            fallback_exploit = f"A method that dynamically adapts representations at test time"
            fallback_fals = f"if the proposed method shows <3% improvement over baselines on standard benchmarks"
            hypotheses = [{
                'flaw': fallback_flaw,
                'mechanism': fallback_mech,
                'exploit': fallback_exploit,
                'falsification': fallback_fals,
                'full': (
                    f"EXISTING_FLAW: {fallback_flaw}\n"
                    f"MECHANISM: {fallback_mech}\n"
                    f"EXPLOIT: {fallback_exploit}\n"
                    f"FALSIFICATION: {fallback_fals}"
                )
            }]

        hypotheses = hypotheses[:5]

        # ── Step 2: parallel adversarial attacks on each structured hypothesis ─
        def attack_hypothesis(hyp_dict: dict) -> dict:
            hyp_full = hyp_dict.get('full', '')
            attack_prompt = (
                "Research topic: " + topic + "\n\n"
                "Hypothesis:\n" + hyp_full + "\n\n"
                "You are an adversarial critic. Attack this hypothesis on EACH dimension:\n"
                "1. Flaw validity: Is the named existing flaw actually a real limitation, "
                "or do existing methods already handle this? Give a concrete counterexample.\n"
                "2. Mechanism accuracy: Is the named mechanism actually responsible for the flaw, "
                "or is there a different underlying cause? Give a specific alternative.\n"
                "3. Exploit feasibility: Does the proposed exploit actually target the mechanism, "
                "or does it just describe a vague improvement? What would break in practice?\n"
                "4. Falsification decisiveness: Is the falsification criterion actually decisive, "
                "or could the hypothesis be true even if this criterion is met/failed?\n\n"
                "Then write a REVISED hypothesis that survives these attacks, keeping the same structure:\n"
                "REVISED EXISTING_FLAW: <text>\n"
                "REVISED MECHANISM: <text>\n"
                "REVISED EXPLOIT: <text>\n"
                "REVISED FALSIFICATION: <text>"
            )
            try:
                attack_text = call_llm(attack_prompt, model, client, temperature=0.6)
                rev_flaw = rev_mech = rev_exploit = rev_fals = ""
                for line in (attack_text or "").split('\n'):
                    line = line.strip()
                    upper = line.upper()
                    if upper.startswith('REVISED EXISTING_FLAW:'):
                        rev_flaw = line.split(':', 1)[1].strip()
                    elif upper.startswith('REVISED MECHANISM:'):
                        rev_mech = line.split(':', 1)[1].strip()
                    elif upper.startswith('REVISED EXPLOIT:'):
                        rev_exploit = line.split(':', 1)[1].strip()
                    elif upper.startswith('REVISED FALSIFICATION:'):
                        rev_fals = line.split(':', 1)[1].strip()

                if rev_mech and rev_fals:
                    if not rev_flaw:
                        rev_flaw = hyp_dict.get('flaw', '')
                    if not rev_exploit:
                        rev_exploit = hyp_dict.get('exploit', '')
                    revised = {
                        'flaw': rev_flaw,
                        'mechanism': rev_mech,
                        'exploit': rev_exploit,
                        'falsification': rev_fals,
                        'full': (
                            f"EXISTING_FLAW: {rev_flaw}\n"
                            f"MECHANISM: {rev_mech}\n"
                            f"EXPLOIT: {rev_exploit}\n"
                            f"FALSIFICATION: {rev_fals}"
                        )
                    }
                else:
                    revised = hyp_dict.copy()

                return {'original': hyp_dict, 'attack': attack_text or '', 'revised': revised}
            except Exception as e:
                return {'original': hyp_dict, 'attack': str(e), 'revised': hyp_dict}

        attack_results = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            futures = [executor.submit(attack_hypothesis, h) for h in hypotheses]
            for f in futures:
                try:
                    attack_results.append(f.result(timeout=120))
                except Exception as e:
                    if hypotheses:
                        attack_results.append({
                            'original': hypotheses[0],
                            'attack': f"timeout: {e}",
                            'revised': hypotheses[0]
                        })

        if not attack_results and hypotheses:
            attack_results = [{'original': hypotheses[0], 'attack': '', 'revised': hypotheses[0]}]

        # ── Step 3: select the strongest surviving hypothesis ─────────────────
        # Selection now explicitly rewards hypotheses that explain why existing methods fail
        pairs_str = ""
        for i, result in enumerate(attack_results):
            orig = result.get('original', {})
            rev = result.get('revised', {})
            atk = result.get('attack', '')
            pairs_str += (
                f"\n--- Hypothesis {i+1} ---\n"
                f"ORIGINAL:\n{orig.get('full', '')}\n\n"
                f"ATTACK SUMMARY:\n{(atk or '')[:400]}\n\n"
                f"REVISED:\n{rev.get('full', '')}\n\n"
            )

        n_results = len(attack_results)
        select_prompt = (
            "Research topic: " + topic + "\n\n"
            "Below are " + str(n_results) + " hypotheses, each attacked and revised:\n"
            + pairs_str + "\n"
            "Select the ONE hypothesis whose REVISED form is strongest. Evaluate each revised hypothesis by:\n"
            "1. FLAW SPECIFICITY: Does it name a real, specific limitation of existing methods "
            "(not a generic 'existing methods are suboptimal' claim)?\n"
            "2. MECHANISM PRECISION: Does it name the exact causal process behind the flaw "
            "(not just a description of the outcome)?\n"
            "3. EXPLOIT CONCRETENESS: Does it describe a technically specific approach that "
            "directly targets the mechanism (not just 'a better method')?\n"
            "4. FALSIFICATION DECISIVENESS: Is the falsification criterion a single clean "
            "measurable threshold that would actually distinguish the hypothesis from alternatives?\n"
            "5. NOVELTY: Is the named mechanism non-obvious and not directly established by prior work?\n\n"
            "IMPORTANT: Prefer hypotheses where the MECHANISM explains WHY existing methods fail "
            "in a way that directly motivates the EXPLOIT. The best hypothesis has a logical chain: "
            "flaw → mechanism → exploit → falsification.\n\n"
            "Respond with:\n"
            "SELECTED: <number 1-" + str(n_results) + ">\n"
            "REVISED EXISTING_FLAW: <exact text from the selected revised hypothesis>\n"
            "REVISED MECHANISM: <exact text from the selected revised hypothesis>\n"
            "REVISED EXPLOIT: <exact text from the selected revised hypothesis>\n"
            "REVISED FALSIFICATION: <exact text from the selected revised hypothesis>\n"
            "REASONING: <1-2 sentences on why this hypothesis has the strongest logical chain>"
        )
        try:
            selection_raw = call_llm(select_prompt, model, client, temperature=0.3)
        except Exception:
            selection_raw = ""

        # Extract selected structured hypothesis
        selected_hyp = attack_results[0].get('revised', attack_results[0].get('original', hypotheses[0] if hypotheses else {}))

        if selection_raw:
            sel_flaw = sel_mech = sel_exploit = sel_fals = ""
            for line in selection_raw.strip().split('\n'):
                line = line.strip()
                upper = line.upper()
                if upper.startswith('REVISED EXISTING_FLAW:'):
                    sel_flaw = line.split(':', 1)[1].strip()
                elif upper.startswith('REVISED MECHANISM:'):
                    sel_mech = line.split(':', 1)[1].strip()
                elif upper.startswith('REVISED EXPLOIT:'):
                    sel_exploit = line.split(':', 1)[1].strip()
                elif upper.startswith('REVISED FALSIFICATION:'):
                    sel_fals = line.split(':', 1)[1].strip()

            if sel_mech and sel_fals:
                if not sel_flaw:
                    sel_flaw = selected_hyp.get('flaw', '') if isinstance(selected_hyp, dict) else ''
                if not sel_exploit:
                    sel_exploit = selected_hyp.get('exploit', '') if isinstance(selected_hyp, dict) else ''
                selected_hyp = {
                    'flaw': sel_flaw,
                    'mechanism': sel_mech,
                    'exploit': sel_exploit,
                    'falsification': sel_fals,
                    'full': (
                        f"EXISTING_FLAW: {sel_flaw}\n"
                        f"MECHANISM: {sel_mech}\n"
                        f"EXPLOIT: {sel_exploit}\n"
                        f"FALSIFICATION: {sel_fals}"
                    )
                }

        if not isinstance(selected_hyp, dict):
            selected_hyp = {
                'flaw': str(selected_hyp),
                'mechanism': 'a specific structural limitation',
                'exploit': 'a targeted method',
                'falsification': 'if baseline matches proposed method',
                'full': str(selected_hyp)
            }
        for key in ('flaw', 'mechanism', 'exploit', 'falsification', 'full'):
            if key not in selected_hyp:
                selected_hyp[key] = ''

        # ── Step 4: construct experimental idea around the structured hypothesis ─
        # Construction prompt is organized around the 4-part hypothesis structure
        context_reminder = (
            "\nExisting work to differentiate from:\n" + sota_context + "\n"
            if sota_context else ""
        )

        construct_prompt = (
            "Research topic: " + topic + "\n"
            + context_reminder + "\n"
            "Core hypothesis to test:\n"
            "  EXISTING_FLAW: " + selected_hyp.get('flaw', '') + "\n"
            "  MECHANISM: " + selected_hyp.get('mechanism', '') + "\n"
            "  EXPLOIT: " + selected_hyp.get('exploit', '') + "\n"
            "  FALSIFICATION: " + selected_hyp.get('falsification', '') + "\n\n"
            "Design a concrete research experiment to PROVE OR DISPROVE this hypothesis.\n\n"
            "Structure your response around the hypothesis components:\n\n"
            "WHY EXISTING METHODS FAIL: Explain the specific flaw and mechanism in concrete terms. "
            "Which existing papers/methods exhibit this flaw? What experimental evidence shows the flaw?\n\n"
            "PROPOSED METHOD: Describe the technical approach that exploits the mechanism. "
            "Name specific architectural choices, algorithmic steps, or training procedures. "
            "Be concrete enough to implement.\n\n"
            "KEY EXPERIMENT: What is the single decisive experiment? Name the exact baseline, "
            "dataset, metric, and threshold from the FALSIFICATION field. "
            "What does a positive result prove? What does a negative result prove?\n\n"
            "EXPERIMENTAL SETUP: Name exact datasets, model sizes, training configurations, "
            "and baselines. Be specific — no generic placeholders.\n\n"
            "Write 3-4 paragraphs total. Be direct — name actual methods, datasets, and numbers."
        )
        try:
            draft = call_llm(construct_prompt, model, client, temperature)
        except Exception:
            draft = (
                f"Research idea for '{topic}' testing the hypothesis that "
                f"{selected_hyp.get('mechanism', 'a specific mechanism')} "
                f"causes existing methods to fail because {selected_hyp.get('flaw', 'a structural limitation')}. "
                f"The proposed approach: {selected_hyp.get('exploit', 'targeting the mechanism directly')}. "
                f"Key falsification: {selected_hyp.get('falsification', 'comparing against baselines')}."
            )

        # ── Step 5: multi-perspective critique ───────────────────────────────
        hyp_summary = (
            "Existing flaw: " + selected_hyp.get('flaw', '') + "\n"
            "Mechanism: " + selected_hyp.get('mechanism', '') + "\n"
            "Exploit: " + selected_hyp.get('exploit', '') + "\n"
            "Falsification: " + selected_hyp.get('falsification', '')
        )

        exp_prompt = (
            "You are a hard-nosed experimentalist reviewing a research proposal about '" + topic + "'.\n\n"
            "Hypothesis being tested:\n" + hyp_summary + "\n\n"
            "Proposed experiment:\n" + draft + "\n\n"
            "Give 2-3 sharp criticisms focusing purely on experimental feasibility:\n"
            "- Is the claimed flaw in existing methods actually demonstrable with the proposed experiments?\n"
            "- Does the proposed method actually target the mechanism, or does it just change hyperparameters?\n"
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
            "- Does the exploit actually target the mechanism, or is the connection hand-wavy?\n"
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
            "- Why is the stated flaw in existing methods probably not as severe as claimed?\n"
            "- Why is the proposed exploit probably not the right way to target the mechanism?\n"
            "- What is the most likely negative result and why?\n"
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
            "the author must make. Focus on: "
            "(1) making the evidence for the existing flaw more concrete and demonstrable, "
            "(2) tightening the connection between the mechanism and the proposed exploit, "
            "(3) sharpening the falsification criterion to be truly decisive. "
            "Be concise and prioritized."
        )
        try:
            synthesis = call_llm(synthesis_prompt, model, client, temperature=0.3)
        except Exception:
            synthesis = (
                "Make the existing flaw demonstrable with concrete experiments, "
                "tighten the mechanism-to-exploit connection, "
                "sharpen the falsification criterion."
            )

        # ── Step 6: final revision ────────────────────────────────────────────
        revise_prompt = (
            "Research topic: " + topic + "\n\n"
            "Core hypothesis:\n" + hyp_summary + "\n\n"
            "Experimental design:\n" + draft + "\n\n"
            "Key improvements required:\n" + synthesis + "\n"
            + context_reminder + "\n"
            "Write the final, improved version of the research idea.\n"
            "Requirements:\n"
            "- Open with WHY existing methods fail: name specific methods and their specific flaw\n"
            "- State the mechanism that causes this failure (the causal process existing methods miss)\n"
            "- Describe the proposed method that directly targets this mechanism\n"
            "- Specify the exact experimental condition (model sizes, data scales, settings)\n"
            "- Name specific datasets, baselines, and metrics throughout\n"
            "- Include the key falsification experiment with its precise threshold\n"
            "- Explain what a positive result proves and what a negative result proves\n"
            "- Be direct and specific — no hedging language\n"
            + IDEA_FORMAT
        )
        try:
            final = call_llm(revise_prompt, model, client, temperature)
            if final and final.strip():
                return final
        except Exception:
            pass

        if draft and draft.strip():
            return draft

        return (
            f"Research idea about {topic}: Testing whether "
            f"{selected_hyp.get('mechanism', 'a specific mechanism')} "
            f"causes existing methods to fail because {selected_hyp.get('flaw', 'a structural limitation')}. "
            f"Proposed approach: {selected_hyp.get('exploit', 'targeting the mechanism directly')}. "
            f"Falsification: {selected_hyp.get('falsification', 'comparison with baselines')}."
        )


GENERATOR = S22_r2Generator()
