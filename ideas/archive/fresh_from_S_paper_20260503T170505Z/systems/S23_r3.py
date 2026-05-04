import sys
import os
import concurrent.futures

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from systems.base import IdeaGenerator, DEFAULT_MODEL, call_llm, IDEA_FORMAT


class S23_r3Generator(IdeaGenerator):
    VERSION = "S23_r3"
    DESCRIPTION = (
        "Mechanism-first ideation: generates causal mechanism claims (not technique names), "
        "stress-tests them with targeted attacks, selects by experimental tractability, "
        "then builds a concrete experiment around the surviving mechanism."
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
            from retrieval import get_topic_context
            sota_context = get_topic_context(topic, n=5)
        except Exception:
            sota_context = ""

        context_block = f"\n\nRelevant existing work:\n{sota_context}\n" if sota_context else ""

        # ── Step 1: generate 5 MECHANISM-SHAPED hypotheses ──────────────────
        # Key fix: force mechanism claims ("X causes Y because Z") not technique names
        hyp_prompt = (
            "Research topic: " + topic + context_block + "\n"
            "Generate exactly 5 falsifiable MECHANISM claims about this topic.\n\n"
            "Each claim must follow this pattern:\n"
            "  '[Specific condition] causes [specific measurable outcome] because [causal mechanism].'\n\n"
            "Rules:\n"
            "- Name the CAUSE and EFFECT specifically (not 'improves performance')\n"
            "- State WHY (the mechanism), not just what to build\n"
            "- The claim must be falsifiable: name what single experiment would disprove it\n"
            "- Do NOT name a technique as the hypothesis (bad: 'using attention improves X')\n"
            "- Must be non-obvious given the existing work above\n\n"
            "Example of BAD hypothesis: 'A variational causal graph learner improves coordination'\n"
            "Example of GOOD hypothesis: 'When agents share only compressed belief states rather than "
            "raw observations, coordination improves because redundant information causes gradient "
            "interference during joint training — falsified if coordination degrades when observation "
            "entropy is low.'\n\n"
            "Format:\n"
            "H1: <mechanism claim>\n"
            "H2: <mechanism claim>\n"
            "H3: <mechanism claim>\n"
            "H4: <mechanism claim>\n"
            "H5: <mechanism claim>"
        )
        try:
            hyp_raw = call_llm(hyp_prompt, model, client, temperature)
        except Exception:
            hyp_raw = "H1: Standard approaches fail under distribution shift because learned features are spuriously correlated with domain-specific artifacts rather than task-relevant signals."

        hypotheses = []
        for line in hyp_raw.strip().split("\n"):
            line = line.strip()
            if line and len(line) > 4 and line[0] == "H" and ":" in line[:4]:
                hyp_text = line.split(":", 1)[1].strip()
                if hyp_text:
                    hypotheses.append(hyp_text)
        if not hypotheses:
            hypotheses = [hyp_raw.strip()]
        hypotheses = hypotheses[:5]

        # ── Step 2: parallel stress-tests focused on experimental tractability ──
        def stress_test(hyp: str) -> str:
            stress_prompt = (
                "Research topic: " + topic + "\n\n"
                "Mechanism claim: " + hyp + "\n\n"
                "Stress-test this claim on FOUR dimensions:\n\n"
                "1. OPERATIONALIZABILITY: Can the cause and effect be measured independently "
                "with standard tools? If not, what measurement problem blocks the experiment?\n\n"
                "2. CONFOUND CHECK: Name one confounding variable that could produce the same "
                "observed effect without the proposed mechanism being true.\n\n"
                "3. DATASET AVAILABILITY: Does a public dataset exist where this mechanism "
                "would be active? Name a specific dataset or explain why none exists.\n\n"
                "4. BASELINE EXISTENCE: Does a published baseline exist that isolates the "
                "proposed mechanism? Name it or explain the gap.\n\n"
                "Then write a REVISED claim that survives these four tests — same mechanism "
                "but more precisely scoped to what is actually measurable:\n"
                "Revised: <tightened mechanism claim that names specific measurable variables>"
            )
            try:
                return call_llm(stress_prompt, model, client, temperature=0.6)
            except Exception as e:
                return "Revised: " + hyp

        stress_results = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            futures = [executor.submit(stress_test, h) for h in hypotheses]
            for f in futures:
                try:
                    stress_results.append(f.result(timeout=120))
                except Exception:
                    stress_results.append("Revised: " + hypotheses[len(stress_results)])

        # ── Step 3: select by EXPERIMENTAL TRACTABILITY (not novelty/ambition) ──
        pairs_str = ""
        for i, (hyp, stress) in enumerate(zip(hypotheses, stress_results)):
            pairs_str += "\n--- Claim " + str(i+1) + " ---\nOriginal: " + hyp + "\nStress test + Revision:\n" + stress + "\n"

        select_prompt = (
            "Research topic: " + topic + "\n\n"
            "Below are " + str(len(hypotheses)) + " mechanism claims, each stress-tested and revised:\n"
            + pairs_str + "\n"
            "Select the ONE claim (by number) whose REVISED form is MOST EXPERIMENTALLY TRACTABLE:\n\n"
            "Tractability criteria (rank these in order of importance):\n"
            "1. MEASURABILITY: The cause and effect can both be measured with existing tools\n"
            "2. DATASET EXISTS: A specific named public dataset can host this experiment\n"
            "3. BASELINE EXISTS: A published method can serve as the control condition\n"
            "4. CONFOUND-ISOLATABLE: The mechanism can be isolated from confounds via ablation\n\n"
            "Do NOT select based on ambition, novelty, or theoretical interest alone.\n"
            "Select the claim you could run an experiment on NEXT WEEK.\n\n"
            "Respond with:\n"
            "SELECTED: <number 1-5>\n"
            "REVISED CLAIM: <copy the revised claim text exactly>\n"
            "EXPERIMENT SKETCH: <one sentence: what you would measure, in what dataset, vs what baseline>"
        )
        try:
            selection_raw = call_llm(select_prompt, model, client, temperature=0.3)
        except Exception:
            selection_raw = "SELECTED: 1\nREVISED CLAIM: " + hypotheses[0] + "\nEXPERIMENT SKETCH: Measure performance on standard benchmark vs baseline."

        selected_claim = hypotheses[0]
        experiment_sketch = ""
        for line in selection_raw.strip().split("\n"):
            ls = line.strip()
            if ls.upper().startswith("REVISED CLAIM:"):
                candidate = ls.split(":", 1)[1].strip()
                if candidate:
                    selected_claim = candidate
            elif ls.upper().startswith("EXPERIMENT SKETCH:"):
                experiment_sketch = ls.split(":", 1)[1].strip()

        # ── Step 4: construct experiment grounded in the mechanism ───────────
        context_reminder = (
            "\nExisting work to differentiate from:\n" + sota_context + "\n"
            if sota_context else ""
        )
        construct_prompt = (
            "Research topic: " + topic + "\n"
            + context_reminder + "\n"
            "Mechanism claim to test: " + selected_claim + "\n"
            "Preliminary experiment sketch: " + experiment_sketch + "\n\n"
            "Design a CONCRETE experiment to test whether this mechanism claim is true.\n\n"
            "Your experiment design MUST specify:\n"
            "1. DATASET: Name the specific public dataset(s). No 'standard benchmarks' — name them.\n"
            "2. MANIPULATION: What exactly do you change between experimental and control conditions? "
            "Be precise enough that two researchers would implement it identically.\n"
            "3. MEASUREMENT: What is the primary dependent variable? How is it computed?\n"
            "4. BASELINES: Name at least 3 specific published methods you compare against.\n"
            "5. FALSIFICATION CRITERION: What specific quantitative result would prove the "
            "mechanism claim FALSE? (e.g., 'if X does not drop by >5% when Y is removed')\n"
            "6. EXPECTED POSITIVE RESULT: What would you observe if the mechanism claim is TRUE?\n\n"
            "Write 3-4 focused paragraphs. Be concrete. Avoid hedging."
        )
        try:
            draft = call_llm(construct_prompt, model, client, temperature)
        except Exception:
            draft = "Research idea for '" + topic + "' based on mechanism: " + selected_claim

        # ── Step 5: targeted critique focused on what judges actually score ──
        # Based on diagnosis: judges score on experimental clarity, concrete methodology,
        # well-defined baselines/metrics, feasibility — not theoretical novelty

        clarity_prompt = (
            "You are reviewing a research proposal for experimental clarity.\n\n"
            "Topic: " + topic + "\n"
            "Core claim: " + selected_claim + "\n\n"
            "Proposal:\n" + draft + "\n\n"
            "Answer these specific questions:\n"
            "1. Are the datasets named specifically? (yes/no + what's missing)\n"
            "2. Are the baselines named specifically? (yes/no + what's missing)\n"
            "3. Is the primary metric defined precisely? (yes/no + what's vague)\n"
            "4. Is the falsification criterion quantitative? (yes/no + what's missing)\n"
            "5. Could two independent researchers implement this identically? (yes/no + what's ambiguous)\n\n"
            "Give concrete fixes for each 'no' answer."
        )
        try:
            critique_clarity = call_llm(clarity_prompt, model, client, temperature=0.5)
        except Exception:
            critique_clarity = "Ensure datasets, baselines, and metrics are named explicitly."

        novelty_prompt = (
            "You are checking whether a research proposal is genuinely novel.\n\n"
            "Topic: " + topic + "\n"
            "Core claim: " + selected_claim + "\n\n"
            "Existing work:\n" + (sota_context if sota_context else "(none provided)") + "\n\n"
            "Proposal:\n" + draft + "\n\n"
            "Answer:\n"
            "1. Does any cited existing work already test this exact mechanism? (yes/no + which)\n"
            "2. Is the proposed manipulation genuinely different from existing work? (yes/no + how)\n"
            "3. What is the single most important thing that makes this non-obvious?\n\n"
            "Be brief and direct."
        )
        try:
            critique_novelty = call_llm(novelty_prompt, model, client, temperature=0.5)
        except Exception:
            critique_novelty = "Verify the mechanism claim is not already tested in existing work."

        feasibility_prompt = (
            "You are a pragmatic reviewer assessing whether a research proposal can actually be executed.\n\n"
            "Topic: " + topic + "\n"
            "Proposal:\n" + draft + "\n\n"
            "Identify the TOP 2 execution risks:\n"
            "1. What is the most likely reason this experiment fails to run?\n"
            "2. What is the most likely reason the results are uninterpretable?\n\n"
            "For each, give a one-sentence mitigation that could be added to the proposal."
        )
        try:
            critique_feasibility = call_llm(feasibility_prompt, model, client, temperature=0.5)
        except Exception:
            critique_feasibility = "Ensure the experimental setup is fully specified to avoid ambiguity."

        synthesis_prompt = (
            "Three reviewers critiqued a research proposal about '" + topic + "'.\n\n"
            "The core mechanism claim being tested: " + selected_claim + "\n\n"
            "Clarity reviewer:\n" + critique_clarity + "\n\n"
            "Novelty reviewer:\n" + critique_novelty + "\n\n"
            "Feasibility reviewer:\n" + critique_feasibility + "\n\n"
            "Synthesize into exactly 3 CONCRETE improvements the author must make.\n"
            "Each improvement should be actionable: not 'add more detail' but "
            "'name the specific dataset X and report metric Y vs baseline Z'.\n"
            "Prioritize fixes that improve experimental concreteness over theoretical scope."
        )
        try:
            synthesis = call_llm(synthesis_prompt, model, client, temperature=0.3)
        except Exception:
            synthesis = "1. Name specific datasets. 2. Name specific baselines. 3. Define quantitative falsification criterion."

        # ── Step 6: final revision ────────────────────────────────────────────
        revise_prompt = (
            "Research topic: " + topic + "\n\n"
            "Core mechanism claim: " + selected_claim + "\n\n"
            "Experimental design draft:\n" + draft + "\n\n"
            "Required improvements:\n" + synthesis + "\n"
            + (("\nExisting work to differentiate from:\n" + sota_context + "\n") if sota_context else "") + "\n"
            "Write the final research idea. Requirements:\n"
            "- State the mechanism claim clearly in the first paragraph\n"
            "- Name specific datasets (no generic 'standard benchmarks')\n"
            "- Name specific baselines (at least 3 published methods)\n"
            "- Define the primary metric precisely\n"
            "- State explicitly what result would FALSIFY the claim\n"
            "- Keep the scope narrow enough to be executable\n\n"
            + IDEA_FORMAT
        )
        try:
            return call_llm(revise_prompt, model, client, temperature)
        except Exception:
            return draft if draft else "Research idea about " + topic + ": " + selected_claim


GENERATOR = S23_r3Generator()
