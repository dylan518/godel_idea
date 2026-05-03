"""S18: Paper-Gap + Feasibility-First Ideation.

Diagnoses S15's core failure: adversarial attack selects for DEFENSIBILITY
(hypotheses survive by becoming hedged and complex), but judges reward
RUNNABILITY (immediate, concrete, named datasets/baselines/metrics).

Fundamental changes from S15:
1. WHERE hypotheses come from: extracted from specific paper gaps/assumptions,
   not free-floating generation. Forces grounding in named work from step 1.
2. HOW hypotheses are selected: runnability score (can it run on a single GPU
   with public data this week?), not "strongest adversarial survivor".
3. CRITIQUE: one experimentalist pass only. The theorist/skeptic passes cause
   ideas to hedge and qualify — complexity as defense mechanism.

Pipeline:
  1. SOTA retrieval (same as S15, n=5)
  2. PAPER ANALYSIS (1 call): for each paper extract claim + assumption + gap
  3. GAP HYPOTHESES (1 call): 5 hypotheses each anchored to a specific paper gap
  4. RUNNABILITY SCORING (5 parallel calls): minimal experiment + numeric score
  5. SELECT highest-scoring (deterministic — no LLM picking)
  6. CONSTRUCT full experiment around winning hypothesis
  7. EXPERIMENTALIST CRITIQUE (1 call)
  8. FINAL REVISION

LLM calls: 1 (analysis) + 1 (hyp) + 5 (scoring) + 1 (construct) + 1 (critique) + 1 (revise) = 10
"""

import sys
import os
import re
import concurrent.futures

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from systems.base import IdeaGenerator, DEFAULT_MODEL, call_llm, IDEA_FORMAT


class S18Generator(IdeaGenerator):
    VERSION = "S18"
    DESCRIPTION = (
        "Paper-gap ideation: extract specific claims/assumptions/gaps from SOTA papers, "
        "generate hypotheses anchored to those gaps, select by runnability score "
        "(not adversarial survival), single experimentalist critique."
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
        sota_papers = ""
        try:
            sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            from retrieval import get_topic_context
            sota_papers = get_topic_context(topic, n=5)
        except Exception:
            sota_papers = ""

        # ── Step 1: extract gaps and assumptions from each paper ─────────────
        # If no papers, fall back to free hypothesis generation (S15-style)
        paper_gaps = ""
        if sota_papers:
            analysis_prompt = (
                f"Research topic: {topic}\n\n"
                f"Here are 5 recent papers on this topic:\n{sota_papers}\n\n"
                "For each paper, extract in ONE line each:\n"
                "CLAIM: the paper's single most important empirical claim\n"
                "ASSUMPTION: one unstated assumption this claim relies on\n"
                "GAP: one specific thing the paper did NOT test but should have\n\n"
                "Format exactly as:\n"
                "P1_CLAIM: ...\nP1_ASSUMPTION: ...\nP1_GAP: ...\n"
                "P2_CLAIM: ...\nP2_ASSUMPTION: ...\nP2_GAP: ...\n"
                "[etc. for all 5 papers]"
            )
            try:
                paper_gaps = call_llm(analysis_prompt, model, client, temperature=0.4)
            except Exception:
                paper_gaps = ""

        # ── Step 2: generate 5 hypotheses anchored to specific paper gaps ────
        if paper_gaps:
            hyp_prompt = (
                f"Research topic: {topic}\n\n"
                f"Specific gaps and untested assumptions in recent work:\n{paper_gaps}\n\n"
                "Generate exactly 5 hypotheses. Each hypothesis MUST:\n"
                "- Directly exploit one of the gaps or assumption violations above "
                "(cite which paper: P1/P2/P3/P4/P5)\n"
                "- Make a claim that CONTRADICTS or EXTENDS a specific paper's result\n"
                "- Be testable with publicly available data and a single GPU\n"
                "- Name the existing method it challenges as a baseline\n\n"
                "Format:\n"
                "H1 [P?]: <hypothesis — what would be true if the gap matters>\n"
                "H2 [P?]: ...\nH3 [P?]: ...\nH4 [P?]: ...\nH5 [P?]: ..."
            )
        else:
            # fallback: free generation with same runnability constraint
            hyp_prompt = (
                f"Research topic: {topic}\n\n"
                "Generate exactly 5 hypotheses. Each must:\n"
                "- Make a specific falsifiable claim (not 'we can improve X')\n"
                "- Name a specific existing method as the baseline it challenges\n"
                "- Be testable with publicly available data and a single GPU\n\n"
                "Format: H1: ...\nH2: ...\nH3: ...\nH4: ...\nH5: ..."
            )
        try:
            hyp_raw = call_llm(hyp_prompt, model, client, temperature)
        except Exception as e:
            hyp_raw = f"H1: Standard approaches to {topic} fail under distribution shift."

        # Parse hypotheses
        hypotheses = []
        for line in hyp_raw.strip().split("\n"):
            line = line.strip()
            m = re.match(r"H\d+\s*(?:\[P\d+\])?\s*[:\-]\s*(.+)", line)
            if m:
                hyp_text = m.group(1).strip()
                if hyp_text:
                    hypotheses.append(hyp_text)
        if not hypotheses:
            hypotheses = [hyp_raw.strip()]
        hypotheses = hypotheses[:5]

        # ── Step 3: score each hypothesis on runnability ─────────────────────
        # For each: generate the minimal experiment, then score 0-3:
        #   +1 if names a specific public dataset
        #   +1 if names a specific existing method as baseline
        #   +1 if states a numeric falsification threshold
        def score_hypothesis(hyp: str) -> tuple:
            score_prompt = (
                f"Research topic: {topic}\n\n"
                f"Hypothesis: {hyp}\n\n"
                "Design the MINIMAL experiment to test this hypothesis:\n"
                "- Must use only publicly available datasets (name them)\n"
                "- Must use an existing method as baseline (name it)\n"
                "- Must complete on a single GPU in under one week\n"
                "- Must state a specific numeric threshold that falsifies the hypothesis\n\n"
                "Then score this experiment:\n"
                "DATASET: <name of public dataset, or MISSING if none named>\n"
                "BASELINE: <name of existing method, or MISSING if none named>\n"
                "THRESHOLD: <numeric falsification threshold, or MISSING if none stated>\n"
                "MINIMAL_EXPERIMENT: <2-3 sentence description of the minimal experiment>"
            )
            try:
                raw = call_llm(score_prompt, model, client, temperature=0.5)
                score = 0
                minimal_exp = ""
                dataset = ""
                baseline = ""
                threshold = ""
                for line in raw.strip().split("\n"):
                    line = line.strip()
                    if line.upper().startswith("DATASET:"):
                        val = line.split(":", 1)[1].strip()
                        dataset = val
                        if val and "MISSING" not in val.upper():
                            score += 1
                    elif line.upper().startswith("BASELINE:"):
                        val = line.split(":", 1)[1].strip()
                        baseline = val
                        if val and "MISSING" not in val.upper():
                            score += 1
                    elif line.upper().startswith("THRESHOLD:"):
                        val = line.split(":", 1)[1].strip()
                        threshold = val
                        if val and "MISSING" not in val.upper():
                            score += 1
                    elif line.upper().startswith("MINIMAL_EXPERIMENT:"):
                        minimal_exp = line.split(":", 1)[1].strip()
                return score, minimal_exp, dataset, baseline, threshold
            except Exception:
                return 0, "", "", "", ""

        scores = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            futures = [executor.submit(score_hypothesis, h) for h in hypotheses]
            for f in futures:
                try:
                    scores.append(f.result(timeout=120))
                except Exception:
                    scores.append((0, "", "", "", ""))

        # ── Step 4: select hypothesis with highest runnability score ──────────
        best_idx = max(range(len(scores)), key=lambda i: scores[i][0])
        selected_hyp = hypotheses[best_idx]
        best_score, best_minimal, best_dataset, best_baseline, best_threshold = scores[best_idx]

        # ── Step 5: construct the full experimental idea ──────────────────────
        context_block = f"\nExisting work to differentiate from:\n{sota_papers}\n" if sota_papers else ""
        construct_prompt = (
            f"Research topic: {topic}\n"
            f"{context_block}\n"
            f"Core hypothesis: {selected_hyp}\n\n"
            f"Minimal experiment already designed:\n{best_minimal}\n"
            f"Dataset: {best_dataset}\n"
            f"Baseline: {best_baseline}\n"
            f"Falsification threshold: {best_threshold}\n\n"
            "Expand this into a full research proposal. Keep the minimal experiment "
            "as the core, but add:\n"
            "- Why this hypothesis is non-obvious (what would most researchers predict?)\n"
            "- Two additional scale-up experiments if the minimal result is positive\n"
            "- Exactly what result would make this publishable vs. a negative result\n\n"
            "Write 3-4 paragraphs. Be direct and specific."
        )
        try:
            draft = call_llm(construct_prompt, model, client, temperature)
        except Exception:
            draft = f"Test hypothesis '{selected_hyp}' using {best_dataset} vs {best_baseline}."

        # ── Step 6: single experimentalist critique ───────────────────────────
        # No theory/skeptic — those cause complexity-as-defense hedging
        crit_prompt = (
            f"You are a hard-nosed experimentalist reviewing a proposal about '{topic}'.\n\n"
            f"Hypothesis: {selected_hyp}\n\n"
            f"Proposed experiment:\n{draft}\n\n"
            "Give 2-3 sharp, specific criticisms ONLY about experimental execution:\n"
            "- Are the measurements well-defined and unambiguous?\n"
            "- What confounds could explain a positive result without the hypothesis being true?\n"
            "- What's the single most likely failure mode in practice?\n\n"
            "Do NOT comment on novelty, theory, or whether the payoff is worth it. "
            "Only execution."
        )
        try:
            critique = call_llm(crit_prompt, model, client, temperature=0.5)
        except Exception:
            critique = "Ensure measurements are unambiguous and confounds are controlled."

        # ── Step 7: final revision ────────────────────────────────────────────
        revise_prompt = (
            f"Research topic: {topic}\n\n"
            f"Core hypothesis: {selected_hyp}\n\n"
            f"Experimental proposal:\n{draft}\n\n"
            f"Experimentalist critique to address:\n{critique}\n"
            f"{context_block}\n"
            "Write the final research idea. The hypothesis must be clearly stated. "
            "The minimal experiment must be immediately runnable (public data, named baseline, "
            "numeric falsification threshold). Address the critique concisely without hedging."
            + IDEA_FORMAT
        )
        try:
            return call_llm(revise_prompt, model, client, temperature)
        except Exception:
            return draft if draft else f"Research idea about {topic}: {selected_hyp}"


GENERATOR = S18Generator()
