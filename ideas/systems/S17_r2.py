"""S17_r2: Feasibility-Anchored Hypothesis Loop.

Fundamentally restructures hypothesis selection to optimise feasibility FIRST,
novelty SECOND — reversing the selection pressure that caused S16's overreach.

Root diagnosis: S15's adversarial attack step pressures hypotheses to become
more ambitious to "survive" criticism, and selection rewards "most novel and
non-obvious" survivor. This created a selection pressure toward intellectually
exciting but practically infeasible ideas (trillion-parameter models,
fault-tolerant quantum hardware). The fix: derive concrete resource constraints
from SOTA *before* generating hypotheses, score on feasibility and novelty as
independent dimensions, and treat feasibility as a hard gate for selection.

Pipeline:
1. SOTA RETRIEVAL: Fetch recent papers for grounding.

2. FEASIBILITY ANCHORS (1 call): Extract from SOTA what datasets, model
   scales, compute budgets, and baselines are standard for this topic.
   Creates grounding constraints that prevent downstream overreach.

3. SCOPE-CONSTRAINED HYPOTHESES (1 call): Generate 5 hypotheses explicitly
   constrained to the feasibility anchors. Prompt forbids anything requiring
   resources beyond standard academic compute.

4. PARALLEL FEASIBILITY+NOVELTY SCORING (5 calls): Each hypothesis scored
   independently on feasibility (1-5) and novelty (1-5). No adversarial
   pressure — purely evaluative. Parallel execution.

5. SELECTION (1 call): Feasibility is a hard gate (≥3); among feasible
   hypotheses, maximise combined score. Explicit instruction not to select
   infeasible ideas even if highly novel.

6. EXPERIMENT CONSTRUCTION (1 call): Build the experiment with feasibility
   anchors visible in the prompt. Produces one fully runnable experiment,
   not a dual-track proxy.

7. MULTI-PERSPECTIVE CRITIQUE (3 parallel + 1 synthesis = 4 calls):
   Experimentalist, novelty skeptic, resource auditor — then synthesis.

8. FINAL REVISION (1 call): Incorporate critique, anchors visible.

LLM calls: 1+1+5+1+1+4+1 = 14
"""

import sys
import os
import concurrent.futures

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from systems.base import IdeaGenerator, DEFAULT_MODEL, call_llm, IDEA_FORMAT


class S17_r2Generator(IdeaGenerator):
    VERSION = "S17_r2"
    DESCRIPTION = (
        "Feasibility-anchored hypothesis loop: extract compute/dataset anchors from SOTA, "
        "generate scope-constrained hypotheses, score on feasibility+novelty independently, "
        "select with feasibility as hard gate, construct grounded experiment, "
        "multi-perspective critique with resource auditor."
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

        # ── Step 1: extract feasibility anchors ──────────────────────────────
        # KEY CHANGE FROM S15/S16: derive resource constraints BEFORE hypothesis
        # generation so all downstream steps are constrained from the start.
        anchor_prompt = (
            f"Research topic: {topic}{context_block}\n"
            "Based on the recent work above (or your knowledge of this field if no context), "
            "describe the STANDARD EXPERIMENTAL SCALE for this topic. "
            "Be conservative — what can a well-resourced academic lab (not a big tech company) "
            "actually run and publish?\n\n"
            "Answer these exactly:\n"
            "DATASETS: 2-3 specific publicly available datasets commonly used "
            "(e.g. 'CIFAR-10, WikiText-103, Penn Treebank')\n"
            "MODEL_SCALE: Standard model sizes for experiments "
            "(e.g. 'GPT-2 125M to 1.3B', 'ResNet-50/101', 'BERT-base/large')\n"
            "COMPUTE: Typical training budget "
            "(e.g. '4-8 GPU days on A100s', '1-2 days on V100s')\n"
            "BASELINES: 2-3 specific go-to methods to compare against\n"
            "METRICS: Primary evaluation metrics\n\n"
            "If a method requires trillion-parameter models or fault-tolerant quantum "
            "hardware, it is out of scope for this topic."
        )
        try:
            anchors_raw = call_llm(anchor_prompt, model, client, temperature=0.3)
        except Exception:
            anchors_raw = (
                f"DATASETS: standard benchmarks for {topic}\n"
                "MODEL_SCALE: medium-scale models (100M–1B parameters)\n"
                "COMPUTE: 4–8 GPU days on A100s\n"
                "BASELINES: standard published baselines\n"
                "METRICS: standard evaluation metrics for the task"
            )

        # ── Step 2: generate 5 scope-constrained hypotheses ──────────────────
        # KEY CHANGE: hypotheses are generated AFTER anchors are set, with the
        # anchors visible in the prompt. Resource overreach is blocked upfront.
        hyp_prompt = (
            f"Research topic: {topic}{context_block}\n"
            f"FEASIBILITY CONSTRAINTS — your hypotheses MUST fit within these:\n"
            f"{anchors_raw}\n\n"
            "Generate exactly 5 falsifiable scientific hypotheses. Each MUST:\n"
            "1. Make a specific, non-obvious claim about a mechanism or relationship\n"
            "2. Be testable using only the datasets, model scales, and compute listed above — "
            "no hypothesis may require resources beyond what is listed\n"
            "3. Include a concrete falsification criterion: one result that would disprove it\n"
            "4. Differ meaningfully from each other and from obvious baselines\n\n"
            "Format:\n"
            "H1: <hypothesis>  [FALSIFY: <what result disproves it>]\n"
            "H2: <hypothesis>  [FALSIFY: <what result disproves it>]\n"
            "H3: <hypothesis>  [FALSIFY: <what result disproves it>]\n"
            "H4: <hypothesis>  [FALSIFY: <what result disproves it>]\n"
            "H5: <hypothesis>  [FALSIFY: <what result disproves it>]"
        )
        try:
            hyp_raw = call_llm(hyp_prompt, model, client, temperature)
        except Exception:
            hyp_raw = (
                f"H1: Standard methods for {topic} underperform on long-tail examples "
                "because of distribution mismatch.  "
                "[FALSIFY: no gap on stratified long-tail test split]"
            )

        # Parse hypotheses
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

        # ── Step 3: parallel feasibility + novelty scoring ───────────────────
        # KEY CHANGE FROM S15: no adversarial attack (which pressures toward
        # ambition). Instead: independent dual-dimension scoring.
        def score_hypothesis(hyp: str) -> dict:
            score_prompt = (
                f"Research topic: {topic}\n\n"
                f"FEASIBILITY CONSTRAINTS:\n{anchors_raw}\n\n"
                f"Hypothesis: {hyp}\n\n"
                "Score this hypothesis on TWO independent dimensions:\n\n"
                "FEASIBILITY (1–5): Can a grad student at a well-resourced university "
                "run the falsification experiment within the constraints above?\n"
                "  5 = yes, uses listed datasets/models with standard tools\n"
                "  3 = probably, some engineering challenge but not a blocker\n"
                "  1 = no — requires unavailable compute, hardware, or data\n\n"
                "NOVELTY (1–5): How non-obvious is the core claim vs. standard methods?\n"
                "  5 = genuinely surprising prediction not in standard literature\n"
                "  3 = incremental but non-trivial extension of known methods\n"
                "  1 = essentially restates known results or obvious baselines\n\n"
                "Reply with ONLY:\n"
                "FEASIBILITY: <integer 1-5>\n"
                "NOVELTY: <integer 1-5>\n"
                "REASON: <one sentence on the main strength or limiting weakness>"
            )
            try:
                raw = call_llm(score_prompt, model, client, temperature=0.3)
                f_score, n_score, reason = 3, 3, ""
                for line in raw.strip().split("\n"):
                    upper = line.strip().upper()
                    if upper.startswith("FEASIBILITY:"):
                        try:
                            f_score = int(line.split(":", 1)[1].strip().split()[0])
                        except Exception:
                            pass
                    elif upper.startswith("NOVELTY:"):
                        try:
                            n_score = int(line.split(":", 1)[1].strip().split()[0])
                        except Exception:
                            pass
                    elif upper.startswith("REASON:"):
                        reason = line.split(":", 1)[1].strip()
                return {"hyp": hyp, "feasibility": f_score, "novelty": n_score, "reason": reason}
            except Exception as e:
                return {"hyp": hyp, "feasibility": 3, "novelty": 3, "reason": str(e)}

        scored = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            futures = [executor.submit(score_hypothesis, h) for h in hypotheses]
            for idx, fut in enumerate(futures):
                try:
                    scored.append(fut.result(timeout=120))
                except Exception as e:
                    hyp_fallback = hypotheses[idx] if idx < len(hypotheses) else ""
                    scored.append({"hyp": hyp_fallback, "feasibility": 2, "novelty": 3, "reason": str(e)})

        # ── Step 4: select best hypothesis with feasibility as hard gate ──────
        # KEY CHANGE: feasibility is a HARD constraint, not just a weighted term.
        scores_str = ""
        for i, s in enumerate(scored):
            combined = s["feasibility"] * 0.5 + s["novelty"] * 0.5
            scores_str += (
                f"\n--- Hypothesis {i+1} ---\n"
                f"Text: {s['hyp']}\n"
                f"Feasibility: {s['feasibility']}/5\n"
                f"Novelty: {s['novelty']}/5\n"
                f"Combined: {combined:.1f}\n"
                f"Note: {s['reason']}\n"
            )

        select_prompt = (
            f"Research topic: {topic}\n\n"
            f"Five hypotheses with independent feasibility and novelty scores:\n{scores_str}\n"
            "SELECTION RULE: Feasibility is a HARD gate — do NOT select any hypothesis "
            "with feasibility < 3, even if it has high novelty. "
            "Among hypotheses with feasibility ≥ 3, prefer the highest combined score. "
            "If all have feasibility < 3, select the one with the highest feasibility.\n\n"
            "Respond with:\n"
            "SELECTED: <number 1-5>\n"
            "FINAL HYPOTHESIS: <copy the hypothesis text, removing [FALSIFY:...] suffix>\n"
            "REASON: <one sentence why this is the best feasible choice>"
        )
        try:
            selection_raw = call_llm(select_prompt, model, client, temperature=0.3)
        except Exception:
            # fallback: pick highest combined score
            best = max(scored, key=lambda s: s["feasibility"] * 0.5 + s["novelty"] * 0.5)
            selection_raw = (
                f"SELECTED: 1\n"
                f"FINAL HYPOTHESIS: {best['hyp']}\n"
                f"REASON: fallback selection"
            )

        selected_hyp = scored[0]["hyp"] if scored else hypotheses[0]
        for line in selection_raw.strip().split("\n"):
            if line.strip().upper().startswith("FINAL HYPOTHESIS:"):
                candidate = line.split(":", 1)[1].strip()
                if candidate:
                    selected_hyp = candidate
                    break

        # ── Step 5: construct experiment with anchors visible ─────────────────
        # KEY CHANGE: feasibility anchors are included in this prompt so the
        # construction step cannot escape the constraints set earlier.
        context_reminder = (
            f"\nExisting work to differentiate from:\n{sota_context}\n"
            if sota_context else ""
        )
        construct_prompt = (
            f"Research topic: {topic}\n"
            f"{context_reminder}\n"
            f"FEASIBILITY CONSTRAINTS (do not exceed these):\n{anchors_raw}\n\n"
            f"Core hypothesis to test: {selected_hyp}\n\n"
            "Design ONE concrete, fully runnable experiment to test this hypothesis.\n"
            "Stay within the feasibility constraints above — no larger models, "
            "no unavailable datasets, no special hardware.\n\n"
            "Include:\n"
            "- Exact datasets by name\n"
            "- Exact model(s) and scale\n"
            "- Exact baseline methods by name\n"
            "- Quantitative success threshold: what result supports the hypothesis, "
            "what result falsifies it?\n"
            "- Key technical innovation: what specifically differs from existing work?\n\n"
            "Write 3–4 focused paragraphs. Every claim should be specific and grounded."
        )
        try:
            draft = call_llm(construct_prompt, model, client, temperature)
        except Exception:
            draft = f"Research idea for '{topic}' based on hypothesis: {selected_hyp}"

        # ── Step 6: multi-perspective critique (3 parallel + synthesis) ───────
        def run_critique(prompt_text: str) -> str:
            try:
                return call_llm(prompt_text, model, client, temperature=0.5)
            except Exception:
                return "No critique available."

        exp_prompt = (
            f"You are a hard-nosed experimentalist reviewing a research proposal "
            f"about '{topic}'.\n\n"
            f"Hypothesis: {selected_hyp}\n\n"
            f"Proposed experiment:\n{draft}\n\n"
            "Give 2–3 sharp criticisms. Focus: Can a grad student run this? "
            "Are measurements well-defined? What controls are missing? "
            "Are the named datasets and baselines actually appropriate for the claim?"
        )

        novelty_prompt = (
            f"You are a skeptical reviewer who has read hundreds of papers "
            f"about '{topic}'.\n\n"
            f"Hypothesis: {selected_hyp}\n\n"
            f"Proposed experiment:\n{draft}\n\n"
            "Give 2–3 sharp criticisms. Focus: Is this genuinely novel? "
            "Name specific existing work this idea overlaps with. "
            "What would make the contribution distinctly different?"
        )

        # KEY CHANGE: new resource auditor critique catches any constraint violations
        resource_prompt = (
            f"You are a resource auditor reviewing a research proposal "
            f"about '{topic}'.\n\n"
            f"FEASIBILITY CONSTRAINTS:\n{anchors_raw}\n\n"
            f"Proposed experiment:\n{draft}\n\n"
            "Flag any specific claims that EXCEED the feasibility constraints: "
            "model sizes too large, datasets unavailable or non-public, baselines "
            "requiring special access, compute beyond the stated budget. "
            "For each flag, suggest a concrete in-scope replacement."
        )

        critiques = {}
        with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
            futures_map = {
                "exp": executor.submit(run_critique, exp_prompt),
                "novelty": executor.submit(run_critique, novelty_prompt),
                "resource": executor.submit(run_critique, resource_prompt),
            }
            for key, fut in futures_map.items():
                try:
                    critiques[key] = fut.result(timeout=120)
                except Exception as e:
                    critiques[key] = f"No {key} critique available."

        synthesis_prompt = (
            f"Three reviewers critiqued a research idea about '{topic}':\n\n"
            f"Experimentalist:\n{critiques.get('exp', 'N/A')}\n\n"
            f"Novelty skeptic:\n{critiques.get('novelty', 'N/A')}\n\n"
            f"Resource auditor:\n{critiques.get('resource', 'N/A')}\n\n"
            "Synthesise into the 3 most important actionable improvements. "
            "Priority order: (1) fix any resource/feasibility violations flagged by "
            "the auditor, (2) strengthen the novelty claim, "
            "(3) tighten the experimental design."
        )
        try:
            synthesis = call_llm(synthesis_prompt, model, client, temperature=0.3)
        except Exception:
            synthesis = "Fix resource constraints, strengthen novelty claim, name explicit baselines."

        # ── Step 7: final revision ────────────────────────────────────────────
        revise_prompt = (
            f"Research topic: {topic}\n\n"
            f"Core hypothesis: {selected_hyp}\n\n"
            f"Experimental design:\n{draft}\n\n"
            f"Key improvements required:\n{synthesis}\n"
            f"{context_reminder}\n"
            f"FEASIBILITY CONSTRAINTS (do not exceed these):\n{anchors_raw}\n\n"
            "Write the final, improved version of this research idea. "
            "The experiment must be fully runnable within the feasibility constraints. "
            "All datasets, baselines, and metrics must be named explicitly. "
            "The hypothesis must be clearly stated and falsifiable."
            + IDEA_FORMAT
        )
        try:
            return call_llm(revise_prompt, model, client, temperature)
        except Exception:
            return draft if draft else f"Research idea about {topic}: {selected_hyp}"


GENERATOR = S17_r2Generator()
