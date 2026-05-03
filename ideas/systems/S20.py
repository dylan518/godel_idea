"""S20_r2: Multi-Framing + Null-Gauntlet + Grounded Experimental Spine.

Root-cause fixes for three failure modes in the prior tree+Elo pipeline:

  (1) CONCEPTUAL NARROWNESS — tree grown from a single topic prompt means every
      candidate shares the same frame; Elo selects among paraphrases.
      FIX: Generate 3 fundamentally different *framings* of the problem
      (mechanism / representation / causal) before any hypothesis is written.
      Each frame spawns hypotheses from a different conceptual starting point.

  (2) SYCOPHANTIC NULL HYPOTHESES — when asked to "attack" its own output, an
      LLM generates nulls just weak enough for the original to survive.
      FIX: Reframe the null-generation task as an adversarial grant-reviewer
      who will RECOMMEND REJECTION. Provide explicit scoring (survivability 1-5)
      so the selection step can rank by genuine null-resistance, not polish.

  (3) HALLUCINATED EXPERIMENTAL SPECS — datasets, baselines, and deltas that
      sound crisp but are ungrounded because the LLM has no access to paper
      numbers.
      FIX: A dedicated grounding step reads the OpenAlex SOTA context and
      extracts ONLY specific names and magnitudes that appear in (or are
      directly implied by) retrieved papers.

Pipeline per idea (~18 LLM calls):
  0. OpenAlex SOTA context retrieval (same as S15 / S_sota)
  1. Multi-framing: 1 call → 3 named framings (mechanism / representation / causal)
  2. Hypothesis generation: 3 parallel calls (2 hyps per frame = up to 6 candidates)
  3. Null-gauntlet: up to 6 parallel calls (adversarial reviewer → null + survivability)
  4. Best-pair selection: 1 call → (hypothesis, null, distinguishing test)
  5. Grounded experimental spine: 1 call, constrained to paper names from context
  6. Multi-perspective critique: 3 parallel calls (experimentalist / theorist / skeptic)
  7. Critique synthesis: 1 call
  8. Final revision to IDEA_FORMAT: 1 call
"""

import os
import sys
import concurrent.futures

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from systems.base import (
    IdeaGenerator,
    DEFAULT_MODEL,
    IDEA_FORMAT,
    call_llm,
)


# ── SOTA retrieval (same as S15) ─────────────────────────────────────────────

def _get_sota_context(topic: str) -> str:
    try:
        from retrieval import get_topic_context
        return get_topic_context(topic, n=5) or ""
    except Exception:
        return ""


# ── Step 1: Multi-framing ────────────────────────────────────────────────────

def _generate_framings(
    topic: str, sota_context: str, model: str, client, temperature: float
) -> list[str]:
    """Return 3 distinct scientific framings of the problem."""
    ctx = f"\n\nRecent work on this topic:\n{sota_context}\n" if sota_context else ""
    prompt = (
        f"Research topic: {topic}{ctx}\n"
        "Generate 3 fundamentally different scientific framings of this research problem. "
        "Each must approach the problem from a DISTINCT conceptual starting point — "
        "they must NOT be paraphrases of each other.\n\n"
        "FRAMING 1 — MECHANISM LEVEL: Describe the problem as a failure of a specific "
        "computational or cognitive mechanism. Name the mechanism and why it fails here.\n"
        "FRAMING 2 — REPRESENTATION LEVEL: Describe the problem as a representation or "
        "measurement limitation. Name what is encoded or measured incorrectly.\n"
        "FRAMING 3 — CAUSAL LEVEL: Describe the problem as a confounding variable or "
        "selection bias that standard approaches ignore. Name the specific confounder.\n\n"
        "For each framing output (use these exact prefixes):\n"
        "FRAMING1_NAME: <2-4 word name>\n"
        "FRAMING1_CLAIM: <what this framing says is actually wrong, 1-2 sentences>\n"
        "FRAMING1_IMPLICATION: <the one experiment that would validate or invalidate this framing>\n"
        "FRAMING2_NAME: ...\n"
        "FRAMING2_CLAIM: ...\n"
        "FRAMING2_IMPLICATION: ...\n"
        "FRAMING3_NAME: ...\n"
        "FRAMING3_CLAIM: ...\n"
        "FRAMING3_IMPLICATION: ..."
    )
    try:
        raw = call_llm(prompt, model, client, temperature, max_tokens=1200)
    except Exception:
        return [
            f"FRAMING1_CLAIM: Mechanism failure in {topic}.",
            f"FRAMING2_CLAIM: Representation limitation in {topic}.",
            f"FRAMING3_CLAIM: Causal confound in {topic}.",
        ]

    framings: list[str] = []
    for n in ("1", "2", "3"):
        parts = [
            line.strip()
            for line in raw.strip().split("\n")
            if line.strip().startswith(f"FRAMING{n}_")
        ]
        if parts:
            framings.append("\n".join(parts))

    if not framings:
        # Fallback: split raw evenly
        lines = raw.strip().split("\n")
        third = max(1, len(lines) // 3)
        framings = [
            "\n".join(lines[:third]),
            "\n".join(lines[third : 2 * third]),
            "\n".join(lines[2 * third :]),
        ]

    return framings[:3] if framings else [raw]


# ── Step 2: Hypotheses per framing ───────────────────────────────────────────

def _generate_hypotheses_for_framing(
    topic: str, framing: str, model: str, client, temperature: float
) -> list[str]:
    """Return 2 falsifiable hypotheses grounded in a specific framing."""
    prompt = (
        f"Research topic: {topic}\n\n"
        f"Scientific framing of the problem:\n{framing}\n\n"
        "Based on this specific framing, generate exactly 2 sharp, falsifiable hypotheses.\n"
        "Each hypothesis must:\n"
        "- Make a specific CAUSAL CLAIM (not 'we can improve X' — name the mechanism)\n"
        "- Be testable: state in ≤1 sentence what result would disprove it\n"
        "- Be non-obvious: must NOT be a restatement of standard existing approaches\n\n"
        "Format (use exact prefixes):\n"
        "H1: <hypothesis statement>\n"
        "H1_DISPROOF: <what specific result would falsify it>\n"
        "H2: <hypothesis statement>\n"
        "H2_DISPROOF: <what specific result would falsify it>"
    )
    try:
        raw = call_llm(prompt, model, client, temperature, max_tokens=600)
    except Exception:
        return [f"{topic}: standard approaches fail due to the mechanism described above."]

    hypotheses: list[str] = []
    for line in raw.strip().split("\n"):
        line = line.strip()
        if (line.startswith("H1:") or line.startswith("H2:")) and ":" in line:
            text = line.split(":", 1)[1].strip()
            if text:
                hypotheses.append(text)

    return hypotheses if hypotheses else [raw.strip()[:300]]


# ── Step 3: Null-gauntlet ─────────────────────────────────────────────────────

def _null_gauntlet(
    topic: str, hypothesis: str, model: str, client, temperature: float
) -> dict:
    """Adversarial grant-reviewer produces strongest null hypothesis."""
    prompt = (
        f"Research topic: {topic}\n\n"
        f"Claim under review: {hypothesis}\n\n"
        "You are writing a peer review recommending REJECTION of a paper making this claim. "
        "Your task: construct the STRONGEST POSSIBLE null hypothesis — the alternative "
        "explanation that, if true, would make this work COMPLETELY UNNECESSARY or WRONG.\n\n"
        "Requirements for the null hypothesis:\n"
        "1. Grounded in a known mechanism or prior experimental result (cite plausible literature)\n"
        "2. Explains AWAY the expected positive result as a confound or trivial artifact\n"
        "3. Independently testable without the proposed method\n\n"
        "Output (use exact prefixes):\n"
        "NULL: <alternative explanation in 1-2 sentences>\n"
        "ARTIFACT: <how the expected positive result could be spurious or trivially explained>\n"
        "DISTINGUISHER: <the one experiment that cleanly separates H from H0>\n"
        "SURVIVABILITY: <integer 1-5; 5=hypothesis clearly survives this attack, "
        "1=hypothesis is completely killed>"
    )
    try:
        raw = call_llm(prompt, model, client, temperature=0.5, max_tokens=600)
    except Exception as e:
        return {
            "null": "Simpler baselines achieve the same result.",
            "artifact": "Dataset-specific artifact.",
            "distinguisher": "Out-of-distribution ablation.",
            "survivability": 3,
            "raw": str(e),
        }

    result: dict = {
        "raw": raw,
        "null": "",
        "artifact": "",
        "distinguisher": "",
        "survivability": 3,
    }
    for line in raw.strip().split("\n"):
        line = line.strip()
        if line.startswith("NULL:"):
            result["null"] = line.split(":", 1)[1].strip()
        elif line.startswith("ARTIFACT:"):
            result["artifact"] = line.split(":", 1)[1].strip()
        elif line.startswith("DISTINGUISHER:"):
            result["distinguisher"] = line.split(":", 1)[1].strip()
        elif line.startswith("SURVIVABILITY:"):
            try:
                result["survivability"] = int(line.split(":", 1)[1].strip()[0])
            except Exception:
                result["survivability"] = 3
    return result


# ── Step 4: Select best (hypothesis, null) pair ───────────────────────────────

def _select_best_pair(
    topic: str,
    hypotheses: list[str],
    nulls: list[dict],
    model: str,
    client,
) -> tuple[str, str, str]:
    """Pick the hypothesis with the clearest null-distinguishing experimental signature."""
    pairs_str = ""
    for i, (hyp, n) in enumerate(zip(hypotheses, nulls)):
        pairs_str += (
            f"\n--- Candidate {i + 1} ---\n"
            f"HYPOTHESIS: {hyp}\n"
            f"NULL: {n.get('null', 'N/A')}\n"
            f"ARTIFACT RISK: {n.get('artifact', 'N/A')}\n"
            f"DISTINGUISHING TEST: {n.get('distinguisher', 'N/A')}\n"
            f"NULL-SURVIVABILITY (1-5): {n.get('survivability', 3)}\n"
        )

    prompt = (
        f"Research topic: {topic}\n\n"
        f"Below are {len(hypotheses)} hypothesis/null-gauntlet pairs.\n"
        f"{pairs_str}\n"
        "Select the ONE candidate that:\n"
        "1. Has the highest null-survivability score\n"
        "2. Has the CLEAREST experimental distinguisher (most concrete and runnable)\n"
        "3. Is the most NOVEL (not a restatement of standard approaches)\n\n"
        "Respond (use exact prefixes):\n"
        "SELECTED: <number 1-N>\n"
        "HYPOTHESIS: <exact text of selected hypothesis>\n"
        "NULL: <exact text of the selected null hypothesis>\n"
        "DISTINGUISHER: <exact text of the distinguishing test>\n"
        "RATIONALE: <1-2 sentences why this is the strongest candidate>"
    )
    try:
        raw = call_llm(prompt, model, client, temperature=0.3, max_tokens=800)
    except Exception:
        return (
            hypotheses[0],
            nulls[0].get("null", "") if nulls else "",
            nulls[0].get("distinguisher", "") if nulls else "",
        )

    selected_hyp = hypotheses[0]
    selected_null = nulls[0].get("null", "") if nulls else ""
    selected_dist = nulls[0].get("distinguisher", "") if nulls else ""

    for line in raw.strip().split("\n"):
        line = line.strip()
        if line.upper().startswith("HYPOTHESIS:"):
            txt = line.split(":", 1)[1].strip()
            if txt:
                selected_hyp = txt
        elif line.upper().startswith("NULL:"):
            txt = line.split(":", 1)[1].strip()
            if txt:
                selected_null = txt
        elif line.upper().startswith("DISTINGUISHER:"):
            txt = line.split(":", 1)[1].strip()
            if txt:
                selected_dist = txt

    return selected_hyp, selected_null, selected_dist


# ── Step 5: SOTA-grounded experimental spine ──────────────────────────────────

def _ground_experimental_spine(
    topic: str,
    hypothesis: str,
    null_hypothesis: str,
    distinguishing_test: str,
    sota_context: str,
    model: str,
    client,
    temperature: float,
) -> str:
    """Extract concrete, paper-grounded datasets / baselines / metrics / thresholds."""
    ctx = f"\nRecent papers on this topic:\n{sota_context}\n" if sota_context else ""
    prompt = (
        f"Research topic: {topic}{ctx}\n"
        f"Research hypothesis: {hypothesis}\n"
        f"Null hypothesis to rule out: {null_hypothesis}\n"
        f"Key distinguishing test: {distinguishing_test}\n\n"
        "Design a concrete experimental spine. Use ONLY specific resource names that appear "
        "in (or are directly implied by) the papers above — do NOT invent fictional benchmarks "
        "or models.\n\n"
        "DATASETS: Name ≥2 real, specific datasets (e.g. 'SQuAD 2.0', 'ImageNet-1k', "
        "'Penn Treebank') — prefer names from the papers above.\n"
        "BASELINES: Name ≥3 real, specific models or methods (e.g. 'BERT-large', "
        "'DeBERTa-v3-large', 'GPT-2 345M') — prefer names from the papers above.\n"
        "PRIMARY_METRIC: The one standard quantitative metric for this area "
        "(e.g. 'Exact Match on SQuAD 2.0', 'top-1 accuracy on ImageNet').\n"
        "EXPECTED_DELTA: A realistic improvement range derived from the magnitude of gains "
        "reported in the papers above (e.g. '1–3% EM improvement over DeBERTa-v3 baseline').\n"
        "FALSIFICATION_THRESHOLD: The specific number or outcome that would DISPROVE the "
        "hypothesis (e.g. '<0.5% improvement over BERT-large at p<0.05').\n"
        "NULL_CONTROL: The ablation or control condition that separates the proposed mechanism "
        "from the null hypothesis above.\n\n"
        "Be specific. Use real names. Never write 'standard benchmarks' or 'common baselines'."
    )
    try:
        return call_llm(prompt, model, client, temperature=0.4, max_tokens=800)
    except Exception:
        return (
            f"DATASETS: Standard benchmarks for {topic}\n"
            "BASELINES: Existing SOTA methods\n"
            "PRIMARY_METRIC: Task-appropriate accuracy metric\n"
            "EXPECTED_DELTA: Moderate improvement over baselines\n"
            "FALSIFICATION_THRESHOLD: No improvement over baselines\n"
            "NULL_CONTROL: Ablation removing proposed component"
        )


# ── Main generator ────────────────────────────────────────────────────────────

class S20Generator(IdeaGenerator):
    VERSION = "S20"
    DESCRIPTION = (
        "Multi-Framing + Null-Gauntlet + Grounded Experimental Spine. "
        "Generates hypotheses from 3 distinct conceptual frames (mechanism / "
        "representation / causal), applies adversarial null-gauntlet ranking, "
        "grounds experimental specs in retrieved SOTA papers, then multi-perspective "
        "critique before final revision to IDEA_FORMAT."
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
        # ── Step 0: SOTA context ──────────────────────────────────────────
        sota = _get_sota_context(topic)
        context_block = f"\n\nExisting work:\n{sota}\n" if sota else ""

        # ── Step 1: Multi-framing (1 call) ───────────────────────────────
        try:
            framings = _generate_framings(topic, sota, model, client, temperature)
        except Exception:
            framings = [
                f"FRAMING1_CLAIM: Mechanism failure in {topic}.",
                f"FRAMING2_CLAIM: Representation limitation in {topic}.",
                f"FRAMING3_CLAIM: Causal confound in {topic}.",
            ]

        # ── Step 2: Hypotheses per framing (3 parallel calls) ────────────
        all_hypotheses: list[str] = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
            futures = [
                pool.submit(
                    _generate_hypotheses_for_framing, topic, f, model, client, temperature
                )
                for f in framings
            ]
            for fut in futures:
                try:
                    all_hypotheses.extend(fut.result(timeout=120))
                except Exception:
                    all_hypotheses.append(
                        f"Standard approaches to {topic} fail due to a mechanism failure."
                    )

        if not all_hypotheses:
            all_hypotheses = [f"Standard approaches to {topic} have a fundamental limitation."]
        all_hypotheses = all_hypotheses[:6]

        # ── Step 3: Null-gauntlet (parallel, up to 6 calls) ──────────────
        nulls: list[dict] = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=6) as pool:
            futures = [
                pool.submit(_null_gauntlet, topic, h, model, client, temperature)
                for h in all_hypotheses
            ]
            for fut in futures:
                try:
                    nulls.append(fut.result(timeout=120))
                except Exception:
                    nulls.append(
                        {
                            "null": "Simpler approach achieves the same result.",
                            "artifact": "",
                            "distinguisher": "",
                            "survivability": 2,
                        }
                    )

        # ── Step 4: Select best (hypothesis, null) pair (1 call) ─────────
        try:
            selected_hyp, selected_null, selected_dist = _select_best_pair(
                topic, all_hypotheses, nulls, model, client
            )
        except Exception:
            selected_hyp = all_hypotheses[0]
            selected_null = nulls[0].get("null", "") if nulls else ""
            selected_dist = nulls[0].get("distinguisher", "") if nulls else ""

        # ── Step 5: Ground experimental spine in SOTA papers (1 call) ────
        try:
            spine = _ground_experimental_spine(
                topic,
                selected_hyp,
                selected_null,
                selected_dist,
                sota,
                model,
                client,
                temperature,
            )
        except Exception:
            spine = f"Experimental design for testing: {selected_hyp}"

        # ── Step 6: Multi-perspective critique (3 parallel calls) ─────────
        exp_prompt = (
            f"You are a hard-nosed experimentalist reviewing a proposal about '{topic}'.\n\n"
            f"Hypothesis: {selected_hyp}\n"
            f"Null hypothesis to rule out: {selected_null}\n\n"
            f"Experimental spine:\n{spine}\n\n"
            "Give 2-3 sharp criticisms on experimental feasibility: Are the named datasets "
            "and baselines appropriate and real? Is the falsification threshold well-defined? "
            "What controls are missing to rule out the null?"
        )
        theory_prompt = (
            f"You are a rigorous theorist reviewing a proposal about '{topic}'.\n\n"
            f"Hypothesis: {selected_hyp}\n"
            f"Null hypothesis: {selected_null}\n\n"
            f"Experimental spine:\n{spine}\n\n"
            "Give 2-3 sharp criticisms on theoretical grounding: Does the distinguishing test "
            "actually separate H from H0? What alternative explanations remain? "
            "Is the novelty claim justified over existing work?"
        )
        skeptic_prompt = (
            f"You are a skeptical reviewer about '{topic}'.\n\n"
            f"Hypothesis: {selected_hyp}\n"
            f"Null hypothesis: {selected_null}\n\n"
            f"Experimental spine:\n{spine}\n\n"
            "Give 2-3 sharp criticisms: Why won't the distinguishing test actually work? "
            "What is the most likely negative result? Is the expected delta realistic "
            "given the literature?"
        )

        critique_exp = critique_theory = critique_skeptic = ""
        with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
            f_exp = pool.submit(call_llm, exp_prompt, model, client, 0.5)
            f_theory = pool.submit(call_llm, theory_prompt, model, client, 0.5)
            f_skeptic = pool.submit(call_llm, skeptic_prompt, model, client, 0.5)
            for key, fut in (("exp", f_exp), ("theory", f_theory), ("skeptic", f_skeptic)):
                try:
                    val = fut.result(timeout=120)
                    if key == "exp":
                        critique_exp = val
                    elif key == "theory":
                        critique_theory = val
                    else:
                        critique_skeptic = val
                except Exception:
                    pass

        # ── Step 7: Synthesis (1 call) ────────────────────────────────────
        synthesis_prompt = (
            f"Three reviewers critiqued a proposal about '{topic}' testing:\n"
            f"HYPOTHESIS: {selected_hyp}\n"
            f"NULL: {selected_null}\n\n"
            f"Experimentalist:\n{critique_exp or 'N/A'}\n\n"
            f"Theorist:\n{critique_theory or 'N/A'}\n\n"
            f"Skeptic:\n{critique_skeptic or 'N/A'}\n\n"
            "Synthesize the 3 most important actionable improvements, focused on:\n"
            "(1) strengthening the null-ruling-out mechanism,\n"
            "(2) improving dataset/baseline specificity,\n"
            "(3) clarifying the falsification threshold."
        )
        try:
            synthesis = call_llm(synthesis_prompt, model, client, temperature=0.3, max_tokens=600)
        except Exception:
            synthesis = (
                "Improve specificity, add concrete baselines, clarify falsification threshold."
            )

        # ── Step 8: Final revision to IDEA_FORMAT (1 call) ───────────────
        context_reminder = (
            f"\nExisting work to differentiate from:\n{sota}\n" if sota else ""
        )
        revise_prompt = (
            f"Research topic: {topic}{context_reminder}\n"
            f"CORE HYPOTHESIS (scientific claim): {selected_hyp}\n\n"
            f"NULL HYPOTHESIS (what this work rules out): {selected_null}\n\n"
            f"EXPERIMENTAL SPINE (pre-committed design):\n{spine}\n\n"
            f"KEY IMPROVEMENTS FROM REVIEW:\n{synthesis}\n\n"
            "Write a complete, polished research idea. Requirements:\n"
            "- Lead with the hypothesis as a concrete scientific claim (not 'we propose X')\n"
            "- Name the null hypothesis explicitly and explain how the experiment rules it out\n"
            "- Use ONLY the specific datasets, baselines, and metrics from the spine\n"
            "- State the quantitative falsification threshold explicitly\n"
            "- Incorporate all three reviewer improvements\n\n"
            + IDEA_FORMAT
        )
        try:
            result = call_llm(revise_prompt, model, client, temperature, max_tokens=2048)
            return result if result else f"Research idea about {topic}: {selected_hyp}"
        except Exception:
            return (
                f"Research idea about {topic}.\n\n"
                f"Hypothesis: {selected_hyp}\n\n"
                f"Experimental design:\n{spine}"
            )


GENERATOR = S20Generator()
