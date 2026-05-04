"""S25_r1: Feasibility-anchored IdeaTreeSearch.

Three-level fix for S12's repeated feasibility failures.  The tournament
selects for Elo (novelty/impact), which surfaces resource-prohibitive leaves;
a single feasibility-framing sentence at expansion time is too weak to fight
that selection pressure.

Fix operates at ALL THREE levels simultaneously:

  1. POST-TOURNAMENT RE-RANKING  (selection pressure)
     After Swiss Elo, one batch call scores all ranked leaves for experimental
     feasibility (1-10).  Leaves are re-ranked by a blended score:
         final = 0.60 * elo_rank_norm + 0.40 * feasibility_norm
     so trillion-param / real-NISQ-hardware leaves are *deprioritized before
     expansion*, not just patched afterward.

  2. FEASIBILITY-FORWARD EXPANSION  (prompt grounding)
     The expansion prompt frames the scientific contribution around the simplest
     experiment that *proves* the core claim on academic-lab resources.  Explicit
     scale ceilings (≤13B params, ≤30-qubit simulators, public datasets) are
     structural requirements, not a trailing note.  The LLM must write an
     EXPERIMENT section that satisfies them — novelty lives in the IDEA,
     APPROACH, and NOVELTY sections.

  3. POST-EXPANSION FEASIBILITY AUDIT & PATCH  (safety net)
     After expansion, a targeted audit call checks the EXPERIMENT section for
     resource-prohibitive proposals.  If found, it replaces ONLY that section
     with a concrete executable alternative while preserving every other section.

LLM calls per topic (n=5 ideas):
  4 (tree-build) + tournament_pairs + 1 (feas-batch) + 5 (expand) + 0-5 (patch)
  ≈ 15-25 total  /  3-5 per idea
"""

import json
import os
import re
import sys
import threading

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from idea_tournament.tree_search import build_idea_tree
from idea_tournament.tournament import run_tournament_ranked

from systems.base import (
    IdeaGenerator,
    DEFAULT_MODEL,
    IDEA_FORMAT,
    _parse_batch_ideas,
    call_llm,
)


# ─── SOTA context ─────────────────────────────────────────────────────────────

def _get_sota_context(topic: str) -> str:
    try:
        from retrieval import get_topic_context
        return get_topic_context(topic, n=5) or ""
    except Exception:
        return ""


# ─── Thread-local pipeline cache ──────────────────────────────────────────────

_tls = threading.local()


def _tls_pipeline(topic: str, client, model: str, temperature: float):
    """Build tree + tournament + feasibility re-rank once per topic per thread."""
    if getattr(_tls, "topic", None) != topic:
        sota = _get_sota_context(topic)
        leaves = build_idea_tree(topic, sota, client, model, temperature=temperature)
        ranked = run_tournament_ranked(topic, leaves, client, model)
        # Level 1: re-rank by blended elo+feasibility score
        ranked = _rerank_with_feasibility(ranked, topic, client, model)
        _tls.topic = topic
        _tls.sota = sota
        _tls.ranked = ranked
        _tls.slot = 0
    return _tls.sota, _tls.ranked


# ─── Level 1: feasibility re-ranking ──────────────────────────────────────────

# Placeholder that won't appear in seed text
_SEEDS_PLACEHOLDER = "__LEAF_SEEDS__"

_FEASIBILITY_SCORE_PROMPT = (
    "You are evaluating research idea seeds for experimental feasibility at a "
    "typical academic lab (4-8 GPU cluster, public datasets, no proprietary hardware).\n\n"
    "Scoring rubric:\n"
    "  10 = Fully runnable with existing public resources; no exotic hardware needed\n"
    "   7 = Feasible with modest effort (fine-tune ≤13B model, standard benchmarks)\n"
    "   4 = Requires substantial resources (train 100B+ model, specialized cluster)\n"
    "   1 = Requires resources outside top-5 labs (trillion-param training, "
    "1000+ real QPUs, inaccessible proprietary data)\n\n"
    "Seeds to evaluate:\n"
    + _SEEDS_PLACEHOLDER
    + "\n\nRespond ONLY with a compact JSON object mapping 0-based integer index "
    "to integer score, e.g. {\"0\": 8, \"1\": 4, \"2\": 7}."
)


def _batch_feasibility_scores(
    leaves: list[dict], topic: str, client, model: str
) -> list[float]:
    """One LLM call → feasibility score per leaf, normalised to [0,1]."""
    if not leaves:
        return []
    seed_lines = []
    for i, leaf in enumerate(leaves):
        title = leaf.get("title") or ""
        desc = (leaf.get("description") or "")[:180]
        seed_lines.append(f"{i}. {title}: {desc}")
    seeds_text = "\n".join(seed_lines)
    prompt = _FEASIBILITY_SCORE_PROMPT.replace(_SEEDS_PLACEHOLDER, seeds_text)
    try:
        raw = call_llm(prompt, model, client, temperature=0.1, max_tokens=256)
        m = re.search(r"\{[^}]+\}", raw, re.DOTALL)
        if m:
            obj = json.loads(m.group())
            scores_raw = {int(k): float(v) for k, v in obj.items()}
            return [
                min(1.0, max(0.0, scores_raw.get(i, 5.0) / 10.0))
                for i in range(len(leaves))
            ]
    except Exception:
        pass
    return [0.5] * len(leaves)  # neutral fallback


def _rerank_with_feasibility(
    ranked: list[dict], topic: str, client, model: str, alpha: float = 0.40
) -> list[dict]:
    """Re-rank by: (1-α)·elo_rank_norm + α·feasibility_norm."""
    if not ranked:
        return ranked
    feas = _batch_feasibility_scores(ranked, topic, client, model)
    n = len(ranked)
    blended = []
    for i, (leaf, f) in enumerate(zip(ranked, feas)):
        rank_norm = 1.0 - i / n  # elo rank: best=1.0, worst→0.0
        score = (1.0 - alpha) * rank_norm + alpha * f
        blended.append((score, i, leaf))  # i as stable tiebreak
    blended.sort(key=lambda x: (-x[0], x[1]))
    return [item[2] for item in blended]


# ─── Level 2: feasibility-forward expansion ───────────────────────────────────

_FEASIBILITY_FRAME = (
    "\n\nFEASIBILITY REQUIREMENT — structure the EXPERIMENT section around a study "
    "executable at a standard academic lab:\n"
    "  • Model scale: use models of ≤13B parameters, or parameter-efficient methods "
    "(LoRA, adapters, probing) applied to larger frozen models.  Do NOT propose "
    "training trillion-parameter models from scratch.\n"
    "  • Quantum computing: use classical quantum-circuit simulators (≤30 qubits) "
    "or small NISQ devices in simulation mode.  Do NOT assume access to "
    "fault-tolerant or 1000+ physical-qubit systems.\n"
    "  • Datasets: cite specific, publicly available datasets by name.\n"
    "  • The scientific claim must be testable with these constraints — if the "
    "original direction requires more resources, *reformulate the hypothesis* "
    "around what IS demonstrable at this scale.  Novelty should live in the "
    "IDEA, APPROACH, and NOVELTY sections, not in the scale of the experiment."
)


def _expand_leaf(
    topic: str,
    leaf: dict,
    sota_context: str,
    model: str,
    client,
    temperature: float,
    variant_index: int = 0,
) -> str:
    title = leaf.get("title") or "Research direction"
    desc = leaf.get("description") or ""
    ctx = f"\n\n{sota_context}\n" if sota_context else ""
    variant_note = ""
    if variant_index > 0:
        variant_note = (
            f"\n\n(Benchmark variant #{variant_index + 1} from the same rank list — "
            "sharpen a *different* experimental emphasis or metric while keeping "
            "the same core insight.)\n"
        )
    prompt = (
        f"Research topic: {topic}{ctx}{variant_note}\n"
        "The following direction was selected after structured idea search and "
        "pairwise ranking among candidate leaves:\n\n"
        f"TITLE: {title}\n"
        f"SUMMARY: {desc}\n\n"
        "Write a complete research idea in the format below. Be concrete: name "
        "specific datasets, baselines, and evaluation metrics."
        + _FEASIBILITY_FRAME
        + "\n\n"
        + IDEA_FORMAT
    )
    return call_llm(prompt, model, client, temperature, max_tokens=2048)


# ─── Level 3: post-expansion feasibility audit & patch ────────────────────────

_IDEA_PLACEHOLDER = "__IDEA_BODY__"

_AUDIT_PROMPT = (
    "Review the EXPERIMENT section of this research idea for feasibility at a "
    "typical academic lab.\n\n"
    "IDEA:\n"
    + _IDEA_PLACEHOLDER
    + "\n\n"
    "Does the EXPERIMENT section propose ANY of the following?\n"
    "  (a) Training or fine-tuning models with >13B parameters end-to-end from scratch\n"
    "  (b) Real quantum hardware at scale (>50 physical qubits for ML, "
    "fault-tolerant systems)\n"
    "  (c) Datasets that are not publicly accessible\n"
    "  (d) Compute unavailable outside top-5 industry labs (>512 GPUs, TPU pods >128)\n\n"
    "If NONE apply, respond with exactly one word: FEASIBLE\n\n"
    "If ANY apply, respond with a replacement EXPERIMENT section only — start your "
    "response with the token EXPERIMENT: and rewrite that section with concrete, "
    "executable alternatives (smaller models, quantum simulators, public benchmarks) "
    "while preserving the scientific contribution.  Do NOT rewrite any other section."
)


def _patch_feasibility(
    idea_text: str, client, model: str, temperature: float
) -> str:
    """Audit and, if necessary, patch the EXPERIMENT section in place."""
    truncated = idea_text[:3500]
    prompt = _AUDIT_PROMPT.replace(_IDEA_PLACEHOLDER, truncated)
    try:
        response = call_llm(prompt, model, client, temperature=0.3, max_tokens=512)
        stripped = response.strip()
        if stripped.upper() == "FEASIBLE" or stripped.upper().startswith("FEASIBLE"):
            return idea_text
        # Expect "EXPERIMENT: ..." replacement
        m = re.search(r"EXPERIMENT\s*:(.*)", stripped, re.DOTALL | re.IGNORECASE)
        if not m:
            return idea_text
        new_exp_body = m.group(1).strip()
        new_exp_section = "EXPERIMENT: " + new_exp_body
        # Replace existing EXPERIMENT section (up to next ALL-CAPS section header or end)
        patched = re.sub(
            r"EXPERIMENT\s*:.*?(?=\n[A-Z]{3,}:|$)",
            new_exp_section + "\n\n",
            idea_text,
            flags=re.DOTALL | re.IGNORECASE,
        )
        return patched if patched != idea_text else idea_text
    except Exception:
        return idea_text


# ─── Generator class ──────────────────────────────────────────────────────────


class S25_r1Generator(IdeaGenerator):
    VERSION = "S25_r1"
    DESCRIPTION = (
        "IdeaTreeSearch + Swiss Elo + feasibility re-ranking (Level 1); "
        "feasibility-forward expansion prompt (Level 2); "
        "post-expansion EXPERIMENT audit & patch (Level 3)."
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
        try:
            sota, ranked = _tls_pipeline(topic, client, model, temperature)
        except Exception:
            sota, ranked = "", []

        if not ranked:
            try:
                return call_llm(
                    self.get_prompt(topic) + "\n\n" + IDEA_FORMAT,
                    model,
                    client,
                    temperature,
                )
            except Exception:
                return f"Research idea about {topic}: [generation failed]"

        i = _tls.slot
        _tls.slot = i + 1
        leaf = ranked[i % len(ranked)]
        variant = i // len(ranked)

        # Level 2: feasibility-forward expansion
        try:
            idea = _expand_leaf(topic, leaf, sota, model, client, temperature, variant)
        except Exception:
            idea = ""

        if not idea:
            try:
                idea = call_llm(
                    self.get_prompt(topic) + "\n\n" + IDEA_FORMAT,
                    model,
                    client,
                    temperature,
                )
            except Exception:
                idea = f"Research idea about {topic}: [expansion failed]"

        # Level 3: audit & patch EXPERIMENT section
        try:
            idea = _patch_feasibility(idea, client, model, temperature)
        except Exception:
            pass  # keep original if audit crashes

        return idea if idea else f"Research idea about {topic}: [empty result]"

    def generate_batch(
        self,
        topic: str,
        client,
        model: str = DEFAULT_MODEL,
        n: int = 5,
        temperature: float = 0.9,
    ) -> list[str]:
        """One tree + one ranked tournament + feasibility rerank; top-n expanded."""
        try:
            sota = _get_sota_context(topic)
            leaves = build_idea_tree(topic, sota, client, model, temperature=temperature)
            ranked = run_tournament_ranked(topic, leaves, client, model)
            ranked = _rerank_with_feasibility(ranked, topic, client, model)
        except Exception:
            ranked = []
            sota = ""

        if not ranked:
            try:
                raw = call_llm(
                    self.get_prompt(topic)
                    + f"\n\nProduce exactly {n} distinct ideas "
                    f"separated by a line containing only ---\n\n"
                    + IDEA_FORMAT,
                    model,
                    client,
                    temperature,
                    max_tokens=n * 800,
                )
                return _parse_batch_ideas(raw, n)
            except Exception:
                return [f"Research idea about {topic}: [batch failed]"] * n

        texts = []
        for i in range(n):
            leaf = ranked[i % len(ranked)]
            variant = i // len(ranked)
            try:
                idea = _expand_leaf(topic, leaf, sota, model, client, temperature, variant)
            except Exception:
                idea = ""
            if not idea:
                try:
                    idea = call_llm(
                        self.get_prompt(topic) + "\n\n" + IDEA_FORMAT,
                        model, client, temperature,
                    )
                except Exception:
                    idea = f"Research idea about {topic}: [idea {i} failed]"
            try:
                idea = _patch_feasibility(idea, client, model, temperature)
            except Exception:
                pass
            texts.append(idea if idea else f"Research idea about {topic}: [idea {i} empty]")
        return texts


GENERATOR = S25_r1Generator()
