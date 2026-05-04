"""S26_r1: IdeaTreeSearch + Swiss Elo + SOTA-grounded experimental anchor.

Key change from S_paper: adds a SOTA-derived experimental anchor step before
leaf expansion.  The anchor is generated from the SOTA literature context (NOT
from the selected leaf), so it reflects actually feasible experimental setups
rather than inheriting the leaf's speculative framing.

The expansion then combines the leaf's novel mechanism with the anchor's
concrete experimental design — structurally separating:
  • "what is novel"   → tournament-selected leaf (mechanism/insight)
  • "how to test it"  → SOTA-grounded anchor  (named datasets / baselines / metrics)

This targets the failure mode where the tournament selects speculative/ambitious
leaves and the expansion inherits and amplifies that speculation.  With the anchor
derived from the SOTA literature, the expansion LLM is constrained to ground its
EXPERIMENT section in named, already-established elements rather than inventing
novel benchmarks or requiring infrastructure not yet available.

LLM calls (typical): ~4 (tree) + tournament pairs + 1 (anchor/topic) + n (expand)
"""

import os
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


def _get_sota_context(topic: str) -> str:
    try:
        from retrieval import get_topic_context

        return get_topic_context(topic, n=5) or ""
    except Exception:
        return ""


# ---------------------------------------------------------------------------
# SOTA-grounded experimental anchor
# ---------------------------------------------------------------------------

_ANCHOR_PROMPT_HEAD = """\
Research topic: {topic}

Below is a summary of recent SOTA literature for this topic.

{sota}

Based ONLY on the papers described above, extract the concrete experimental \
infrastructure that already exists in this field:

DATASETS: Name the specific benchmark datasets mentioned in the literature \
(exact names, e.g. "SQuAD 2.0", "IBM Quantum ibmq_toronto 27-qubit device", \
"StarCraft II benchmark"). If none are explicitly named, write "none identified".

BASELINES: Name the specific methods or systems cited as comparison points \
(exact names, e.g. "BERT-large", "Surface Code with MWPM decoder", \
"Rainbow DQN"). If none are explicitly named, write "none identified".

METRICS: State the 1-2 primary evaluation metrics used in this field \
(e.g. "logical error rate", "F1 score", "win rate"). \
If none are named, write "none identified".

SETUP: One sentence on the standard experimental scale / hardware already \
established in this literature. If not clear, write "none identified".

Only report what appears in the text above. Do not invent or infer beyond it.\
"""


def _extract_sota_anchor(
    topic: str,
    sota_context: str,
    model: str,
    client,
    temperature: float,
) -> str:
    """Generate a concrete experimental anchor from SOTA literature.

    Critically: derived from the SOTA context, NOT from the selected leaf.
    This prevents the anchor from inheriting the leaf's speculative framing.
    If SOTA context is empty or the call fails, returns "" (no anchor applied).
    """
    if not sota_context or not sota_context.strip():
        return ""

    prompt = _ANCHOR_PROMPT_HEAD.replace("{topic}", topic).replace(
        "{sota}", sota_context
    )
    try:
        return call_llm(prompt, model, client, temperature=0.1, max_tokens=400)
    except Exception:
        return ""


# Runner calls generate_idea() n times per topic (not generate_batch). Use TLS so
# parallel topic workers each build the tree + tournament once per topic.
_tls = threading.local()


def _tls_pipeline(topic: str, client, model: str, temperature: float):
    if getattr(_tls, "topic", None) != topic:
        sota = _get_sota_context(topic)
        leaves = build_idea_tree(topic, sota, client, model, temperature=temperature)
        ranked = run_tournament_ranked(topic, leaves, client, model)
        # Anchor is derived from SOTA, not from any leaf — must run after sota is set.
        anchor = _extract_sota_anchor(topic, sota, model, client, temperature)
        _tls.topic = topic
        _tls.sota = sota
        _tls.ranked = ranked
        _tls.anchor = anchor
        _tls.slot = 0
    return _tls.sota, _tls.ranked, _tls.anchor


# ---------------------------------------------------------------------------
# Leaf expansion with SOTA-grounded anchor
# ---------------------------------------------------------------------------

def _expand_leaf(
    topic: str,
    leaf: dict,
    sota_context: str,
    sota_anchor: str,
    model: str,
    client,
    temperature: float,
    variant_index: int = 0,
) -> str:
    title = leaf.get("title") or "Research direction"
    desc = leaf.get("description") or ""
    ctx = ("\n\n" + sota_context + "\n") if sota_context else ""

    variant_note = ""
    if variant_index > 0:
        variant_note = (
            "\n\n(Variant #"
            + str(variant_index + 1)
            + " from the same ranked list — apply the same core mechanism to a "
            "different angle while keeping the experimental setup grounded in "
            "the anchor below.)\n"
        )

    # Anchor block: forces the EXPERIMENT section to name concrete elements
    # drawn from the SOTA literature, not from the leaf's framing.
    anchor_block = ""
    if sota_anchor and sota_anchor.strip():
        anchor_block = (
            "\n\nEXPERIMENTAL ANCHOR"
            " (extracted from SOTA literature — not from the direction below):\n"
            + sota_anchor
            + "\n\nCONSTRAINT: Your EXPERIMENT section MUST name the specific "
            "datasets, baselines, and metrics listed in the anchor above. "
            "The source of novelty is the mechanism described in the selected "
            "direction — not a new benchmark domain or infrastructure not yet "
            "established in the literature.\n"
        )

    prompt = (
        "Research topic: "
        + topic
        + ctx
        + variant_note
        + "\nThe following direction was selected by Elo tournament from "
        "structured idea search:\n\n"
        "TITLE: "
        + title
        + "\nSUMMARY: "
        + desc
        + anchor_block
        + "\n\nWrite a complete research idea. "
        "The IDEA and NOVELTY sections should capture the mechanism/insight "
        "from the selected direction above. "
        "The EXPERIMENT section must be grounded in the named datasets, "
        "baselines, and metrics from the anchor — do not require novel "
        "benchmarks or infrastructure not already established in the field.\n"
        + IDEA_FORMAT
    )

    try:
        return call_llm(prompt, model, client, temperature, max_tokens=2048)
    except Exception:
        # Fallback: simple single-call prompt with just the topic
        fallback = (
            "Research topic: "
            + topic
            + "\nGenerate a concrete, testable research idea.\n"
            + IDEA_FORMAT
        )
        try:
            return call_llm(fallback, model, client, temperature, max_tokens=1024)
        except Exception:
            return (
                "IDEA: Novel approach to " + topic + "\n\n"
                "BACKGROUND: Open problem in " + topic + "\n\n"
                "APPROACH: Proposed method\n\n"
                "EXPERIMENT: Evaluate on standard benchmarks with named baselines\n\n"
                "NOVELTY: Differs from prior work"
            )


class S26_r1Generator(IdeaGenerator):
    VERSION = "S26_r1"
    DESCRIPTION = (
        "IdeaTreeSearch + Swiss Elo + SOTA-grounded experimental anchor. "
        "Anchor is derived from SOTA literature (not from the selected leaf) "
        "to prevent speculation from propagating into the expansion step."
    )

    def get_prompt(self, topic: str) -> str:
        return "Generate a novel research idea about: " + topic

    def generate_idea(
        self,
        topic: str,
        client,
        model: str = DEFAULT_MODEL,
        temperature: float = 0.8,
    ) -> str:
        sota, ranked, anchor = _tls_pipeline(topic, client, model, temperature)
        if not ranked:
            try:
                return call_llm(
                    self.get_prompt(topic) + "\n\n" + IDEA_FORMAT,
                    model,
                    client,
                    temperature,
                )
            except Exception:
                return (
                    "IDEA: Novel approach to " + topic + "\n\n"
                    "BACKGROUND: Open problem in " + topic + "\n\n"
                    "APPROACH: Proposed method\n\n"
                    "EXPERIMENT: Evaluate on standard benchmarks\n\n"
                    "NOVELTY: Differs from prior work"
                )
        i = _tls.slot
        _tls.slot = i + 1
        leaf = ranked[i % len(ranked)]
        variant = i // len(ranked)
        return _expand_leaf(
            topic, leaf, sota, anchor, model, client, temperature,
            variant_index=variant,
        )

    def generate_batch(
        self,
        topic: str,
        client,
        model: str = DEFAULT_MODEL,
        n: int = 5,
        temperature: float = 0.9,
    ) -> list[str]:
        """One tree + one ranked tournament + one anchor; top-n ranked leaves expanded."""
        sota = _get_sota_context(topic)
        leaves = build_idea_tree(topic, sota, client, model, temperature=temperature)
        ranked = run_tournament_ranked(topic, leaves, client, model)
        anchor = _extract_sota_anchor(topic, sota, model, client, temperature)
        if not ranked:
            try:
                raw = call_llm(
                    self.get_prompt(topic)
                    + "\n\nProduce exactly "
                    + str(n)
                    + " distinct ideas separated by a line containing only ---\n\n"
                    + IDEA_FORMAT,
                    model,
                    client,
                    temperature,
                    max_tokens=n * 800,
                )
                return _parse_batch_ideas(raw, n)
            except Exception:
                fallback_idea = (
                    "IDEA: Novel approach to " + topic + "\n\n"
                    "BACKGROUND: Open problem in " + topic + "\n\n"
                    "APPROACH: Proposed method\n\n"
                    "EXPERIMENT: Evaluate on standard benchmarks\n\n"
                    "NOVELTY: Differs from prior work"
                )
                return [fallback_idea] * n

        texts = []
        for i in range(n):
            leaf = ranked[i % len(ranked)]
            variant = i // len(ranked)
            texts.append(
                _expand_leaf(
                    topic, leaf, sota, anchor, model, client, temperature,
                    variant_index=variant,
                )
            )
        return texts


GENERATOR = S26_r1Generator()
