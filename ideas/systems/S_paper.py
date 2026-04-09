"""S_paper: IdeaTreeSearch + Elo wired to repo-root ``skills/`` (idea-tournament + research-ideation).

Uses ``ideas/idea_tournament/`` Python (tree_search + tournament). Prompts load canonical
Markdown from ``skills/idea-tournament/references/*.md`` and
``skills/research-ideation/references/literature-tree.md`` (same sources as Claude Code).

Per benchmark topic (``generate_idea`` × n_ideas uses thread-local cache):
  1. OpenAlex SOTA context (same retrieval as S_sota / S15)
  2. ``build_idea_tree`` — L1→L2→L3 JSON tree + review (4 LLM calls), once per topic
  3. ``run_tournament_ranked`` — Swiss Elo on leaf dicts (paper-style judge), once per topic
  4. Each idea slot: expand one ranked leaf to ``IDEA_FORMAT`` (1 LLM call each)

``generate_batch`` runs steps 1–3 once, then step 4 for ``n`` leaves (fresh-eval path).

This is **not** the LangGraph EvoScientist agent from the main package; it is the
paper's *idea-search* subroutine, runnable under ``godel_loop.py`` / ``runner.py``.

LLM calls (typical): ~4 (tree) + tournament pairs + n (expand). Re-benchmark vs S15
after editing ``skills/…`` rubrics or ``idea_tournament/prompts.py``.
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


# Runner calls generate_idea() n times per topic (not generate_batch). Use TLS so
# parallel topic workers each build the tree + tournament once per topic.
_tls = threading.local()


def _tls_pipeline(topic: str, client, model: str, temperature: float):
    if getattr(_tls, "topic", None) != topic:
        sota = _get_sota_context(topic)
        leaves = build_idea_tree(topic, sota, client, model, temperature=temperature)
        ranked = run_tournament_ranked(topic, leaves, client, model)
        _tls.topic = topic
        _tls.sota = sota
        _tls.ranked = ranked
        _tls.slot = 0
    return _tls.sota, _tls.ranked


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
            f"\n\n(This is benchmark variant #{variant_index + 1} from the same "
            "tree-tournament rank list — sharpen a *different* experimental emphasis "
            "or metric choice while keeping the same core insight.)\n"
        )
    prompt = (
        f"Research topic: {topic}{ctx}{variant_note}\n"
        "The following direction was selected after structured idea search and "
        "pairwise ranking among candidate leaves:\n\n"
        f"TITLE: {title}\n"
        f"SUMMARY: {desc}\n\n"
        "Write a complete research idea in the format below. Be concrete: name "
        "datasets, baselines, or evaluation metrics where applicable.\n"
        + IDEA_FORMAT
    )
    return call_llm(prompt, model, client, temperature, max_tokens=2048)


class S_paperGenerator(IdeaGenerator):
    VERSION = "S_paper"
    DESCRIPTION = (
        "IdeaTreeSearch + Swiss Elo; prompts append repo skills/idea-tournament + "
        "skills/research-ideation references; expansion to IDEA_FORMAT."
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
        sota, ranked = _tls_pipeline(topic, client, model, temperature)
        if not ranked:
            return call_llm(
                self.get_prompt(topic) + "\n\n" + IDEA_FORMAT,
                model,
                client,
                temperature,
            )
        i = _tls.slot
        _tls.slot = i + 1
        leaf = ranked[i % len(ranked)]
        variant = i // len(ranked)
        return _expand_leaf(
            topic, leaf, sota, model, client, temperature, variant_index=variant
        )

    def generate_batch(
        self,
        topic: str,
        client,
        model: str = DEFAULT_MODEL,
        n: int = 5,
        temperature: float = 0.9,
    ) -> list[str]:
        """One tree + one ranked tournament; top-n ranked leaves expanded separately."""
        sota = _get_sota_context(topic)
        leaves = build_idea_tree(topic, sota, client, model, temperature=temperature)
        ranked = run_tournament_ranked(topic, leaves, client, model)
        if not ranked:
            raw = call_llm(
                self.get_prompt(topic) + f"\n\nProduce exactly {n} distinct ideas "
                f"separated by a line containing only ---\n\n" + IDEA_FORMAT,
                model,
                client,
                temperature,
                max_tokens=n * 800,
            )
            return _parse_batch_ideas(raw, n)

        texts = []
        for i in range(n):
            leaf = ranked[i % len(ranked)]
            variant = i // len(ranked)
            texts.append(
                _expand_leaf(
                    topic, leaf, sota, model, client, temperature, variant_index=variant
                )
            )
        return texts


GENERATOR = S_paperGenerator()
