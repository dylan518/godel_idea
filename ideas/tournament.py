"""Elo tournament for intra-topic idea selection.

Given N candidate ideas for the same topic, runs Swiss-system Elo rounds
to identify the strongest idea. The winner replaces the N independent ideas
before cross-system pairwise comparison — raising quality before the "finals".

Usage:
    from tournament import run_tournament
    best_idea = run_tournament(topic, ideas, client, model)
"""

import json
import random
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import log as _log

logger = _log.setup("tournament")

TOURNAMENT_JUDGE_PROMPT = """\
You are a rigorous scientific reviewer comparing two research ideas on the same topic.

Topic: {topic}

---
IDEA A:
{idea_a}

---
IDEA B:
{idea_b}

---
Evaluate both ideas on these four dimensions (0-10 each):
1. Novelty — genuinely new angle, not obvious incremental work
2. Scientific usefulness — addresses real open problems, actionable
3. Experimental clarity — concrete methodology with testable predictions
4. Feasibility — achievable with current or near-future resources

Respond with ONLY this JSON:
{{
  "scores_a": {{"novelty": <0-10>, "usefulness": <0-10>, "clarity": <0-10>, "feasibility": <0-10>}},
  "scores_b": {{"novelty": <0-10>, "usefulness": <0-10>, "clarity": <0-10>, "feasibility": <0-10>}},
  "winner": "<A|B|tie>"
}}
Declare a tie if totals differ by 2 or fewer points. No other text."""

ELO_K = 32
ELO_START = 1500


def _elo_expected(rating_a: float, rating_b: float) -> float:
    return 1.0 / (1.0 + 10 ** ((rating_b - rating_a) / 400))


def _elo_update(rating_a: float, rating_b: float, score_a: float) -> tuple[float, float]:
    """score_a: 1=A wins, 0=B wins, 0.5=tie."""
    ea = _elo_expected(rating_a, rating_b)
    eb = _elo_expected(rating_b, rating_a)
    return (
        rating_a + ELO_K * (score_a - ea),
        rating_b + ELO_K * ((1 - score_a) - eb),
    )


def _compare_pair(topic: str, idea_a: str, idea_b: str, client, model: str) -> str:
    """Compare two ideas; returns 'A', 'B', or 'tie'."""
    from systems.base import call_llm

    flipped = random.random() < 0.5
    presented_a, presented_b = (idea_b, idea_a) if flipped else (idea_a, idea_b)

    prompt = TOURNAMENT_JUDGE_PROMPT.format(
        topic=topic, idea_a=presented_a, idea_b=presented_b
    )
    try:
        raw = call_llm(prompt, model, client, temperature=0.1, max_tokens=256)
        raw = re.sub(r"^```(?:json)?\s*", "", raw.strip())
        raw = re.sub(r"\s*```$", "", raw)
        verdict = json.loads(raw)
        winner = verdict.get("winner", "tie")
    except Exception as e:
        logger.warning("Tournament comparison failed: %s — defaulting to tie", e)
        winner = "tie"

    if flipped:
        if winner == "A":
            winner = "B"
        elif winner == "B":
            winner = "A"

    return winner


def run_tournament(topic: str, ideas: list[str], client, model: str,
                   rounds: int = 3) -> str:
    """Run Swiss Elo tournament among ideas. Returns the winning idea text.

    Uses the generator model (cheap, fast) for intra-topic ranking —
    a lighter touch than the full primary judge.

    rounds: number of Swiss-system rounds (default 3 for N≤5 ideas)
    """
    if len(ideas) == 0:
        raise ValueError("No ideas to tournament")
    if len(ideas) == 1:
        return ideas[0]

    n = len(ideas)
    ratings = {i: float(ELO_START) for i in range(n)}
    matchups: set[tuple] = set()

    for rnd in range(rounds):
        # Swiss pairing: sort by current rating, pair adjacent
        ranked = sorted(ratings, key=lambda i: ratings[i], reverse=True)
        pairs = []
        paired = set()
        for idx in range(len(ranked)):
            a = ranked[idx]
            if a in paired:
                continue
            for jdx in range(idx + 1, len(ranked)):
                b = ranked[jdx]
                if b in paired:
                    continue
                if (min(a, b), max(a, b)) not in matchups:
                    pairs.append((a, b))
                    paired.add(a)
                    paired.add(b)
                    matchups.add((min(a, b), max(a, b)))
                    break

        if not pairs:
            break  # all pairs exhausted

        for a, b in pairs:
            winner = _compare_pair(topic, ideas[a], ideas[b], client, model)
            if winner == "A":
                score_a = 1.0
            elif winner == "B":
                score_a = 0.0
            else:
                score_a = 0.5

            ratings[a], ratings[b] = _elo_update(ratings[a], ratings[b], score_a)
            logger.debug("Tournament [%s] r%d: idea%d vs idea%d → %s (ratings: %.0f vs %.0f)",
                         topic[:30], rnd, a, b, winner, ratings[a], ratings[b])

    best_idx = max(ratings, key=lambda i: ratings[i])
    logger.info("Tournament winner: idea %d (Elo=%.0f) out of %d for topic '%s'",
                best_idx, ratings[best_idx], n, topic[:40])
    return ideas[best_idx]
