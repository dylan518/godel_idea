"""Elo tournament for leaf idea ranking.

Implements Phase 2 of the EvoScientist idea-tournament skill.
Reference: ideas/idea-tournament/SKILL.md, references/elo-ranking-guide.md

Algorithm:
  - Swiss-system pairing (avoid rematches, pair similar Elo)
  - K=32, starting Elo=1500
  - 4 dimensions: Novelty, Feasibility, Relevance, Clarity (equal weight)
  - Rounds: 4 for ≥10 candidates, 3 for fewer

Edit targets:
  - ELO_K, ELO_START: rating system parameters
  - N_ROUNDS_LARGE, N_ROUNDS_SMALL: number of tournament rounds
  - TOURNAMENT_JUDGE_PROMPT in prompts.py: scoring criteria
"""

import json
import random
import re
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# ── Tournament parameters (edit these to tune) ────────────────────────────────
ELO_K = 32          # Rating change per match (higher = more volatile)
ELO_START = 1500.0  # Starting Elo for all candidates
N_ROUNDS_LARGE = 4  # Rounds for ≥10 candidates (paper recommends 4-5)
N_ROUNDS_SMALL = 3  # Rounds for <10 candidates


def _parse_json(raw: str) -> dict:
    raw = re.sub(r"^```(?:json)?\s*\n?", "", raw.strip())
    raw = re.sub(r"\n?```\s*$", "", raw)
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        m = re.search(r"\{[\s\S]*\}", raw)
        if m:
            return json.loads(m.group(0))
        raise


def _compare_pair(topic: str, idea_a: dict, idea_b: dict,
                  client, model: str) -> str:
    """Compare two leaf ideas. Returns 'A', 'B', or 'tie'.

    Randomizes A/B position to prevent positional bias.
    """
    import idea_tournament.prompts as P
    from systems.base import call_llm

    text_a = f"{idea_a.get('title', '')}\n{idea_a.get('description', '')}"
    text_b = f"{idea_b.get('title', '')}\n{idea_b.get('description', '')}"

    flipped = random.random() < 0.5
    pa, pb = (text_b, text_a) if flipped else (text_a, text_b)

    try:
        raw = call_llm(
            P.TOURNAMENT_JUDGE_PROMPT.format(topic=topic, idea_a=pa, idea_b=pb),
            model, client, temperature=0.1, max_tokens=300,
        )
        verdict = _parse_json(raw)
        winner = verdict.get("winner", "tie")
    except Exception:
        winner = "tie"

    if flipped:
        if winner == "A":
            winner = "B"
        elif winner == "B":
            winner = "A"
    return winner


def run_tournament(topic: str, leaves: list[dict], client, model: str) -> dict:
    """Swiss-system Elo tournament. Returns the winning leaf idea dict.

    Uses N_ROUNDS_LARGE or N_ROUNDS_SMALL depending on candidate count.
    Stops early if top-3 rankings stabilize (same as previous round).
    """
    import log as _log
    logger = _log.setup("tournament")

    if not leaves:
        return {}
    if len(leaves) == 1:
        return leaves[0]

    n = len(leaves)
    rounds = N_ROUNDS_LARGE if n >= 10 else N_ROUNDS_SMALL
    ratings = {i: ELO_START for i in range(n)}
    matchups: set = set()
    prev_top3 = None

    for rnd in range(rounds):
        # Swiss pairing: sort by Elo, pair adjacent unmatched
        ranked = sorted(ratings, key=lambda i: ratings[i], reverse=True)
        pairs, paired = [], set()
        for a in ranked:
            if a in paired:
                continue
            for b in ranked:
                if b == a or b in paired:
                    continue
                key = (min(a, b), max(a, b))
                if key not in matchups:
                    pairs.append((a, b))
                    paired.add(a)
                    paired.add(b)
                    matchups.add(key)
                    break

        if not pairs:
            break

        for a, b in pairs:
            winner = _compare_pair(topic, leaves[a], leaves[b], client, model)
            ea = 1.0 / (1.0 + 10 ** ((ratings[b] - ratings[a]) / 400))
            score_a = 1.0 if winner == "A" else (0.0 if winner == "B" else 0.5)
            ratings[a] += ELO_K * (score_a - ea)
            ratings[b] += ELO_K * ((1 - score_a) - (1 - ea))
            logger.debug("r%d: %s vs %s → %s (%.0f vs %.0f)",
                         rnd, leaves[a].get("id", a), leaves[b].get("id", b),
                         winner, ratings[a], ratings[b])

        # Early stop if top-3 stable
        top3 = tuple(sorted(ratings, key=lambda i: ratings[i], reverse=True)[:3])
        if top3 == prev_top3:
            logger.debug("Top-3 stable after round %d — stopping early", rnd + 1)
            break
        prev_top3 = top3

    best = max(ratings, key=lambda i: ratings[i])
    logger.info("Tournament: %s wins (Elo=%.0f) from %d candidates for '%s'",
                leaves[best].get("id", "?"), ratings[best], n, topic[:40])
    return leaves[best]
