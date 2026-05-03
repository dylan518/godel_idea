"""Elo tournament for leaf idea ranking.

Implements Phase 2 of the EvoScientist idea-tournament skill.
Canonical rubric: repo ``skills/idea-tournament/references/elo-ranking-guide.md`` (appended in ``prompts.format_tournament_judge_prompt``).

Algorithm:
  - Swiss-system pairing (avoid rematches, pair similar Elo)
  - K=32, starting Elo=1500
  - 4 dimensions: Novelty, Feasibility, Relevance, Clarity (equal weight)
  - Rounds: 4 for ≥10 candidates, 3 for fewer

Edit targets:
  - ELO_K, ELO_START: rating system parameters
  - N_ROUNDS_LARGE, N_ROUNDS_SMALL: number of tournament rounds
  - ``skills/idea-tournament/references/elo-ranking-guide.md`` and ``prompts.format_tournament_judge_prompt``
"""

import json
import random
import re
import sys
import os
from concurrent.futures import ThreadPoolExecutor, as_completed

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# ── Tournament parameters (edit these to tune) ────────────────────────────────
ELO_K = 32          # Rating change per match (higher = more volatile)
ELO_START = 1500.0  # Starting Elo for all candidates
N_ROUNDS_LARGE = 4  # Rounds for ≥10 candidates (paper recommends 4-5)
N_ROUNDS_SMALL = 3  # Rounds for <10 candidates


def _parallel_workers(n_pairs: int) -> int:
    """Tournament pair concurrency for a single Swiss round."""
    try:
        requested = int(os.environ.get("IDEAS_TOURNAMENT_WORKERS", "1"))
    except ValueError:
        requested = 1
    return min(max(1, requested), max(1, n_pairs))


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
            P.format_tournament_judge_prompt(topic, pa, pb),
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


def _run_swiss_elo(
    topic: str,
    leaves: list[dict],
    client,
    model: str,
    logger,
) -> dict[int, float]:
    """Run Swiss Elo rounds; return index → rating."""
    n = len(leaves)
    rounds = N_ROUNDS_LARGE if n >= 10 else N_ROUNDS_SMALL
    ratings = {i: ELO_START for i in range(n)}
    matchups: set = set()
    prev_top3 = None

    for rnd in range(rounds):
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

        winners: dict[tuple[int, int], str] = {}
        workers = _parallel_workers(len(pairs))
        if workers <= 1:
            for a, b in pairs:
                winners[(a, b)] = _compare_pair(topic, leaves[a], leaves[b], client, model)
        else:
            with ThreadPoolExecutor(max_workers=workers) as executor:
                future_map = {
                    executor.submit(_compare_pair, topic, leaves[a], leaves[b], client, model): (a, b)
                    for a, b in pairs
                }
                for future in as_completed(future_map):
                    pair = future_map[future]
                    try:
                        winners[pair] = future.result()
                    except Exception:
                        winners[pair] = "tie"

        # Apply Elo updates in deterministic pair order after all parallel calls
        # for the round finish, preserving Swiss-round semantics.
        for a, b in pairs:
            winner = winners.get((a, b), "tie")
            ea = 1.0 / (1.0 + 10 ** ((ratings[b] - ratings[a]) / 400))
            score_a = 1.0 if winner == "A" else (0.0 if winner == "B" else 0.5)
            ratings[a] += ELO_K * (score_a - ea)
            ratings[b] += ELO_K * ((1 - score_a) - (1 - ea))
            logger.debug("r%d: %s vs %s → %s (%.0f vs %.0f)",
                         rnd, leaves[a].get("id", a), leaves[b].get("id", b),
                         winner, ratings[a], ratings[b])

        top3 = tuple(sorted(ratings, key=lambda i: ratings[i], reverse=True)[:3])
        if top3 == prev_top3:
            logger.debug("Top-3 stable after round %d — stopping early", rnd + 1)
            break
        prev_top3 = top3

    return ratings


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
    ratings = _run_swiss_elo(topic, leaves, client, model, logger)
    best = max(ratings, key=lambda i: ratings[i])
    logger.info("Tournament: %s wins (Elo=%.0f) from %d candidates for '%s'",
                leaves[best].get("id", "?"), ratings[best], n, topic[:40])
    return leaves[best]


def run_tournament_ranked(topic: str, leaves: list[dict], client, model: str) -> list[dict]:
    """Same pairing/Elo as ``run_tournament``, but return all leaves sorted best-first.

    Used when a benchmark needs multiple distinct ideas per topic (e.g. n_ideas=5)
    without re-running IdeaTreeSearch for each slot.
    """
    import log as _log
    logger = _log.setup("tournament")

    if not leaves:
        return []
    if len(leaves) == 1:
        return list(leaves)

    ratings = _run_swiss_elo(topic, leaves, client, model, logger)
    order = sorted(ratings.keys(), key=lambda i: ratings[i], reverse=True)
    ranked = [leaves[i] for i in order]
    logger.info("Tournament ranked %d leaves for '%s' (best id=%s)",
                len(ranked), topic[:40], ranked[0].get("id", "?"))
    return ranked
