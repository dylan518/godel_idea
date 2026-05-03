"""Judge: LLM pairwise evaluation of research ideas.

Primary judge: deepseek-chat / DeepSeek V3  (used for accept/reject decisions)
Blind judge:   gemini-flash-lite-latest     (independent canary — never optimized against)

Usage:
    python ideas/judge.py --results-a results/S0/ideas.json --results-b results/S1/ideas.json
"""

import argparse
import json
import os
import random
import re
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import log as _log

logger = _log.setup("judge")

JUDGE_MODEL = os.environ.get("IDEAS_JUDGE_MODEL", "deepseek-chat")
JUDGE_TEMPERATURE = float(os.environ.get("IDEAS_JUDGE_TEMPERATURE", "0.2"))

BLIND_JUDGE_MODEL = os.environ.get("IDEAS_BLIND_JUDGE_MODEL", "gemini-flash-lite-latest")
BLIND_JUDGE_TEMPERATURE = float(os.environ.get("IDEAS_BLIND_JUDGE_TEMPERATURE", "0.2"))


JUDGE_PROMPT_TEMPLATE = """\
You are an expert scientific reviewer evaluating two research ideas on the same topic.

Topic: {topic}

---
IDEA A:
{idea_a}

---
IDEA B:
{idea_b}

---
Score each idea on the following criteria (0-10 each):
1. Novelty: Is this a genuinely new angle, not obvious incremental work?
2. Scientific usefulness: Does it address real open problems, actionable by researchers?
3. Experimental clarity: Is the methodology clear with testable predictions?
4. Feasibility: Is it achievable with current or near-future resources?

Provide your evaluation in the following JSON format exactly:
{{
  "scores_a": {{"novelty": <0-10>, "scientific_usefulness": <0-10>, "experimental_clarity": <0-10>, "feasibility": <0-10>}},
  "scores_b": {{"novelty": <0-10>, "scientific_usefulness": <0-10>, "experimental_clarity": <0-10>, "feasibility": <0-10>}},
  "winner": "<A|B|tie>",
  "reasoning": "<2-3 sentence explanation of your verdict>"
}}

Base the winner on total score across all criteria. Declare a tie if totals differ by 2 or fewer points.
Respond with only the JSON object, no other text."""


def _parse_verdict(raw: str) -> dict:
    raw = re.sub(r"^```(?:json)?\s*", "", raw.strip())
    raw = re.sub(r"\s*```$", "", raw)
    try:
        verdict = json.loads(raw)
    except json.JSONDecodeError as e:
        raise ValueError(f"Judge returned invalid JSON: {e}\nRaw:\n{raw}") from e
    required = {"scores_a", "scores_b", "winner", "reasoning"}
    missing = required - verdict.keys()
    if missing:
        raise ValueError(f"Judge response missing fields: {missing}")
    if verdict["winner"] not in ("A", "B", "tie"):
        raise ValueError(f"Invalid winner value: {verdict['winner']!r}")
    return verdict


def judge_pair(topic: str, idea_a: str, idea_b: str, client, model: str = JUDGE_MODEL) -> dict:
    """Compare two ideas. Randomly shuffles A/B presentation to eliminate positional bias."""
    sys.path.insert(0, str(Path(__file__).parent / "systems"))
    from base import call_llm

    flipped = random.random() < 0.5
    presented_a, presented_b = (idea_b, idea_a) if flipped else (idea_a, idea_b)

    prompt = JUDGE_PROMPT_TEMPLATE.format(topic=topic, idea_a=presented_a, idea_b=presented_b)
    temperature = BLIND_JUDGE_TEMPERATURE if model == BLIND_JUDGE_MODEL else JUDGE_TEMPERATURE
    verdict = None
    prompt_curr = prompt
    for attempt in range(3):
        raw = call_llm(prompt_curr, model, client, temperature, max_tokens=2048)
        logger.debug("Raw %s response for '%s' (flipped=%s): %s", model, topic[:40], flipped, raw[:200])
        try:
            verdict = _parse_verdict(raw)
            break
        except ValueError:
            if attempt == 2:
                raise
            prompt_curr = (
                prompt
                + "\n\nYour previous response was not valid JSON. "
                + "Return ONLY a single valid JSON object with all fields present. "
                + "Do not include markdown fences or any extra text."
            )

    assert verdict is not None

    if flipped:
        verdict["scores_a"], verdict["scores_b"] = verdict["scores_b"], verdict["scores_a"]
        if verdict["winner"] == "A":
            verdict["winner"] = "B"
        elif verdict["winner"] == "B":
            verdict["winner"] = "A"

    logger.debug("Scores (flipped=%s) — A total=%d  B total=%d  winner=%s",
                 flipped, sum(verdict["scores_a"].values()),
                 sum(verdict["scores_b"].values()), verdict["winner"])
    return verdict


def _group_by_topic(results: list[dict]) -> dict[str, list[dict]]:
    groups: dict[str, list[dict]] = {}
    for entry in results:
        groups.setdefault(entry["topic_id"], []).append(entry)
    return groups


def _judge_topic(topic_id: str, topic: str, entries_a: list, entries_b: list,
                 client, model: str) -> list[dict]:
    """Judge all pairs for a single topic. Returns list of verdict dicts."""
    n_pairs = min(len(entries_a), len(entries_b))
    logger.info("Judging %s [%s] (%d pair(s)): %s",
                topic_id, model.split("-")[0], n_pairs, topic[:55])
    results = []
    for pair_idx in range(n_pairs):
        ea = entries_a[pair_idx]
        eb = entries_b[pair_idx]
        try:
            verdict = judge_pair(topic, ea["text"], eb["text"], client, model)
        except Exception as e:
            logger.error("Judge failed for %s pair %d: %s", topic_id, pair_idx, e)
            continue
        logger.info("  %s[%d] → %s | %s", topic_id, pair_idx, verdict["winner"],
                    verdict["reasoning"][:80])
        results.append({
            "topic_id": topic_id,
            "topic": topic,
            "idea_index": pair_idx,
            "system_a": ea["system_version"],
            "system_b": eb["system_version"],
            "judge_model": model,
            **verdict,
        })
    return results


def _check_early_stop(
    wins_b: int, wins_a: int, ties: int,
    n_total: int, threshold: float, min_pairs: int = 8,
) -> tuple[bool, str]:
    """Return (should_stop, reason).

    Stops early in either direction using Wilson score bounds (z=1.2816, ~90th pct):
      - Early reject: P(win_rate >= threshold) < 10%  — candidate can't recover
      - Early accept: P(win_rate <  threshold) < 10%  — candidate can't fall below

    min_pairs: don't fire before this many pairs have been judged (avoid noise).
    """
    k = wins_a + wins_b + ties
    if k < min_pairs or n_total <= 0:
        return False, ""

    remaining = n_total - k
    effective = wins_b + 0.5 * ties  # ties count as half-wins

    # Hard reject: even winning every remaining pair can't reach threshold
    if (effective + remaining) / n_total < threshold:
        max_rate = (effective + remaining) / n_total
        return True, (
            f"early stop (reject) — max achievable {max_rate:.1%} < threshold {threshold:.0%} "
            f"({k}/{n_total} pairs judged, {wins_b}W/{wins_a}L/{ties}T)"
        )

    # Hard accept: even losing every remaining pair stays above threshold
    if effective / n_total > threshold:
        min_rate = effective / n_total
        return True, (
            f"early stop (accept) — min achievable {min_rate:.1%} > threshold {threshold:.0%} "
            f"({k}/{n_total} pairs judged, {wins_b}W/{wins_a}L/{ties}T)"
        )

    # Wilson score bounds at 90th percentile (one-tailed, z=1.2816)
    p = effective / k
    z = 1.2816
    denom = 1.0 + z * z / k
    center = (p + z * z / (2.0 * k)) / denom
    spread = z * ((p * (1.0 - p) + z * z / (4.0 * k)) / k) ** 0.5 / denom
    p_upper = min(center + spread, 1.0)
    p_lower = max(center - spread, 0.0)

    if p_upper < threshold:
        return True, (
            f"early stop (reject) — P(win_rate≥{threshold:.0%}) < 10% "
            f"(90th-pct upper bound {p_upper:.1%} after {k}/{n_total} pairs, "
            f"{wins_b}W/{wins_a}L/{ties}T)"
        )

    if p_lower > threshold:
        return True, (
            f"early stop (accept) — P(win_rate<{threshold:.0%}) < 10% "
            f"(90th-pct lower bound {p_lower:.1%} after {k}/{n_total} pairs, "
            f"{wins_b}W/{wins_a}L/{ties}T)"
        )

    return False, ""


def compare_systems(results_a: list[dict], results_b: list[dict], client,
                    model: str = JUDGE_MODEL, workers: int = 1,
                    early_stop_threshold: float | None = None) -> dict:
    """Pairwise comparison across all shared topics.

    workers > 1 judges individual topic×idea pairs concurrently for faster throughput.
    early_stop_threshold: if set, stop judging early when P(win_rate >= threshold) < 10%.
      Pass the acceptance threshold (e.g. 0.55) to skip wasted judging on clear losers.
    """
    groups_a = _group_by_topic(results_a)
    groups_b = _group_by_topic(results_b)
    all_topics = sorted(set(groups_a) | set(groups_b))

    # Filter out topics missing from one side
    valid_topics = []
    for topic_id in all_topics:
        if not groups_a.get(topic_id) or not groups_b.get(topic_id):
            logger.warning("[SKIP] %s: missing results for one system", topic_id)
        else:
            valid_topics.append(topic_id)

    # Total pairs to judge (for early stopping denominator)
    n_total_pairs = sum(
        min(len(groups_a[t]), len(groups_b[t])) for t in valid_topics
    )

    wins_a = wins_b = ties = 0
    verdicts = []
    stopped_early = False

    def _collect(verdict):
        nonlocal wins_a, wins_b, ties
        if verdict is None:
            return
        if verdict["winner"] == "A":
            wins_a += 1
        elif verdict["winner"] == "B":
            wins_b += 1
        else:
            ties += 1
        verdicts.append(verdict)

    def _should_stop():
        if early_stop_threshold is None:
            return False, ""
        return _check_early_stop(wins_b, wins_a, ties, n_total_pairs,
                                 early_stop_threshold)

    work_items = []
    for topic_id in valid_topics:
        entries_a = groups_a[topic_id]
        entries_b = groups_b[topic_id]
        topic = entries_a[0]["topic"]
        for pair_idx in range(min(len(entries_a), len(entries_b))):
            work_items.append((topic_id, topic, pair_idx, entries_a[pair_idx], entries_b[pair_idx]))

    def _judge_one(item) -> dict | None:
        topic_id, topic, pair_idx, ea, eb = item
        logger.info("Judging %s[%d] [%s]: %s",
                    topic_id, pair_idx, model.split("-")[0], topic[:55])
        try:
            verdict = judge_pair(topic, ea["text"], eb["text"], client, model)
        except Exception as e:
            logger.error("Judge failed for %s pair %d: %s", topic_id, pair_idx, e)
            return None
        logger.info("  %s[%d] → %s | %s", topic_id, pair_idx, verdict["winner"],
                    verdict["reasoning"][:80])
        return {
            "topic_id": topic_id,
            "topic": topic,
            "idea_index": pair_idx,
            "system_a": ea["system_version"],
            "system_b": eb["system_version"],
            "judge_model": model,
            **verdict,
        }

    effective_workers = min(max(1, workers), len(work_items)) if work_items else 1
    if effective_workers <= 1:
        for item in work_items:
            _collect(_judge_one(item))
            stop, reason = _should_stop()
            if stop:
                logger.info("Early stop: %s", reason)
                stopped_early = True
                break
    else:
        with ThreadPoolExecutor(max_workers=effective_workers) as executor:
            futures = {executor.submit(_judge_one, item): item for item in work_items}
            for future in as_completed(futures):
                try:
                    _collect(future.result())
                except Exception as e:
                    logger.error("Judge future failed: %s", e)
                stop, reason = _should_stop()
                if stop:
                    logger.info("Early stop: %s", reason)
                    stopped_early = True
                    # Cancel any queued (not yet started) futures
                    for f in futures:
                        f.cancel()
                    break

    total = wins_a + wins_b + ties
    win_rate_b = (wins_b + 0.5 * ties) / total if total > 0 else 0.0
    suffix = f" [early stop after {total}/{n_total_pairs}]" if stopped_early else ""
    logger.info("Done [%s] — A:%d  B:%d  ties:%d  B win rate: %.1f%%%s",
                model.split("-")[0], wins_a, wins_b, ties, win_rate_b * 100, suffix)

    return {"wins_a": wins_a, "wins_b": wins_b, "ties": ties,
            "total_judged": total, "win_rate_b": win_rate_b,
            "verdicts": verdicts, "stopped_early": stopped_early}


def blind_compare_systems(results_a: list[dict], results_b: list[dict],
                          n_sample: int | None = None, workers: int = 1) -> dict:
    """Run blind Gemini judge on topics, using idea[0] per topic.

    n_sample=None (default) runs on ALL shared topics for a full confusion matrix.
    Pass an int to subsample (legacy behaviour).

    workers > 1 runs independent topic judgments concurrently.

    Deliberately isolated — creates its own client, never touches primary judge.
    Results are for tracking only, never used for accept/reject.
    Each verdict stores idea_index=0 so it can be matched against primary verdicts.
    """
    sys.path.insert(0, str(Path(__file__).parent / "systems"))
    from base import make_client

    client = make_client(BLIND_JUDGE_MODEL)  # None for Gemini (handled in call_llm)

    groups_a = _group_by_topic(results_a)
    groups_b = _group_by_topic(results_b)
    shared = sorted(set(groups_a) & set(groups_b))

    if n_sample is None:
        sample = shared          # all topics
    else:
        sample = random.sample(shared, min(n_sample, len(shared)))
    work_order = sorted(sample)
    logger.info("Blind judge [%s] running %d/%d topics (workers=%d)",
                BLIND_JUDGE_MODEL, len(sample), len(shared), max(1, workers))

    wins_a = wins_b = ties = 0
    verdicts = []

    def _one(topic_id: str) -> dict | None:
        topic = groups_a[topic_id][0]["topic"]
        logger.info("Blind judging %s: %s", topic_id, topic[:55])
        try:
            verdict = judge_pair(topic, groups_a[topic_id][0]["text"],
                                 groups_b[topic_id][0]["text"], client, BLIND_JUDGE_MODEL)
        except Exception as e:
            logger.error("Blind judge failed for %s: %s", topic_id, e)
            return None
        logger.info("  blind %s → %s | %s", topic_id, verdict["winner"],
                    verdict["reasoning"][:80])
        return {
            "topic_id": topic_id,
            "topic": topic,
            "idea_index": 0,
            "system_a": groups_a[topic_id][0]["system_version"],
            "system_b": groups_b[topic_id][0]["system_version"],
            "judge_model": BLIND_JUDGE_MODEL,
            **verdict,
        }

    effective_workers = min(max(1, workers), len(work_order)) if work_order else 1
    results_flat: list[dict | None]
    if effective_workers <= 1:
        results_flat = [_one(tid) for tid in work_order]
    else:
        results_flat = [None] * len(work_order)
        with ThreadPoolExecutor(max_workers=effective_workers) as executor:
            future_map = {executor.submit(_one, tid): i for i, tid in enumerate(work_order)}
            for fut in as_completed(future_map):
                i = future_map[fut]
                try:
                    results_flat[i] = fut.result()
                except Exception as e:
                    logger.error("Blind judge future failed: %s", e)

    for row in results_flat:
        if row is None:
            continue
        winner = row["winner"]
        if winner == "A":
            wins_a += 1
        elif winner == "B":
            wins_b += 1
        else:
            ties += 1
        verdicts.append(row)

    total = wins_a + wins_b + ties
    win_rate_b = (wins_b + 0.5 * ties) / total if total > 0 else 0.0
    logger.info("Blind done — A:%d  B:%d  ties:%d  blind win rate: %.1f%%",
                wins_a, wins_b, ties, win_rate_b * 100)

    return {"wins_a": wins_a, "wins_b": wins_b, "ties": ties,
            "total_judged": total, "win_rate_b": win_rate_b,
            "sampled_topics": sample, "verdicts": verdicts}


def compute_judge_confusion_matrix(
    primary_verdicts: list[dict],
    blind_verdicts: list[dict],
) -> dict:
    """Compute agreement between primary and blind judge on the same pairs.

    Matches by (topic_id, idea_index). Blind always uses idea_index=0.

    Returns:
      n_compared        — number of matched pairs
      agreement_rate    — fraction where both judges agree (same winner)
      flip_rate         — fraction where one says A and the other says B
                          (the key Goodhart signal — high flip means gaming)
      both_say_B        — both judges think candidate is better
      primary_only_B    — primary thinks B wins, blind thinks A wins
      blind_only_B      — blind thinks B wins, primary thinks A wins
      both_say_A        — both judges think champion is better
      matrix            — raw {primary_blind: count} dict
    """
    from collections import Counter

    primary_by_key = {
        (v["topic_id"], v.get("idea_index", 0)): v["winner"]
        for v in primary_verdicts
    }
    blind_by_key = {
        (v["topic_id"], v.get("idea_index", 0)): v["winner"]
        for v in blind_verdicts
    }

    overlap = set(primary_by_key) & set(blind_by_key)
    if not overlap:
        return {
            "n_compared": 0,
            "agreement_rate": None,
            "flip_rate": None,
            "both_say_B": 0, "primary_only_B": 0,
            "blind_only_B": 0, "both_say_A": 0,
            "matrix": {},
        }

    matrix: Counter = Counter()
    for key in overlap:
        p = primary_by_key[key]
        b = blind_by_key[key]
        matrix[f"{p}_{b}"] += 1

    n = len(overlap)
    agree = sum(c for k, c in matrix.items() if k.split("_")[0] == k.split("_")[1])
    flip  = matrix.get("A_B", 0) + matrix.get("B_A", 0)

    return {
        "n_compared": n,
        "agreement_rate": round(agree / n, 3),
        "flip_rate": round(flip / n, 3),
        "both_say_B":     matrix.get("B_B", 0),
        "primary_only_B": matrix.get("B_A", 0),   # primary→B, blind→A  (suspicious)
        "blind_only_B":   matrix.get("A_B", 0),   # primary→A, blind→B
        "both_say_A":     matrix.get("A_A", 0),
        "matrix": dict(matrix),
    }


def load_results(path: str) -> list[dict]:
    with open(path) as f:
        return json.load(f)


def main():
    parser = argparse.ArgumentParser(description="LLM pairwise judge for idea generator systems")
    parser.add_argument("--results-a", required=True)
    parser.add_argument("--results-b", required=True)
    parser.add_argument("--blind-n", type=int, default=2,
                        help="Topics to sample for blind Gemini judge (default: 2, 0 to skip)")
    args = parser.parse_args()

    sys.path.insert(0, str(Path(__file__).parent / "systems"))
    from base import make_client

    results_a = load_results(args.results_a)
    results_b = load_results(args.results_b)
    version_a = results_a[0]["system_version"] if results_a else "A"
    version_b = results_b[0]["system_version"] if results_b else "B"

    logger.info("Primary judge (%s): %s vs %s", JUDGE_MODEL, version_a, version_b)
    client = make_client(JUDGE_MODEL)
    comparison = compare_systems(results_a, results_b, client)

    threshold = 0.60
    logger.info("=" * 50)
    logger.info("%s wins %d, %s wins %d, ties %d (of %d)",
                version_a, comparison["wins_a"], version_b, comparison["wins_b"],
                comparison["ties"], comparison["total_judged"])
    logger.info("%s win rate: %.1f%%", version_b, comparison["win_rate_b"] * 100)
    if comparison["win_rate_b"] > threshold:
        logger.info("PASS: %s exceeds %.0f%% threshold", version_b, threshold * 100)
    else:
        logger.info("FAIL: does not meet %.0f%% threshold — keep %s", threshold * 100, version_a)

    if args.blind_n > 0:
        logger.info("Running blind judge (%s) on %d topics...", BLIND_JUDGE_MODEL, args.blind_n)
        blind = blind_compare_systems(results_a, results_b, n_sample=args.blind_n)
        logger.info("Blind win rate: %.1f%%", blind["win_rate_b"] * 100)


if __name__ == "__main__":
    main()
