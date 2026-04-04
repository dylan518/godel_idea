"""Judge: LLM pairwise evaluation of research ideas.

Primary judge: claude-haiku-4-5  (used for accept/reject decisions)
Blind judge:   gemini-2.0-flash  (independent canary — never optimized against)

Usage:
    python ideas/judge.py --results-a results/S0/ideas.json --results-b results/S1/ideas.json
"""

import argparse
import json
import random
import re
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import log as _log

logger = _log.setup("judge")

JUDGE_MODEL = "claude-haiku-4-5-20251001"
JUDGE_TEMPERATURE = 0.2

BLIND_JUDGE_MODEL = "gemini-flash-lite-latest"
BLIND_JUDGE_TEMPERATURE = 0.2


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
    raw = call_llm(prompt, model, client, temperature, max_tokens=1024)
    logger.debug("Raw %s response for '%s' (flipped=%s): %s", model, topic[:40], flipped, raw[:200])

    verdict = _parse_verdict(raw)

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


def compare_systems(results_a: list[dict], results_b: list[dict], client,
                    model: str = JUDGE_MODEL, workers: int = 1) -> dict:
    """Pairwise comparison across all shared topics.

    workers > 1 judges multiple topics concurrently for faster throughput.
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

    wins_a = wins_b = ties = 0
    verdicts = []

    def _collect(topic_verdicts):
        nonlocal wins_a, wins_b, ties
        for v in topic_verdicts:
            if v["winner"] == "A":
                wins_a += 1
            elif v["winner"] == "B":
                wins_b += 1
            else:
                ties += 1
            verdicts.append(v)

    effective_workers = min(workers, len(valid_topics))
    if effective_workers <= 1:
        for topic_id in valid_topics:
            topic = groups_a[topic_id][0]["topic"]
            _collect(_judge_topic(topic_id, topic, groups_a[topic_id],
                                  groups_b[topic_id], client, model))
    else:
        with ThreadPoolExecutor(max_workers=effective_workers) as executor:
            futures = {
                executor.submit(
                    _judge_topic, topic_id, groups_a[topic_id][0]["topic"],
                    groups_a[topic_id], groups_b[topic_id], client, model
                ): topic_id
                for topic_id in valid_topics
            }
            for future in as_completed(futures):
                _collect(future.result())

    total = wins_a + wins_b + ties
    win_rate_b = (wins_b + 0.5 * ties) / total if total > 0 else 0.0
    logger.info("Done [%s] — A:%d  B:%d  ties:%d  B win rate: %.1f%%",
                model.split("-")[0], wins_a, wins_b, ties, win_rate_b * 100)

    return {"wins_a": wins_a, "wins_b": wins_b, "ties": ties,
            "total_judged": total, "win_rate_b": win_rate_b, "verdicts": verdicts}


def blind_compare_systems(results_a: list[dict], results_b: list[dict],
                          n_sample: int = 2) -> dict:
    """Run blind Gemini judge on a random sample of topics.

    Deliberately isolated — creates its own client, never touches primary judge.
    Results are for tracking only, never used for accept/reject.
    """
    sys.path.insert(0, str(Path(__file__).parent / "systems"))
    from base import make_client

    client = make_client(BLIND_JUDGE_MODEL)  # None for Gemini (handled in call_llm)

    groups_a = _group_by_topic(results_a)
    groups_b = _group_by_topic(results_b)
    shared = sorted(set(groups_a) & set(groups_b))

    sample = random.sample(shared, min(n_sample, len(shared)))
    logger.info("Blind judge [%s] sampling %d/%d topics: %s",
                BLIND_JUDGE_MODEL, len(sample), len(shared), sample)

    wins_a = wins_b = ties = 0
    verdicts = []

    for topic_id in sample:
        topic = groups_a[topic_id][0]["topic"]
        logger.info("Blind judging %s: %s", topic_id, topic[:55])
        try:
            verdict = judge_pair(topic, groups_a[topic_id][0]["text"],
                                 groups_b[topic_id][0]["text"], client, BLIND_JUDGE_MODEL)
        except Exception as e:
            logger.error("Blind judge failed for %s: %s", topic_id, e)
            continue

        winner = verdict["winner"]
        if winner == "A":
            wins_a += 1
        elif winner == "B":
            wins_b += 1
        else:
            ties += 1

        logger.info("  blind %s → %s | %s", topic_id, winner, verdict["reasoning"][:80])
        verdicts.append({
            "topic_id": topic_id,
            "topic": topic,
            "system_a": groups_a[topic_id][0]["system_version"],
            "system_b": groups_b[topic_id][0]["system_version"],
            "judge_model": BLIND_JUDGE_MODEL,
            **verdict,
        })

    total = wins_a + wins_b + ties
    win_rate_b = (wins_b + 0.5 * ties) / total if total > 0 else 0.0
    logger.info("Blind done — A:%d  B:%d  ties:%d  blind win rate: %.1f%%",
                wins_a, wins_b, ties, win_rate_b * 100)

    return {"wins_a": wins_a, "wins_b": wins_b, "ties": ties,
            "total_judged": total, "win_rate_b": win_rate_b,
            "sampled_topics": sample, "verdicts": verdicts}


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
