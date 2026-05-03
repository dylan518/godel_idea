#!/usr/bin/env python3
"""Compute topic-block bootstrap CIs for compare_* result files.

Workshop-safe standalone script to avoid conflicts with existing bootstrap_ci.py.
"""

from __future__ import annotations

import argparse
import json
import random
from dataclasses import dataclass
from pathlib import Path

from workshop_result_excludes import PRIMARY_COMPARE_EXCLUDE, skip_compare_path


ROOT = Path(__file__).resolve().parent
RESULTS_DIR = ROOT / "results"


@dataclass
class Summary:
    file: str
    n_pairs: int
    n_topics: int
    point_win_rate_b: float
    ci95_low: float
    ci95_high: float
    wins_a: int | None = None
    wins_b: int | None = None
    ties: int | None = None


def _winner_to_b_win(winner: str | None) -> float:
    if winner == "B":
        return 1.0
    if winner == "tie":
        return 0.5
    return 0.0


def _quantile(sorted_vals: list[float], q: float) -> float:
    n = len(sorted_vals)
    if n == 0:
        raise ValueError("Cannot compute quantile of empty list")
    if n == 1:
        return sorted_vals[0]
    idx = q * (n - 1)
    lo = int(idx)
    hi = min(lo + 1, n - 1)
    frac = idx - lo
    return sorted_vals[lo] * (1.0 - frac) + sorted_vals[hi] * frac


def summarize_verdict_json(path: Path, iters: int, rng: random.Random) -> Summary:
    obj = json.loads(path.read_text())
    verdicts = obj.get("verdicts", [])
    if not verdicts:
        raise ValueError(f"No verdicts found in {path}")

    by_topic: dict[str, list[float]] = {}
    for row in verdicts:
        topic_id = str(row.get("topic_id", "UNKNOWN"))
        by_topic.setdefault(topic_id, []).append(_winner_to_b_win(row.get("winner")))

    topics = sorted(by_topic.keys())
    n_topics = len(topics)
    if n_topics == 0:
        raise ValueError(f"No topic ids found in verdicts for {path}")

    point_num = sum(sum(by_topic[t]) for t in topics)
    point_den = sum(len(by_topic[t]) for t in topics)
    point = point_num / point_den if point_den else 0.0

    boots: list[float] = []
    for _ in range(iters):
        sample_topics = [topics[rng.randrange(n_topics)] for _ in range(n_topics)]
        num = 0.0
        den = 0
        for tid in sample_topics:
            vals = by_topic[tid]
            num += sum(vals)
            den += len(vals)
        boots.append(num / den if den else 0.0)

    boots.sort()
    return Summary(
        file=str(path.relative_to(ROOT)),
        n_pairs=point_den,
        n_topics=n_topics,
        point_win_rate_b=point,
        ci95_low=_quantile(boots, 0.025),
        ci95_high=_quantile(boots, 0.975),
        wins_a=obj.get("wins_a"),
        wins_b=obj.get("wins_b"),
        ties=obj.get("ties"),
    )


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Topic-block bootstrap CI summary")
    ap.add_argument("--results-dir", type=Path, default=RESULTS_DIR)
    ap.add_argument("--pattern", default="compare_*.json")
    ap.add_argument("--iters", type=int, default=3000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--output", type=Path, default=RESULTS_DIR / "ci_summary.json")
    return ap.parse_args()


def main() -> int:
    args = parse_args()
    files = sorted(
        p for p in args.results_dir.glob(args.pattern) if not skip_compare_path(p.name)
    )
    if not files:
        raise SystemExit(
            f"No files found in {args.results_dir} with pattern {args.pattern!r}"
        )

    rng = random.Random(args.seed)
    summaries = [summarize_verdict_json(p, args.iters, rng) for p in files]
    out = {
        "meta": {
            "bootstrap": "topic_block",
            "iters": args.iters,
            "seed": args.seed,
            "pattern": args.pattern,
            "excluded_compare_files": sorted(PRIMARY_COMPARE_EXCLUDE),
        },
        "summaries": [s.__dict__ for s in summaries],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(out, indent=2) + "\n")
    print(f"Wrote {len(summaries)} summaries to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
