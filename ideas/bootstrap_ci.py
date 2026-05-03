#!/usr/bin/env python3
"""Compute topic-block bootstrap CIs for compare_* results files.

Usage:
  python3 ideas/bootstrap_ci.py
  python3 ideas/bootstrap_ci.py --pattern "compare_S15_vs_*.json" --iters 5000
"""

from __future__ import annotations

import argparse
import json
import random
from dataclasses import dataclass
from pathlib import Path


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
    return 0.0


def _quantile(sorted_vals: list[float], q: float) -> float:
    if not sorted_vals:
        raise ValueError("Cannot compute quantile of empty list")
    n = len(sorted_vals)
    if n == 1:
        return sorted_vals[0]
    idx = q * (n - 1)
    lo = int(idx)
    hi = min(lo + 1, n - 1)
    frac = idx - lo
    return sorted_vals[lo] * (1.0 - frac) + sorted_vals[hi] * frac


def _summarize_compare(path: Path, iters: int, rng: random.Random) -> Summary:
    obj = json.loads(path.read_text())
    verdicts = obj.get("verdicts", [])
    if not verdicts:
        raise ValueError(f"No verdicts found in {path}")

    by_topic: dict[str, list[float]] = {}
    for row in verdicts:
        topic_id = str(row.get("topic_id", "UNKNOWN"))
        winner = row.get("winner")
        by_topic.setdefault(topic_id, []).append(_winner_to_b_win(winner))

    topics = sorted(by_topic.keys())
    n_topics = len(topics)
    if n_topics == 0:
        raise ValueError(f"No topic ids found in verdicts for {path}")

    point_num = sum(sum(by_topic[t]) for t in topics)
    point_den = sum(len(by_topic[t]) for t in topics)
    point = point_num / point_den if point_den else 0.0

    boot: list[float] = []
    for _ in range(iters):
        sample_topics = [topics[rng.randrange(n_topics)] for _ in range(n_topics)]
        num = 0.0
        den = 0
        for tid in sample_topics:
            vals = by_topic[tid]
            num += sum(vals)
            den += len(vals)
        boot.append(num / den if den else 0.0)

    boot.sort()
    ci_low = _quantile(boot, 0.025)
    ci_high = _quantile(boot, 0.975)

    return Summary(
        file=str(path.relative_to(ROOT)),
        n_pairs=point_den,
        n_topics=n_topics,
        point_win_rate_b=point,
        ci95_low=ci_low,
        ci95_high=ci_high,
        wins_a=obj.get("wins_a"),
        wins_b=obj.get("wins_b"),
        ties=obj.get("ties"),
    )


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Topic-block bootstrap CI summary for compare files"
    )
    ap.add_argument(
        "--results-dir",
        type=Path,
        default=RESULTS_DIR,
        help=f"Directory containing compare files (default: {RESULTS_DIR})",
    )
    ap.add_argument(
        "--pattern",
        default="compare_*.json",
        help='Glob pattern under --results-dir (default: "compare_*.json")',
    )
    ap.add_argument(
        "--iters",
        type=int,
        default=3000,
        help="Bootstrap iterations per file (default: 3000)",
    )
    ap.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for bootstrap reproducibility (default: 42)",
    )
    ap.add_argument(
        "--output",
        type=Path,
        default=RESULTS_DIR / "ci_summary.json",
        help="Output JSON path",
    )
    return ap.parse_args()


def main() -> int:
    args = parse_args()
    results_dir: Path = args.results_dir
    files = sorted(results_dir.glob(args.pattern))
    if not files:
        raise SystemExit(
            f"No files found in {results_dir} with pattern {args.pattern!r}"
        )

    rng = random.Random(args.seed)
    summaries: list[Summary] = []
    for f in files:
        summaries.append(_summarize_compare(f, args.iters, rng))

    out = {
        "meta": {
            "bootstrap": "topic_block",
            "iters": args.iters,
            "seed": args.seed,
            "pattern": args.pattern,
        },
        "summaries": [s.__dict__ for s in summaries],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(out, indent=2) + "\n")
    print(f"Wrote {len(summaries)} summaries to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
#!/usr/bin/env python3
"""Block-bootstrap 95% confidence intervals for pairwise win rate.

Block = topic_id. Resamples topics with replacement, aggregates verdicts within
each resampled topic, recomputes win_rate_b = (wins_b + 0.5 * ties) / total.

Usage::

    python3 ideas/bootstrap_ci.py                       # all compare_*.json
    python3 ideas/bootstrap_ci.py results/compare_S15_vs_S16.json
    python3 ideas/bootstrap_ci.py --n-boot 20000 --seed 7

Writes ``ideas/results/ci_summary.json`` + prints a table.
"""

from __future__ import annotations

import argparse
import glob
import json
import random
from collections import defaultdict
from pathlib import Path

IDEAS_DIR = Path(__file__).resolve().parent
RESULTS_DIR = IDEAS_DIR / "results"
DEFAULT_OUT = RESULTS_DIR / "ci_summary.json"


def _bucket_by_topic(verdicts: list[dict]) -> dict[str, list[dict]]:
    buckets: dict[str, list[dict]] = defaultdict(list)
    for v in verdicts:
        buckets[v["topic_id"]].append(v)
    return buckets


def _win_rate_b(verdicts: list[dict]) -> float | None:
    total = 0
    score = 0.0
    for v in verdicts:
        w = v.get("winner")
        if w == "B":
            score += 1.0
            total += 1
        elif w == "A":
            total += 1
        elif w == "tie":
            score += 0.5
            total += 1
    if total == 0:
        return None
    return score / total


def block_bootstrap_ci(
    verdicts: list[dict],
    n_boot: int = 10000,
    seed: int = 0,
    alpha: float = 0.05,
) -> dict:
    buckets = _bucket_by_topic(verdicts)
    topic_ids = list(buckets.keys())
    n_topics = len(topic_ids)
    rng = random.Random(seed)
    point = _win_rate_b(verdicts)
    if n_topics == 0 or point is None:
        return {
            "point": point,
            "ci_lo": None,
            "ci_hi": None,
            "n_verdicts": len(verdicts),
            "n_topics": n_topics,
            "n_boot": 0,
        }

    samples: list[float] = []
    for _ in range(n_boot):
        picks = [topic_ids[rng.randrange(n_topics)] for _ in range(n_topics)]
        acc: list[dict] = []
        for tid in picks:
            acc.extend(buckets[tid])
        wr = _win_rate_b(acc)
        if wr is not None:
            samples.append(wr)

    samples.sort()
    if not samples:
        return {
            "point": point,
            "ci_lo": None,
            "ci_hi": None,
            "n_verdicts": len(verdicts),
            "n_topics": n_topics,
            "n_boot": 0,
        }
    lo_idx = max(0, int((alpha / 2.0) * len(samples)))
    hi_idx = min(len(samples) - 1, int((1.0 - alpha / 2.0) * len(samples)))
    return {
        "point": round(point, 4),
        "ci_lo": round(samples[lo_idx], 4),
        "ci_hi": round(samples[hi_idx], 4),
        "n_verdicts": len(verdicts),
        "n_topics": n_topics,
        "n_boot": len(samples),
    }


def _summarize_compare(path: Path, n_boot: int, seed: int) -> dict:
    data = json.loads(path.read_text())
    verdicts = data.get("verdicts") or []
    ci = block_bootstrap_ci(verdicts, n_boot=n_boot, seed=seed)
    judges = sorted({v.get("judge_model") for v in verdicts if v.get("judge_model")})
    return {
        "file": str(path.relative_to(IDEAS_DIR)),
        "current": data.get("current"),
        "candidate": data.get("candidate"),
        "judges": judges,
        **ci,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("files", nargs="*", help="compare_*.json paths (default: all).")
    ap.add_argument("--n-boot", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    args = ap.parse_args()

    if args.files:
        paths = [Path(f).resolve() for f in args.files]
    else:
        paths = sorted(Path(p) for p in glob.glob(str(RESULTS_DIR / "compare_*.json")))

    rows = []
    for p in paths:
        if not p.is_file():
            print(f"[skip] not a file: {p}")
            continue
        row = _summarize_compare(p, args.n_boot, args.seed)
        rows.append(row)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"n_boot": args.n_boot, "seed": args.seed, "rows": rows}, indent=2))

    hdr = f"{'compare':<46} {'judge':<30} {'N':>4} {'topics':>7}  win_rate_b [95% CI]"
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        tag = f"{r['current']}_vs_{r['candidate']}"
        judges = ",".join(r["judges"]) if r["judges"] else "?"
        point = r.get("point")
        lo = r.get("ci_lo")
        hi = r.get("ci_hi")
        if point is None:
            body = "N/A"
        else:
            body = f"{point:.3f} [{lo:.3f}, {hi:.3f}]"
        print(f"{tag:<46} {judges:<30} {r['n_verdicts']:>4} {r['n_topics']:>7}  {body}")
    print(f"\nWrote: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
