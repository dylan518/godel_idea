#!/usr/bin/env python3
"""Aggregate human_blind_scores.jsonl by rater_id → human_blind_by_rater.json."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

IDEAS_DIR = Path(__file__).resolve().parent
DEFAULT_IN = IDEAS_DIR / "results" / "human_blind_scores.jsonl"
DEFAULT_OUT = IDEAS_DIR / "results" / "human_blind_by_rater.json"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input", type=Path, default=DEFAULT_IN)
    ap.add_argument("--output", type=Path, default=DEFAULT_OUT)
    args = ap.parse_args()

    rows = [json.loads(l) for l in args.input.read_text().strip().splitlines()]
    by_rater: dict[str, list] = defaultdict(list)
    for r in rows:
        by_rater[str(r["rater_id"])].append(r)

    def tally(rs: list) -> tuple[int, int, int, int, float, float]:
        s15 = sp = t = 0
        for r in rs:
            if r.get("winner_label") == "tie" or r.get("winner_version") is None:
                t += 1
            elif r["winner_version"] == "S15":
                s15 += 1
            elif r["winner_version"] == "S_paper":
                sp += 1
        n = len(rs)
        wr_b = (sp + 0.5 * t) / n if n else 0.0
        s15_share = (s15 + 0.5 * t) / n if n else 0.0
        return n, s15, sp, t, wr_b, s15_share

    per = []
    for rid in sorted(by_rater.keys(), key=lambda x: (-len(by_rater[x]), x)):
        n, s15, sp, t, wr_b, s15s = tally(by_rater[rid])
        per.append({
            "rater_id": rid,
            "n": n,
            "S15_wins": s15,
            "S_paper_wins": sp,
            "ties": t,
            "win_rate_b": round(wr_b, 4),
            "s15_share": round(s15s, 4),
        })

    nT, s15T, spT, tT, wr_pool, s15_pool = tally(rows)
    out = {
        "source": str(args.input.relative_to(IDEAS_DIR)),
        "total_comparisons": len(rows),
        "pooled": {
            "n": nT,
            "S15_wins": s15T,
            "S_paper_wins": spT,
            "ties": tT,
            "win_rate_b": round(wr_pool, 4),
            "s15_share": round(s15_pool, 4),
        },
        "per_rater": per,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(out, indent=2) + "\n")
    print(f"Wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
