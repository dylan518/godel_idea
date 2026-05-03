#!/usr/bin/env python3
"""Compute human-vs-judge topic-level agreement on the CUSTOM slice.

Inputs: the ``summary.json`` produced by ``run_custom_slice_judge.py`` which
contains (a) per-topic human win rates on the 12 CUSTOM topics and (b) per-topic
judge win rates for each judge evaluated on freshly-regenerated ideas for the
same topics + same systems.

Outputs ``results/custom_slice_human_vs_judge/human_vs_judge.json`` with:
    - per_judge_overall_gap: |judge_winrate_b - human_winrate_b|
    - per_judge_pearson_r: topic-level correlation between human win rate and judge win rate
    - per_judge_kendall_tau: rank correlation (robust alternative)
    - per_judge_win_agreement_rate: fraction of topics where human side and judge side both picked the same system (majority)
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


def _pearson(xs: list[float], ys: list[float]) -> float | None:
    n = len(xs)
    if n < 2 or len(ys) != n:
        return None
    mx = sum(xs) / n
    my = sum(ys) / n
    num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    dx2 = sum((x - mx) ** 2 for x in xs)
    dy2 = sum((y - my) ** 2 for y in ys)
    denom = math.sqrt(dx2 * dy2)
    if denom < 1e-12:
        return None
    return num / denom


def _kendall_tau(xs: list[float], ys: list[float]) -> float | None:
    n = len(xs)
    if n < 2 or len(ys) != n:
        return None
    concordant = 0
    discordant = 0
    ties_x = 0
    ties_y = 0
    for i in range(n):
        for j in range(i + 1, n):
            dx = xs[i] - xs[j]
            dy = ys[i] - ys[j]
            if dx == 0 and dy == 0:
                continue
            if dx == 0:
                ties_x += 1
                continue
            if dy == 0:
                ties_y += 1
                continue
            if (dx > 0 and dy > 0) or (dx < 0 and dy < 0):
                concordant += 1
            else:
                discordant += 1
    denom = math.sqrt(
        (concordant + discordant + ties_x) * (concordant + discordant + ties_y)
    )
    if denom < 1e-12:
        return None
    return (concordant - discordant) / denom


def _side(win_rate_b: float | None) -> str | None:
    if win_rate_b is None:
        return None
    if win_rate_b > 0.5 + 1e-9:
        return "B"
    if win_rate_b < 0.5 - 1e-9:
        return "A"
    return "tie"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--summary", type=Path,
                    default=Path("ideas/results/custom_slice_human_vs_judge/summary.json"))
    ap.add_argument("--output", type=Path,
                    default=Path("ideas/results/custom_slice_human_vs_judge/human_vs_judge.json"))
    args = ap.parse_args()

    summary = json.loads(args.summary.read_text())
    human_by_topic = summary["human"]["by_topic"]
    human_overall = summary["human"]["overall_win_rate_b"]

    judges = summary["judges"]
    rows = []
    for tid, h in human_by_topic.items():
        hwb = h.get("human_win_rate_b")
        row = {"topic_id": tid, "n_human": h["n"], "human_win_rate_b": hwb, "human_side": _side(hwb)}
        for jm, jdata in judges.items():
            pt = jdata["per_topic"].get(tid)
            jwb = pt.get("judge_win_rate_b") if pt else None
            row[f"judge__{jm}__win_rate_b"] = jwb
            row[f"judge__{jm}__side"] = _side(jwb)
            row[f"judge__{jm}__n"] = pt["n"] if pt else 0
        rows.append(row)

    per_judge = {}
    for jm in judges:
        xs, ys = [], []
        side_match = 0
        side_total = 0
        for r in rows:
            h = r["human_win_rate_b"]
            j = r.get(f"judge__{jm}__win_rate_b")
            if h is not None and j is not None:
                xs.append(h)
                ys.append(j)
            hs = r["human_side"]
            js = r.get(f"judge__{jm}__side")
            if hs is not None and js is not None:
                side_total += 1
                if hs == js:
                    side_match += 1
        per_judge[jm] = {
            "n_topics": len(xs),
            "pearson_r": _pearson(xs, ys),
            "kendall_tau": _kendall_tau(xs, ys),
            "side_agreement_rate": side_match / side_total if side_total else None,
            "overall_win_rate_b": judges[jm]["overall_win_rate_b"],
            "gap_vs_human_overall": (
                abs(judges[jm]["overall_win_rate_b"] - human_overall)
                if human_overall is not None and judges[jm]["overall_win_rate_b"] is not None
                else None
            ),
        }

    out = {
        "pair": summary["pair"],
        "n_topics": len(rows),
        "human_overall_win_rate_b": human_overall,
        "per_judge": per_judge,
        "per_topic": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(out, indent=2) + "\n")
    print(f"Wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
