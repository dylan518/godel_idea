#!/usr/bin/env python3
"""Compute pairwise judge agreement (rate, flip, Cohen's kappa) across N judges
on the same set of pairs.

Usage::

    python3 ideas/compute_multi_judge_agreement.py \
        --pairs ideas/results/publish_eval_20260413T163806Z/judge_claude-sonnet-4-6.json \
                ideas/results/publish_eval_20260413T163806Z/judge_gemini-3-flash-preview.json \
                ideas/results/publish_eval_20260413T163806Z/judge_gpt-5.4.json \
                ideas/results/blind_S15_vs_S_paper__deepseek-chat.json \
        --output ideas/results/multi_judge_agreement_S15_vs_Spaper.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def _load_verdicts(path: Path) -> tuple[str, dict[tuple[str, int], str]]:
    obj = json.loads(path.read_text())
    model = obj.get("judge_model") or path.stem
    out: dict[tuple[str, int], str] = {}
    for v in obj.get("verdicts", []):
        key = (str(v["topic_id"]), int(v["idea_index"]))
        out[key] = str(v.get("winner"))
    return model, out


def _counts(labels: list[str]) -> dict[str, int]:
    c = {"A": 0, "B": 0, "tie": 0}
    for x in labels:
        if x in c:
            c[x] += 1
    return c


def _cohen_kappa(a: list[str], b: list[str]) -> float | None:
    if not a or len(a) != len(b):
        return None
    n = len(a)
    cats = ("A", "B", "tie")
    p0 = sum(1 for x, y in zip(a, b) if x == y) / n
    pa = {c: sum(1 for x in a if x == c) / n for c in cats}
    pb = {c: sum(1 for x in b if x == c) / n for c in cats}
    pe = sum(pa[c] * pb[c] for c in cats)
    denom = 1.0 - pe
    if abs(denom) < 1e-12:
        return None
    return (p0 - pe) / denom


def _win_rate_b(labels: list[str]) -> float:
    if not labels:
        return 0.0
    return sum(1.0 if x == "B" else 0.5 if x == "tie" else 0.0 for x in labels) / len(labels)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pairs", nargs="+", required=True, help="Paths to judge JSON files")
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()

    judges: list[tuple[str, dict[tuple[str, int], str]]] = [
        _load_verdicts(Path(p)) for p in args.pairs
    ]

    shared_keys = None
    for _, d in judges:
        keys = set(d.keys())
        shared_keys = keys if shared_keys is None else shared_keys & keys
    shared_keys = sorted(shared_keys or set())

    per_judge = []
    for model, d in judges:
        labels = [d[k] for k in shared_keys]
        per_judge.append({
            "model": model,
            "n_pairs": len(shared_keys),
            "counts": _counts(labels),
            "win_rate_b": _win_rate_b(labels),
        })

    pairwise = []
    for i in range(len(judges)):
        mi, di = judges[i]
        li = [di[k] for k in shared_keys]
        for j in range(i + 1, len(judges)):
            mj, dj = judges[j]
            lj = [dj[k] for k in shared_keys]
            agreement = sum(1 for x, y in zip(li, lj) if x == y) / max(len(li), 1)
            decisive = [(x, y) for x, y in zip(li, lj) if x in ("A", "B") and y in ("A", "B")]
            flip = sum(1 for x, y in decisive if x != y) / max(len(decisive), 1) if decisive else None
            kappa = _cohen_kappa(li, lj)
            pairwise.append({
                "judge_a": mi,
                "judge_b": mj,
                "n_overlap": len(shared_keys),
                "agreement_rate": agreement,
                "decisive_overlap_n": len(decisive),
                "flip_rate_decisive": flip,
                "cohen_kappa": kappa,
            })

    out = {
        "n_shared_pairs": len(shared_keys),
        "per_judge": per_judge,
        "pairwise": pairwise,
        "pair_keys_sample": shared_keys[:5],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(out, indent=2) + "\n")
    print(f"Wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
