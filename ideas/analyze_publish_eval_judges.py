#!/usr/bin/env python3
"""Summarize all judge_*.json in a publish_eval folder: marginals, rule compliance, pairwise κ."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path


def _verdict_key(v: dict) -> tuple[str, int]:
    return (str(v["topic_id"]), int(v["idea_index"]))


def _load_verdict_file(path: Path) -> tuple[str, dict[tuple[str, int], dict]] | None:
    obj = json.loads(path.read_text())
    if not obj.get("verdicts"):
        return None
    model = str(obj.get("judge_model", path.stem))
    if "_rerun" in path.name and "__rerun" not in model:
        model = f"{model}__rerun"
    by = {_verdict_key(v): v for v in obj["verdicts"]}
    return model, by


def _rule_winner(v: dict) -> str:
    sa, sb = sum(v["scores_a"].values()), sum(v["scores_b"].values())
    if abs(sa - sb) <= 2:
        return "tie"
    return "A" if sa > sb else "B"


def _cohen_kappa_3(a: list[str], b: list[str]) -> float | None:
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


def _cohen_kappa_bin(a: list[str], b: list[str]) -> float | None:
    if not a:
        return None
    n = len(a)
    p0 = sum(1 for x, y in zip(a, b) if x == y) / n
    pa = sum(1 for x in a if x == "B") / n
    pb = sum(1 for x in b if x == "B") / n
    pe = pa * pb + (1 - pa) * (1 - pb)
    denom = 1.0 - pe
    if abs(denom) < 1e-12:
        return None
    return (p0 - pe) / denom


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--eval-dir", type=Path, required=True)
    ap.add_argument("--output", type=Path, default=None)
    args = ap.parse_args()
    eval_dir = args.eval_dir.resolve()
    _skip = frozenset({"judge_analysis.json", "judge_rerun_meta.json"})
    judge_files = sorted(
        p for p in eval_dir.glob("judge_*.json") if p.name not in _skip
    )
    if len(judge_files) < 1:
        raise SystemExit(f"No verdict judge_*.json in {eval_dir}")

    judges: list[tuple[str, dict[tuple[str, int], dict]]] = []
    for p in judge_files:
        loaded = _load_verdict_file(p)
        if loaded is not None:
            judges.append(loaded)
    if not judges:
        raise SystemExit(f"No judge files with non-empty verdicts in {eval_dir}")

    shared: set[tuple[str, int]] | None = None
    for _, d in judges:
        shared = set(d.keys()) if shared is None else shared & set(d.keys())
    assert shared is not None
    keys = sorted(shared)

    per_judge = []
    for model, d in judges:
        mismatch = 0
        counts: Counter[str] = Counter()
        for k in keys:
            v = d[k]
            counts[v["winner"]] += 1
            if str(v.get("winner")) != _rule_winner(v):
                mismatch += 1
        per_judge.append({
            "model": model,
            "n_pairs": len(keys),
            "winner_counts": dict(counts),
            "win_rate_b": (counts.get("B", 0) + 0.5 * counts.get("tie", 0)) / max(len(keys), 1),
            "declared_vs_score_rule_mismatch_n": mismatch,
            "declared_vs_score_rule_mismatch_rate": mismatch / max(len(keys), 1),
        })

    pairwise = []
    for i in range(len(judges)):
        mi, di = judges[i]
        for j in range(i + 1, len(judges)):
            mj, dj = judges[j]
            la = [di[k]["winner"] for k in keys]
            lb = [dj[k]["winner"] for k in keys]
            agr = sum(1 for x, y in zip(la, lb) if x == y) / max(len(keys), 1)
            dec_idx = [k for k in keys if di[k]["winner"] in ("A", "B") and dj[k]["winner"] in ("A", "B")]
            la_d = [di[k]["winner"] for k in dec_idx]
            lb_d = [dj[k]["winner"] for k in dec_idx]
            flips = sum(1 for x, y in zip(la_d, lb_d) if x != y) / max(len(dec_idx), 1) if dec_idx else None
            pairwise.append({
                "judge_a": mi,
                "judge_b": mj,
                "n_overlap": len(keys),
                "agreement_rate": agr,
                "cohen_kappa_3way": _cohen_kappa_3(la, lb),
                "decisive_overlap_n": len(dec_idx),
                "flip_rate_decisive": flips,
                "cohen_kappa_binary_decisive": _cohen_kappa_bin(la_d, lb_d) if dec_idx else None,
            })

    ideas_results = eval_dir.parent
    try:
        rel = str(eval_dir.relative_to(ideas_results))
    except ValueError:
        rel = str(eval_dir)
    out = {
        "eval_dir": rel,
        "n_shared_pairs": len(keys),
        "per_judge": per_judge,
        "pairwise": pairwise,
    }
    out_path = args.output or (eval_dir / "judge_analysis.json")
    out_path.write_text(json.dumps(out, indent=2) + "\n")
    print(f"Wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
