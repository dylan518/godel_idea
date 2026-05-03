#!/usr/bin/env python3
"""Pair-level agreement: human blind log vs frozen judge JSON (same topic_id, idea_index).

Use after humans rate ``fixed_benchmark_S15`` vs ``fixed_benchmark_S_paper`` (see
``prepare_fixed_benchmark_human_eval.py``). Each judge verdict must use the same
keys as ``compare_`` / ``judge_`` outputs: topic_id, idea_index, winner, system_a, system_b.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path


IDEAS_DIR = Path(__file__).resolve().parent


def _winner_to_system(v: dict) -> str | None:
    w = v.get("winner")
    if w == "tie":
        return "tie"
    if w == "A":
        return str(v.get("system_a", ""))
    if w == "B":
        return str(v.get("system_b", ""))
    return None


def _human_winner(row: dict) -> str | None:
    lab = row.get("winner_label") or row.get("winner")
    if lab == "tie":
        return "tie"
    return row.get("winner_version")


def _load_human_index(
    path: Path,
    left_names: frozenset[str],
    right_names: frozenset[str],
) -> dict[tuple[str, int], str]:
    """Map (topic_id, idea_index) -> winning system or 'tie'."""
    idx: dict[tuple[str, int], str] = {}
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        lv, rv = row.get("left_version"), row.get("right_version")
        if lv not in left_names or rv not in right_names:
            if lv not in right_names or rv not in left_names:
                continue
        tid = str(row.get("topic_id", ""))
        try:
            ii = int(row["idea_index"])
        except (KeyError, TypeError, ValueError):
            continue
        w = _human_winner(row)
        if w is None:
            w = "tie"
        idx[(tid, ii)] = w
    return idx


def _cohen_kappa(labels_a: list[str], labels_b: list[str]) -> float | None:
    if not labels_a or len(labels_a) != len(labels_b):
        return None
    n = len(labels_a)
    cats = sorted(set(labels_a) | set(labels_b))
    if len(cats) < 2:
        return None
    # confusion counts
    joint: dict[tuple[str, str], int] = Counter()
    for a, b in zip(labels_a, labels_b):
        joint[(a, b)] += 1
    p_o = sum(joint[(c, c)] for c in cats) / n
    marg_a = Counter(labels_a)
    marg_b = Counter(labels_b)
    p_e = sum((marg_a[c] / n) * (marg_b[c] / n) for c in cats)
    if 1.0 - p_e < 1e-12:
        return None
    return (p_o - p_e) / (1.0 - p_e)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--human-log", type=Path, required=True)
    ap.add_argument("--judge", type=Path, nargs="+", required=True, help="Judge JSON files")
    ap.add_argument(
        "--left-version",
        action="append",
        dest="left_versions",
        default=None,
        help="Human log left_version (repeatable; default: fixed_benchmark_S15)",
    )
    ap.add_argument(
        "--right-version",
        action="append",
        dest="right_versions",
        default=None,
        help="Human log right_version (repeatable; default: fixed_benchmark_S_paper)",
    )
    ap.add_argument("--output", type=Path, default=None)
    return ap.parse_args()


def main() -> int:
    args = parse_args()
    left_names = frozenset(args.left_versions or ["fixed_benchmark_S15"])
    right_names = frozenset(args.right_versions or ["fixed_benchmark_S_paper"])
    human_idx = _load_human_index(args.human_log, left_names, right_names)

    report: dict = {
        "meta": {
            "human_log": str(args.human_log),
            "human_pairs_indexed": len(human_idx),
            "left_versions": sorted(left_names),
            "right_versions": sorted(right_names),
        },
        "per_judge": [],
    }

    for jpath in args.judge:
        obj = json.loads(jpath.read_text())
        verdicts = obj.get("verdicts", [])
        jname = obj.get("judge_model") or jpath.name
        keys_h: list[str] = []
        keys_j: list[str] = []
        for v in verdicts:
            tid = str(v.get("topic_id", ""))
            try:
                ii = int(v["idea_index"])
            except (KeyError, TypeError, ValueError):
                continue
            k = (tid, ii)
            if k not in human_idx:
                continue
            hj = _winner_to_system(v)
            if hj is None:
                continue
            hh = human_idx[k]
            keys_h.append(hh)
            keys_j.append(hj)

        n = len(keys_h)
        agree = sum(1 for a, b in zip(keys_h, keys_j) if a == b) / n if n else 0.0
        kappa = _cohen_kappa(keys_h, keys_j) if n else None
        report["per_judge"].append(
            {
                "judge_file": str(jpath),
                "judge_model": jname,
                "overlap_n": n,
                "agreement_rate": agree,
                "cohen_kappa": kappa,
            }
        )

    text = json.dumps(report, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text)
        print(f"Wrote {args.output}")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
