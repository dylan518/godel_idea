#!/usr/bin/env python3
"""Aggregate 4-dimension rubric scores from compare_*.json and human_blind_scores.jsonl.

Writes ideas/results/rubric_summary.json (no API calls).
"""

from __future__ import annotations

import json
import statistics
from pathlib import Path

IDEAS_DIR = Path(__file__).resolve().parent
RESULTS_DIR = IDEAS_DIR / "results"

DIMS = ("novelty", "scientific_usefulness", "experimental_clarity", "feasibility")


def _mean_scores(rows: list[dict]) -> dict[str, float]:
    if not rows:
        return {d: float("nan") for d in DIMS}
    return {d: statistics.mean(r[d] for r in rows) for d in DIMS}


def _collect_side(verdicts: list[dict], key: str) -> list[dict]:
    out: list[dict] = []
    for v in verdicts:
        s = v.get(key) or {}
        if all(d in s for d in DIMS):
            out.append({d: int(s[d]) for d in DIMS})
    return out


def summarize_compare(path: Path) -> dict:
    obj = json.loads(path.read_text())
    verdicts = obj.get("verdicts") or []
    a_rows = _collect_side(verdicts, "scores_a")
    b_rows = _collect_side(verdicts, "scores_b")
    cur, cand = obj.get("current"), obj.get("candidate")
    judge = None
    if verdicts:
        judge = verdicts[0].get("judge_model")
    ma, mb = _mean_scores(a_rows), _mean_scores(b_rows)
    ta = sum(sum(r.values()) for r in a_rows) / len(a_rows) if a_rows else float("nan")
    tb = sum(sum(r.values()) for r in b_rows) / len(b_rows) if b_rows else float("nan")
    return {
        "file": str(path.relative_to(IDEAS_DIR)),
        "current": cur,
        "candidate": cand,
        "judge_model": judge,
        "n_pairs": len(verdicts),
        "n_scored_a": len(a_rows),
        "n_scored_b": len(b_rows),
        "mean_scores_a": ma,
        "mean_scores_b": mb,
        "mean_total_a": ta,
        "mean_total_b": tb,
    }


def human_s15_spaper(path: Path) -> dict:
    s15_rows: list[dict] = []
    sp_rows: list[dict] = []
    n_tie = 0
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        if r.get("winner_label") == "tie":
            n_tie += 1
        sa, sb = r.get("scores_a"), r.get("scores_b")
        if not sa or not sb or not all(d in sa and d in sb for d in DIMS):
            continue
        swap = bool(r.get("swap"))
        # swap=false: a=left=S15, b=right=S_paper; swap=true: a=S_paper, b=S15
        if not swap:
            s15 = {d: int(sa[d]) for d in DIMS}
            sp = {d: int(sb[d]) for d in DIMS}
        else:
            sp = {d: int(sa[d]) for d in DIMS}
            s15 = {d: int(sb[d]) for d in DIMS}
        s15_rows.append(s15)
        sp_rows.append(sp)

    ms15 = _mean_scores(s15_rows)
    msp = _mean_scores(sp_rows)
    # Per-dimension paired diffs (S15 - S_paper)
    diffs = {
        d: statistics.mean(s15_rows[i][d] - sp_rows[i][d] for i in range(len(s15_rows)))
        for d in DIMS
    }
    totals_s15 = [sum(r.values()) for r in s15_rows]
    totals_sp = [sum(r.values()) for r in sp_rows]
    return {
        "path": str(path.relative_to(IDEAS_DIR)),
        "n_ratings": len(s15_rows),
        "n_ties": n_tie,
        "mean_scores_S15": ms15,
        "mean_scores_S_paper": msp,
        "mean_total_S15": statistics.mean(totals_s15) if totals_s15 else float("nan"),
        "mean_total_S_paper": statistics.mean(totals_sp) if totals_sp else float("nan"),
        "mean_paired_diff_S15_minus_S_paper": diffs,
    }


def summarize_blind(path: Path) -> dict:
    """Blind JSON uses same verdict shape as compare (scores_a/b, current/candidate)."""
    obj = json.loads(path.read_text())
    verdicts = obj.get("verdicts") or []
    a_rows = _collect_side(verdicts, "scores_a")
    b_rows = _collect_side(verdicts, "scores_b")
    cur, cand = obj.get("current"), obj.get("candidate")
    judge = verdicts[0].get("judge_model") if verdicts else obj.get("judge_model")
    ma, mb = _mean_scores(a_rows), _mean_scores(b_rows)
    ta = sum(sum(r.values()) for r in a_rows) / len(a_rows) if a_rows else float("nan")
    tb = sum(sum(r.values()) for r in b_rows) / len(b_rows) if b_rows else float("nan")
    return {
        "file": str(path.relative_to(IDEAS_DIR)),
        "role": obj.get("role"),
        "current": cur,
        "candidate": cand,
        "judge_model": judge,
        "n_pairs": len(verdicts),
        "mean_scores_a": ma,
        "mean_scores_b": mb,
        "mean_total_a": ta,
        "mean_total_b": tb,
    }


def main() -> int:
    compares = sorted(RESULTS_DIR.glob("compare_*.json"))
    per_compare = [summarize_compare(p) for p in compares]

    blind_extra = []
    for name in (
        "blind_S_paper_vs_S12__deepseek-chat.json",
        "blind_S12_vs_S14__deepseek-chat.json",
        "blind_S12_vs_S15__deepseek-chat.json",
    ):
        p = RESULTS_DIR / name
        if p.is_file():
            blind_extra.append(summarize_blind(p))

    # Per system: pool all appearances (as A or B) with rubric scores
    by_system: dict[str, list[dict]] = {}
    for p in compares:
        obj = json.loads(p.read_text())
        verdicts = obj.get("verdicts") or []
        cur, cand = obj.get("current"), obj.get("candidate")
        for v in verdicts:
            w = v.get("winner")
            if w not in ("A", "B", "tie"):
                continue
            sa = v.get("scores_a") or {}
            sb = v.get("scores_b") or {}
            if cur and all(d in sa for d in DIMS):
                by_system.setdefault(str(cur), []).append({d: int(sa[d]) for d in DIMS})
            if cand and all(d in sb for d in DIMS):
                by_system.setdefault(str(cand), []).append({d: int(sb[d]) for d in DIMS})

    system_pooled = {
        sys: {
            "n_cells": len(rows),
            "mean_scores": _mean_scores(rows),
            "mean_total": statistics.mean(sum(r.values()) for r in rows) if rows else float("nan"),
        }
        for sys, rows in sorted(by_system.items())
    }

    human_path = RESULTS_DIR / "human_blind_scores.jsonl"
    human = human_s15_spaper(human_path) if human_path.is_file() else None

    out = {
        "meta": {
            "dims": list(DIMS),
            "compare_files": [x["file"] for x in per_compare],
            "note": "compare rows: mean_scores_a/b are side A (champion/current) vs B (candidate) for that file.",
        },
        "per_compare": per_compare,
        "per_blind_sample": blind_extra,
        "pooled_by_system_version": system_pooled,
        "human_pilot_s15_vs_s_paper": human,
    }
    out_path = RESULTS_DIR / "rubric_summary.json"
    out_path.write_text(json.dumps(out, indent=2) + "\n")
    print(f"Wrote {out_path.relative_to(IDEAS_DIR)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
