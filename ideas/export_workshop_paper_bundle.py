#!/usr/bin/env python3
"""Merge frozen workshop JSON into paper-ready CSV + JSON (no API calls).

Reads:
  - results/ci_summary.json (primary loop, topic-block bootstrap)
  - results/blind_*__deepseek-chat.json (blind judge rerun)
  - results/judge_agreement_deepseek_blind.json
  - results/human_blind_scores.jsonl

Writes results/workshop_paper_bundle/:
  - trajectory_table.csv
  - human_pilot_summary.json
  - bundle_meta.json
"""

from __future__ import annotations

import argparse
import csv
import json
import random
from collections import Counter
from pathlib import Path

from bootstrap_ci_workshop import RESULTS_DIR, summarize_verdict_json

IDEAS_DIR = Path(__file__).resolve().parent
OUT_DIR = RESULTS_DIR / "workshop_paper_bundle"

# Paper row order: (label, primary_compare_basename or None, blind_basename)
TRAJECTORY_ROWS: list[tuple[str, str | None, str]] = [
    ("S_paper → S12", None, "blind_S_paper_vs_S12__deepseek-chat.json"),
    ("S12 → S14", "compare_S12_vs_S14.json", "blind_S12_vs_S14__deepseek-chat.json"),
    ("S12 → S15", "compare_S12_vs_S15.json", "blind_S12_vs_S15__deepseek-chat.json"),
    ("S15 → S16", "compare_S15_vs_S16.json", "blind_S15_vs_S16__deepseek-chat.json"),
    ("S15 → S17", "compare_S15_vs_S17.json", "blind_S15_vs_S17__deepseek-chat.json"),
    ("S15 → S18", "compare_S15_vs_S18.json", "blind_S15_vs_S18__deepseek-chat.json"),
    ("S15 → S19", "compare_S15_vs_S19.json", "blind_S15_vs_S19__deepseek-chat.json"),
    ("S15 → S13", "compare_S15_vs_S13.json", "blind_S15_vs_S13__deepseek-chat.json"),
    ("S15 vs S_paper (blind only)", None, "blind_S15_vs_S_paper__deepseek-chat.json"),
]


def _load_ci_map() -> dict[str, dict]:
    path = RESULTS_DIR / "ci_summary.json"
    if not path.is_file():
        return {}
    obj = json.loads(path.read_text())
    out: dict[str, dict] = {}
    for row in obj.get("summaries", []):
        fname = Path(row["file"]).name
        out[fname] = row
    return out


def _load_kappa_map() -> dict[tuple[str, str], float | None]:
    path = RESULTS_DIR / "judge_agreement_deepseek_blind.json"
    if not path.is_file():
        return {}
    obj = json.loads(path.read_text())
    m: dict[tuple[str, str], float | None] = {}
    for t in obj.get("transitions", []):
        cur, cand = t.get("current"), t.get("candidate")
        if cur is not None and cand is not None:
            m[(str(cur), str(cand))] = t.get("cohen_kappa")
    return m


def _transition_key(label: str) -> tuple[str, str] | None:
    if "→" not in label:
        return None
    left, right = [x.strip() for x in label.split("→", 1)]
    return left, right


def _human_pilot(path: Path) -> dict:
    counts: Counter[str] = Counter()
    n = 0
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        w = r.get("winner_version")
        if w is None:
            counts["tie"] += 1
        else:
            counts[str(w)] += 1
        n += 1
    ties = counts.get("tie", 0)
    decisive = n - ties
    s15 = counts.get("S15", 0)
    spaper = counts.get("S_paper", 0)
    return {
        "path": str(path.relative_to(IDEAS_DIR)),
        "n_ratings": n,
        "counts_by_winner_version": dict(counts),
        "s15_share_decisive_pairs": (s15 / decisive) if decisive else None,
        "s15_share_including_half_ties": (s15 + 0.5 * ties) / n if n else None,
        "notes": (
            "Single rater (dylan_pilot); winner_version is the preferred system "
            "after UI swap normalization."
        ),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--iters", type=int, default=3000, help="Bootstrap iters for blind CIs")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    rng = random.Random(args.seed)

    ci_map = _load_ci_map()
    kappa_map = _load_kappa_map()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    csv_path = OUT_DIR / "trajectory_table.csv"
    rows_out: list[dict[str, str | float | int | None]] = []

    for label, primary_name, blind_name in TRAJECTORY_ROWS:
        key = _transition_key(label)
        kappa = kappa_map.get(key) if key else None

        pr: dict[str, str | float | int | None] = {
            "transition": label,
            "primary_compare_file": primary_name or "",
            "primary_n": "",
            "primary_win_rate_b": "",
            "primary_ci95_low": "",
            "primary_ci95_high": "",
            "blind_file": blind_name,
            "blind_n": "",
            "blind_win_rate_b": "",
            "blind_ci95_low": "",
            "blind_ci95_high": "",
            "cohen_kappa_primary_vs_blind_deepseek": kappa if kappa is not None else "",
        }

        if primary_name and primary_name in ci_map:
            c = ci_map[primary_name]
            pr["primary_n"] = c["n_pairs"]
            pr["primary_win_rate_b"] = round(float(c["point_win_rate_b"]), 6)
            pr["primary_ci95_low"] = round(float(c["ci95_low"]), 6)
            pr["primary_ci95_high"] = round(float(c["ci95_high"]), 6)

        blind_path = RESULTS_DIR / blind_name
        if blind_path.is_file():
            bsum = summarize_verdict_json(blind_path, args.iters, rng)
            pr["blind_n"] = bsum.n_pairs
            pr["blind_win_rate_b"] = round(float(bsum.point_win_rate_b), 6)
            pr["blind_ci95_low"] = round(float(bsum.ci95_low), 6)
            pr["blind_ci95_high"] = round(float(bsum.ci95_high), 6)
        rows_out.append(pr)

    fieldnames = list(rows_out[0].keys())
    with csv_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for row in rows_out:
            w.writerow(row)

    human_path = RESULTS_DIR / "human_blind_scores.jsonl"
    human = _human_pilot(human_path) if human_path.is_file() else {"error": "missing file"}

    meta = {
        "generated_by": "ideas/export_workshop_paper_bundle.py",
        "bootstrap": {"method": "topic_block", "iters": args.iters, "seed": args.seed},
        "outputs": {
            "trajectory_table_csv": str(csv_path.relative_to(IDEAS_DIR)),
            "human_pilot_summary_json": "results/workshop_paper_bundle/human_pilot_summary.json",
        },
    }
    (OUT_DIR / "human_pilot_summary.json").write_text(json.dumps(human, indent=2) + "\n")
    (OUT_DIR / "bundle_meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    print(f"Wrote {csv_path} and human_pilot_summary.json under {OUT_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
