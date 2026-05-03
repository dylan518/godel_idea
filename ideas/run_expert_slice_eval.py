#!/usr/bin/env python3
"""Run fresh expert-topic eval slice + export human-scoring pair lists.

This script:
1) Generates fresh ideas for system A and B on a custom expert topics file
2) Judges all pairs with primary + blind models
3) Exports summary + human pair picks (disagreement/agreement)

Example:
  python3 ideas/run_expert_slice_eval.py \
    --topics ideas/expert_topics.json \
    --system-a S15 --system-b S_paper \
    --n-ideas 3 --workers 6
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

IDEAS_DIR = Path(__file__).resolve().parent
RESULTS_DIR = IDEAS_DIR / "results"

import sys

if str(IDEAS_DIR) not in sys.path:
    sys.path.insert(0, str(IDEAS_DIR))
if str(IDEAS_DIR / "systems") not in sys.path:
    sys.path.insert(0, str(IDEAS_DIR / "systems"))

import log as _log

_log.load_dotenv()

from runner import load_topics, run_system
from judge import compare_systems
from systems.base import make_client


def _pair_key(v: dict) -> tuple[str, int]:
    return (str(v["topic_id"]), int(v["idea_index"]))


def _build_pair_export(
    primary_report: dict,
    blind_report: dict,
    ideas_a: list[dict],
    ideas_b: list[dict],
    disagreement_target: int,
    agreement_target: int,
) -> dict:
    mp = {_pair_key(v): v for v in primary_report.get("verdicts", [])}
    mb = {_pair_key(v): v for v in blind_report.get("verdicts", [])}
    keys = sorted(set(mp) & set(mb))

    idx_a = {_pair_key(r): r for r in ideas_a}
    idx_b = {_pair_key(r): r for r in ideas_b}

    disagreement = []
    agreement = []
    for k in keys:
        pa = mp[k].get("winner")
        pb = mb[k].get("winner")
        rec = {
            "topic_id": k[0],
            "idea_index": k[1],
            "topic": mp[k].get("topic", ""),
            "primary_winner": pa,
            "blind_winner": pb,
            "primary_reasoning": mp[k].get("reasoning", ""),
            "blind_reasoning": mb[k].get("reasoning", ""),
            "idea_a_text": idx_a.get(k, {}).get("text", ""),
            "idea_b_text": idx_b.get(k, {}).get("text", ""),
        }
        if pa == pb:
            agreement.append(rec)
        else:
            disagreement.append(rec)

    return {
        "meta": {
            "overlap_pairs": len(keys),
            "disagreement_pairs_available": len(disagreement),
            "agreement_pairs_available": len(agreement),
            "disagreement_target": disagreement_target,
            "agreement_target": agreement_target,
        },
        "disagreement_pairs": disagreement[:disagreement_target],
        "agreement_pairs": agreement[:agreement_target],
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--topics", required=True, help="Path to expert topics json")
    ap.add_argument("--system-a", default="S15")
    ap.add_argument("--system-b", default="S_paper")
    ap.add_argument("--generator-model", default="gpt-4.1-mini")
    ap.add_argument("--primary-judge", default="deepseek-chat")
    ap.add_argument("--blind-judge", default="gemini-flash-lite-latest")
    ap.add_argument("--n-ideas", type=int, default=3)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--disagreement-target", type=int, default=10)
    ap.add_argument("--agreement-target", type=int, default=10)
    ap.add_argument("--out-dir", default="")
    args = ap.parse_args()

    topics_path = Path(args.topics)
    if not topics_path.is_file():
        raise SystemExit(f"Topics file not found: {topics_path}")

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_dir = (
        Path(args.out_dir)
        if args.out_dir
        else RESULTS_DIR / f"expert_eval_{args.system_a}_vs_{args.system_b}_{stamp}"
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    topics = load_topics(str(topics_path))
    systems_dir = str(IDEAS_DIR / "systems")

    out_a = out_dir / args.system_a
    out_b = out_dir / args.system_b
    ideas_a = run_system(
        version=args.system_a,
        topics=topics,
        output_dir=str(out_a),
        model=args.generator_model,
        n_ideas=args.n_ideas,
        systems_dir=systems_dir,
        workers=args.workers,
    )
    ideas_b = run_system(
        version=args.system_b,
        topics=topics,
        output_dir=str(out_b),
        model=args.generator_model,
        n_ideas=args.n_ideas,
        systems_dir=systems_dir,
        workers=args.workers,
    )

    primary_client = make_client(args.primary_judge)
    blind_client = make_client(args.blind_judge)

    primary = compare_systems(
        ideas_a, ideas_b, primary_client, model=args.primary_judge, workers=args.workers
    )
    blind = compare_systems(
        ideas_a, ideas_b, blind_client, model=args.blind_judge, workers=args.workers
    )

    primary_report = {
        "judge_model": args.primary_judge,
        "current": args.system_a,
        "candidate": args.system_b,
        **primary,
    }
    blind_report = {
        "judge_model": args.blind_judge,
        "current": args.system_a,
        "candidate": args.system_b,
        **blind,
    }

    (out_dir / "compare_primary.json").write_text(json.dumps(primary_report, indent=2))
    (out_dir / "compare_blind.json").write_text(json.dumps(blind_report, indent=2))

    pairs = _build_pair_export(
        primary_report,
        blind_report,
        ideas_a,
        ideas_b,
        disagreement_target=args.disagreement_target,
        agreement_target=args.agreement_target,
    )
    (out_dir / "human_pairs_expert.json").write_text(json.dumps(pairs, indent=2))

    summary = {
        "topics_file": str(topics_path),
        "n_topics": len(topics),
        "n_ideas_per_topic": args.n_ideas,
        "system_a": args.system_a,
        "system_b": args.system_b,
        "generator_model": args.generator_model,
        "primary_judge": args.primary_judge,
        "blind_judge": args.blind_judge,
        "primary_win_rate_b": primary.get("win_rate_b"),
        "blind_win_rate_b": blind.get("win_rate_b"),
        "out_dir": str(out_dir),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    print(f"Wrote pair export: {out_dir / 'human_pairs_expert.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
