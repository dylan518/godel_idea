#!/usr/bin/env python3
"""Install publish_eval idea snapshots as top-level result dirs for human_judge_ui.

``human_judge_ui`` only loads ``results/<version>/ideas.json`` for each side.
The frozen T1–T15 × 2-idea benchmark lives under
``results/publish_eval_<stamp>/S15`` and ``.../S_paper``, which the UI does not
discover. This script copies those files to stable names so blind human
ratings use the **same** texts the multi-judge run saw.

Creates (default):

  results/fixed_benchmark_S15/ideas.json
  results/fixed_benchmark_S_paper/ideas.json
  (+ run_config.json when present)

Then run the UI, e.g.::

    python3 ideas/human_judge_ui.py --port 8765

Open::

    http://127.0.0.1:8765/?left=fixed_benchmark_S15&right=fixed_benchmark_S_paper

Pick a benchmark topic filter or leave unfiltered — you should get **30** shared
slots (15 topics × 2 idea_index). Do **not** fill “custom topic” (that
regenerates ideas).

After collecting votes, pair-level agreement vs judges::

    python3 ideas/compute_human_judge_agreement_fixed.py \\
        --human-log results/human_blind_scores.jsonl \\
        --judge results/publish_eval_20260413T163806Z/judge_claude-sonnet-4-6.json \\
                results/publish_eval_20260413T163806Z/judge_gemini-3-flash-preview.json \\
                results/publish_eval_20260413T163806Z/judge_gpt-5.4.json \\
                results/blind_S15_vs_S_paper__deepseek-chat.json
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path


IDEAS_DIR = Path(__file__).resolve().parent
RESULTS_DIR = IDEAS_DIR / "results"


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--publish-dir",
        type=Path,
        default=RESULTS_DIR / "publish_eval_20260413T163806Z",
        help="Directory containing S15/ and S_paper/ with ideas.json",
    )
    ap.add_argument(
        "--dest-left",
        default="fixed_benchmark_S15",
        help="results/<name> for S15 ideas copy",
    )
    ap.add_argument(
        "--dest-right",
        default="fixed_benchmark_S_paper",
        help="results/<name> for S_paper ideas copy",
    )
    return ap.parse_args()


def _copy_side(publish: Path, side: str, dest_name: str) -> Path:
    src_dir = publish / side
    src_ideas = src_dir / "ideas.json"
    if not src_ideas.is_file():
        raise SystemExit(f"Missing {src_ideas}")
    out_dir = RESULTS_DIR / dest_name
    out_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src_ideas, out_dir / "ideas.json")
    src_rc = src_dir / "run_config.json"
    if src_rc.is_file():
        shutil.copy2(src_rc, out_dir / "run_config.json")
    meta = {
        "role": "fixed_benchmark_human_eval",
        "source_publish_dir": str(publish.relative_to(IDEAS_DIR)),
        "source_side": side,
        "n_ideas_file": len(json.loads((out_dir / "ideas.json").read_text())),
    }
    (out_dir / "fixed_benchmark_meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    return out_dir


def main() -> int:
    args = parse_args()
    pub = args.publish_dir
    if not pub.is_dir():
        raise SystemExit(f"Publish dir not found: {pub}")
    _copy_side(pub, "S15", args.dest_left)
    _copy_side(pub, "S_paper", args.dest_right)
    print(f"Installed benchmark ideas into results/{args.dest_left}/ and results/{args.dest_right}/")
    print(
        "UI URL (example):\n"
        f"  http://127.0.0.1:8765/?left={args.dest_left}&right={args.dest_right}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
