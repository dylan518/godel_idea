#!/usr/bin/env python3
"""Run one judge on ideas already produced under a publish_eval_* directory.

Example::

    python3 ideas/judge_eval_folder.py \\
        --eval-dir ideas/results/publish_eval_n75_gemini3_deepseek_S15_Spaper \\
        --judge-model gemini-flash-lite-latest \\
        --workers 30
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

IDEAS_DIR = Path(__file__).resolve().parent
REPO_ROOT = IDEAS_DIR.parent
sys.path.insert(0, str(IDEAS_DIR))
sys.path.insert(0, str(IDEAS_DIR / "systems"))

import log as _log  # noqa: E402

_log.load_dotenv()

import judge as judge_mod  # noqa: E402
from base import make_client  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--eval-dir", type=Path, required=True)
    ap.add_argument("--system-a", default="S15")
    ap.add_argument("--system-b", default="S_paper")
    ap.add_argument("--judge-model", required=True)
    ap.add_argument("--workers", type=int, default=15)
    ap.add_argument(
        "--out-name",
        default="",
        help="Output filename under eval-dir (default: judge_<model>.json)",
    )
    args = ap.parse_args()

    eval_dir = args.eval_dir.resolve()
    path_a = eval_dir / args.system_a / "ideas.json"
    path_b = eval_dir / args.system_b / "ideas.json"
    if not path_a.is_file() or not path_b.is_file():
        raise SystemExit(f"Missing ideas.json under {eval_dir}/{{{args.system_a},{args.system_b}}}/")

    ideas_a = json.loads(path_a.read_text())
    ideas_b = json.loads(path_b.read_text())
    client = make_client(args.judge_model)
    comparison = judge_mod.compare_systems(
        ideas_a,
        ideas_b,
        client,
        model=args.judge_model,
        workers=args.workers,
        early_stop_threshold=None,
    )
    if args.out_name:
        out_path = eval_dir / args.out_name
    else:
        out_path = eval_dir / f"judge_{args.judge_model.replace('/', '_')}.json"
    report = {
        "judge_model": args.judge_model,
        "current": args.system_a,
        "candidate": args.system_b,
        "wins_a": comparison["wins_a"],
        "wins_b": comparison["wins_b"],
        "ties": comparison["ties"],
        "total_judged": comparison["total_judged"],
        "win_rate_b": comparison["win_rate_b"],
        "stopped_early": comparison.get("stopped_early", False),
        "verdicts": comparison["verdicts"],
    }
    out_path.write_text(json.dumps(report, indent=2) + "\n")
    print(f"Wrote {out_path} ({comparison['total_judged']} verdicts)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
