#!/usr/bin/env python3
"""Re-run only the judge panel on an existing publish_eval directory (no generation).

Loads ``<eval-dir>/S15/ideas.json`` and ``<eval-dir>/S_paper/ideas.json``, runs
``judge.compare_systems`` for each requested judge model, and writes
``judge_<model>.json`` in the same directory (or ``judge_<model>_<suffix>.json`` when
``--out-suffix`` is set — use this for test–retest without overwriting the primary run).

Use after you already have frozen ideas (e.g. ``publish_eval_*`` bundles with the full benchmark pair list, typically 75 slots).

Example::

    python3 ideas/rerun_publish_eval_judges.py \\
      --eval-dir ideas/results/publish_eval_n75_gemini3_deepseek_S15_Spaper \\
      --workers 30

Default judges: ``deepseek-chat``, ``gemini-3-flash-preview`` (no Gemini Flash Lite).

Env: same as ``judge.py`` / ``blind_backfill.py`` (``.env`` via ``log.load_dotenv``).
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

IDEAS_DIR = Path(__file__).resolve().parent
if str(IDEAS_DIR) not in sys.path:
    sys.path.insert(0, str(IDEAS_DIR))
SYSTEMS_DIR = IDEAS_DIR / "systems"
if str(SYSTEMS_DIR) not in sys.path:
    sys.path.insert(0, str(SYSTEMS_DIR))

import log as _log  # noqa: E402

_log.load_dotenv()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--eval-dir",
        type=Path,
        required=True,
        help="publish_eval folder containing S15/ and S_paper/ with ideas.json",
    )
    ap.add_argument(
        "--judges",
        default="deepseek-chat,gemini-3-flash-preview",
        help="Comma-separated judge model ids.",
    )
    ap.add_argument(
        "--out-suffix",
        default="",
        help="If set (e.g. rerun), write judge_<id>_<suffix>.json and set judge_model to <id>__<suffix> in JSON for analysis.",
    )
    ap.add_argument(
        "--workers",
        type=int,
        default=30,
        help="Topic-parallel workers passed to compare_systems (default 30).",
    )
    ap.add_argument(
        "--archive",
        action="store_true",
        help="Move existing judge_*.json (not *_rerun) to _archive_judges_<UTC>/ first.",
    )
    args = ap.parse_args()

    eval_dir = args.eval_dir.resolve()
    dir_a = eval_dir / "S15"
    dir_b = eval_dir / "S_paper"
    for d in (dir_a, dir_b):
        if not (d / "ideas.json").is_file():
            raise SystemExit(f"Missing {d / 'ideas.json'}")

    ideas_a = json.loads((dir_a / "ideas.json").read_text())
    ideas_b = json.loads((dir_b / "ideas.json").read_text())
    n_a, n_b = len(ideas_a), len(ideas_b)
    judges = [j.strip() for j in args.judges.split(",") if j.strip()]
    if not judges:
        raise SystemExit("No judges in --judges")

    if args.archive:
        arch = eval_dir / f"_archive_judges_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
        arch.mkdir(parents=True, exist_ok=True)
        for p in sorted(eval_dir.glob("judge_*.json")):
            if "_rerun" in p.name:
                continue
            shutil.copy2(p, arch / p.name)
        print(f"[archive] copied primary judge files to {arch.relative_to(eval_dir.parent)}")

    sys.path.insert(0, str(IDEAS_DIR / "systems"))
    from base import make_client  # noqa: E402

    import judge as judge_mod  # noqa: E402

    meta = {
        "rerun_utc": datetime.now(timezone.utc).isoformat(),
        "eval_dir": str(eval_dir.relative_to(IDEAS_DIR)),
        "ideas_counts": {"S15": n_a, "S_paper": n_b},
        "judges": judges,
        "workers": args.workers,
        "out_suffix": args.out_suffix or None,
        "note": "Judge-only rerun; ideas unchanged.",
    }
    (eval_dir / "judge_rerun_meta.json").write_text(json.dumps(meta, indent=2) + "\n")

    for jm in judges:
        client = make_client(jm)
        print(f"[judge] {jm}  (workers={args.workers})", flush=True)
        comparison = judge_mod.compare_systems(
            ideas_a,
            ideas_b,
            client,
            model=jm,
            workers=args.workers,
            early_stop_threshold=None,
        )
        safe = jm.replace("/", "_")
        if args.out_suffix:
            out = eval_dir / f"judge_{safe}_{args.out_suffix}.json"
            model_label = f"{jm}__{args.out_suffix}"
        else:
            out = eval_dir / f"judge_{safe}.json"
            model_label = jm
        report = {
            "judge_model": model_label,
            "api_model": jm,
            "current": "S15",
            "candidate": "S_paper",
            "wins_a": comparison["wins_a"],
            "wins_b": comparison["wins_b"],
            "ties": comparison["ties"],
            "total_judged": comparison["total_judged"],
            "win_rate_b": comparison["win_rate_b"],
            "stopped_early": comparison.get("stopped_early", False),
            "verdicts": comparison["verdicts"],
            "rerun_note": "ideas from publish_eval dir; judge-only rerun",
        }
        out.write_text(json.dumps(report, indent=2) + "\n")
        print(
            f"[judge] done {jm}: N={comparison['total_judged']} "
            f"win_rate_b={comparison['win_rate_b']:.4f} -> {out.name}",
            flush=True,
        )

    # Refresh aggregate if analyzer exists
    try:
        import subprocess

        subprocess.run(
            [
                sys.executable,
                str(IDEAS_DIR / "analyze_publish_eval_judges.py"),
                "--eval-dir",
                str(eval_dir),
            ],
            check=False,
        )
    except Exception:
        pass

    print(f"[done] wrote judge files under {eval_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
