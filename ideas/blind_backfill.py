#!/usr/bin/env python3
"""Run a single consistent blind judge (default: gemini-flash-lite-latest) on
existing idea pairs and write ``results/blind_<A>_vs_<B>.json``.

Purpose: normalize the "I switched blind judge half way through" problem so
every transition in the trajectory has a comparable primary/blind pair.

Pairs must already have their ``results/<VER>/ideas.json`` on disk (same
topic_id/idea_index overlap as the primary ``compare_*.json``).

Example::

    python3 ideas/blind_backfill.py S12 S15
    python3 ideas/blind_backfill.py --workers 6 --model gemini-flash-lite-latest S15 S16 S17 S19
    python3 ideas/blind_backfill.py --all-vs S15 S16 S17 S19

``--all-vs A B C D ...`` expands to (A,B), (A,C), (A,D) comparisons.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

IDEAS_DIR = Path(__file__).resolve().parent
if str(IDEAS_DIR) not in sys.path:
    sys.path.insert(0, str(IDEAS_DIR))
SYSTEMS_DIR = IDEAS_DIR / "systems"
if str(SYSTEMS_DIR) not in sys.path:
    sys.path.insert(0, str(SYSTEMS_DIR))

import log as _log

_log.load_dotenv()

from base import make_client
import judge as judge_mod

RESULTS_DIR = IDEAS_DIR / "results"


def _load_ideas(ver: str) -> list[dict]:
    p = RESULTS_DIR / ver / "ideas.json"
    if not p.is_file():
        raise FileNotFoundError(f"Missing {p}. Restore {ver}/ideas.json before backfilling.")
    return json.loads(p.read_text())


def _out_path(a: str, b: str, model: str) -> Path:
    safe = model.replace("/", "_")
    return RESULTS_DIR / f"blind_{a}_vs_{b}__{safe}.json"


def _run_pair(a: str, b: str, model: str, workers: int) -> Path:
    ideas_a = _load_ideas(a)
    ideas_b = _load_ideas(b)
    client = make_client(model)
    print(f"[backfill] {a} vs {b}  |  judge={model}  |  workers={workers}", flush=True)
    started = datetime.now(timezone.utc).isoformat()
    result = judge_mod.compare_systems(
        ideas_a,
        ideas_b,
        client,
        model=model,
        workers=workers,
        early_stop_threshold=None,
    )
    finished = datetime.now(timezone.utc).isoformat()
    report = {
        "role": "blind_backfill",
        "judge_model": model,
        "current": a,
        "candidate": b,
        "started_utc": started,
        "finished_utc": finished,
        "wins_a": result["wins_a"],
        "wins_b": result["wins_b"],
        "ties": result["ties"],
        "total_judged": result["total_judged"],
        "win_rate_b": result["win_rate_b"],
        "verdicts": result["verdicts"],
    }
    out = _out_path(a, b, model)
    out.write_text(json.dumps(report, indent=2))
    print(
        f"[backfill] done {a} vs {b}: N={result['total_judged']} "
        f"win_rate_b={result['win_rate_b']:.3f} -> {out.name}",
        flush=True,
    )
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("versions", nargs="+", help="Pairs: A B [C D ...] or with --all-vs: A B C D ...")
    ap.add_argument(
        "--model",
        default="gemini-flash-lite-latest",
        help="Blind judge model id (default: gemini-flash-lite-latest).",
    )
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument(
        "--all-vs",
        action="store_true",
        help="Treat first arg as base; judge base vs every other arg.",
    )
    args = ap.parse_args()

    pairs: list[tuple[str, str]] = []
    if args.all_vs:
        if len(args.versions) < 2:
            ap.error("--all-vs needs at least BASE and one candidate")
        base, *rest = args.versions
        pairs = [(base, c) for c in rest]
    else:
        if len(args.versions) < 2 or len(args.versions) % 2 != 0:
            ap.error("Provide an even number of versions, or use --all-vs")
        pairs = list(zip(args.versions[0::2], args.versions[1::2]))

    for a, b in pairs:
        try:
            _run_pair(a, b, args.model, args.workers)
        except FileNotFoundError as e:
            print(f"[skip] {a} vs {b}: {e}", flush=True)
        except Exception as e:
            print(f"[error] {a} vs {b}: {type(e).__name__}: {e}", flush=True)
            raise
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
