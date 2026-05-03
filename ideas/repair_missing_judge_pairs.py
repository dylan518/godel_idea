#!/usr/bin/env python3
"""Re-run only missing pairwise verdicts in a saved ``judge_*.json`` and refresh ``summary.json``.

Use when a judge (often Gemini) skipped pairs after invalid JSON. Recomputes wins /
``win_rate_b`` and rewrites ``summary.json`` for that publish_eval directory.

Example::

    python3 ideas/repair_missing_judge_pairs.py \\
        --run-dir ideas/results/publish_eval_20260413T163806Z \\
        --judge-model gemini-3-flash-preview
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

IDEAS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(IDEAS_DIR))

import log as _log  # noqa: E402

_log.load_dotenv()


def _verdict_key(v: dict) -> tuple[str, int]:
    return (v["topic_id"], int(v["idea_index"]))


def wilson_95_interval(successes: float, trials: int) -> tuple[float, float] | None:
    if trials <= 0:
        return None
    z = 1.96
    p = successes / trials
    denom = 1.0 + z * z / trials
    center = (p + z * z / (2.0 * trials)) / denom
    half = z * math.sqrt((p * (1.0 - p) / trials + z * z / (4.0 * trials * trials))) / denom
    return (max(0.0, center - half), min(1.0, center + half))


def _pairwise_agreement(verdicts_a: list[dict], verdicts_b: list[dict]) -> float | None:
    ma = {_verdict_key(v): v["winner"] for v in verdicts_a}
    mb = {_verdict_key(v): v["winner"] for v in verdicts_b}
    keys = sorted(set(ma) & set(mb))
    if not keys:
        return None
    return sum(1 for k in keys if ma[k] == mb[k]) / len(keys)


def _group_ideas(entries: list[dict]) -> dict[str, list[dict]]:
    g: dict[str, list[dict]] = {}
    for e in entries:
        g.setdefault(e["topic_id"], []).append(e)
    for tid in g:
        g[tid].sort(key=lambda x: int(x["idea_index"]))
    return g


def _expected_pairs(ga: dict[str, list[dict]], gb: dict[str, list[dict]]) -> set[tuple[str, int]]:
    keys: set[tuple[str, int]] = set()
    for tid in sorted(set(ga) & set(gb)):
        n = min(len(ga[tid]), len(gb[tid]))
        for i in range(n):
            keys.add((tid, i))
    return keys


def _aggregate(verdicts: list[dict]) -> dict:
    wa = wb = ties = 0
    for v in verdicts:
        w = v["winner"]
        if w == "A":
            wa += 1
        elif w == "B":
            wb += 1
        else:
            ties += 1
    total = wa + wb + ties
    win_b = (wb + 0.5 * ties) / total if total else 0.0
    eff = wb + 0.5 * ties
    return {
        "wins_a": wa,
        "wins_b": wb,
        "ties": ties,
        "total_judged": total,
        "win_rate_b": win_b,
        "wilson_95_win_rate_b": wilson_95_interval(eff, total) if total else None,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run-dir", required=True, help="publish_eval_* directory")
    ap.add_argument("--judge-model", required=True, help="e.g. gemini-3-flash-preview")
    ap.add_argument("--outer-retries", type=int, default=5, help="Retries per pair on failure")
    args = ap.parse_args()

    run_dir = Path(args.run_dir)
    if not run_dir.is_dir():
        raise SystemExit(f"not a directory: {run_dir}")

    judge_path = run_dir / f"judge_{args.judge_model.replace('/', '_')}.json"
    if not judge_path.exists():
        raise SystemExit(f"missing {judge_path}")

    protocol_path = run_dir / "protocol.json"
    if protocol_path.exists():
        protocol = json.load(open(protocol_path))
        system_a = protocol["system_a"]
        system_b = protocol["system_b"]
    else:
        data0 = json.load(open(judge_path))
        system_a = data0["current"]
        system_b = data0["candidate"]

    ideas_a = json.load(open(run_dir / system_a / "ideas.json"))
    ideas_b = json.load(open(run_dir / system_b / "ideas.json"))
    ga, gb = _group_ideas(ideas_a), _group_ideas(ideas_b)
    expected = _expected_pairs(ga, gb)

    report = json.load(open(judge_path))
    verdicts = list(report["verdicts"])
    have = {_verdict_key(v) for v in verdicts}
    missing = sorted(expected - have)
    if not missing:
        print("No missing pairs; nothing to do.")
        return 0

    print(f"Missing {len(missing)} pair(s): {missing}")

    sys.path.insert(0, str(IDEAS_DIR / "systems"))
    from base import make_client  # noqa: E402

    import judge as judge_mod  # noqa: E402

    client = make_client(args.judge_model)

    for tid, idx in missing:
        ea = next(x for x in ga[tid] if int(x["idea_index"]) == idx)
        eb = next(x for x in gb[tid] if int(x["idea_index"]) == idx)
        topic = ea["topic"]
        last_err = None
        for attempt in range(args.outer_retries):
            try:
                verdict = judge_mod.judge_pair(
                    topic, ea["text"], eb["text"], client, args.judge_model
                )
                break
            except Exception as e:
                last_err = e
                print(f"  attempt {attempt + 1}/{args.outer_retries} failed {tid}[{idx}]: {e}")
        else:
            raise SystemExit(f"gave up on {tid}[{idx}]: {last_err}") from last_err

        verdicts.append({
            "topic_id": tid,
            "topic": topic,
            "idea_index": idx,
            "system_a": ea["system_version"],
            "system_b": eb["system_version"],
            "judge_model": args.judge_model,
            **verdict,
        })
        print(f"  ok {tid}[{idx}] → {verdict['winner']}")

    verdicts.sort(key=lambda v: (_verdict_key(v)[0], _verdict_key(v)[1]))
    agg = _aggregate(verdicts)
    report["verdicts"] = verdicts
    report["stopped_early"] = False
    for k, v in agg.items():
        if k != "wilson_95_win_rate_b":
            report[k] = v
    with open(judge_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"Wrote {judge_path} ({agg['total_judged']} verdicts)")

    # --- Refresh summary.json ---
    summary_path = run_dir / "summary.json"
    prev: dict = {}
    if summary_path.exists():
        prev = json.load(open(summary_path))

    judge_files = sorted(run_dir.glob("judge_*.json"))
    if not judge_files:
        return 0

    by_model: dict[str, dict] = {}
    for p in judge_files:
        data = json.load(open(p))
        by_model[data["judge_model"]] = data

    judge_summaries = {}
    prev_judges = (prev.get("judges") if prev else None) or {}
    for model, data in by_model.items():
        tot = data["total_judged"]
        wb, wa, ties = data["wins_b"], data["wins_a"], data["ties"]
        eff = wb + 0.5 * ties
        pprev = prev_judges.get(model, {})
        judge_path_rel = run_dir / f"judge_{model.replace('/', '_')}.json"
        judge_summaries[model] = {
            "skipped": False,
            "report_path": str(judge_path_rel.relative_to(IDEAS_DIR)),
            "wins_a": wa,
            "wins_b": wb,
            "ties": ties,
            "total_judged": tot,
            "win_rate_b": data["win_rate_b"],
            "heuristic_usd_block": pprev.get("heuristic_usd_block", 0.0),
            "wilson_95_win_rate_b": wilson_95_interval(eff, tot) if tot else None,
        }

    models = list(judge_summaries.keys())
    agreement: dict[str, dict[str, float | None]] = {m: {} for m in models}
    for i, m1 in enumerate(models):
        v1 = by_model[m1]["verdicts"]
        for m2 in models[i + 1 :]:
            v2 = by_model[m2]["verdicts"]
            agr = _pairwise_agreement(v1, v2)
            agreement[m1][m2] = agr
            agreement[m2][m1] = agr

    summary = {
        "out_dir": str(run_dir.relative_to(IDEAS_DIR)),
        "heuristic_spend_usd_total": prev.get("heuristic_spend_usd_total", 0.0),
        "heuristic_spend_note": prev.get(
            "heuristic_spend_note",
            "updated after repair_missing_judge_pairs",
        ),
        "judges": judge_summaries,
        "judge_pair_agreement": agreement,
        "default_primary_judge_env": prev.get("default_primary_judge_env", ""),
        "repaired_judge": args.judge_model,
        "repaired_pairs": [list(x) for x in missing],
    }
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Wrote {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
