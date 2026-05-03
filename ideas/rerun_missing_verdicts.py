#!/usr/bin/env python3
"""Re-run only missing pairwise verdicts for a saved judge report and merge back.

Example::

    python3 ideas/rerun_missing_verdicts.py \\
      --publish-dir ideas/results/publish_eval_20260413T163806Z \\
      --judge-model gemini-3-flash-preview

Expects ``S15/ideas.json``, ``S_paper/ideas.json``, and ``judge_<model>.json`` under
``--publish-dir``. Updates that judge JSON and ``summary.json``.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

IDEAS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(IDEAS_DIR))

import log as _log  # noqa: E402

_log.load_dotenv()

from judge import judge_pair, _group_by_topic  # noqa: E402
from publish_multi_judge import (  # noqa: E402
    _pairwise_agreement,
    _verdict_key,
    wilson_95_interval,
)


def sync_summary_from_judge_files(pub: Path) -> None:
    """Refresh ``summary.json`` judge stats and agreement from on-disk judge_*.json."""
    summary_path = pub / "summary.json"
    if not summary_path.exists():
        return
    summary = json.loads(summary_path.read_text())
    judges_meta = summary.get("judges", {})
    for jm, info in list(judges_meta.items()):
        if info.get("skipped"):
            continue
        path = pub / f"judge_{jm.replace('/', '_')}.json"
        if not path.exists():
            continue
        rep = json.loads(path.read_text())
        stats = _recount(rep["verdicts"])
        eff = stats["wins_b"] + 0.5 * stats["ties"]
        judges_meta[jm] = {
            **info,
            **stats,
            "wilson_95_win_rate_b": wilson_95_interval(eff, stats["total_judged"]),
        }
    summary["judges"] = judges_meta

    judges_done = [j for j, info in judges_meta.items() if not info.get("skipped")]
    loaded: dict[str, list] = {}
    for j in judges_done:
        p = pub / f"judge_{j.replace('/', '_')}.json"
        if p.exists():
            loaded[j] = json.loads(p.read_text())["verdicts"]
    agreement: dict[str, dict[str, float | None]] = {}
    for i, j1 in enumerate(judges_done):
        agreement[j1] = {}
        for j2 in judges_done[i + 1 :]:
            agr = _pairwise_agreement(loaded[j1], loaded[j2])
            rate = agr["agreement_rate"]
            agreement[j1][j2] = rate
            agreement.setdefault(j2, {})
            agreement[j2][j1] = rate
    summary["judge_pair_agreement"] = agreement
    summary_path.write_text(json.dumps(summary, indent=2))
    print(f"synced {summary_path}")


def _sync_summary(pub: Path) -> None:
    """Recompute ``summary.json`` judge stats + agreement from on-disk judge JSON."""
    summary_path = pub / "summary.json"
    if not summary_path.exists():
        return
    summary = json.loads(summary_path.read_text())
    judges_meta = summary.get("judges", {})
    loaded: dict[str, list] = {}
    for jm, info in judges_meta.items():
        if info.get("skipped"):
            continue
        rel = info.get("report_path", "")
        p = IDEAS_DIR / rel if rel else pub / f"judge_{jm.replace('/', '_')}.json"
        if not p.exists():
            p = pub / f"judge_{jm.replace('/', '_')}.json"
        if not p.exists():
            continue
        rep = json.loads(p.read_text())
        verdicts = rep["verdicts"]
        stats = _recount(verdicts)
        eff = stats["wins_b"] + 0.5 * stats["ties"]
        judges_meta[jm].update(
            {
                "wins_a": stats["wins_a"],
                "wins_b": stats["wins_b"],
                "ties": stats["ties"],
                "total_judged": stats["total_judged"],
                "win_rate_b": stats["win_rate_b"],
                "wilson_95_win_rate_b": wilson_95_interval(
                    eff, stats["total_judged"]
                ),
            }
        )
        loaded[jm] = verdicts
    judges_done = [j for j in judges_meta if not judges_meta[j].get("skipped")]
    agreement: dict[str, dict[str, float | None]] = {}
    for i, j1 in enumerate(judges_done):
        agreement[j1] = {}
        for j2 in judges_done[i + 1 :]:
            if j1 in loaded and j2 in loaded:
                agr = _pairwise_agreement(loaded[j1], loaded[j2])
                rate = agr["agreement_rate"]
                agreement[j1][j2] = rate
                agreement.setdefault(j2, {})
                agreement[j2][j1] = rate
    summary["judges"] = judges_meta
    summary["judge_pair_agreement"] = agreement
    summary_path.write_text(json.dumps(summary, indent=2))
    print(f"synced {summary_path}")


def _recount(verdicts: list[dict]) -> dict:
    wins_a = wins_b = ties = 0
    for v in verdicts:
        w = v["winner"]
        if w == "A":
            wins_a += 1
        elif w == "B":
            wins_b += 1
        else:
            ties += 1
    total = wins_a + wins_b + ties
    win_rate_b = (wins_b + 0.5 * ties) / total if total else 0.0
    return {
        "wins_a": wins_a,
        "wins_b": wins_b,
        "ties": ties,
        "total_judged": total,
        "win_rate_b": win_rate_b,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--publish-dir",
        required=True,
        help="Directory containing S15/, S_paper/, judge_*.json, summary.json",
    )
    parser.add_argument(
        "--judge-model",
        default="gemini-3-flash-preview",
        help="Model id (must match judge_<model>.json file name)",
    )
    parser.add_argument(
        "--max-outer-retries",
        type=int,
        default=6,
        help="Extra attempts per missing pair (beyond judge_pair internal retries)",
    )
    parser.add_argument(
        "--sync-summary-only",
        action="store_true",
        help="Only rebuild summary.json from existing judge_*.json (no API calls).",
    )
    args = parser.parse_args()

    pub = Path(args.publish_dir).resolve()
    if not pub.is_dir():
        raise SystemExit(f"not a directory: {pub}")

    if args.sync_summary_only:
        _sync_summary(pub)
        return 0

    ideas_a = json.loads((pub / "S15" / "ideas.json").read_text())
    ideas_b = json.loads((pub / "S_paper" / "ideas.json").read_text())
    system_a = ideas_a[0]["system_version"] if ideas_a else "S15"
    system_b = ideas_b[0]["system_version"] if ideas_b else "S_paper"

    judge_path = pub / f"judge_{args.judge_model.replace('/', '_')}.json"
    if not judge_path.exists():
        raise SystemExit(f"missing {judge_path}")

    report = json.loads(judge_path.read_text())
    verdicts: list[dict] = report["verdicts"]
    have = {_verdict_key(v) for v in verdicts}

    ga = _group_by_topic(ideas_a)
    gb = _group_by_topic(ideas_b)
    all_topics = sorted(set(ga) & set(gb))

    missing: list[tuple[str, str, int, str, str]] = []
    for tid in all_topics:
        n = min(len(ga[tid]), len(gb[tid]))
        topic_title = ga[tid][0]["topic"]
        for pair_idx in range(n):
            if (tid, pair_idx) not in have:
                ea = ga[tid][pair_idx]
                eb = gb[tid][pair_idx]
                missing.append((tid, topic_title, pair_idx, ea["text"], eb["text"]))

    if not missing:
        print("no missing pairs; syncing summary from disk")
        _sync_summary(pub)
        return 0

    print(f"re-judging {len(missing)} missing pair(s): {[(m[0], m[2]) for m in missing]}")

    sys.path.insert(0, str(IDEAS_DIR / "systems"))
    from base import make_client  # noqa: E402

    client = make_client(args.judge_model)
    new_rows: list[dict] = []
    for tid, topic_title, pair_idx, text_a, text_b in missing:
        last_err: Exception | None = None
        verdict = None
        for attempt in range(args.max_outer_retries):
            try:
                verdict = judge_pair(
                    topic_title, text_a, text_b, client, args.judge_model
                )
                break
            except Exception as e:
                last_err = e
                print(f"  attempt {attempt + 1}/{args.max_outer_retries} {tid}[{pair_idx}]: {e}")
        if verdict is None:
            raise SystemExit(f"gave up on {tid} pair {pair_idx}: {last_err}")

        new_rows.append(
            {
                "topic_id": tid,
                "topic": topic_title,
                "idea_index": pair_idx,
                "system_a": system_a,
                "system_b": system_b,
                "judge_model": args.judge_model,
                **verdict,
            }
        )
        print(f"  ok {tid}[{pair_idx}] → {verdict['winner']}")

    verdicts.extend(new_rows)
    verdicts.sort(key=lambda v: (v["topic_id"], int(v["idea_index"])))
    report["verdicts"] = verdicts
    stats = _recount(verdicts)
    report.update(stats)
    report["stopped_early"] = False
    judge_path.write_text(json.dumps(report, indent=2))
    print(f"wrote {judge_path} ({stats['total_judged']} verdicts)")

    _sync_summary(pub)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
