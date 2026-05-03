#!/usr/bin/env python3
"""Regenerate ideas on the CUSTOM topics humans rated, and run multiple
judges so we can compute human-vs-judge agreement at the topic level.

Why: the human UI generated custom-topic ideas on-the-fly and did not
persist the idea texts. We can't reconstruct the exact pairs humans saw,
but we CAN regenerate on the same topics + systems with the same generator,
then judge with strong models, and correlate per-topic win rates with the
per-topic human win rates. This is a legitimate topic-level agreement study
(with the caveat noted in the paper's limitations).

Outputs under ``results/custom_slice_human_vs_judge/``:
    - topics.json         (list of {id, topic, domain} from human log)
    - ideas_<VER>.json    (ideas list, one per system)
    - judge_<MODEL>.json  (compare_systems output, per judge)
    - summary.json        (topic-level human win rates + each judge's win rates)

Typical usage::

    python3 ideas/run_custom_slice_judge.py \
        --systems S15 S_paper \
        --gen-model gpt-4.1-mini \
        --n-ideas 3 \
        --judges deepseek-chat gpt-5.4 gemini-3-flash-preview \
        --workers 4
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
from runner import load_system

RESULTS_DIR = IDEAS_DIR / "results"
OUT_DIR = RESULTS_DIR / "custom_slice_human_vs_judge"


def _load_human_log(path: Path, pair_versions: tuple[str, str]) -> list[dict]:
    rows: list[dict] = []
    with path.open() as f:
        for ln in f:
            ln = ln.strip()
            if not ln:
                continue
            d = json.loads(ln)
            lv = d.get("left_version")
            rv = d.get("right_version")
            if {lv, rv} == set(pair_versions):
                rows.append(d)
    return rows


def _topics_from_human_rows(rows: list[dict]) -> list[dict]:
    seen: dict[str, dict] = {}
    for r in rows:
        tid = r.get("topic_id")
        if not tid or not str(tid).startswith("CUSTOM"):
            continue
        if tid in seen:
            continue
        seen[tid] = {
            "id": tid,
            "topic": r.get("topic") or tid.split(":", 1)[-1],
            "domain": r.get("domain") or "",
        }
    return list(seen.values())


def _human_topic_summary(rows: list[dict], pair_versions: tuple[str, str]) -> dict[str, dict]:
    a, b = pair_versions
    out: dict[str, dict] = {}
    for r in rows:
        tid = r.get("topic_id")
        if not tid or not str(tid).startswith("CUSTOM"):
            continue
        w = r.get("winner_version")
        rec = out.setdefault(tid, {"n": 0, "wins_a": 0, "wins_b": 0, "ties": 0})
        rec["n"] += 1
        if r.get("winner_label") == "tie" or w is None:
            rec["ties"] += 1
        elif w == a:
            rec["wins_a"] += 1
        elif w == b:
            rec["wins_b"] += 1
        else:
            rec["ties"] += 1
    for tid, rec in out.items():
        n = rec["n"]
        rec["human_win_rate_b"] = (
            (rec["wins_b"] + 0.5 * rec["ties"]) / n if n else None
        )
    return out


def _generate_ideas(system_version: str, topics: list[dict], gen_model: str, n_ideas: int) -> list[dict]:
    generator = load_system(system_version, str(SYSTEMS_DIR))
    client = make_client(gen_model)
    all_rows: list[dict] = []
    for t in topics:
        topic_text = t["topic"]
        try:
            ideas = generator.generate_batch(topic_text, client, model=gen_model, n=n_ideas)
        except Exception:
            ideas = [
                generator.generate_idea(topic_text, client, model=gen_model)
                for _ in range(n_ideas)
            ]
        for i in range(n_ideas):
            all_rows.append({
                "topic_id": t["id"],
                "topic": topic_text,
                "domain": t.get("domain", ""),
                "idea_index": i,
                "text": ideas[i] if i < len(ideas) else "ERROR: missing idea",
                "system_version": system_version,
                "model": gen_model,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            })
        print(f"[gen] {system_version} topic={t['id']!s}: {len(ideas)} ideas", flush=True)
    return all_rows


def _run_judge(a_rows: list[dict], b_rows: list[dict], model: str, workers: int) -> dict:
    client = make_client(model)
    result = judge_mod.compare_systems(
        a_rows,
        b_rows,
        client,
        model=model,
        workers=workers,
        early_stop_threshold=None,
    )
    return result


def _winrate_b_per_topic(verdicts: list[dict]) -> dict[str, dict]:
    by: dict[str, dict] = {}
    for v in verdicts:
        tid = v.get("topic_id")
        if not tid:
            continue
        rec = by.setdefault(tid, {"n": 0, "wins_a": 0, "wins_b": 0, "ties": 0})
        rec["n"] += 1
        w = v.get("winner")
        if w == "A":
            rec["wins_a"] += 1
        elif w == "B":
            rec["wins_b"] += 1
        else:
            rec["ties"] += 1
    for tid, rec in by.items():
        n = rec["n"]
        rec["judge_win_rate_b"] = (rec["wins_b"] + 0.5 * rec["ties"]) / n if n else None
    return by


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--systems", nargs=2, required=True, metavar=("A", "B"),
                    help="Two system versions, must match left_version/right_version in human log")
    ap.add_argument("--gen-model", default="gpt-4.1-mini")
    ap.add_argument("--n-ideas", type=int, default=3)
    ap.add_argument("--judges", nargs="+", required=True)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--human-log", type=Path, default=RESULTS_DIR / "human_blind_scores.jsonl")
    ap.add_argument("--out-dir", type=Path, default=OUT_DIR)
    args = ap.parse_args()

    a, b = args.systems
    pair = (a, b)

    human_rows = _load_human_log(args.human_log, pair)
    if not human_rows:
        raise SystemExit(f"No human rows for pair {pair} in {args.human_log}")
    topics = _topics_from_human_rows(human_rows)
    if not topics:
        raise SystemExit("No CUSTOM topics found in human log for this pair")
    human_topic_sum = _human_topic_summary(human_rows, pair)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "topics.json").write_text(json.dumps(topics, indent=2) + "\n")
    (args.out_dir / "human_topic_summary.json").write_text(
        json.dumps({"pair": list(pair), "by_topic": human_topic_sum}, indent=2) + "\n"
    )

    ideas_by_ver: dict[str, list[dict]] = {}
    for ver in pair:
        fp = args.out_dir / f"ideas_{ver}.json"
        if fp.exists():
            print(f"[gen] reuse {fp}", flush=True)
            ideas_by_ver[ver] = json.loads(fp.read_text())
        else:
            rows = _generate_ideas(ver, topics, args.gen_model, args.n_ideas)
            fp.write_text(json.dumps(rows, indent=2) + "\n")
            ideas_by_ver[ver] = rows

    judge_summaries: dict[str, dict] = {}
    for jm in args.judges:
        safe = jm.replace("/", "_")
        fp = args.out_dir / f"judge_{safe}.json"
        if fp.exists():
            print(f"[judge] reuse {fp}", flush=True)
            obj = json.loads(fp.read_text())
        else:
            print(f"[judge] {jm} on {a} vs {b}", flush=True)
            res = _run_judge(ideas_by_ver[a], ideas_by_ver[b], jm, args.workers)
            obj = {
                "judge_model": jm,
                "current": a,
                "candidate": b,
                "gen_model": args.gen_model,
                "wins_a": res["wins_a"],
                "wins_b": res["wins_b"],
                "ties": res["ties"],
                "total_judged": res["total_judged"],
                "win_rate_b": res["win_rate_b"],
                "verdicts": res["verdicts"],
            }
            fp.write_text(json.dumps(obj, indent=2) + "\n")
        judge_summaries[jm] = {
            "overall_win_rate_b": obj["win_rate_b"],
            "per_topic": _winrate_b_per_topic(obj["verdicts"]),
        }

    human_overall_n = sum(v["n"] for v in human_topic_sum.values())
    human_overall_wb = (
        sum(v["wins_b"] + 0.5 * v["ties"] for v in human_topic_sum.values()) / human_overall_n
        if human_overall_n else None
    )

    summary = {
        "pair": list(pair),
        "gen_model": args.gen_model,
        "n_ideas_per_system_per_topic": args.n_ideas,
        "n_topics": len(topics),
        "human": {
            "n_total_ratings": human_overall_n,
            "overall_win_rate_b": human_overall_wb,
            "by_topic": human_topic_sum,
        },
        "judges": judge_summaries,
    }
    (args.out_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(f"Wrote {args.out_dir / 'summary.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
