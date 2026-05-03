#!/usr/bin/env python3
"""Multi-judge pairwise evaluation harness (publishable artifacts).

Runs two idea generators on the same topic list, then judges every matched
pair with one or more judge models. Writes JSON per judge plus a summary
with cross-judge agreement and rough USD estimates (heuristic — verify
against your provider invoices).

Designed for bounded cost: default 15 topics × 2 ideas × 3 judges = 90
judge API calls (plus generation on ``--generator-model``).

Usage (repo root)::

    python3 ideas/publish_multi_judge.py --max-spend-usd 10 --workers 15

Parallelism is **per topic** (generation and judging). Default is one worker per
topic up to 32, so a 15-topic run uses 15-way parallelism unless you cap it.

By default, **different judge models run concurrently** (one thread per judge,
each with ``--judge-workers`` topic parallelism). Use ``--sequential-judges`` to
force the old one-judge-at-a-time behavior.

Env: loads ``.env`` via ``ideas/log.py`` (same as runner/judge).
"""

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

IDEAS_DIR = Path(__file__).resolve().parent
REPO_ROOT = IDEAS_DIR.parent
sys.path.insert(0, str(IDEAS_DIR))

import log as _log  # noqa: E402

_log.load_dotenv()

from runner import load_topics, run_system  # noqa: E402

# Heuristic $/1M input tokens, $/1M output tokens (edit when pricing changes).
# Used only for budget guarding — not billing truth.
DEFAULT_TOKEN_USD_PER_MILLION: dict[str, tuple[float, float]] = {
    "deepseek-chat": (0.14, 0.28),
    "gemini-3-flash-preview": (0.10, 0.40),
    "gemini-flash-lite-latest": (0.05, 0.20),
    "claude-sonnet-4-6": (3.0, 15.0),
    "gpt-5.4": (5.0, 15.0),
    "gpt-4.1-mini": (0.40, 1.60),
}


def _chars_to_tokens(n: int) -> float:
    return max(n, 0) / 4.0


def _estimate_call_usd(model: str, prompt: str, completion: str) -> float:
    rates = DEFAULT_TOKEN_USD_PER_MILLION.get(model)
    if not rates:
        return 0.0
    tin, tout = rates
    it = _chars_to_tokens(len(prompt))
    ot = _chars_to_tokens(len(completion))
    return (it / 1_000_000.0) * tin + (ot / 1_000_000.0) * tout


def _verdict_key(v: dict) -> tuple[str, int]:
    return (v["topic_id"], int(v["idea_index"]))


def _pairwise_agreement(verdicts_a: list[dict], verdicts_b: list[dict]) -> dict:
    ma = {_verdict_key(v): v["winner"] for v in verdicts_a}
    mb = {_verdict_key(v): v["winner"] for v in verdicts_b}
    keys = sorted(set(ma) & set(mb))
    if not keys:
        return {"n": 0, "agreement_rate": None}
    agree = sum(1 for k in keys if ma[k] == mb[k])
    return {"n": len(keys), "agreement_rate": agree / len(keys)}


def wilson_95_interval(successes: float, trials: int) -> tuple[float, float] | None:
    """Wilson score interval for p = successes / trials (ties-as-half allowed)."""
    if trials <= 0:
        return None
    z = 1.96
    p = successes / trials
    denom = 1.0 + z * z / trials
    center = (p + z * z / (2.0 * trials)) / denom
    half = z * math.sqrt((p * (1.0 - p) / trials + z * z / (4.0 * trials * trials))) / denom
    return (max(0.0, center - half), min(1.0, center + half))


def _run_one_judge(
    judge_model: str,
    ideas_a: list,
    ideas_b: list,
    judge_workers: int,
    out_root: Path,
    system_a: str,
    system_b: str,
    n_pairs: int,
    est_block: float,
) -> tuple[str, dict, float]:
    """Run ``compare_systems`` for one judge; thread-safe (no global judge reload)."""
    sys.path.insert(0, str(IDEAS_DIR / "systems"))
    from base import make_client  # noqa: E402

    import judge as judge_mod  # noqa: E402

    client = make_client(judge_model)
    comparison = judge_mod.compare_systems(
        ideas_a,
        ideas_b,
        client,
        model=judge_model,
        workers=judge_workers,
        early_stop_threshold=None,
    )
    path = out_root / f"judge_{judge_model.replace('/', '_')}.json"
    report = {
        "judge_model": judge_model,
        "current": system_a,
        "candidate": system_b,
        "wins_a": comparison["wins_a"],
        "wins_b": comparison["wins_b"],
        "ties": comparison["ties"],
        "total_judged": comparison["total_judged"],
        "win_rate_b": comparison["win_rate_b"],
        "stopped_early": comparison.get("stopped_early", False),
        "verdicts": comparison["verdicts"],
    }
    with open(path, "w") as f:
        json.dump(report, f, indent=2)

    frac = comparison["total_judged"] / max(n_pairs, 1) if n_pairs else 0.0
    heur = est_block * frac
    summary = {
        "skipped": False,
        "report_path": str(path.resolve().relative_to(IDEAS_DIR.resolve())),
        "wins_a": comparison["wins_a"],
        "wins_b": comparison["wins_b"],
        "ties": comparison["ties"],
        "total_judged": comparison["total_judged"],
        "win_rate_b": comparison["win_rate_b"],
        "heuristic_usd_block": round(heur, 4),
    }
    return judge_model, summary, heur


def _git_rev() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=str(REPO_ROOT),
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--system-a", default="S15")
    parser.add_argument("--system-b", default="S_paper")
    parser.add_argument(
        "--topics",
        default=str(IDEAS_DIR / "benchmark_topics.json"),
        help="benchmark_topics.json path",
    )
    parser.add_argument("--n-ideas", type=int, default=2, dest="n_ideas")
    parser.add_argument(
        "--generator-model",
        default="deepseek-chat",
        help="Model for both generators (cheap; judges are separate).",
    )
    parser.add_argument(
        "--judges",
        default="gemini-3-flash-preview,claude-sonnet-4-6,gpt-5.4",
        help="Comma-separated judge model ids (cheapest first recommended).",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=0,
        help="If >0, set both --gen-workers and --judge-workers to min(workers, 64, n_topics).",
    )
    parser.add_argument(
        "--gen-workers",
        type=int,
        default=0,
        dest="gen_workers",
        help="Parallel topic workers for generation (0 = auto: min(32, n_topics)).",
    )
    parser.add_argument(
        "--judge-workers",
        type=int,
        default=0,
        dest="judge_workers",
        help="Parallel topic workers for judging (0 = auto: min(32, n_topics)).",
    )
    parser.add_argument(
        "--max-spend-usd",
        type=float,
        default=10.0,
        help="Skip remaining judges if cumulative heuristic spend exceeds this.",
    )
    parser.add_argument(
        "--sequential-judges",
        action="store_true",
        help="Run judge models one after another instead of all at once.",
    )
    parser.add_argument(
        "--out",
        default="",
        help="Output directory (default: ideas/results/publish_eval_<UTC timestamp>)",
    )
    args = parser.parse_args()

    topics_path = Path(args.topics)
    topics = load_topics(str(topics_path))
    n_topics = len(topics)
    judges = [j.strip() for j in args.judges.split(",") if j.strip()]

    auto_workers = min(32, n_topics)
    if args.workers > 0:
        gen_workers = judge_workers = min(args.workers, 64, n_topics)
    else:
        gen_workers = auto_workers if args.gen_workers <= 0 else min(args.gen_workers, 64, n_topics)
        judge_workers = auto_workers if args.judge_workers <= 0 else min(args.judge_workers, 64, n_topics)

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_root = Path(args.out) if args.out else IDEAS_DIR / "results" / f"publish_eval_{stamp}"
    out_root.mkdir(parents=True, exist_ok=True)

    systems_dir = str(IDEAS_DIR / "systems")
    dir_a = out_root / args.system_a
    dir_b = out_root / args.system_b

    protocol = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "git_rev": _git_rev(),
        "topics_file": str(topics_path.resolve()),
        "n_topics": n_topics,
        "n_ideas": args.n_ideas,
        "generator_model": args.generator_model,
        "system_a": args.system_a,
        "system_b": args.system_b,
        "judges_planned": judges,
        "gen_workers": gen_workers,
        "judge_workers": judge_workers,
        "parallel_judge_models": not args.sequential_judges,
        "max_spend_usd_heuristic": args.max_spend_usd,
        "note": (
            "USD fields are heuristic token-pricing estimates only; "
            "use provider dashboards for true cost."
        ),
    }
    with open(out_root / "protocol.json", "w") as f:
        json.dump(protocol, f, indent=2)

    # --- Generation (same topics, same n_ideas, same gen model) ---
    run_system(
        args.system_a,
        topics,
        str(dir_a),
        args.generator_model,
        args.n_ideas,
        systems_dir,
        workers=gen_workers,
        fresh=False,
    )
    run_system(
        args.system_b,
        topics,
        str(dir_b),
        args.generator_model,
        args.n_ideas,
        systems_dir,
        workers=gen_workers,
        fresh=False,
    )

    with open(dir_a / "ideas.json") as f:
        ideas_a = json.load(f)
    with open(dir_b / "ideas.json") as f:
        ideas_b = json.load(f)

    gen_pairs = sum(
        min(
            sum(1 for x in ideas_a if x["topic_id"] == t["id"]),
            sum(1 for x in ideas_b if x["topic_id"] == t["id"]),
        )
        for t in topics
    )
    _per_idea = _estimate_call_usd(
        args.generator_model,
        "x" * 4000,
        "y" * 2500,
    )
    gen_est = (len(ideas_a) + len(ideas_b)) * _per_idea

    spent = gen_est
    judge_reports: dict[str, dict] = {}

    # Per-judge rough cost before running: n_pairs * est one call
    est_per_pair = {}
    for j in judges:
        est_per_pair[j] = _estimate_call_usd(j, "z" * 9000, "w" * 1200)

    n_pairs = gen_pairs
    to_run: list[tuple[str, float]] = []
    for j in judges:
        est_block = n_pairs * est_per_pair.get(j, 0.01)
        if spent + est_block > args.max_spend_usd:
            judge_reports[j] = {
                "skipped": True,
                "reason": f"heuristic budget: {spent:.2f} + ~{est_block:.2f} > {args.max_spend_usd}",
            }
            continue
        to_run.append((j, est_block))
        spent += est_block

    if args.sequential_judges or len(to_run) <= 1:
        spent = gen_est
        for j, est_block in to_run:
            _, summary, heur = _run_one_judge(
                j,
                ideas_a,
                ideas_b,
                judge_workers,
                out_root,
                args.system_a,
                args.system_b,
                n_pairs,
                est_block,
            )
            judge_reports[j] = summary
            spent += heur
    else:
        spent = gen_est
        with ThreadPoolExecutor(max_workers=len(to_run)) as executor:
            future_map = {
                executor.submit(
                    _run_one_judge,
                    j,
                    ideas_a,
                    ideas_b,
                    judge_workers,
                    out_root,
                    args.system_a,
                    args.system_b,
                    n_pairs,
                    est_block,
                ): j
                for j, est_block in to_run
            }
            for fut in as_completed(future_map):
                j = future_map[fut]
                try:
                    jm, summary, heur = fut.result()
                except Exception as e:
                    judge_reports[j] = {
                        "skipped": True,
                        "reason": f"judge task failed: {type(e).__name__}: {e}",
                    }
                    continue
                judge_reports[jm] = summary
                spent += heur

    # Agreement matrix
    completed = [(j, r) for j, r in judge_reports.items() if not r.get("skipped")]
    agreement_matrix: dict[str, dict[str, float | None | int]] = {}
    for i, (j1, _) in enumerate(completed):
        agreement_matrix[j1] = {}
        for j2, _ in completed[i + 1 :]:
            v1 = json.load(open(out_root / f"judge_{j1.replace('/', '_')}.json"))["verdicts"]
            v2 = json.load(open(out_root / f"judge_{j2.replace('/', '_')}.json"))["verdicts"]
            agr = _pairwise_agreement(v1, v2)
            agreement_matrix[j1][j2] = agr["agreement_rate"]
            if j2 not in agreement_matrix:
                agreement_matrix[j2] = {}
            agreement_matrix[j2][j1] = agr["agreement_rate"]

    # Wilson CI on candidate win rate (ties split 50/50, same as win_rate_b)
    summary_judges = {}
    for j, r in judge_reports.items():
        if r.get("skipped"):
            summary_judges[j] = r
            continue
        data = json.load(open(out_root / f"judge_{j.replace('/', '_')}.json"))
        total = data["total_judged"]
        wb = data["wins_b"]
        wa = data["wins_a"]
        ties = data["ties"]
        eff = wb + 0.5 * ties
        ci = wilson_95_interval(eff, total) if total else None
        summary_judges[j] = {
            **r,
            "wilson_95_win_rate_b": ci,
        }

    summary = {
        "out_dir": str(out_root.resolve().relative_to(IDEAS_DIR.resolve())),
        "heuristic_spend_usd_total": round(spent, 4),
        "heuristic_spend_note": "includes generation guess + per-judge blocks",
        "judges": summary_judges,
        "judge_pair_agreement": agreement_matrix,
        "default_primary_judge_env": os.environ.get("IDEAS_JUDGE_MODEL", "deepseek-chat"),
    }
    with open(out_root / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
