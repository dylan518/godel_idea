"""Gödel loop CLI: manage the self-improving idea generator evolution.

Subcommands:
    status                          Print current version, available systems, cached results
    benchmark [--version V]         Run specified (or current) version on all benchmark topics
    compare --candidate S1          Run candidate vs current, judge, print verdict
    accept S1 [--force]             Accept a candidate version (write CURRENT_VERSION + log)

Usage:
    python ideas/godel_loop.py status
    python ideas/godel_loop.py benchmark --version S0
    python ideas/godel_loop.py compare --candidate S1
    python ideas/godel_loop.py accept S1
    python ideas/godel_loop.py accept S1 --force   # skip threshold guard
"""

import argparse
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

IDEAS_DIR = Path(__file__).parent
sys.path.insert(0, str(IDEAS_DIR))

import log as _log

_log.load_dotenv()
logger = _log.setup("godel")

ACCEPTANCE_THRESHOLD = 0.55
DEFAULT_MODEL = "gpt-4.1-mini"
DEFAULT_N_IDEAS = 3


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _read_current_version() -> str:
    return (IDEAS_DIR / "CURRENT_VERSION").read_text().strip()


def _write_current_version(version: str):
    (IDEAS_DIR / "CURRENT_VERSION").write_text(version + "\n")


def _available_systems() -> list[str]:
    stems = set()
    for pattern in ("S*.py", "H*.py"):
        stems.update(p.stem for p in (IDEAS_DIR / "systems").glob(pattern)
                     if p.stem != "__init__")
    return sorted(stems)


def _cached_results_path(version: str) -> Path:
    return IDEAS_DIR / "results" / version / "ideas.json"


def _has_cached_results(version: str) -> bool:
    return _cached_results_path(version).exists()


def _load_results(version: str) -> list[dict]:
    with open(_cached_results_path(version)) as f:
        return json.load(f)


def _load_topics() -> list[dict]:
    with open(IDEAS_DIR / "benchmark_topics.json") as f:
        return json.load(f)["topics"]


# ---------------------------------------------------------------------------
# Subcommands
# ---------------------------------------------------------------------------

def cmd_status(args):
    current = _read_current_version()
    available = _available_systems()
    topics = _load_topics()

    logger.info("Current version : %s", current)
    logger.info("Available systems: %s", ", ".join(available))
    logger.info("Benchmark topics : %d", len(topics))

    any_cached = False
    for v in available:
        if _has_cached_results(v):
            results = _load_results(v)
            logger.info("  Cached %s: %d ideas", v, len(results))
            any_cached = True
    if not any_cached:
        logger.info("  Cached results: (none)")

    evo_log = IDEAS_DIR / "results" / "evolution_log.jsonl"
    if evo_log.exists():
        with open(evo_log) as f:
            entries = [json.loads(line) for line in f if line.strip()]
        logger.info("Evolution log: %d accepted version(s)", len(entries))
        for e in entries:
            logger.info("  %s  %s -> %s  (win rate: %.1f%%)",
                        e["timestamp"][:19], e["from_version"], e["to_version"],
                        (e["win_rate"] or 0) * 100)
    else:
        logger.info("Evolution log: (empty)")


def cmd_benchmark(args):
    from runner import run_system, load_topics

    version = args.version or _read_current_version()
    systems_dir = str(IDEAS_DIR / "systems")
    output_dir = str(IDEAS_DIR / "results" / version)
    model = args.model
    n_ideas = args.n_ideas
    workers = args.workers

    logger.info("Benchmarking %s  model=%s  n_ideas=%d  workers=%d",
                version, model, n_ideas, workers)
    topics = load_topics(str(IDEAS_DIR / "benchmark_topics.json"))
    run_system(version=version, topics=topics, output_dir=output_dir,
               model=model, n_ideas=n_ideas, systems_dir=systems_dir, workers=workers)


def cmd_compare(args):
    from runner import run_system, load_topics
    from judge import compare_systems, blind_compare_systems, JUDGE_MODEL

    candidate = args.candidate
    current = _read_current_version()
    model = args.model
    n_ideas = args.n_ideas
    workers = args.workers
    systems_dir = str(IDEAS_DIR / "systems")

    if candidate == current:
        logger.error("Candidate %s is already the current version.", candidate)
        sys.exit(1)

    available = _available_systems()
    if candidate not in available:
        logger.error("Candidate %s not found. Available: %s", candidate, ", ".join(available))
        sys.exit(1)

    topics = load_topics(str(IDEAS_DIR / "benchmark_topics.json"))

    from runner import cache_is_valid
    n_topics = len(topics)
    current_output_dir = str(IDEAS_DIR / "results" / current)
    if cache_is_valid(current_output_dir, model, n_ideas, n_topics):
        logger.info("Using cached results for %s (model=%s, n_ideas=%d, n_topics=%d)",
                    current, model, n_ideas, n_topics)
        results_current = _load_results(current)
    else:
        logger.info("Running %s on benchmark topics...", current)
        results_current = run_system(
            version=current, topics=topics,
            output_dir=str(IDEAS_DIR / "results" / current),
            model=model, n_ideas=n_ideas, systems_dir=systems_dir,
        )

    logger.info("Running candidate %s on benchmark topics...", candidate)
    results_candidate = run_system(
        version=candidate, topics=topics,
        output_dir=str(IDEAS_DIR / "results" / candidate),
        model=model, n_ideas=n_ideas, systems_dir=systems_dir, workers=workers,
    )

    logger.info("Primary judge (%s): %s (A) vs %s (B)...", JUDGE_MODEL, current, candidate)
    from systems.base import make_client as _make_client
    judge_client = _make_client(JUDGE_MODEL)
    comparison = compare_systems(results_current, results_candidate, judge_client,
                                 workers=workers)

    wins_a = comparison["wins_a"]
    wins_b = comparison["wins_b"]
    ties = comparison["ties"]
    win_rate = comparison["win_rate_b"]
    total = comparison["total_judged"]

    logger.info("=" * 60)
    logger.info("%s (current) wins: %d/%d", current, wins_a, total)
    logger.info("%s (candidate) wins: %d/%d", candidate, wins_b, total)
    logger.info("Ties: %d/%d", ties, total)
    logger.info("%s win rate: %.1f%%  (threshold: %.0f%%)",
                candidate, win_rate * 100, ACCEPTANCE_THRESHOLD * 100)
    logger.info("=" * 60)

    if win_rate > ACCEPTANCE_THRESHOLD:
        logger.info("VERDICT: %s wins %.0f%% — ACCEPTED (run `accept %s` to apply)",
                    candidate, win_rate * 100, candidate)
    else:
        logger.info("VERDICT: %s win rate %.0f%% is below threshold — keep %s",
                    candidate, win_rate * 100, current)

    # Blind judge — independent Gemini sample, never used for accept/reject
    blind = None
    blind_n = getattr(args, "blind_n", 2)
    if blind_n > 0:
        logger.info("Running blind judge on %d sampled topic(s)...", blind_n)
        try:
            blind = blind_compare_systems(results_current, results_candidate, n_sample=blind_n)
            logger.info("Blind win rate: %.1f%%  (primary: %.1f%%)",
                        blind["win_rate_b"] * 100, win_rate * 100)
            divergence = abs(win_rate - blind["win_rate_b"])
            if divergence > 0.30:
                logger.warning("GOODHART ALERT: primary=%.0f%% vs blind=%.0f%% (divergence %.0f%%)",
                               win_rate * 100, blind["win_rate_b"] * 100, divergence * 100)
        except Exception as e:
            logger.error("Blind judge failed: %s", e)

    report_path = IDEAS_DIR / "results" / f"compare_{current}_vs_{candidate}.json"
    with open(report_path, "w") as f:
        json.dump({
            "current": current,
            "candidate": candidate,
            **comparison,
            "blind": blind,
        }, f, indent=2)
    logger.info("Detailed report saved to: %s", report_path)


def _next_version_name() -> str:
    """Return the next unused S{n} name (e.g. 'S6' if S0-S5 exist)."""
    existing = _available_systems()
    nums = []
    for s in existing:
        m = re.match(r"S(\d+)$", s)
        if m:
            nums.append(int(m.group(1)))
    return f"S{max(nums) + 1}" if nums else "S1"


def cmd_generate(args):
    """Generate the next system file using the meta-LLM."""
    from meta import generate_next_system

    next_version = args.version or _next_version_name()
    out_path = IDEAS_DIR / "systems" / f"{next_version}.py"

    if out_path.exists() and not args.force:
        logger.error("%s already exists. Use --force to overwrite.", out_path)
        sys.exit(1)

    code = generate_next_system(IDEAS_DIR, next_version)

    # Basic sanity check: must define GENERATOR
    if "GENERATOR" not in code:
        logger.error("Generated code missing GENERATOR singleton — regenerate or fix manually")
        sys.exit(1)

    out_path.write_text(code)
    logger.info("Wrote %s (%d chars) → %s", next_version, len(code), out_path)


def cmd_evolve(args):
    """Auto-evolve from current version to S{target}, generating and testing each iteration."""
    from runner import run_system, load_topics, cache_is_valid
    from judge import compare_systems, blind_compare_systems, JUDGE_MODEL
    from meta import generate_next_system
    from systems.base import make_client as _make_client

    target = args.target
    model = args.model
    n_ideas = args.n_ideas
    workers = args.workers
    systems_dir = str(IDEAS_DIR / "systems")
    topics = load_topics(str(IDEAS_DIR / "benchmark_topics.json"))
    n_topics = len(topics)

    current = _read_current_version()
    m = re.match(r"S(\d+)$", current)
    start = int(m.group(1)) + 1 if m else 1

    logger.info("Evolve loop: %s → S%d  (workers=%d, n_ideas=%d)",
                current, target, workers, n_ideas)

    for i in range(start, target + 1):
        next_version = f"S{i}"
        logger.info("=" * 60)
        logger.info("ITERATION %d/%d: testing %s vs current champion %s",
                    i - start + 1, target - start + 1, next_version, current)

        # --- Generate system file if needed ---
        sys_path = IDEAS_DIR / "systems" / f"{next_version}.py"
        if sys_path.exists():
            logger.info("%s already exists — skipping generation", next_version)
        else:
            try:
                code = generate_next_system(IDEAS_DIR, next_version)
                if "GENERATOR" not in code:
                    logger.error("%s: generated code missing GENERATOR — skipping iteration",
                                 next_version)
                    continue
                sys_path.write_text(code)
                logger.info("Generated %s (%d chars)", next_version, len(code))
            except Exception as e:
                logger.error("Failed to generate %s: %s", next_version, e)
                continue

        # --- Check for existing comparison report ---
        report_path = IDEAS_DIR / "results" / f"compare_{current}_vs_{next_version}.json"
        if report_path.exists():
            logger.info("Comparison report already exists for %s vs %s — loading",
                        current, next_version)
            with open(report_path) as f:
                existing = json.load(f)
            win_rate = existing.get("win_rate_b", 0)
        else:
            # --- Benchmark current if cache missing ---
            current_output_dir = str(IDEAS_DIR / "results" / current)
            if cache_is_valid(current_output_dir, model, n_ideas, n_topics):
                logger.info("Using cached results for %s", current)
                results_current = _load_results(current)
            else:
                logger.info("Benchmarking %s...", current)
                results_current = run_system(
                    version=current, topics=topics,
                    output_dir=current_output_dir,
                    model=model, n_ideas=n_ideas,
                    systems_dir=systems_dir, workers=workers,
                )

            # --- Benchmark candidate ---
            logger.info("Benchmarking %s...", next_version)
            results_candidate = run_system(
                version=next_version, topics=topics,
                output_dir=str(IDEAS_DIR / "results" / next_version),
                model=model, n_ideas=n_ideas,
                systems_dir=systems_dir, workers=workers,
            )

            # --- Primary judge ---
            logger.info("Judging %s vs %s...", current, next_version)
            judge_client = _make_client(JUDGE_MODEL)
            comparison = compare_systems(results_current, results_candidate,
                                         judge_client, workers=workers)
            win_rate = comparison["win_rate_b"]
            wins_a = comparison["wins_a"]
            wins_b = comparison["wins_b"]
            ties = comparison["ties"]
            total = comparison["total_judged"]

            logger.info("=" * 60)
            logger.info("%s wins: %d/%d  |  %s wins: %d/%d  |  ties: %d/%d",
                        current, wins_a, total, next_version, wins_b, total, ties, total)
            logger.info("%s win rate: %.1f%%  (threshold: %.0f%%)",
                        next_version, win_rate * 100, ACCEPTANCE_THRESHOLD * 100)

            # --- Blind judge ---
            blind = None
            blind_n = getattr(args, "blind_n", 5)
            if blind_n > 0:
                try:
                    blind = blind_compare_systems(results_current, results_candidate,
                                                  n_sample=blind_n)
                    logger.info("Blind win rate: %.1f%%  (primary: %.1f%%)",
                                blind["win_rate_b"] * 100, win_rate * 100)
                    divergence = abs(win_rate - blind["win_rate_b"])
                    if divergence > 0.30:
                        logger.warning(
                            "GOODHART ALERT: primary=%.0f%% vs blind=%.0f%%",
                            win_rate * 100, blind["win_rate_b"] * 100)
                except Exception as e:
                    logger.error("Blind judge failed: %s", e)

            # Save report
            with open(report_path, "w") as f:
                json.dump({"current": current, "candidate": next_version,
                           **comparison, "blind": blind}, f, indent=2)
            logger.info("Report saved to %s", report_path)

        # --- Accept if threshold met ---
        if win_rate > ACCEPTANCE_THRESHOLD:
            logger.info("VERDICT: %s ACCEPTED (%.1f%%) — new champion",
                        next_version, win_rate * 100)
            _write_current_version(next_version)
            log_path = IDEAS_DIR / "results" / "evolution_log.jsonl"
            blind_win_rate = None
            if report_path.exists():
                with open(report_path) as rf:
                    rdata = json.load(rf)
                blind_win_rate = (rdata.get("blind") or {}).get("win_rate_b")
            entry = {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "from_version": current,
                "to_version": next_version,
                "win_rate": win_rate,
                "blind_win_rate": blind_win_rate,
            }
            with open(log_path, "a") as f:
                f.write(json.dumps(entry) + "\n")
            current = next_version
        else:
            logger.info("VERDICT: %s REJECTED (%.1f%%) — keeping %s",
                        next_version, win_rate * 100, current)

    logger.info("=" * 60)
    logger.info("Evolve loop complete. Current champion: %s", _read_current_version())


def cmd_swe_evolve(args):
    """SWE agent loop: multi-turn targeted edits to evolve from current champion."""
    from swe_agent import run_swe_loop
    from runner import run_system, load_topics, cache_is_valid
    from judge import compare_systems, blind_compare_systems, JUDGE_MODEL
    from systems.base import make_client as _make_client

    target = args.target
    model = args.model
    n_ideas = args.n_ideas
    workers = args.workers
    systems_dir = str(IDEAS_DIR / "systems")
    topics = load_topics(str(IDEAS_DIR / "benchmark_topics.json"))
    n_topics = len(topics)

    current = _read_current_version()

    # Find the next unused S{n} number, always above all existing systems
    existing = _available_systems()
    used_nums = []
    for s in existing:
        mm = re.match(r"S(\d+)$", s)
        if mm:
            used_nums.append(int(mm.group(1)))
    start = (max(used_nums) + 1) if used_nums else 1

    logger.info("SWE-evolve loop: %s → S%d  (workers=%d, n_ideas=%d, swe_rounds=%d)",
                current, target, workers, n_ideas, args.swe_rounds)

    for i in range(start, target + 1):
        next_version = f"S{i}"
        logger.info("=" * 60)
        logger.info("SWE ITERATION %d/%d: %s → %s",
                    i - start + 1, target - start + 1, current, next_version)

        sys_path = IDEAS_DIR / "systems" / f"{next_version}.py"
        report_path = IDEAS_DIR / "results" / f"compare_{current}_vs_{next_version}.json"

        # Find most recent compare report for failure analysis
        existing_reports = sorted(IDEAS_DIR.glob(f"results/compare_{current}_vs_*.json"))
        last_report = existing_reports[-1] if existing_reports else None

        # --- SWE agent: multi-turn edit loop ---
        if sys_path.exists():
            logger.info("%s already exists — skipping SWE loop", next_version)
        else:
            try:
                run_swe_loop(
                    ideas_dir=IDEAS_DIR,
                    next_version=next_version,
                    champion_version=current,
                    compare_report_path=last_report,
                    max_rounds=args.swe_rounds,
                    max_failures=args.swe_failures,
                    generator_model=model,
                )
            except Exception as e:
                logger.error("SWE loop failed for %s: %s — skipping", next_version, e)
                continue

        # --- Full eval: benchmark + judge ---
        if report_path.exists():
            logger.info("Comparison report exists for %s — loading", next_version)
            with open(report_path) as f:
                existing = json.load(f)
            win_rate = existing.get("win_rate_b", 0)
        else:
            current_output_dir = str(IDEAS_DIR / "results" / current)
            if cache_is_valid(current_output_dir, model, n_ideas, n_topics):
                logger.info("Using cached results for %s", current)
                results_current = _load_results(current)
            else:
                results_current = run_system(
                    version=current, topics=topics,
                    output_dir=current_output_dir,
                    model=model, n_ideas=n_ideas,
                    systems_dir=systems_dir, workers=workers,
                )

            results_candidate = run_system(
                version=next_version, topics=topics,
                output_dir=str(IDEAS_DIR / "results" / next_version),
                model=model, n_ideas=n_ideas,
                systems_dir=systems_dir, workers=workers,
            )

            judge_client = _make_client(JUDGE_MODEL)
            comparison = compare_systems(results_current, results_candidate,
                                         judge_client, workers=workers)
            win_rate = comparison["win_rate_b"]

            logger.info("=" * 60)
            logger.info("%s wins: %d/%d  |  %s wins: %d/%d  |  ties: %d/%d",
                        current, comparison["wins_a"], comparison["total_judged"],
                        next_version, comparison["wins_b"], comparison["total_judged"],
                        comparison["ties"], comparison["total_judged"])
            logger.info("%s win rate: %.1f%%  (threshold: %.0f%%)",
                        next_version, win_rate * 100, ACCEPTANCE_THRESHOLD * 100)

            blind = None
            blind_n = getattr(args, "blind_n", 5)
            if blind_n > 0:
                try:
                    blind = blind_compare_systems(results_current, results_candidate,
                                                  n_sample=blind_n)
                    divergence = abs(win_rate - blind["win_rate_b"])
                    logger.info("Blind win rate: %.1f%%  (divergence: %.0f%%)",
                                blind["win_rate_b"] * 100, divergence * 100)
                    if divergence > 0.30:
                        logger.warning("GOODHART ALERT: primary=%.0f%% vs blind=%.0f%%",
                                       win_rate * 100, blind["win_rate_b"] * 100)
                except Exception as e:
                    logger.error("Blind judge failed: %s", e)

            with open(report_path, "w") as f:
                json.dump({"current": current, "candidate": next_version,
                           **comparison, "blind": blind}, f, indent=2)

        if win_rate > ACCEPTANCE_THRESHOLD:
            logger.info("VERDICT: %s ACCEPTED (%.1f%%)", next_version, win_rate * 100)
            _write_current_version(next_version)
            log_path = IDEAS_DIR / "results" / "evolution_log.jsonl"
            blind_win_rate = None
            if report_path.exists():
                with open(report_path) as rf:
                    rdata = json.load(rf)
                blind_win_rate = (rdata.get("blind") or {}).get("win_rate_b")
            entry = {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "from_version": current,
                "to_version": next_version,
                "win_rate": win_rate,
                "blind_win_rate": blind_win_rate,
            }
            with open(log_path, "a") as f:
                f.write(json.dumps(entry) + "\n")
            current = next_version
        else:
            logger.info("VERDICT: %s REJECTED (%.1f%%)", next_version, win_rate * 100)

    logger.info("=" * 60)
    logger.info("SWE-evolve complete. Champion: %s", _read_current_version())


def cmd_reset_sota(args):
    """Reset evolution to start fresh from S_sota as the new baseline."""
    sota_path = IDEAS_DIR / "systems" / "S_sota.py"
    if not sota_path.exists():
        logger.error("S_sota.py not found — cannot reset")
        sys.exit(1)

    if not args.force:
        logger.error("This will reset CURRENT_VERSION to S_sota. Use --force to confirm.")
        sys.exit(1)

    _write_current_version("S_sota")
    logger.info("CURRENT_VERSION reset to S_sota")
    logger.info("Next: benchmark S_sota, then use swe-evolve to improve it")


def cmd_accept(args):
    candidate = args.version
    force = args.force
    current = _read_current_version()
    available = _available_systems()

    if candidate not in available:
        logger.error("Version %s not found. Available: %s", candidate, ", ".join(available))
        sys.exit(1)

    if candidate == current:
        logger.info("%s is already the current version.", candidate)
        sys.exit(0)

    report_path = IDEAS_DIR / "results" / f"compare_{current}_vs_{candidate}.json"
    win_rate = None

    if report_path.exists():
        with open(report_path) as f:
            report = json.load(f)
        win_rate = report.get("win_rate_b")

        if win_rate is not None and win_rate <= ACCEPTANCE_THRESHOLD:
            if force:
                logger.warning(
                    "%s win rate %.1f%% is below threshold (%.0f%%) — accepting anyway (--force)",
                    candidate, win_rate * 100, ACCEPTANCE_THRESHOLD * 100,
                )
            else:
                logger.error(
                    "%s win rate %.1f%% is below acceptance threshold (%.0f%%). "
                    "Use --force to override.",
                    candidate, win_rate * 100, ACCEPTANCE_THRESHOLD * 100,
                )
                sys.exit(1)
    else:
        if force:
            logger.warning("No comparison report found for %s vs %s — accepting anyway (--force)",
                           current, candidate)
        else:
            logger.error(
                "No comparison report found for %s vs %s. "
                "Run `compare --candidate %s` first, or use --force to skip.",
                current, candidate, candidate,
            )
            sys.exit(1)

    _write_current_version(candidate)

    log_path = IDEAS_DIR / "results" / "evolution_log.jsonl"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    blind_win_rate = None
    if report_path.exists():
        with open(report_path) as rf:
            rdata = json.load(rf)
        blind_win_rate = (rdata.get("blind") or {}).get("win_rate_b")

    entry = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "from_version": current,
        "to_version": candidate,
        "win_rate": win_rate,
        "blind_win_rate": blind_win_rate,
    }
    with open(log_path, "a") as f:
        f.write(json.dumps(entry) + "\n")

    logger.info("Accepted %s as new current version (was %s)", candidate, current)
    logger.info("Logged to %s", log_path)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Gödel loop: self-improving idea generator evolution"
    )
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("status", help="Print current version, available systems, cached results")

    bench = sub.add_parser("benchmark", help="Run a version on all benchmark topics")
    bench.add_argument("--version", default=None, help="Version to run (default: current)")
    bench.add_argument("--model", default=DEFAULT_MODEL)
    bench.add_argument("--n-ideas", type=int, default=DEFAULT_N_IDEAS, dest="n_ideas")
    bench.add_argument("--workers", type=int, default=1,
                       help="Parallel topic workers (default: 1)")

    comp = sub.add_parser("compare", help="Compare candidate vs current, judge, print verdict")
    comp.add_argument("--candidate", required=True, help="Candidate version, e.g. S1")
    comp.add_argument("--model", default=DEFAULT_MODEL)
    comp.add_argument("--n-ideas", type=int, default=DEFAULT_N_IDEAS, dest="n_ideas")
    comp.add_argument("--blind-n", type=int, default=5, dest="blind_n",
                      help="Topics to sample for blind Gemini judge (default: 5, 0 to skip)")
    comp.add_argument("--workers", type=int, default=1,
                       help="Parallel topic workers for generation and judging (default: 1)")

    acc = sub.add_parser("accept", help="Accept a candidate version as the new current")
    acc.add_argument("version", help="Version to accept, e.g. S1")
    acc.add_argument("--force", action="store_true",
                     help="Skip threshold guard (for headless / agent use)")

    gen = sub.add_parser("generate", help="Use meta-LLM to write the next system file")
    gen.add_argument("--version", default=None,
                     help="Version name to generate (default: auto-increments)")
    gen.add_argument("--force", action="store_true", help="Overwrite if file already exists")

    evo = sub.add_parser("evolve", help="Auto-evolve: generate+test each iteration up to --target")
    evo.add_argument("--target", type=int, required=True,
                     help="Run iterations up to S{target}, e.g. --target 20")
    evo.add_argument("--model", default=DEFAULT_MODEL)
    evo.add_argument("--n-ideas", type=int, default=DEFAULT_N_IDEAS, dest="n_ideas")
    evo.add_argument("--workers", type=int, default=3,
                     help="Parallel topic workers (default: 3)")
    evo.add_argument("--blind-n", type=int, default=5, dest="blind_n",
                     help="Topics for blind judge per iteration (default: 5, 0 to skip)")

    swe = sub.add_parser("swe-evolve",
                          help="SWE agent: multi-turn targeted edits per iteration up to --target")
    swe.add_argument("--target", type=int, required=True,
                     help="Evolve up to S{target}")
    swe.add_argument("--model", default=DEFAULT_MODEL)
    swe.add_argument("--n-ideas", type=int, default=DEFAULT_N_IDEAS, dest="n_ideas")
    swe.add_argument("--workers", type=int, default=3)
    swe.add_argument("--swe-rounds", type=int, default=6, dest="swe_rounds",
                     help="Max edit rounds per iteration (default: 6)")
    swe.add_argument("--swe-failures", type=int, default=3, dest="swe_failures",
                     help="Max consecutive failed mini-evals before stopping (default: 3)")
    swe.add_argument("--blind-n", type=int, default=5, dest="blind_n")

    rst = sub.add_parser("reset-sota", help="Reset CURRENT_VERSION to S_sota baseline")
    rst.add_argument("--force", action="store_true", help="Required to confirm reset")

    args = parser.parse_args()
    {"status": cmd_status, "benchmark": cmd_benchmark, "compare": cmd_compare,
     "accept": cmd_accept, "generate": cmd_generate, "evolve": cmd_evolve,
     "swe-evolve": cmd_swe_evolve, "reset-sota": cmd_reset_sota,
     }[args.command](args)


if __name__ == "__main__":
    main()
