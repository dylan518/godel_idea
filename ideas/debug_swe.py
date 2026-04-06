#!/usr/bin/env python3
"""
Debug runner: one SWE cycle, full visibility.

Usage:
    python3 ideas/debug_swe.py [--version S_debug] [--revert]

Runs a single SWE round against the current champion, prints every step,
then reverts (deletes the output file) unless --no-revert is passed.
"""

import argparse
import difflib
import json
import shutil
import sys
from pathlib import Path

# ── Setup ─────────────────────────────────────────────────────────────────────
IDEAS_DIR = Path(__file__).parent
sys.path.insert(0, str(IDEAS_DIR))
sys.path.insert(0, str(IDEAS_DIR / "systems"))

import log as _log
_log.load_dotenv()   # must call before any API clients are created
import swe_agent

# Force DEBUG level so we see everything
import logging
logging.getLogger().setLevel(logging.DEBUG)
for name in ["swe_agent", "tree_search_v2", "tournament", "runner", "retrieval"]:
    logging.getLogger(name).setLevel(logging.DEBUG)

SEP = "─" * 70


def banner(title: str) -> None:
    print(f"\n{SEP}\n  {title}\n{SEP}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--version", default="S_debug", help="Output version name")
    parser.add_argument("--no-revert", action="store_true", help="Keep the output file")
    args = parser.parse_args()

    next_version = args.version
    systems_dir  = IDEAS_DIR / "systems"
    output_path  = systems_dir / f"{next_version}.py"

    # ── 1. Find current champion ───────────────────────────────────────────────
    banner("STEP 1: Current champion")
    current_version = (IDEAS_DIR / "CURRENT_VERSION").read_text().strip()
    champion_path   = systems_dir / f"{current_version}.py"
    print(f"  Champion  : {current_version}  ({champion_path})")
    print(f"  Output    : {next_version}  ({output_path})")
    print(f"  Claude Code mode: {swe_agent.USE_CLAUDE_CODE}")

    # ── 2. Find comparison report ──────────────────────────────────────────────
    banner("STEP 2: Failure report")
    reports = sorted(IDEAS_DIR.glob(f"results/compare_*_vs_{current_version}.json"))
    reports += sorted(IDEAS_DIR.glob(f"results/compare_{current_version}_vs_*.json"))
    compare_report_path = reports[-1] if reports else None
    print(f"  Using report: {compare_report_path}")

    losing_verdicts = swe_agent._extract_losing_verdicts(compare_report_path)
    print("\n  Top losing verdicts (where champion loses):")
    for line in losing_verdicts.splitlines()[:5]:
        print(f"    {line}")

    # ── 3. Build code bundle ───────────────────────────────────────────────────
    banner("STEP 3: Code bundle")
    code_bundle = swe_agent._bundle_editable_context(IDEAS_DIR, current_version)
    print(f"  Bundle size: {len(code_bundle):,} chars ({len(code_bundle)//4:,} tokens est.)")

    # ── 4. Failure analysis ────────────────────────────────────────────────────
    banner("STEP 4: Failure analysis (calling claude-sonnet-4-6)")
    swe_context = swe_agent.load_swe_context(IDEAS_DIR)
    swe_context_str = swe_agent.format_swe_context(swe_context, IDEAS_DIR, current_version)

    analysis_prompt = swe_agent.ANALYZE_PROMPT.format(
        version=current_version,
        code_bundle=code_bundle,
        total=50,
        win_rate=0.5,
        losing_verdicts=losing_verdicts,
        edit_history="(none yet)",
        swe_context=swe_context_str,
    )
    print(f"  Prompt size: {len(analysis_prompt):,} chars")
    print("  Calling SWE LLM for failure analysis...")

    try:
        failure_analysis = swe_agent._call_swe_llm(analysis_prompt)
        import re
        failure_analysis = re.sub(r"```.*?```", "", failure_analysis, flags=re.DOTALL).strip()
        print(f"\n  FAILURE ANALYSIS ({len(failure_analysis)} chars):")
        print("  " + "\n  ".join(failure_analysis.splitlines()))
    except Exception as e:
        print(f"  FAILED: {e}")
        failure_analysis = "Focus on improving experimental clarity and named baselines."
        print(f"  Using fallback: {failure_analysis}")

    # ── 5. Improvement tournament ──────────────────────────────────────────────
    banner("STEP 5: Improvement tournament (gpt-4.1-mini)")
    print("  Running IdeaTreeSearch on improvement strategies...")
    memory = swe_agent.load_swe_memory(IDEAS_DIR)
    memory_str = swe_agent.format_memory_str(memory)

    try:
        tournament_winner, n_candidates = swe_agent._run_improvement_tournament(
            failure_analysis=failure_analysis,
            code_bundle=code_bundle,
            edit_history_str="(none yet)",
            memory_str=swe_context_str,
            ideas_dir=IDEAS_DIR,
            model="gpt-4.1-mini",
        )
        print(f"\n  Tournament winner ({n_candidates} candidates):")
        print("  " + "\n  ".join(tournament_winner.splitlines()))
    except Exception as e:
        print(f"  FAILED: {e}")
        tournament_winner = failure_analysis
        n_candidates = 0

    # ── 6. Claude Code edit ────────────────────────────────────────────────────
    banner("STEP 6: Claude Code edit")
    task_description = (
        f"Tournament-selected strategy ({n_candidates} candidates evaluated):\n"
        f"{tournament_winner}"
    )
    print(f"  Task (first 400 chars):\n  {task_description[:400]}")
    print(f"\n  Calling _call_claude_code_edit → {output_path.name} ...")

    # Stream claude output live so we can watch it work
    import subprocess, os as _os
    shutil.copy(champion_path, output_path)

    repo_root = IDEAS_DIR.parent
    rel_output = output_path.relative_to(repo_root)
    task_prompt = f"""You are improving a Python research idea generator. Make ONE focused, surgical improvement.

## Improvement strategy
{task_description}

## Why the current generator is losing
{failure_analysis}

## What to do
1. Read `{rel_output}` — already a copy of the champion, your starting point
2. Read `ideas/idea_tournament/prompts.py`, `ideas/idea_tournament/tree_search.py`, `ideas/idea_tournament/tournament.py` for full context
3. Implement the improvement strategy above with TARGETED edits to `{rel_output}`

## Hard constraints
- Class name MUST be `{next_version}Generator(IdeaGenerator)`
- `VERSION = "{next_version}"`
- File MUST end with `GENERATOR = {next_version}Generator()`
- Any code you MODIFY from `idea_tournament/` must be inlined (not imported)
- Every `call_llm()` call must be in try/except
- `generate_idea()` must always return a non-empty string"""

    clean_env = {k: v for k, v in _os.environ.items() if k != "CLAUDECODE"}
    print(f"  Streaming claude --print output:\n")
    try:
        proc = subprocess.Popen(
            ["claude", "--print", "--allowedTools", "Read,Edit,Write"],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            cwd=str(repo_root),
            env=clean_env,
        )
        proc.stdin.write(task_prompt)
        proc.stdin.close()
        output_lines = []
        for line in proc.stdout:
            print(f"  | {line}", end="", flush=True)
            output_lines.append(line)
        proc.wait(timeout=600)
        if proc.returncode != 0:
            raise RuntimeError(f"claude returned rc={proc.returncode}")
    except Exception as e:
        print(f"\n  FAILED: {e}")
        if not args.no_revert:
            output_path.unlink(missing_ok=True)
        sys.exit(1)

    if not output_path.exists():
        print("  FAILED: Claude Code did not write the output file")
        sys.exit(1)

    new_code = output_path.read_text()
    champion_content = champion_path.read_text()
    if new_code == champion_content:
        print("  FAILED: Claude Code made no changes")
        output_path.unlink(missing_ok=True)
        sys.exit(1)

    print(f"\n  Edit produced {len(new_code):,} chars")

    # ── 7. Smoke test ──────────────────────────────────────────────────────────
    banner("STEP 7: Smoke test")
    err = swe_agent._smoke_test(output_path, IDEAS_DIR)
    if err:
        print(f"  SMOKE TEST FAILED:\n{err}")
        if not args.no_revert:
            output_path.unlink(missing_ok=True)
        sys.exit(1)
    print("  Smoke test PASSED ✓")

    # ── 8. Constraint check ────────────────────────────────────────────────────
    banner("STEP 8: Constraint check")
    checks = {
        f"class {next_version}Generator":          f"class {next_version}Generator" in new_code,
        f'VERSION = "{next_version}"':             f'VERSION = "{next_version}"' in new_code,
        f"GENERATOR = {next_version}Generator()":  f"GENERATOR = {next_version}Generator()" in new_code,
        "try/except around call_llm":              new_code.count("try:") >= 3,
    }
    all_ok = True
    for label, ok in checks.items():
        mark = "✓" if ok else "✗"
        print(f"  {mark}  {label}")
        if not ok:
            all_ok = False

    # ── 9. Diff ────────────────────────────────────────────────────────────────
    banner("STEP 9: Diff vs champion")
    champion_lines = champion_path.read_text().splitlines(keepends=True)
    new_lines      = new_code.splitlines(keepends=True)
    diff = list(difflib.unified_diff(
        champion_lines, new_lines,
        fromfile=f"{current_version}.py", tofile=f"{next_version}.py",
    ))
    adds    = sum(1 for l in diff if l.startswith('+') and not l.startswith('+++'))
    removes = sum(1 for l in diff if l.startswith('-') and not l.startswith('---'))
    print(f"  +{adds} lines added, -{removes} lines removed")
    print()
    for line in diff[:100]:
        print(line, end="")
    if len(diff) > 100:
        print(f"\n  ... ({len(diff) - 100} more diff lines)")

    # ── 10. Revert ─────────────────────────────────────────────────────────────
    banner("STEP 10: Cleanup")
    if args.no_revert:
        print(f"  Kept: {output_path}")
    else:
        output_path.unlink(missing_ok=True)
        print(f"  Reverted: deleted {output_path.name} ✓")

    banner("DONE")
    status = "ALL CONSTRAINTS MET ✓" if all_ok else "SOME CONSTRAINTS FAILED ✗"
    print(f"  {status}")
    print(f"  Smoke test: {'PASS ✓' if not err else 'FAIL ✗'}")
    print(f"  Diff: +{adds}/-{removes} lines from {current_version}")


if __name__ == "__main__":
    main()
