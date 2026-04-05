"""SWE Agent: multi-turn self-editing loop for the idea generator pipeline.

Instead of writing whole new S{n}.py files from scratch, the SWE agent
makes targeted, surgical edits to the current champion's code, tests each
edit with a fast mini-eval, and iterates until it can't improve further.

This is the proper Gödel loop: the agent reads its own failure modes,
proposes specific code changes, tests them, and accumulates improvements.

Loop per iteration:
  1. Analyze failures: read comparison report, extract what ideas lost and why
  2. Propose edit: ask meta-LLM for ONE specific targeted code change
  3. Apply edit: meta-LLM writes complete new version of the file
  4. Mini-eval: run 3 topics × 3 ideas = 9 pairs against current champion
  5. Accept (keep edit, continue) or reject (revert, try again)
  6. Stop when: max_rounds reached, or max_failures consecutive rejections,
     or mini-eval win rate hasn't improved for 2 consecutive rounds

After the loop: the accumulated edits form the new candidate S{n}.py,
which is then evaluated with a FULL 75-pair comparison against the champion.
"""

import json
import re
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).parent))
import log as _log

logger = _log.setup("swe_agent")

# ── Meta-model (writes the edits) ─────────────────────────────────────────────
SWE_MODEL = "claude-sonnet-4-6"
SWE_TIMEOUT = 180

# ── Stopping criteria ─────────────────────────────────────────────────────────
DEFAULT_MAX_ROUNDS = 6       # max accepted edits per session
DEFAULT_MAX_FAILURES = 3     # stop if this many consecutive mini-evals fail
MINI_IMPROVEMENT_THRESHOLD = 0.52  # mini-eval must show >52% to count as improvement
MINI_N_TOPICS = 3
MINI_N_IDEAS = 3

# ── Prompts ───────────────────────────────────────────────────────────────────

ANALYZE_PROMPT = """\
You are analyzing why a research idea generator is losing pairwise comparisons.

## Current generator code ({version})
```python
{code}
```

## Comparison results summary
- Total pairs judged: {total}
- Win rate: {win_rate:.1%}
- Top losing patterns from judge reasoning:

{losing_verdicts}

## Edit history this session
{edit_history}

## Your task
Identify the SINGLE most important failure pattern from the reasoning above.
What specific aspect of the generation pipeline is causing ideas to lose?
Be concrete and specific — quote from the judge reasoning where possible.

Output: 2-3 sentences describing the root cause of failures.
"""

PROPOSE_EDIT_PROMPT = """\
You are a software engineer improving a research idea generator.
Make ONE targeted edit to fix the identified failure pattern.

## Current generator code ({version})
```python
{code}
```

## Failure analysis
{failure_analysis}

## Edit history this session (DO NOT repeat these)
{edit_history}

## Your task
Propose ONE specific, targeted improvement to the generator code.
The edit should:
- Directly address the failure pattern identified above
- Be a surgical change (modify/add/remove specific prompts or steps)
- NOT be a complete rewrite — build on what works
- NOT repeat edits already tried this session

Think step by step about what change would most directly fix the identified failure.
Then write the COMPLETE updated Python file for {next_version}.

The file must:
1. Use VERSION = "{next_version}"
2. Keep the same import structure
3. Define GENERATOR = {next_version}Generator() at module level
4. End the final prompt with IDEA_FORMAT

Respond with ONLY the complete Python file — no markdown fences, no preamble.
"""

REFLECT_PROMPT = """\
The last edit to the research idea generator did NOT improve performance
(mini-eval win rate: {win_rate:.1%}, needed >{threshold:.1%}).

## Edit that was tried
{edit_description}

## Why it likely didn't help
Think about: did it address the root cause? Was it too minor? Did it overcomplicate?

## Current generator code
```python
{code}
```

## Failure analysis (unchanged)
{failure_analysis}

Suggest a DIFFERENT approach to fixing the same failure pattern.
Then write the COMPLETE updated Python file for {next_version}.
Respond with ONLY the complete Python file.
"""


# ── Helpers ───────────────────────────────────────────────────────────────────

def _extract_losing_verdicts(report_path: Path, n: int = 8) -> str:
    """Extract top losing verdicts (with judge reasoning) from a comparison report."""
    if report_path is None or not report_path.exists():
        return "(no comparison report available — first iteration from new baseline)"
    try:
        with open(report_path) as f:
            report = json.load(f)
        verdicts = report.get("verdicts", [])
        losses = [v for v in verdicts if v.get("winner") == "A"]  # A=current beat B=candidate
        losses.sort(key=lambda v: sum(v.get("scores_a", {}).values()) -
                    sum(v.get("scores_b", {}).values()), reverse=True)
        lines = []
        for v in losses[:n]:
            topic = v.get("topic", "")[:45]
            reasoning = v.get("reasoning", "")[:200]
            sa = sum(v.get("scores_a", {}).values())
            sb = sum(v.get("scores_b", {}).values())
            lines.append(f"- [{topic}] A={sa}/40 vs B={sb}/40: {reasoning}")
        return "\n".join(lines) if lines else "(no losses found)"
    except Exception as e:
        return f"(error reading report: {e})"


def _call_swe_llm(prompt: str) -> str:
    """Call the SWE meta-model. Strips accidental markdown fences."""
    sys.path.insert(0, str(Path(__file__).parent / "systems"))
    from base import make_client, call_llm
    client = make_client(SWE_MODEL)
    raw = call_llm(prompt, SWE_MODEL, client, temperature=0.7,
                   max_tokens=4096, timeout=SWE_TIMEOUT)
    raw = re.sub(r"^```python\s*\n", "", raw.strip())
    raw = re.sub(r"^```\s*\n", "", raw)
    raw = re.sub(r"\n```\s*$", "", raw)
    return raw.strip() + "\n"


def _run_mini_eval(
    ideas_dir: Path,
    candidate_path: Path,
    champion_version: str,
    n_topics: int = MINI_N_TOPICS,
    n_ideas: int = MINI_N_IDEAS,
    model: str = "gpt-4.1-mini",
    workers: int = 3,
) -> float:
    """Run a fast comparison of candidate vs champion on a topic subset.

    Returns win_rate_b (candidate win rate, 0..1).
    Returns 0.0 on failure to avoid false positives.
    """
    import concurrent.futures
    import importlib.util
    import random

    sys.path.insert(0, str(ideas_dir / "systems"))
    from base import make_client
    from runner import load_topics, run_system
    from judge import compare_systems, JUDGE_MODEL

    topics = load_topics(str(ideas_dir / "benchmark_topics.json"))
    sampled = random.sample(topics, min(n_topics, len(topics)))

    # Load candidate module dynamically
    try:
        spec = importlib.util.spec_from_file_location("_candidate_tmp", candidate_path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        candidate_version = mod.GENERATOR.VERSION
    except Exception as e:
        logger.error("Failed to load candidate %s: %s", candidate_path, e)
        return 0.0

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)

        # Copy candidate file to a temp systems dir for loading
        tmp_systems = tmp / "systems"
        shutil.copytree(str(ideas_dir / "systems"), str(tmp_systems))
        shutil.copy(str(candidate_path), str(tmp_systems / f"{candidate_version}.py"))

        # Run champion and candidate in parallel, each with topic-level workers
        champion_out = str(tmp / "champion")
        candidate_out = str(tmp / "candidate")

        results_champion = None
        results_candidate = None
        err_champion = None
        err_candidate = None

        def _run_champion():
            return run_system(
                version=champion_version,
                topics=sampled,
                output_dir=champion_out,
                model=model,
                n_ideas=n_ideas,
                systems_dir=str(ideas_dir / "systems"),
                workers=workers,
            )

        def _run_candidate():
            return run_system(
                version=candidate_version,
                topics=sampled,
                output_dir=candidate_out,
                model=model,
                n_ideas=n_ideas,
                systems_dir=str(tmp_systems),
                workers=workers,
            )

        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
            fut_champ = pool.submit(_run_champion)
            fut_cand = pool.submit(_run_candidate)
            try:
                results_champion = fut_champ.result()
            except Exception as e:
                err_champion = e
            try:
                results_candidate = fut_cand.result()
            except Exception as e:
                err_candidate = e

        if err_champion:
            logger.error("Mini-eval champion run failed: %s", err_champion)
            return 0.0
        if err_candidate:
            logger.error("Mini-eval candidate run failed: %s", err_candidate)
            return 0.0

        # Judge (parallel across topics)
        try:
            judge_client = make_client(JUDGE_MODEL)
            result = compare_systems(results_champion, results_candidate, judge_client,
                                     workers=workers)
            win_rate = result["win_rate_b"]
            logger.info("Mini-eval: %s vs %s → win_rate=%.1f%% (%d pairs)",
                        champion_version, candidate_version, win_rate * 100,
                        result["total_judged"])
            return win_rate
        except Exception as e:
            logger.error("Mini-eval judging failed: %s", e)
            return 0.0


# ── Main SWE agent loop ───────────────────────────────────────────────────────

def run_swe_loop(
    ideas_dir: Path,
    next_version: str,
    champion_version: str,
    compare_report_path: Optional[Path] = None,
    max_rounds: int = DEFAULT_MAX_ROUNDS,
    max_failures: int = DEFAULT_MAX_FAILURES,
    generator_model: str = "gpt-4.1-mini",
) -> Path:
    """Multi-turn SWE agent loop.

    Starts from the champion's code and makes iterative targeted edits,
    testing each with a mini-eval. Returns path to the final candidate file.

    Arguments:
        ideas_dir: root of the ideas/ directory
        next_version: name for the output system (e.g. "S21")
        champion_version: current champion (e.g. "S5")
        compare_report_path: path to comparison report for failure analysis
        max_rounds: max accepted edits
        max_failures: max consecutive failed mini-evals before stopping
        generator_model: model used for idea generation
    """
    systems_dir = ideas_dir / "systems"
    champion_path = systems_dir / f"{champion_version}.py"
    output_path = systems_dir / f"{next_version}.py"

    if not champion_path.exists():
        raise FileNotFoundError(f"Champion {champion_version}.py not found")

    # Find most relevant comparison report if not specified
    if compare_report_path is None:
        reports = sorted(ideas_dir.glob(f"results/compare_*_vs_{champion_version}.json"))
        reports += sorted(ideas_dir.glob(f"results/compare_{champion_version}_vs_*.json"))
        compare_report_path = reports[-1] if reports else None

    champion_code = champion_path.read_text()
    current_code = champion_code
    current_version_name = champion_version

    edit_history: list[dict] = []
    failures = 0
    last_win_rate: Optional[float] = None
    stall_count = 0

    winning_verdicts_str = _extract_losing_verdicts(compare_report_path)

    # Initial win_rate for context
    initial_win_rate = 0.5
    if compare_report_path and compare_report_path.exists():
        with open(compare_report_path) as f:
            rdata = json.load(f)
        initial_win_rate = rdata.get("win_rate_b", 0.5)

    logger.info("SWE loop starting: %s → %s  (max_rounds=%d, max_failures=%d)",
                champion_version, next_version, max_rounds, max_failures)

    for rnd in range(1, max_rounds + 1):
        if failures >= max_failures:
            logger.info("SWE loop stopping: %d consecutive failures", failures)
            break

        logger.info("── SWE round %d/%d ──────────────────────────────────", rnd, max_rounds)

        edit_history_str = "\n".join(
            f"Round {i+1}: {e['description'][:100]} → mini-eval {e['win_rate']:.1%}"
            for i, e in enumerate(edit_history)
        ) or "(none yet)"

        # ── Step 1: analyze failures ─────────────────────────────────────────
        analysis_prompt = ANALYZE_PROMPT.format(
            version=current_version_name,
            code=current_code,
            total=int(initial_win_rate * 100),  # approximate
            win_rate=initial_win_rate,
            losing_verdicts=winning_verdicts_str,
            edit_history=edit_history_str,
        )
        try:
            failure_analysis = _call_swe_llm(analysis_prompt)
            # failure_analysis is prose, not code — just the analysis text
            # Strip any code fences that might appear
            failure_analysis = re.sub(r"```.*?```", "", failure_analysis, flags=re.DOTALL).strip()
            logger.info("Failure analysis: %s", failure_analysis[:150])
        except Exception as e:
            logger.error("Failure analysis failed: %s", e)
            failure_analysis = "Unknown failure pattern — try improving experimental clarity."

        # ── Step 2: propose + apply edit ─────────────────────────────────────
        tmp_version = f"{next_version}_r{rnd}"

        if failures > 0 and edit_history:
            # Reflect on why last edit failed
            last = edit_history[-1] if edit_history else {}
            propose_prompt = REFLECT_PROMPT.format(
                win_rate=last.get("win_rate", 0.0),
                threshold=MINI_IMPROVEMENT_THRESHOLD,
                edit_description=last.get("description", "unknown"),
                code=current_code,
                failure_analysis=failure_analysis,
                next_version=tmp_version,
            )
        else:
            propose_prompt = PROPOSE_EDIT_PROMPT.format(
                version=current_version_name,
                code=current_code,
                failure_analysis=failure_analysis,
                edit_history=edit_history_str,
                next_version=tmp_version,
            )

        try:
            new_code = _call_swe_llm(propose_prompt)
            if "GENERATOR" not in new_code:
                raise ValueError("Generated code missing GENERATOR singleton")
        except Exception as e:
            logger.error("Edit proposal failed: %s", e)
            failures += 1
            continue

        # Write to temp file
        tmp_path = systems_dir / f"{tmp_version}.py"
        tmp_path.write_text(new_code)
        logger.info("Wrote candidate %s (%d chars)", tmp_version, len(new_code))

        # ── Step 3: mini-eval ────────────────────────────────────────────────
        try:
            win_rate = _run_mini_eval(
                ideas_dir=ideas_dir,
                candidate_path=tmp_path,
                champion_version=champion_version,  # always compare against original champion
                n_topics=MINI_N_TOPICS,
                n_ideas=MINI_N_IDEAS,
                model=generator_model,
            )
        except Exception as e:
            logger.error("Mini-eval crashed: %s", e)
            win_rate = 0.0
        finally:
            # Clean up temp file regardless
            if tmp_path.exists():
                tmp_path.unlink()

        improved = win_rate > MINI_IMPROVEMENT_THRESHOLD
        edit_history.append({
            "round": rnd,
            "description": failure_analysis[:200],
            "win_rate": win_rate,
            "accepted": improved,
            "code_snippet": new_code[:300],
        })

        if improved:
            logger.info("Round %d ACCEPTED (mini-eval %.1f%% > %.0f%%)",
                        rnd, win_rate * 100, MINI_IMPROVEMENT_THRESHOLD * 100)
            current_code = new_code
            current_version_name = tmp_version
            failures = 0

            # Check for improvement stall (winning but not by much more each time)
            if last_win_rate is not None and abs(win_rate - last_win_rate) < 0.02:
                stall_count += 1
                if stall_count >= 2:
                    logger.info("SWE loop stopping: win rate stalled at %.1f%%", win_rate * 100)
                    break
            else:
                stall_count = 0
            last_win_rate = win_rate
        else:
            logger.info("Round %d REJECTED (mini-eval %.1f%% ≤ %.0f%%)",
                        rnd, win_rate * 100, MINI_IMPROVEMENT_THRESHOLD * 100)
            failures += 1

    # Write final output
    # Update VERSION in code to final name
    final_code = re.sub(
        r'VERSION\s*=\s*["\'].*?["\']',
        f'VERSION = "{next_version}"',
        current_code,
    )
    # Update class name
    old_class = re.search(r'class\s+(S\w+Generator)', champion_code)
    new_class = f"class {next_version}Generator"
    if old_class:
        final_code = final_code.replace(old_class.group(0), new_class)
    # Update GENERATOR singleton
    final_code = re.sub(
        r'GENERATOR\s*=\s*\w+\(\)',
        f'GENERATOR = {next_version}Generator()',
        final_code,
    )

    output_path.write_text(final_code)
    logger.info("SWE loop complete: wrote %s (%d chars, %d rounds, %d accepted edits)",
                output_path, len(final_code), rnd,
                sum(1 for e in edit_history if e["accepted"]))

    # Save edit log
    log_path = ideas_dir / "results" / f"swe_log_{next_version}.json"
    with open(log_path, "w") as f:
        json.dump({
            "champion": champion_version,
            "output": next_version,
            "rounds": rnd,
            "edits": edit_history,
        }, f, indent=2)
    logger.info("Edit log saved to %s", log_path)

    return output_path
