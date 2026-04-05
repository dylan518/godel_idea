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

SWE_MEMORY_FILE = "results/swe_memory.json"

# ── Persistent cross-iteration memory ─────────────────────────────────────────

def load_swe_memory(ideas_dir: Path) -> list[dict]:
    """Load cross-iteration memory. Returns list of past iteration records."""
    path = ideas_dir / SWE_MEMORY_FILE
    if not path.exists():
        return []
    try:
        with open(path) as f:
            return json.load(f).get("iterations", [])
    except Exception:
        return []


def update_swe_memory(ideas_dir: Path, record: dict) -> None:
    """Append or update an iteration record in swe_memory.json.

    record keys: version, champion, mini_eval_best, accepted_edits,
                 failed_edits, full_eval_win_rate (optional), accepted (optional)
    """
    path = ideas_dir / SWE_MEMORY_FILE
    memory = load_swe_memory(ideas_dir)
    # Replace existing record for this version if present, else append
    existing = [i for i, r in enumerate(memory) if r.get("version") == record["version"]]
    if existing:
        memory[existing[0]].update(record)
    else:
        memory.append(record)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump({"iterations": memory}, f, indent=2)


def format_memory_str(memory: list[dict]) -> str:
    """Format memory as a concise summary for injection into prompts."""
    if not memory:
        return "(no previous iterations — this is the first run)"
    lines = []
    for r in memory[-6:]:  # show last 6 iterations to keep context manageable
        v = r.get("version", "?")
        champ = r.get("champion", "?")
        full_wr = r.get("full_eval_win_rate")
        accepted = r.get("accepted")
        status = ""
        if full_wr is not None:
            verdict = "ACCEPTED" if accepted else "REJECTED"
            status = f" → full eval {full_wr:.0%} ({verdict})"
        lines.append(f"\n### {v} (from {champ}){status}")
        for e in r.get("accepted_edits", []):
            lines.append(f"  ✓ WORKED  ({e['win_rate']:.0%}): {e['description'][:120]}")
        for e in r.get("failed_edits", [])[:4]:
            lines.append(f"  ✗ FAILED  ({e['win_rate']:.0%}): {e['description'][:120]}")
    return "\n".join(lines)


# ── Meta-model (writes the edits) ─────────────────────────────────────────────
SWE_MODEL = "claude-sonnet-4-6"
SWE_TIMEOUT = 180

# ── Stopping criteria ─────────────────────────────────────────────────────────
DEFAULT_MAX_ROUNDS = 6       # max accepted edits per session
DEFAULT_MAX_FAILURES = 3     # stop if this many consecutive mini-evals fail
MINI_IMPROVEMENT_THRESHOLD = 0.52  # mini-eval must show >52% to count as improvement
MINI_N_TOPICS = 3
MINI_N_IDEAS = 3

# ── Editable file list (relative to ideas_dir) ────────────────────────────────
EDITABLE_FILES = [
    "idea_tournament/prompts.py",
    "idea_tournament/tree_search.py",
    "idea_tournament/tournament.py",
]


def _bundle_editable_context(ideas_dir: Path, champion_version: str) -> str:
    """Read champion wrapper + all idea_tournament modules into one formatted bundle.

    This gives the SWE agent full visibility into the real generation logic,
    not just the thin S_sota.py wrapper.
    """
    parts = []

    # Champion wrapper (thin orchestrator — shows how modules are called)
    champion_path = ideas_dir / "systems" / f"{champion_version}.py"
    if champion_path.exists():
        parts.append(
            f"### FILE: systems/{champion_version}.py\n"
            f"```python\n{champion_path.read_text()}```"
        )

    # All editable idea_tournament modules
    for rel_path in EDITABLE_FILES:
        p = ideas_dir / rel_path
        if p.exists():
            parts.append(f"### FILE: {rel_path}\n```python\n{p.read_text()}```")

    return "\n\n".join(parts)


# ── Prompts ───────────────────────────────────────────────────────────────────

ANALYZE_PROMPT = """\
You are analyzing why a research idea generator is losing pairwise comparisons.

## Full generator codebase ({version})
The champion consists of a thin wrapper + three editable modules shown below.
The REAL logic lives in idea_tournament/ — that is what you should target.

{code_bundle}

## Comparison results summary
- Total pairs judged: {total}
- Win rate: {win_rate:.1%}
- Top losing patterns from judge reasoning:

{losing_verdicts}

## Edit history this session
{edit_history}

## Cross-iteration memory (ALL prior iterations — do not repeat what failed)
{memory}

## Your task
Identify the SINGLE most important failure pattern from the reasoning above.
What specific aspect of the generation pipeline is causing ideas to lose?
Reference the actual code (prompts, algorithm, parameters) where relevant.
Do NOT suggest approaches already tried in the memory above.

Output: 2-3 sentences describing the root cause of failures.
"""

_OUTPUT_FORMAT = """\
## Output format — CRITICAL isolation rule
Your output is a SINGLE complete Python file: systems/{next_version}.py

The champion S_sota.py imports from idea_tournament/ at runtime.
Your candidate will be evaluated HEAD-TO-HEAD against S_sota.
For a fair comparison, you MUST inline any code you are modifying:

  - Changing a prompt?  → Define the new prompt string as a local variable in
    generate_idea(), do NOT use the idea_tournament.prompts version.
  - Changing tree_search or tournament logic?  → Copy + modify the relevant
    functions inline inside generate_idea() or as module-level helpers.
  - NOT changing something?  → You may still import it from idea_tournament/.

If you import from idea_tournament/ for code you modified, both systems will
run the SAME code and the comparison will be meaningless.

The file must:
1. Import: from systems.base import IdeaGenerator, DEFAULT_MODEL, call_llm, IDEA_FORMAT
2. Define: class {next_version}Generator(IdeaGenerator) with VERSION = "{next_version}"
3. Implement: generate_idea(self, topic, client, model, temperature) with all logic inline
4. End with: GENERATOR = {next_version}Generator()

Respond with ONLY the complete Python file — no markdown fences, no preamble.
"""

PROPOSE_EDIT_PROMPT = """\
You are a software engineer improving a research idea generator.
Make ONE targeted edit to fix the identified failure pattern.

## Full generator codebase ({version})
{code_bundle}

## Failure analysis
{failure_analysis}

## Edit history this session (DO NOT repeat these)
{edit_history}

## Cross-iteration memory (ALL prior attempts — do not repeat what failed)
{memory}

## Your task
Propose ONE specific, targeted improvement. The edit should:
- Directly address the failure pattern identified above
- NOT repeat any approach in the memory or edit history above
- Prefer editing a prompt or parameter over restructuring the algorithm
- Be a surgical change — build on what works

Think step by step about what change would most directly fix the identified failure.
Then write the output file following the format below.

""" + _OUTPUT_FORMAT

REFLECT_PROMPT = """\
The last edit to the research idea generator did NOT improve performance
(mini-eval win rate: {win_rate:.1%}, needed >{threshold:.1%}).

## Edit that was tried
{edit_description}

## Why it likely didn't help
Think about: did it address the root cause? Was it too minor? Did it overcomplicate?

## Full generator codebase (current state)
{code_bundle}

## Failure analysis (unchanged)
{failure_analysis}

Suggest a DIFFERENT approach to fixing the same failure pattern.
Then write the output file following the format below.

""" + _OUTPUT_FORMAT


# ── Helpers ───────────────────────────────────────────────────────────────────

_TOPIC_GEN_PROMPT = """\
Generate {n} diverse research topics for evaluating a scientific idea generator.

Rules:
- Each topic must be a specific, active research area (not a broad field)
- Cover different scientific domains — no two topics from the same area
- Do NOT use any of these (already used as benchmark): {exclude}

Respond with ONLY a JSON array of objects, no other text:
[
  {{"id": "E1", "topic": "<specific research topic>", "domain": "<field>"}},
  ...
]"""

_BENCHMARK_TOPICS = [
    "Scaling laws for Large Language Models",
    "Protein structure prediction beyond AlphaFold",
    "Quantum error correction in NISQ devices",
    "Causal discovery from observational data",
    "Energy-efficient neuromorphic computing",
    "Generative AI for drug discovery and molecular design",
    "Federated learning with heterogeneous and non-IID data",
    "Deep learning emulators for climate model acceleration",
    "Uncertainty quantification in deep neural networks",
    "Foundation models for genomics and single-cell biology",
    "Emergent communication in multi-agent systems",
    "Mechanistic interpretability of transformer models",
    "Zero-shot generalization in reinforcement learning",
    "Topological methods for high-dimensional data analysis",
    "Quantum advantage in machine learning tasks",
]


def _generate_mini_eval_topics(n: int = 5, model: str = "gpt-4.1-mini") -> list[dict]:
    """Generate n fresh random research topics for a mini-eval round.

    Uses a cheap LLM call so topics are never reused — prevents overfitting
    to any fixed dev set. Falls back to random sampling from dev_topics.json
    on failure.
    """
    import json as _json
    import re as _re
    sys.path.insert(0, str(Path(__file__).parent / "systems"))
    from base import make_client, call_llm

    exclude_str = "; ".join(_BENCHMARK_TOPICS[:8])  # keep prompt short
    prompt = _TOPIC_GEN_PROMPT.format(n=n, exclude=exclude_str)
    try:
        client = make_client(model)
        raw = call_llm(prompt, model, client, temperature=1.0, max_tokens=512, timeout=30)
        raw = _re.sub(r"^```(?:json)?\s*\n?", "", raw.strip())
        raw = _re.sub(r"\n?```\s*$", "", raw)
        topics = _json.loads(raw)
        if isinstance(topics, list) and topics and "topic" in topics[0]:
            logger.debug("Generated %d fresh mini-eval topics", len(topics))
            return topics[:n]
    except Exception as e:
        logger.warning("Topic generation failed (%s) — falling back to dev_topics.json", e)

    # Fallback: load from dev_topics.json
    dev_path = Path(__file__).parent / "dev_topics.json"
    if dev_path.exists():
        with open(dev_path) as f:
            return _json.load(f).get("topics", [])[:n]
    return [{"id": f"E{i}", "topic": t, "domain": "ML"}
            for i, t in enumerate(_BENCHMARK_TOPICS[-n:], 1)]


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
    from runner import run_system
    from judge import compare_systems, JUDGE_MODEL

    # Generate fresh topics per mini-eval round — prevents overfitting to any fixed set
    sampled = _generate_mini_eval_topics(n=n_topics, model=model)

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
    # Full code bundle (wrapper + all idea_tournament modules) — used in prompts
    code_bundle = _bundle_editable_context(ideas_dir, champion_version)

    edit_history: list[dict] = []
    failures = 0
    last_win_rate: Optional[float] = None
    stall_count = 0

    # Load cross-iteration memory
    memory = load_swe_memory(ideas_dir)
    memory_str = format_memory_str(memory)
    logger.info("Loaded SWE memory: %d prior iterations", len(memory))

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
            code_bundle=code_bundle,
            total=int(initial_win_rate * 100),  # approximate
            win_rate=initial_win_rate,
            losing_verdicts=winning_verdicts_str,
            edit_history=edit_history_str,
            memory=memory_str,
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
                code_bundle=code_bundle,
                failure_analysis=failure_analysis,
                next_version=tmp_version,
            )
        else:
            propose_prompt = PROPOSE_EDIT_PROMPT.format(
                version=current_version_name,
                code_bundle=code_bundle,
                failure_analysis=failure_analysis,
                edit_history=edit_history_str,
                memory=memory_str,
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
            # Update code_bundle so subsequent rounds see the accumulated changes
            code_bundle = (
                f"### FILE: systems/{tmp_version}.py\n```python\n{new_code}```\n\n"
                + "\n\n".join(
                    f"### FILE: {rel}\n```python\n{(ideas_dir / rel).read_text()}```"
                    for rel in EDITABLE_FILES if (ideas_dir / rel).exists()
                )
            )
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

    # Update cross-iteration memory
    accepted_edits = [
        {"description": e["description"], "win_rate": e["win_rate"]}
        for e in edit_history if e["accepted"]
    ]
    failed_edits = [
        {"description": e["description"], "win_rate": e["win_rate"]}
        for e in edit_history if not e["accepted"]
    ]
    best_mini = max((e["win_rate"] for e in edit_history), default=0.0)
    update_swe_memory(ideas_dir, {
        "version": next_version,
        "champion": champion_version,
        "mini_eval_best": best_mini,
        "accepted_edits": accepted_edits,
        "failed_edits": failed_edits,
        # full_eval_win_rate and accepted filled in later by cmd_swe_evolve
    })
    logger.info("SWE memory updated for %s", next_version)

    return output_path
