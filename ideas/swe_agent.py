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
SWE_CONTEXT_FILE = "results/swe_context.json"

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

# When True, use `claude --print` (Claude Code CLI) for the actual code editing
# instead of a raw API call. Claude Code reads files with its own tools and makes
# surgical edits, avoiding the "write full file from scratch" failure mode.
USE_CLAUDE_CODE = True

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


# ── Rich cross-iteration context ──────────────────────────────────────────────
# swe_context.json stores three things that the SWE agent needs across iterations:
#   1. pipeline_overview  — concise description of how the current champion works
#   2. judge_profile      — patterns accumulated from ALL judge verdicts ever seen
#   3. iteration_log      — what changed each time and the key lesson learned

def load_swe_context(ideas_dir: Path) -> dict:
    path = ideas_dir / SWE_CONTEXT_FILE
    if not path.exists():
        return {"pipeline_overview": "", "judge_profile": {}, "iteration_log": []}
    try:
        with open(path) as f:
            return json.load(f)
    except Exception:
        return {"pipeline_overview": "", "judge_profile": {}, "iteration_log": []}


def save_swe_context(ideas_dir: Path, ctx: dict) -> None:
    path = ideas_dir / SWE_CONTEXT_FILE
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(ctx, f, indent=2)


def extract_judge_patterns(report_path: Path, n_quotes: int = 6) -> dict:
    """Parse a comparison report and extract what the judge consistently rewards/penalizes.

    Returns:
      rewards   — phrases/patterns in winning verdicts (champion beat candidate)
      penalizes — phrases/patterns in losing verdicts (candidate beat champion)
      quotes    — verbatim judge reasoning snippets (both sides)
    """
    if report_path is None or not report_path.exists():
        return {}
    try:
        with open(report_path) as f:
            report = json.load(f)
    except Exception:
        return {}

    verdicts = report.get("verdicts", [])
    if not verdicts:
        return {}

    # A wins = champion beat candidate; B wins = candidate beat champion
    a_wins = [v for v in verdicts if v.get("winner") == "A"]
    b_wins = [v for v in verdicts if v.get("winner") == "B"]

    # Collect reasoning text from each side
    a_reasoning = [v.get("reasoning", "")[:300] for v in a_wins if v.get("reasoning")]
    b_reasoning = [v.get("reasoning", "")[:300] for v in b_wins if v.get("reasoning")]

    # Top quotes: champion wins first (what the current system does well),
    # then candidate wins (what the new direction achieved)
    quotes = []
    for r in a_reasoning[:n_quotes // 2]:
        quotes.append(f"[champion won] {r}")
    for r in b_reasoning[:n_quotes // 2]:
        quotes.append(f"[candidate won] {r}")

    return {
        "champion_win_count": len(a_wins),
        "candidate_win_count": len(b_wins),
        "champion_win_reasoning": a_reasoning[:3],
        "candidate_win_reasoning": b_reasoning[:3],
        "quotes": quotes,
    }


def update_swe_context(
    ideas_dir: Path,
    version: str,
    champion: str,
    full_eval_win_rate: float,
    accepted: bool,
    code_change_summary: str,
    compare_report_path: Optional[Path],
) -> None:
    """Update swe_context.json after a full eval completes.

    Accumulates judge patterns across all iterations into a growing profile.
    """
    ctx = load_swe_context(ideas_dir)

    # Extract patterns from this comparison
    patterns = extract_judge_patterns(compare_report_path)

    # Accumulate judge profile across all iterations
    profile = ctx.get("judge_profile", {})
    all_champ_reasoning = profile.get("all_champion_win_reasoning", [])
    all_cand_reasoning = profile.get("all_candidate_win_reasoning", [])
    if patterns.get("champion_win_reasoning"):
        all_champ_reasoning.extend(patterns["champion_win_reasoning"])
    if patterns.get("candidate_win_reasoning"):
        all_cand_reasoning.extend(patterns["candidate_win_reasoning"])
    # Keep the most recent 20 of each to avoid unbounded growth
    profile["all_champion_win_reasoning"] = all_champ_reasoning[-20:]
    profile["all_candidate_win_reasoning"] = all_cand_reasoning[-20:]
    profile["last_comparison"] = patterns
    ctx["judge_profile"] = profile

    # Append iteration log entry
    log_entry = {
        "version": version,
        "champion_at_time": champion,
        "full_eval_win_rate": full_eval_win_rate,
        "accepted": accepted,
        "code_change_summary": code_change_summary,
        "judge_patterns": {
            "champion_wins": patterns.get("champion_win_count", 0),
            "candidate_wins": patterns.get("candidate_win_count", 0),
            "sample_champion_wins": patterns.get("champion_win_reasoning", [])[:2],
            "sample_candidate_wins": patterns.get("candidate_win_reasoning", [])[:2],
        },
    }
    ctx.setdefault("iteration_log", []).append(log_entry)

    save_swe_context(ideas_dir, ctx)
    logger.info("SWE context updated for %s (full_eval=%.1f%%, accepted=%s)",
                version, full_eval_win_rate * 100, accepted)


def build_pipeline_overview(ideas_dir: Path, champion_version: str) -> str:
    """Generate a concise human-readable overview of the current champion pipeline.

    Reads the champion file and idea_tournament modules to produce a short
    description the SWE agent can use without parsing all the raw code.
    """
    lines = [f"Current champion: {champion_version}"]
    lines.append("Pipeline: IdeaTreeSearch (L1→L2→L3, ~12 candidates) → Elo tournament → Expansion")
    lines.append("")

    # Check what the champion has customized vs vanilla S_sota
    champion_path = ideas_dir / "systems" / f"{champion_version}.py"
    if champion_path.exists():
        code = champion_path.read_text()
        if "EXPAND_WINNER_PROMPT_V2" in code or "EXPAND_WINNER_PROMPT" in code:
            # Find the custom prompt name
            m = re.search(r"(EXPAND_WINNER_PROMPT\w*)\s*=\s*\"\"\"", code)
            if m:
                lines.append(f"  • Expansion prompt: custom {m.group(1)} (overrides idea_tournament default)")
        for func, module in [("build_idea_tree", "tree_search"), ("run_tournament", "tournament")]:
            if f"def {func}" in code:
                lines.append(f"  • {func}: inlined custom version in champion file")
            else:
                lines.append(f"  • {func}: using idea_tournament/{module}.py")
        if "def generate_idea" in code:
            # Count approximate LLM calls (each call_llm = 1 call)
            n_calls = code.count("call_llm(")
            lines.append(f"  • generate_idea: ~{n_calls} direct call_llm() calls + tree/tournament calls")

    lines.append("")
    lines.append("Editable modules (primary targets for improvement):")
    for rel in EDITABLE_FILES:
        p = ideas_dir / rel
        if p.exists():
            n_lines = len(p.read_text().splitlines())
            lines.append(f"  • {rel} ({n_lines} lines)")

    return "\n".join(lines)


def format_swe_context(ctx: dict, ideas_dir: Path, champion_version: str) -> str:
    """Format the full context for injection into SWE agent prompts.

    Returns a structured string with three sections:
      1. Pipeline overview
      2. Judge profile (accumulated preferences)
      3. Experiment log (what worked/failed and why)
    """
    sections = []

    # ── Section 1: Pipeline overview ──────────────────────────────────────────
    overview = ctx.get("pipeline_overview") or build_pipeline_overview(ideas_dir, champion_version)
    sections.append("### 1. Current Pipeline\n" + overview)

    # ── Section 2: Judge profile (what the judge consistently rewards) ─────────
    profile = ctx.get("judge_profile", {})
    champ_reasons = profile.get("all_champion_win_reasoning", [])
    cand_reasons = profile.get("all_candidate_win_reasoning", [])

    judge_lines = ["### 2. Accumulated Judge Preferences"]
    if champ_reasons:
        judge_lines.append("\nWhen the CHAMPION wins, judges say things like:")
        for r in champ_reasons[-4:]:
            judge_lines.append(f"  > {r}")
    if cand_reasons:
        judge_lines.append("\nWhen the CANDIDATE wins (good — what to aim for):")
        for r in cand_reasons[-4:]:
            judge_lines.append(f"  > {r}")
    if not champ_reasons and not cand_reasons:
        judge_lines.append("(no judge data accumulated yet)")
    sections.append("\n".join(judge_lines))

    # ── Section 3: Experiment log ──────────────────────────────────────────────
    log = ctx.get("iteration_log", [])
    log_lines = ["### 3. Experiment Log"]
    if not log:
        log_lines.append("(no completed iterations yet)")
    for entry in log[-8:]:  # show last 8
        v = entry.get("version", "?")
        wr = entry.get("full_eval_win_rate", 0)
        acc = "✓ ACCEPTED" if entry.get("accepted") else "✗ REJECTED"
        summary = entry.get("code_change_summary", "")[:150]
        log_lines.append(f"\n{v} vs {entry.get('champion_at_time','?')}: {wr:.0%} — {acc}")
        if summary:
            log_lines.append(f"  Changed: {summary}")
        jp = entry.get("judge_patterns", {})
        sw = jp.get("sample_candidate_wins", [])
        if sw:
            log_lines.append(f"  Judge when this won: {sw[0][:150]}")
    sections.append("\n".join(log_lines))

    return "\n\n".join(sections)


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

DIAGNOSE_PROMPT = """\
You are diagnosing exactly why a research idea generator is losing pairwise evaluations.

## Generation pipeline — the code that produced the losing ideas
{champion_code}

## Concrete failing examples — where our ideas lost

{grounded_failures}

## What has already been tried and failed this session
{failed_attempts}

## Context: judge preferences and experiment history
{swe_context}

## Your task
Study the losing ideas above carefully. Identify the specific step in the pipeline
where quality broke down — was the hypothesis too generic? Did selection pick the safe
option? Did the critique fail to add experimental specificity? Did the revision water
things down?

Diagnose the root cause, then propose ONE targeted concrete fix.

Output EXACTLY in this format (no other text):
DIAGNOSIS: <which specific step failed and precisely why — reference the actual examples>
FIX: <the single concrete change — which call, what the prompt should do differently>
EXPECTED_IMPACT: <why this addresses the failure pattern seen above>
"""

ATTACK_PROMPT = """\
A proposed fix for an underperforming research idea generator:

DIAGNOSIS: {diagnosis}
FIX: {proposed_fix}

Attack this fix on 3 dimensions:
1. ROOT CAUSE: Does this actually address the root cause, or just a symptom?
2. ASSUMPTION: What unstated assumption does this fix make that might not hold?
3. RESIDUAL FAILURE: What would still fail after making this change?

Then write a REVISED fix that is stronger, more targeted, and addresses all three attacks.

REVISED_FIX: <concrete, strengthened version — specific enough to implement directly as code>
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
You are a software engineer implementing a targeted improvement to a research idea generator.

## Full generator codebase ({version})
{code_bundle}

## Diagnosis and proposed fix
{refined_fix}

## Context: pipeline, judge preferences, experiment history
{swe_context}

## Edit history this session (DO NOT repeat these)
{edit_history}

## Your task
Implement the proposed fix as a NEW generate_idea() method.
Make the minimal targeted change that addresses the diagnosis — do not restructure
unrelated parts of the pipeline.

CRITICAL robustness requirements:
- Every call_llm() call MUST be wrapped in try/except with a sensible fallback
- Never call len(), enumerate(), or index into a variable that might be None or empty
- The generate_idea() method must ALWAYS return a non-empty string, even on full failure
- Budget ~10-15 LLM calls per idea

Then write the output file following the format below.

""" + _OUTPUT_FORMAT

REFLECT_PROMPT = """\
The last edit did NOT improve performance (mini-eval: {win_rate:.1%}, needed >{threshold:.1%}).

## Edit that was tried
{edit_description}

## Revised diagnosis and fix
{refined_fix}

## Full generator codebase (current state)
{code_bundle}

## Your task
Implement the revised fix. The previous attempt failed — the refined fix above
addresses why. Make a targeted change that differs from what was tried.

CRITICAL: Wrap every call_llm() in try/except. generate_idea() must always return a string.

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

def _extract_grounded_failures(report_path: Path, ideas_dir: Path, n: int = 4) -> str:
    """Extract concrete failing examples with actual idea text for grounded diagnosis.

    Shows champion idea vs candidate idea side-by-side with scores and full judge
    reasoning — so the SWE agent diagnoses from real outputs, not abstractions.
    """
    if report_path is None or not report_path.exists():
        return "(no comparison report — first iteration from new baseline)"
    try:
        with open(report_path) as f:
            report = json.load(f)
        verdicts = report.get("verdicts", [])
        losses = [v for v in verdicts if v.get("winner") == "A"]

        # Detect infrastructure failures: B score=0 on most losses → quota error
        zero_score_losses = [v for v in losses if sum(v.get("scores_b", {}).values()) == 0]
        if len(losses) > 0 and len(zero_score_losses) / len(losses) > 0.5:
            return (
                "⚠️  INFRASTRUCTURE FAILURE: candidate scored 0/40 on "
                f"{len(zero_score_losses)}/{len(losses)} comparisons — API quota error, "
                "not a quality failure. Focus on improving idea quality, not error handling."
            )

        losses.sort(key=lambda v: sum(v.get("scores_a", {}).values()) -
                    sum(v.get("scores_b", {}).values()), reverse=True)

        # Load actual idea texts from results JSONs
        system_a = losses[0]["system_a"] if losses else None
        system_b = losses[0]["system_b"] if losses else None
        ideas_a: dict = {}
        ideas_b: dict = {}
        for sys_name, store in [(system_a, ideas_a), (system_b, ideas_b)]:
            if not sys_name:
                continue
            p = ideas_dir / "results" / sys_name / "ideas.json"
            if p.exists():
                try:
                    for item in json.load(open(p)):
                        store[(item["topic_id"], item.get("idea_index", 0))] = item.get("text", "")
                except Exception:
                    pass

        lines = []
        for v in losses[:n]:
            topic = v.get("topic", "")
            tid = v.get("topic_id", "")
            idx = v.get("idea_index", 0)
            sa = sum(v.get("scores_a", {}).values())
            sb = sum(v.get("scores_b", {}).values())
            reasoning = v.get("reasoning", "")

            champ_text = ideas_a.get((tid, idx), "(text unavailable)")
            cand_text  = ideas_b.get((tid, idx), "(text unavailable)")

            lines.append(f"### Topic: {topic}")
            lines.append(f"**Champion ({system_a}) — {sa}/40 — WON:**")
            lines.append(champ_text[:600])
            lines.append(f"\n**Candidate ({system_b}) — {sb}/40 — LOST:**")
            lines.append(cand_text[:600])
            lines.append(f"\n**Judge:** {reasoning}")
            lines.append("")

        return "\n".join(lines) if lines else "(no losses found)"
    except Exception as e:
        return f"(error reading report: {e})"


def _call_swe_llm(prompt: str) -> str:
    """Call the SWE meta-model. Extracts Python code block from response."""
    sys.path.insert(0, str(Path(__file__).parent / "systems"))
    from base import make_client, call_llm
    client = make_client(SWE_MODEL)
    raw = call_llm(prompt, SWE_MODEL, client, temperature=0.7,
                   max_tokens=4096, timeout=SWE_TIMEOUT)
    raw = raw.strip()

    # If response contains a fenced Python block, extract it (handles explanatory preamble)
    fence_match = re.search(r"```python\s*\n(.*?)```", raw, re.DOTALL)
    if fence_match:
        return fence_match.group(1).strip() + "\n"

    # Fallback: try any fenced block
    fence_match = re.search(r"```\s*\n(.*?)```", raw, re.DOTALL)
    if fence_match:
        block = fence_match.group(1).strip()
        if "class " in block and "def generate_idea" in block:
            return block + "\n"

    # Last resort: strip leading/trailing fences and return
    raw = re.sub(r"^```python\s*\n", "", raw)
    raw = re.sub(r"^```\s*\n", "", raw)
    raw = re.sub(r"\n```\s*$", "", raw)
    return raw.strip() + "\n"


def _call_swe_llm_prose(prompt: str, max_tokens: int = 1024) -> str:
    """Call the SWE meta-model for prose output (diagnosis, attack). No code extraction."""
    sys.path.insert(0, str(Path(__file__).parent / "systems"))
    from base import make_client, call_llm
    client = make_client(SWE_MODEL)
    raw = call_llm(prompt, SWE_MODEL, client, temperature=0.7,
                   max_tokens=max_tokens, timeout=SWE_TIMEOUT)
    return raw.strip()


def _parse_diagnose_output(raw: str) -> tuple[str, str, str]:
    """Parse DIAGNOSIS / FIX / EXPECTED_IMPACT fields from DIAGNOSE_PROMPT output."""
    diagnosis = fix = impact = ""
    for line in raw.splitlines():
        if line.startswith("DIAGNOSIS:"):
            diagnosis = line[len("DIAGNOSIS:"):].strip()
        elif line.startswith("FIX:"):
            fix = line[len("FIX:"):].strip()
        elif line.startswith("EXPECTED_IMPACT:"):
            impact = line[len("EXPECTED_IMPACT:"):].strip()
    # Multi-line fallback: if fields span multiple lines, grab everything after the label
    if not diagnosis:
        m = re.search(r"DIAGNOSIS:\s*(.+?)(?=FIX:|EXPECTED_IMPACT:|$)", raw, re.DOTALL)
        if m:
            diagnosis = m.group(1).strip()
    if not fix:
        m = re.search(r"FIX:\s*(.+?)(?=EXPECTED_IMPACT:|$)", raw, re.DOTALL)
        if m:
            fix = m.group(1).strip()
    if not impact:
        m = re.search(r"EXPECTED_IMPACT:\s*(.+?)$", raw, re.DOTALL)
        if m:
            impact = m.group(1).strip()
    return diagnosis or raw[:300], fix or raw[:300], impact


def _parse_revised_fix(raw: str) -> str:
    """Extract REVISED_FIX from ATTACK_PROMPT output."""
    m = re.search(r"REVISED_FIX:\s*(.+?)$", raw, re.DOTALL)
    if m:
        return m.group(1).strip()
    return raw.strip()  # fallback: use full response


def _call_claude_code_edit(
    champion_path: Path,
    output_path: Path,
    task_description: str,
    failure_analysis: str,
    ideas_dir: Path,
    next_version: str,
) -> str:
    """Invoke Claude Code CLI to implement a targeted improvement.

    Copies champion to output_path, then runs `claude --print` with a structured
    task. Claude Code reads the files with its own tools and makes surgical edits
    rather than generating a whole new file from scratch.

    Returns the content of the edited file.
    Raises RuntimeError if the 'claude' CLI is unavailable or the edit fails.
    """
    import subprocess

    # Start from champion as a base so Claude Code can diff/edit rather than rewrite
    shutil.copy(champion_path, output_path)

    repo_root = ideas_dir.parent
    try:
        rel_output = output_path.relative_to(repo_root)
        rel_prompts = Path("ideas/idea_tournament/prompts.py")
        rel_tree    = Path("ideas/idea_tournament/tree_search.py")
        rel_tourn   = Path("ideas/idea_tournament/tournament.py")
    except ValueError:
        rel_output = output_path
        rel_prompts = ideas_dir / "idea_tournament/prompts.py"
        rel_tree    = ideas_dir / "idea_tournament/tree_search.py"
        rel_tourn   = ideas_dir / "idea_tournament/tournament.py"

    task_prompt = f"""You are improving a Python research idea generator. Make ONE focused, surgical improvement.

## Diagnosis and proposed fix
{task_description}

## Why the current generator is losing to the baseline
{failure_analysis}

## What to do
1. Read `{rel_output}` — this is already a copy of the champion, your starting point
2. Read the relevant `ideas/idea_tournament/` modules for full context on what the pipeline does
3. Implement the improvement strategy above — make TARGETED edits to `{rel_output}`

## Hard constraints (failure to meet these causes the system to crash or be disqualified)
- Class name MUST be `{next_version}Generator(IdeaGenerator)`
- `VERSION = "{next_version}"` (update from whatever the champion has)
- File MUST end with `GENERATOR = {next_version}Generator()`
- Any code you MODIFY from `idea_tournament/` must be inlined in the file (not imported)
- Every `call_llm()` call must be wrapped in try/except with a sensible fallback
- `generate_idea(self, topic, client, model, temperature)` must always return a non-empty string
- Keep total LLM calls per idea: ~10-15 (same budget as champion, just use them differently)

## What the judge rewards (optimise for this, not pipeline complexity)
- Concrete, specific, well-grounded ideas
- Named datasets, baselines, and quantitative metrics
- Clear problem statements with named failure modes
- Novelty relative to cited SOTA

Make the minimal change needed to implement the strategy. Do NOT restructure the file unnecessarily."""

    # Strip CLAUDECODE from the subprocess env so nested Claude Code sessions work.
    # Claude Code blocks nested launches unless this var is absent.
    import os as _os
    clean_env = {k: v for k, v in _os.environ.items() if k != "CLAUDECODE"}

    try:
        result = subprocess.run(
            ["claude", "--print", "--allowedTools", "Read,Edit,Write"],
            input=task_prompt,
            cwd=str(repo_root),
            capture_output=True,
            text=True,
            timeout=600,
            env=clean_env,
        )
        logger.info(
            "Claude Code edit done (rc=%d): %s",
            result.returncode,
            (result.stdout or result.stderr or "")[:200],
        )
        if result.returncode != 0:
            raise RuntimeError(
                f"claude --print returned rc={result.returncode}: "
                f"{result.stderr[:400]}"
            )
    except subprocess.TimeoutExpired:
        raise RuntimeError("Claude Code edit timed out after 600s")
    except FileNotFoundError:
        raise RuntimeError(
            "'claude' CLI not found — install Claude Code or set USE_CLAUDE_CODE=False"
        )

    if not output_path.exists():
        raise ValueError(f"Claude Code did not write {output_path}")

    content = output_path.read_text()
    if not content.strip():
        raise ValueError("Claude Code produced an empty file")

    # Verify it's actually different from the champion (not a no-op)
    champion_content = champion_path.read_text()
    if content == champion_content:
        raise ValueError("Claude Code made no changes to the file")

    logger.info("Claude Code wrote %s (%d chars)", output_path.name, len(content))
    return content


def _smoke_test(candidate_path: Path, ideas_dir: Path) -> str | None:
    """Import the candidate module in a subprocess and verify it's well-formed.

    Returns None on success, or an error string on failure.
    Checks: syntax, imports, GENERATOR attribute, generate_idea method.
    Does NOT run a full idea generation (too expensive).
    """
    import subprocess
    script = f"""
import sys
sys.path.insert(0, {str(ideas_dir)!r})
sys.path.insert(0, {str(ideas_dir / 'systems')!r})
import importlib.util, traceback
try:
    spec = importlib.util.spec_from_file_location("_smoke", {str(candidate_path)!r})
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    assert hasattr(mod, "GENERATOR"), "Missing GENERATOR singleton"
    g = mod.GENERATOR
    assert hasattr(g, "generate_idea"), "Missing generate_idea method"
    assert hasattr(g, "VERSION"), "Missing VERSION attribute"
    print("OK:", g.VERSION)
except Exception:
    traceback.print_exc()
    sys.exit(1)
"""
    try:
        result = subprocess.run(
            [sys.executable, "-c", script],
            capture_output=True, text=True, timeout=30,
            env={**__import__("os").environ},
        )
        if result.returncode != 0:
            return (result.stdout + result.stderr).strip()
        return None
    except subprocess.TimeoutExpired:
        return "Smoke test timed out (30s)"
    except Exception as e:
        return f"Smoke test runner error: {e}"


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
                                     workers=workers,
                                     early_stop_threshold=MINI_IMPROVEMENT_THRESHOLD)
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

    # Load cross-iteration memory and rich context
    memory = load_swe_memory(ideas_dir)
    logger.info("Loaded SWE memory: %d prior iterations", len(memory))
    swe_ctx = load_swe_context(ideas_dir)
    swe_context_str = format_swe_context(swe_ctx, ideas_dir, champion_version)

    grounded_failures_str = _extract_grounded_failures(compare_report_path, ideas_dir)

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

        # ── Step 1: diagnose failures from concrete examples ─────────────────
        champion_code_text = champion_path.read_text()
        failed_attempts_str = "\n".join(
            f"Round {e['round']}: {e['description'][:120]} → {e['win_rate']:.1%}"
            for e in edit_history if not e.get("accepted")
        ) or "(none yet)"

        diagnose_prompt = DIAGNOSE_PROMPT.format(
            champion_code=champion_code_text,
            grounded_failures=grounded_failures_str,
            failed_attempts=failed_attempts_str,
            swe_context=swe_context_str,
        )
        try:
            diagnose_raw = _call_swe_llm_prose(diagnose_prompt, max_tokens=800)
            diagnosis, proposed_fix, expected_impact = _parse_diagnose_output(diagnose_raw)
            logger.info("Diagnosis:\n%s\n\nProposed fix:\n%s", diagnosis, proposed_fix)
        except Exception as e:
            logger.error("Diagnosis failed: %s", e)
            diagnosis = "Unknown failure — try improving experimental specificity."
            proposed_fix = "Add concrete dataset and baseline names to the final revision prompt."

        # ── Step 2: adversarially refine the proposed fix ────────────────────
        tmp_version = f"{next_version}_r{rnd}"
        tmp_path = systems_dir / f"{tmp_version}.py"

        try:
            attack_prompt = ATTACK_PROMPT.format(
                diagnosis=diagnosis,
                proposed_fix=proposed_fix,
            )
            attack_raw = _call_swe_llm_prose(attack_prompt, max_tokens=600)
            refined_fix = _parse_revised_fix(attack_raw)
            logger.info("Refined fix:\n%s", refined_fix)
        except Exception as e:
            logger.error("Attack/refine failed: %s", e)
            refined_fix = proposed_fix  # fall back to unrefined fix

        task_description = (
            f"DIAGNOSIS: {diagnosis}\n\n"
            f"REFINED FIX: {refined_fix}\n\n"
            f"EXPECTED IMPACT: {expected_impact}\n\n"
            f"Edit history this session:\n{edit_history_str}"
        )

        try:
            if USE_CLAUDE_CODE:
                new_code = _call_claude_code_edit(
                    champion_path=champion_path,
                    output_path=tmp_path,
                    task_description=task_description,
                    failure_analysis=failure_analysis,
                    ideas_dir=ideas_dir,
                    next_version=tmp_version,
                )
            else:
                # Fallback: direct LLM call (original behaviour)
                if failures > 0 and edit_history:
                    last = edit_history[-1] if edit_history else {}
                    propose_prompt = REFLECT_PROMPT.format(
                        win_rate=last.get("win_rate", 0.0),
                        threshold=MINI_IMPROVEMENT_THRESHOLD,
                        edit_description=last.get("description", "unknown"),
                        code_bundle=code_bundle,
                        refined_fix=refined_fix,
                        next_version=tmp_version,
                    )
                else:
                    propose_prompt = PROPOSE_EDIT_PROMPT.format(
                        version=current_version_name,
                        code_bundle=code_bundle,
                        refined_fix=refined_fix,
                        edit_history=edit_history_str,
                        swe_context=swe_context_str,
                        next_version=tmp_version,
                    )
                new_code = _call_swe_llm(propose_prompt)
                # Auto-repair missing GENERATOR singleton
                if "GENERATOR" not in new_code:
                    cls_match = (
                        re.search(r"class\s+(S\w+Generator)\s*\(", new_code)
                        or re.search(r"class\s+(\w+Generator)\s*\(\s*IdeaGenerator", new_code)
                        or re.search(r"class\s+(\w+)\s*\(\s*IdeaGenerator\s*\)", new_code)
                        or re.search(r"class\s+(\w+Generator)\s*\(", new_code)
                    )
                    if cls_match:
                        new_code = new_code.rstrip() + f"\n\nGENERATOR = {cls_match.group(1)}()\n"
                        logger.warning("Auto-repaired missing GENERATOR → %s()", cls_match.group(1))
                    else:
                        raise ValueError("Generated code missing GENERATOR singleton")
                tmp_path.write_text(new_code)
                logger.info("Wrote candidate %s (%d chars)", tmp_version, len(new_code))
        except Exception as e:
            logger.error("Edit proposal failed: %s", e)
            failures += 1
            continue

        # ── Smoke test: validate the file imports cleanly before spending mini-eval ──
        smoke_err = _smoke_test(tmp_path, ideas_dir)
        if smoke_err:
            logger.error("Smoke test FAILED for %s — skipping mini-eval:\n%s", tmp_version, smoke_err[-600:])
            tmp_path.unlink(missing_ok=True)
            failures += 1
            continue
        logger.info("Smoke test passed for %s", tmp_version)

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
            pass  # keep temp file — rejected rounds are preserved for inspection

        improved = win_rate > MINI_IMPROVEMENT_THRESHOLD
        edit_history.append({
            "round": rnd,
            "description": task_description[:1200],
            "failure_analysis": failure_analysis[:2000],
            "win_rate": win_rate,
            "accepted": improved,
            "code_snippet": new_code[:2000],
        })

        if improved:
            logger.info("Round %d ACCEPTED (mini-eval %.1f%% > %.0f%%)",
                        rnd, win_rate * 100, MINI_IMPROVEMENT_THRESHOLD * 100)
            # Save intermediate accepted round as a permanent snapshot
            round_path = systems_dir / f"{tmp_version}.py"
            try:
                round_path.write_text(new_code)
                logger.info("Saved intermediate snapshot: %s", round_path.name)
            except Exception as _rpe:
                logger.warning("Could not save round snapshot %s: %s", round_path, _rpe)
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
    # Update class name — search the ACCUMULATED final_code (not champion_code),
    # since intermediate rounds may have renamed the class (e.g. S3_r2Generator)
    old_class = (
        re.search(r'class\s+(\w+Generator)\s*\(\s*IdeaGenerator', final_code)
        or re.search(r'class\s+(\w+Generator)\s*\(', final_code)
    )
    new_cls_name = f"{next_version}Generator"
    if old_class and old_class.group(1) != new_cls_name:
        old_cls_name = old_class.group(1)
        final_code = final_code.replace(f"class {old_cls_name}", f"class {new_cls_name}")
        # Also fix any direct references in GENERATOR singleton
        final_code = final_code.replace(f"GENERATOR = {old_cls_name}()", f"GENERATOR = {new_cls_name}()")
    # Ensure GENERATOR singleton is correct regardless
    final_code = re.sub(
        r'GENERATOR\s*=\s*\w+\(\)',
        f'GENERATOR = {new_cls_name}()',
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
