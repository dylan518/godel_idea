"""SWE Agent: multi-turn self-editing loop for the idea generator pipeline.

Instead of writing whole new S{n}.py files from scratch, the SWE agent
makes one targeted, surgical edit to the current champion's code, then tests it
against a fixed SWE holdout set.

This is the proper Gödel loop: the agent reads its own failure modes,
proposes specific code changes, tests them, and accumulates improvements.

Loop per iteration:
  1. Analyze failures: read comparison report, extract what ideas lost and why
  2. Propose edit: ask meta-LLM for ONE specific targeted code change
  3. Apply edit: meta-LLM writes complete new version of the file
  4. Holdout eval: run fixed dev topics against current champion
  5. Accept (write candidate) or reject (do not write final S{n}.py)
  6. Stop after the one edit attempt by default

After the loop: the accumulated edits form the new candidate S{n}.py,
which is then evaluated with a FULL 75-pair comparison against the champion.
"""

import ast
import json
import re
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).parent))
import log as _log
from canonical_skills import ideas_path_prefix, workspace_root

_log.load_dotenv()
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
SWE_CODE_MAX_TOKENS = 12000

# When True, use `claude --print` (Claude Code CLI) for the actual code editing
# instead of a raw API call. Claude Code reads files with its own tools and makes
# surgical edits, avoiding the "write full file from scratch" failure mode.
USE_CLAUDE_CODE = True

# ── Stopping criteria ─────────────────────────────────────────────────────────
DEFAULT_MAX_ROUNDS = 1       # one edit attempt; repeated rounds overfit the eval
DEFAULT_MAX_FAILURES = 1     # stop immediately after a failed holdout eval
MINI_IMPROVEMENT_THRESHOLD = 0.52  # holdout eval must show >52% to count as improvement
MINI_N_TOPICS = 10
MINI_N_IDEAS = 1
VALIDATION_N_TOPICS = 0
VALIDATION_N_IDEAS = 0
DEFAULT_SWE_WORKERS = 50
MAX_INVALID_IDEAS_FOR_ACCEPT = 0
SWE_HOLDOUT_TOPICS_FILE = "dev_topics.json"

# ── Editable file list (relative to ideas_dir) ────────────────────────────────
EDITABLE_FILES = [
    "idea_tournament/prompts.py",
    "idea_tournament/tree_search.py",
    "idea_tournament/tournament.py",
    "canonical_skills.py",
    # Repo-root skills (same files Claude Code edits for the live agent)
    "../skills/idea-tournament/references/tree-search-protocol.md",
    "../skills/idea-tournament/references/elo-ranking-guide.md",
    "../skills/idea-tournament/references/proposal-extension.md",
    "../skills/research-ideation/references/literature-tree.md",
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

    The champion source is the source of truth. Older runs stored stale summaries
    in swe_context.json, so this overview is rebuilt from systems/{version}.py
    whenever prompts are formatted.
    """
    lines = [f"Current champion: {champion_version}"]

    champion_path = ideas_dir / "systems" / f"{champion_version}.py"
    if not champion_path.exists():
        lines.append(f"Source file missing: systems/{champion_version}.py")
    else:
        code = champion_path.read_text(encoding="utf-8", errors="replace")
        lines.append(f"Source: systems/{champion_version}.py")

        try:
            module_doc = ast.get_docstring(ast.parse(code)) or ""
        except SyntaxError:
            module_doc = ""
        if module_doc:
            doc = module_doc.strip()
            if len(doc) > 1200:
                doc = doc[:1200].rstrip() + "..."
            lines.append("")
            lines.append("Pipeline from champion module docstring:")
            lines.extend(f"  {line}" for line in doc.splitlines())
        else:
            lines.append("")
            lines.append("Pipeline from champion module docstring: (none found; inspect source directly)")

        lines.append("")
        lines.append("Source-derived implementation notes:")
        if "EXPAND_WINNER_PROMPT_V2" in code or "EXPAND_WINNER_PROMPT" in code:
            # Find the custom prompt name
            m = re.search(r"(EXPAND_WINNER_PROMPT\w*)\s*=\s*\"\"\"", code)
            if m:
                lines.append(f"  • Expansion prompt: custom {m.group(1)} (overrides idea_tournament default)")
        for func in ["build_idea_tree", "run_tournament", "run_tournament_ranked"]:
            if f"def {func}" in code:
                lines.append(f"  • {func}: inlined custom version in champion file")
            elif func in code:
                lines.append(f"  • {func}: referenced/imported by champion")
        if "def generate_idea" in code:
            # Count approximate LLM calls (each call_llm = 1 call)
            n_calls = code.count("call_llm(")
            lines.append(f"  • generate_idea: ~{n_calls} direct call_llm() call sites")
        if "idea_tournament" not in code:
            lines.append("  • idea_tournament modules are not referenced by this champion")

    lines.append("")
    lines.append("Shared editable modules (only modify when the champion imports/uses them):")
    for rel in EDITABLE_FILES:
        p = (ideas_dir / rel).resolve()
        if p.exists():
            n_lines = len(p.read_text(encoding="utf-8", errors="replace").splitlines())
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
    # Always rebuild from the live champion source. Stored pipeline_overview values
    # have gone stale across promotions and can mislead the SWE agent.
    overview = build_pipeline_overview(ideas_dir, champion_version)
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
        p = (ideas_dir / rel_path).resolve()
        if not p.exists():
            continue
        text = p.read_text(encoding="utf-8", errors="replace")
        fence = "markdown" if p.suffix.lower() == ".md" else "python"
        parts.append(f"### FILE: {rel_path}\n```{fence}\n{text}\n```")

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
Study the losing ideas above carefully. Diagnose WHY they lose, starting from the
actual champion code above as the source of truth. Preserve the current working
scaffold unless the failures specifically implicate that scaffold.

Prefer the smallest structural change that should change outcomes: a new gate,
selection criterion, construction substep, or information flow inside the existing
pipeline. Do NOT replace the whole algorithm unless the evidence says the current
algorithmic structure is the root cause.

Examples of the level of change we want:
- If hypothesis generation is too generic, add contradiction-mining across fetched papers
- If selection over-rewards novelty, add a feasibility or clarity gate before selection
- If construction drifts from the selected hypothesis, add a mechanism-locking substep
- If hypotheses are too one-sided, keep two competing hypotheses through construction
- If the pipeline is truly wrong, replace the implicated stage, not unrelated stages

Output EXACTLY in this format (no other text):
DIAGNOSIS: <what structural property of the losing ideas caused them to lose>
FIX: <the smallest structural algorithm/prompt-flow change that addresses it — describe it concretely enough to implement>
EXPECTED_IMPACT: <how this changes the KIND of ideas that come out, not just their polish>
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

The current champion may or may not import from idea_tournament/ at runtime.
Your candidate will be evaluated HEAD-TO-HEAD against the current champion.
For a fair comparison, you MUST inline any code you are modifying:

  - Changing a prompt?  → Define the new prompt string as a local variable in
    generate_idea(), do NOT use the idea_tournament.prompts version.
  - Changing tree_search or tournament logic?  → Copy + modify the relevant
    functions inline inside generate_idea() or as module-level helpers.
  - NOT changing something?  → You may still import it from idea_tournament/ if
    the champion already does so.

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
- Avoid brittle giant mandatory output templates. Prefer targeted prompt changes over
  escalating into stricter formatting machinery unless malformed output is the diagnosed failure.
- Do NOT call .format() on large prompt strings that may contain literal braces from
  IDEA_FORMAT, math notation, JSON, or examples. Use f-strings with escaped braces or
  simple .replace() on explicit placeholders instead.
- Budget ~10-15 LLM calls per idea

Then write the output file following the format below.

""" + _OUTPUT_FORMAT

REFLECT_PROMPT = """\
The last edit did NOT improve performance (holdout eval: {win_rate:.1%}, needed >{threshold:.1%}).

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
Avoid adding giant mandatory output templates or using .format() on prompt strings that may
contain literal braces; those changes are brittle and have previously caused invalid ideas.

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


def _load_swe_holdout_topics(
    ideas_dir: Path,
    n: int = MINI_N_TOPICS,
    offset: int = 0,
) -> list[dict]:
    """Load a fixed SWE holdout slice.

    This intentionally does not generate topics with an LLM. The eval questions
    should be stable and outside the agent's control; repeated generated
    mini-evals were too easy to overfit.
    """
    import json as _json

    holdout_path = ideas_dir / SWE_HOLDOUT_TOPICS_FILE
    if holdout_path.exists():
        with open(holdout_path) as f:
            holdout_topics = _json.load(f).get("topics", [])
        if holdout_topics:
            start = offset % len(holdout_topics)
            return [holdout_topics[(start + i) % len(holdout_topics)] for i in range(n)]

    logger.warning("SWE holdout file missing (%s); falling back to benchmark tail", holdout_path)
    return [{"id": f"E{i}", "topic": t, "domain": "ML"}
            for i, t in enumerate(_BENCHMARK_TOPICS[-n:], 1)]

def _extract_grounded_failures(
    report_path: Path,
    ideas_dir: Path,
    target_version: str | None = None,
    n: int = 4,
) -> str:
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
        if target_version:
            losses = [
                v for v in verdicts
                if (
                    v.get("system_a") == target_version and v.get("winner") == "B"
                ) or (
                    v.get("system_b") == target_version and v.get("winner") == "A"
                )
            ]
        else:
            # Backwards-compatible fallback for old compare_current_vs_candidate reports.
            losses = [v for v in verdicts if v.get("winner") == "A"]

        def _target_scores(v: dict) -> dict:
            if target_version and v.get("system_a") == target_version:
                return v.get("scores_a", {})
            return v.get("scores_b", {})

        def _winner_scores(v: dict) -> dict:
            if v.get("winner") == "A":
                return v.get("scores_a", {})
            if v.get("winner") == "B":
                return v.get("scores_b", {})
            return {}

        # Detect infrastructure failures: target score=0 on most losses → quota error
        zero_score_losses = [v for v in losses if sum(_target_scores(v).values()) == 0]
        if len(losses) > 0 and len(zero_score_losses) / len(losses) > 0.5:
            return (
                "⚠️  INFRASTRUCTURE FAILURE: target system scored 0/40 on "
                f"{len(zero_score_losses)}/{len(losses)} comparisons — API quota error, "
                "not a quality failure. Focus on improving idea quality, not error handling."
            )

        losses.sort(
            key=lambda v: sum(_winner_scores(v).values()) - sum(_target_scores(v).values()),
            reverse=True,
        )

        # Load actual idea texts from results JSONs
        systems = sorted({
            s for v in losses for s in (v.get("system_a"), v.get("system_b")) if s
        })
        ideas_a: dict = {}
        ideas_b: dict = {}
        ideas_by_system: dict[str, dict] = {}
        for sys_name in systems:
            if not sys_name:
                continue
            p = ideas_dir / "results" / sys_name / "ideas.json"
            store: dict = {}
            if p.exists():
                try:
                    for item in json.load(open(p)):
                        store[(item["topic_id"], item.get("idea_index", 0))] = item.get("text", "")
                except Exception:
                    pass
            ideas_by_system[sys_name] = store

        lines = []
        for v in losses[:n]:
            topic = v.get("topic", "")
            tid = v.get("topic_id", "")
            idx = v.get("idea_index", 0)
            reasoning = v.get("reasoning", "")

            target_side = "A" if v.get("system_a") == target_version else "B"
            winner_side = "B" if target_side == "A" else "A"
            target_system = v.get(f"system_{target_side.lower()}")
            winner_system = v.get(f"system_{winner_side.lower()}")
            target_scores = v.get(f"scores_{target_side.lower()}", {})
            winner_scores = v.get(f"scores_{winner_side.lower()}", {})
            target_total = sum(target_scores.values())
            winner_total = sum(winner_scores.values())

            target_text = ideas_by_system.get(target_system, {}).get((tid, idx), "(text unavailable)")
            winner_text = ideas_by_system.get(winner_system, {}).get((tid, idx), "(text unavailable)")

            lines.append(f"### Topic: {topic}")
            lines.append(f"**Comparison winner ({winner_system}) — {winner_total}/40 — WON:**")
            lines.append(winner_text[:600])
            lines.append(f"\n**Current target ({target_system}) — {target_total}/40 — LOST:**")
            lines.append(target_text[:600])
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
                   max_tokens=SWE_CODE_MAX_TOKENS, timeout=SWE_TIMEOUT)
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


def _build_self_improvement_topic(
    champion_version: str,
    champion_path: Path,
    grounded_failures_str: str,
    failed_attempts_str: str,
) -> str:
    """Build a topic string for generate_idea() that frames self-improvement as research.

    Uses the champion's module docstring (max 400 chars) instead of full source code
    to keep the topic within API limits while still conveying the pipeline structure.
    """
    # Extract docstring from champion file (first triple-quoted string)
    champion_text = champion_path.read_text()
    doc_match = re.search(r'"""(.*?)"""', champion_text, re.DOTALL)
    doc = doc_match.group(1).strip()[:400] if doc_match else f"{champion_version} idea generator pipeline"

    topic = (
        f"Self-improvement of a research idea generation pipeline ({champion_version}).\n\n"
        f"CURRENT PIPELINE OVERVIEW:\n{doc}\n\n"
        f"RECENT FAILURES (ideas where the current system lost to a weaker baseline):\n"
        f"{grounded_failures_str[:1500]}\n\n"
        f"PREVIOUSLY ATTEMPTED FIXES THAT DID NOT WORK:\n"
        f"{failed_attempts_str[:600]}\n\n"
        f"Your task: propose a targeted ideation-strategy change for the APPROACH section.\n"
        f"Use the current pipeline overview above as the source of truth. Preserve working\n"
        f"stages unless the failures specifically implicate them.\n\n"
        f"Examples of the level of change we want:\n"
        f"- If hypothesis generation is too generic, add contradiction-mining from recent papers\n"
        f"- If selection over-rewards novelty, add a feasibility or clarity gate before selection\n"
        f"- If construction drifts from the selected hypothesis, add a mechanism-locking substep\n"
        f"- If hypotheses are too one-sided, keep two competing hypotheses through construction\n\n"
        f"Describe the smallest concrete algorithm or information-flow change likely to help."
    )
    return topic


def _parse_idea_output(raw: str) -> dict:
    """Parse IDEA_FORMAT output into field dict.

    Returns keys: idea, background, approach, experiment, novelty.
    Falls back to DOTALL multiline search per field.
    """
    fields = ["IDEA", "BACKGROUND", "APPROACH", "EXPERIMENT", "NOVELTY"]
    result: dict = {}

    # Try splitting on field headers
    pattern = r"(?:^|\n)(IDEA|BACKGROUND|APPROACH|EXPERIMENT|NOVELTY)\s*[:\-]\s*"
    parts = re.split(pattern, raw, flags=re.IGNORECASE)
    if len(parts) > 1:
        # parts = [pre, field1, content1, field2, content2, ...]
        for i in range(1, len(parts) - 1, 2):
            key = parts[i].upper()
            val = parts[i + 1].strip() if i + 1 < len(parts) else ""
            result[key.lower()] = val

    # Fill any missing fields via DOTALL search
    for field in fields:
        key = field.lower()
        if key not in result or not result[key]:
            m = re.search(
                rf"{field}\s*[:\-]\s*(.+?)(?=(?:IDEA|BACKGROUND|APPROACH|EXPERIMENT|NOVELTY)\s*[:\-]|$)",
                raw,
                re.DOTALL | re.IGNORECASE,
            )
            if m:
                result[key] = m.group(1).strip()

    # Final fallback: store full raw under "idea" if nothing parsed
    if not result:
        result["idea"] = raw.strip()

    return result


def _call_generate_idea_proposal(
    champion_version: str,
    ideas_dir: Path,
    topic: str,
) -> tuple:
    """Call champion's generate_idea() to produce a self-improvement proposal.

    Loads the champion module dynamically (same pattern as _run_mini_eval),
    then calls generate_idea(topic, ...) with temperature=0.7 for precision.

    Returns (raw_idea_str, parsed_fields_dict).
    Raises RuntimeError on any failure — caller should catch and fall back.
    """
    import importlib.util

    champion_path = ideas_dir / "systems" / f"{champion_version}.py"
    if not champion_path.exists():
        raise RuntimeError(f"Champion file not found: {champion_path}")

    # Insert paths so the champion module can import its dependencies
    for p in [str(ideas_dir), str(ideas_dir / "systems")]:
        if p not in sys.path:
            sys.path.insert(0, p)

    spec = importlib.util.spec_from_file_location("_champ_selfimprove", champion_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    generator = mod.GENERATOR
    sys.path.insert(0, str(ideas_dir / "systems"))
    from base import make_client, DEFAULT_MODEL  # noqa: F401
    client = make_client(DEFAULT_MODEL)

    raw = generator.generate_idea(topic, client, model=DEFAULT_MODEL, temperature=0.7)
    if not isinstance(raw, str) or len(raw) < 50:
        raise RuntimeError(f"generate_idea() returned unexpectedly short output: {raw!r}")

    parsed = _parse_idea_output(raw)
    if not parsed.get("background") and not parsed.get("approach"):
        raise RuntimeError(
            f"generate_idea() output missing BACKGROUND and APPROACH fields. Raw: {raw[:200]}"
        )

    return raw, parsed


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

    wr = workspace_root(ideas_dir)
    ip = ideas_path_prefix(ideas_dir, wr)
    try:
        rel_output = output_path.relative_to(wr)
        rel_prompts = Path(ip) / "idea_tournament/prompts.py"
        rel_tree = Path(ip) / "idea_tournament/tree_search.py"
        rel_tourn = Path(ip) / "idea_tournament/tournament.py"
    except ValueError:
        rel_output = output_path
        rel_prompts = ideas_dir / "idea_tournament/prompts.py"
        rel_tree = ideas_dir / "idea_tournament/tree_search.py"
        rel_tourn = ideas_dir / "idea_tournament/tournament.py"

    task_prompt = f"""You are redesigning the ideation algorithm inside a Python research idea generator.

## What needs to change and why
{task_description}

## Where the current generator is losing
{failure_analysis}

## What to do
1. Read `{rel_output}` — this is already a copy of the champion, your starting point
2. Treat `{rel_output}` as the source of truth for the current pipeline. Only read
   `{ip}idea_tournament/` or repo-root `skills/idea-tournament/` if `{rel_output}`
   imports or references them.
3. Implement the new ideation strategy above — rewrite or add to `generate_idea()` in `{rel_output}`
4. After editing, run cheap local checks yourself, at minimum:
   `python3 -m py_compile {rel_output}`.
   You may run small import/smoke checks, but do not run expensive idea generation
   or judge evaluations from inside this edit step.

## What we want
Preserve the champion's working scaffold unless the diagnosis identifies it as the
root cause. Prefer targeted changes to the specific failing stage: generation,
attack/revision, selection, construction, critique, or final revision. The change
should affect what KIND of ideas come out, not just how polished they are.

## Hard constraints (failure to meet these causes the system to crash or be disqualified)
- Class name MUST be `{next_version}Generator(IdeaGenerator)`
- `VERSION = "{next_version}"` (update from whatever the champion has)
- File MUST end with `GENERATOR = {next_version}Generator()`
- Any code you MODIFY from `idea_tournament/` must be inlined in the champion file (not imported). You MAY edit `skills/**/*.md` in place when changing rubrics those prompts load.
- Every `call_llm()` call must be wrapped in try/except with a sensible fallback
- `generate_idea(self, topic, client, model, temperature)` must always return a non-empty string
- Avoid brittle giant mandatory output templates. Prefer one targeted mechanism change over
  accumulating stricter formatting rules unless the diagnosed failure is malformed output.
- Do NOT call `.format()` on large prompt strings that may contain literal braces from
  `IDEA_FORMAT`, math notation, JSON examples, or generated text. Use f-strings with escaped
  braces or explicit `.replace()` placeholders instead.
- Keep total LLM calls per idea: ~10-20 (can use more than champion if the strategy warrants it)

## What the judge rewards (optimise for this)
- Concrete, specific, well-grounded ideas
- Named datasets, baselines, and quantitative metrics
- Clear problem statements with named failure modes
- Genuine novelty — ideas the judge hasn't seen before"""

    # Strip CLAUDECODE from the subprocess env so nested Claude Code sessions work.
    # Claude Code blocks nested launches unless this var is absent.
    import os as _os
    clean_env = {k: v for k, v in _os.environ.items() if k != "CLAUDECODE"}

    try:
        result = subprocess.run(
            ["claude", "--print", "--allowedTools", "Read,Edit,Write,Bash"],
            input=task_prompt,
            cwd=str(wr),
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


def _idea_output_health(results: list[dict]) -> dict:
    """Summarize invalid generated ideas that would poison evaluation results."""
    invalid = []
    for rec in results or []:
        text = str(rec.get("text", "") if isinstance(rec, dict) else rec).strip()
        if (
            not text
            or len(text) < 200
            or text.startswith("ERROR:")
            or text.startswith("TIMEOUT:")
            or text in {"-", "—"}
        ):
            invalid.append({
                "topic_id": rec.get("topic_id") if isinstance(rec, dict) else None,
                "idea_index": rec.get("idea_index") if isinstance(rec, dict) else None,
                "text": text[:120],
            })
    return {
        "total": len(results or []),
        "invalid_count": len(invalid),
        "invalid_examples": invalid[:5],
    }


def _rename_candidate_code(code: str, next_version: str) -> str:
    """Rename a generated snapshot's class/version/singleton to the final system name."""
    final_code = re.sub(
        r'VERSION\s*=\s*["\'].*?["\']',
        f'VERSION = "{next_version}"',
        code,
    )
    old_class = (
        re.search(r'class\s+(\w+Generator)\s*\(\s*IdeaGenerator', final_code)
        or re.search(r'class\s+(\w+Generator)\s*\(', final_code)
    )
    new_cls_name = f"{next_version}Generator"
    if old_class and old_class.group(1) != new_cls_name:
        old_cls_name = old_class.group(1)
        final_code = final_code.replace(f"class {old_cls_name}", f"class {new_cls_name}")
        final_code = final_code.replace(f"GENERATOR = {old_cls_name}()", f"GENERATOR = {new_cls_name}()")
    final_code = re.sub(
        r'GENERATOR\s*=\s*\w+\(\)',
        f'GENERATOR = {new_cls_name}()',
        final_code,
    )
    return final_code


def _run_mini_eval(
    ideas_dir: Path,
    candidate_path: Path,
    champion_version: str,
    n_topics: int = MINI_N_TOPICS,
    n_ideas: int = MINI_N_IDEAS,
    model: str = "gpt-4.1-mini",
    workers: int = DEFAULT_SWE_WORKERS,
    topic_offset: int = 0,
) -> dict:
    """Run a fast comparison of candidate vs champion on a topic subset.

    Returns a dict with win_rate_b and candidate output-health metadata.
    Returns win_rate=0.0 on failure to avoid false positives.
    """
    import concurrent.futures
    import importlib.util
    import random

    sys.path.insert(0, str(ideas_dir / "systems"))
    from base import make_client
    from runner import run_system
    from judge import compare_systems, JUDGE_MODEL

    # Fixed holdout questions; the SWE agent must not generate its own eval set.
    sampled = _load_swe_holdout_topics(ideas_dir, n=n_topics, offset=topic_offset)

    # Load candidate module dynamically
    try:
        spec = importlib.util.spec_from_file_location("_candidate_tmp", candidate_path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        candidate_version = mod.GENERATOR.VERSION
    except Exception as e:
        logger.error("Failed to load candidate %s: %s", candidate_path, e)
        return {"win_rate": 0.0, "health": {"total": 0, "invalid_count": 1, "invalid_examples": [{"text": str(e)[:120]}]}}

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
            logger.error("Holdout eval champion run failed: %s", err_champion)
            return {"win_rate": 0.0, "health": {"total": 0, "invalid_count": 1, "invalid_examples": [{"text": str(err_champion)[:120]}]}}
        if err_candidate:
            logger.error("Holdout eval candidate run failed: %s", err_candidate)
            return {"win_rate": 0.0, "health": {"total": 0, "invalid_count": 1, "invalid_examples": [{"text": str(err_candidate)[:120]}]}}

        health = _idea_output_health(results_candidate or [])
        if health["invalid_count"]:
            logger.warning(
                "Holdout eval output health: %s produced %d/%d invalid ideas; examples=%s",
                candidate_version,
                health["invalid_count"],
                health["total"],
                health["invalid_examples"],
            )

        # Judge (parallel across topics)
        try:
            judge_client = make_client(JUDGE_MODEL)
            result = compare_systems(results_champion, results_candidate, judge_client,
                                     workers=workers,
                                     early_stop_threshold=MINI_IMPROVEMENT_THRESHOLD)
            win_rate = result["win_rate_b"]
            logger.info("Holdout eval: %s vs %s → win_rate=%.1f%% (%d pairs)",
                        champion_version, candidate_version, win_rate * 100,
                        result["total_judged"])
            return {"win_rate": win_rate, "health": health, "comparison": result}
        except Exception as e:
            logger.error("Holdout eval judging failed: %s", e)
            return {"win_rate": 0.0, "health": health}


# ── Main SWE agent loop ───────────────────────────────────────────────────────

def run_swe_loop(
    ideas_dir: Path,
    next_version: str,
    champion_version: str,
    compare_report_path: Optional[Path] = None,
    max_rounds: int = DEFAULT_MAX_ROUNDS,
    max_failures: int = DEFAULT_MAX_FAILURES,
    generator_model: str = "gpt-4.1-mini",
    workers: int = DEFAULT_SWE_WORKERS,
) -> Path:
    """Multi-turn SWE agent loop.

    Starts from the champion's code and makes targeted edits, testing each with
    the fixed SWE holdout. Returns path to the final candidate file.

    Arguments:
        ideas_dir: root of the ideas/ directory
        next_version: name for the output system (e.g. "S21")
        champion_version: current champion (e.g. "S5")
        compare_report_path: path to comparison report for failure analysis
        max_rounds: max accepted edits
        max_failures: max consecutive failed holdout evals before stopping
        generator_model: model used for idea generation
        workers: parallel topic/judge workers for holdout eval
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

    grounded_failures_str = _extract_grounded_failures(
        compare_report_path,
        ideas_dir,
        target_version=champion_version,
    )

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
            f"Round {i+1}: {e['description'][:100]} → holdout {e['win_rate']:.1%}"
            for i, e in enumerate(edit_history)
        ) or "(none yet)"

        # ── Step 1: diagnose failures and propose new ideation strategy ─────────
        champion_code_text = champion_path.read_text()
        failed_attempts_str = "\n".join(
            f"Round {e['round']}: {e['description'][:120]} → {e['win_rate']:.1%}"
            for e in edit_history if not e.get("accepted")
        ) or "(none yet)"

        tmp_version = f"{next_version}_r{rnd}"
        tmp_path = systems_dir / f"{tmp_version}.py"

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
            expected_impact = ""

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
            refined_fix = proposed_fix

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
                    failure_analysis=grounded_failures_str,
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

        # ── Smoke test: validate the file imports cleanly before spending holdout eval ──
        smoke_err = _smoke_test(tmp_path, ideas_dir)
        if smoke_err:
            logger.error("Smoke test FAILED for %s — skipping holdout eval:\n%s", tmp_version, smoke_err[-600:])
            tmp_path.unlink(missing_ok=True)
            failures += 1
            continue
        logger.info("Smoke test passed for %s", tmp_version)

        # ── Step 3: fixed holdout eval ───────────────────────────────────────
        try:
            primary_eval = _run_mini_eval(
                ideas_dir=ideas_dir,
                candidate_path=tmp_path,
                champion_version=champion_version,  # always compare against original champion
                n_topics=MINI_N_TOPICS,
                n_ideas=MINI_N_IDEAS,
                model=generator_model,
                workers=workers,
                topic_offset=(rnd - 1) * MINI_N_TOPICS,
            )
        except Exception as e:
            logger.error("Holdout eval crashed: %s", e)
            primary_eval = {"win_rate": 0.0, "health": {"total": 0, "invalid_count": 1, "invalid_examples": [{"text": str(e)[:120]}]}}
        finally:
            pass  # keep temp file — rejected rounds are preserved for inspection

        win_rate = float(primary_eval.get("win_rate", 0.0))
        health = primary_eval.get("health", {})
        health_ok = health.get("invalid_count", 0) <= MAX_INVALID_IDEAS_FOR_ACCEPT

        if not health_ok:
            logger.info(
                "%s failed output-health gate (%d/%d invalid ideas)",
                tmp_version,
                health.get("invalid_count", 0),
                health.get("total", 0),
            )

        validation_win_rate = win_rate
        validation_health = health
        validation_health_ok = health_ok
        validation_score = win_rate
        improved = (
            win_rate > MINI_IMPROVEMENT_THRESHOLD
            and health_ok
        )

        edit_history.append({
            "round": rnd,
            "description": task_description[:1200],
            "failure_analysis": grounded_failures_str[:2000],
            "win_rate": win_rate,
            "validation_win_rate": validation_win_rate,
            "validation_score": validation_score,
            "health": health,
            "validation_health": validation_health,
            "accepted": improved,
            "code_snippet": new_code[:2000],
        })

        if improved:
            logger.info(
                "Round %d ACCEPTED (holdout %.1f%%, invalid ideas %d/%d)",
                rnd,
                win_rate * 100,
                health.get("invalid_count", 0),
                health.get("total", 0),
            )
            # Save intermediate accepted round as a permanent snapshot
            round_path = systems_dir / f"{tmp_version}.py"
            try:
                round_path.write_text(new_code)
                logger.info("Saved intermediate snapshot: %s", round_path.name)
                edit_history[-1]["snapshot_path"] = str(round_path)
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
            if last_win_rate is not None and abs(validation_score - last_win_rate) < 0.02:
                stall_count += 1
                if stall_count >= 2:
                    logger.info("SWE loop stopping: validation score stalled at %.1f%%", validation_score * 100)
                    break
            else:
                stall_count = 0
            last_win_rate = validation_score
        else:
            logger.info(
                "Round %d REJECTED (holdout %.1f%%, invalid ideas %d/%d)",
                rnd,
                win_rate * 100,
                health.get("invalid_count", 0),
                health.get("total", 0),
            )
            failures += 1

    # Write final output only if at least one edit survived holdout eval. Writing a
    # renamed copy of the champion creates misleading no-op candidates.
    if not any(e.get("accepted") for e in edit_history):
        log_path = ideas_dir / "results" / f"swe_log_{next_version}.json"
        with open(log_path, "w") as f:
            json.dump({
                "champion": champion_version,
                "output": next_version,
                "rounds": rnd,
                "edits": edit_history,
            }, f, indent=2)
        if edit_history:
            best_mini = max((e["win_rate"] for e in edit_history), default=0.0)
            update_swe_memory(ideas_dir, {
                "version": next_version,
                "champion": champion_version,
                "mini_eval_best": best_mini,
                "accepted_edits": [],
                "failed_edits": [
                    {"description": e["description"], "win_rate": e["win_rate"]}
                    for e in edit_history if not e.get("accepted")
                ],
            })
        raise RuntimeError(f"No accepted SWE edits for {next_version}; not writing no-op candidate")

    # Write the accepted snapshot. With the default one-shot loop there is only
    # one; the ranking logic remains for manual multi-round runs.
    accepted_entries = [e for e in edit_history if e.get("accepted")]
    best_entry = max(
        accepted_entries,
        key=lambda e: (e.get("validation_score", 0.0), e.get("validation_win_rate", 0.0), e.get("win_rate", 0.0)),
    )
    best_snapshot = Path(best_entry.get("snapshot_path", ""))
    if best_snapshot.is_file():
        selected_code = best_snapshot.read_text()
    else:
        selected_code = current_code
        logger.warning("Best snapshot path missing (%s); falling back to latest accepted code", best_snapshot)
    logger.info(
        "Selected best accepted snapshot for final %s: round %s (holdout %.1f%%)",
        next_version,
        best_entry.get("round"),
        best_entry.get("win_rate", 0.0) * 100,
    )

    final_code = _rename_candidate_code(selected_code, next_version)

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
        {
            "description": e["description"],
            "win_rate": e["win_rate"],
            "validation_win_rate": e.get("validation_win_rate"),
            "validation_score": e.get("validation_score"),
            "health": e.get("health"),
            "validation_health": e.get("validation_health"),
        }
        for e in edit_history if e["accepted"]
    ]
    failed_edits = [
        {
            "description": e["description"],
            "win_rate": e["win_rate"],
            "validation_win_rate": e.get("validation_win_rate"),
            "health": e.get("health"),
            "validation_health": e.get("validation_health"),
        }
        for e in edit_history if not e["accepted"]
    ]
    best_mini = max((e["win_rate"] for e in edit_history), default=0.0)
    best_validation = max((e.get("validation_win_rate", 0.0) for e in edit_history), default=0.0)
    update_swe_memory(ideas_dir, {
        "version": next_version,
        "champion": champion_version,
        "mini_eval_best": best_mini,
        "validation_eval_best": best_validation,
        "selected_round": best_entry.get("round"),
        "accepted_edits": accepted_edits,
        "failed_edits": failed_edits,
        # full_eval_win_rate and accepted filled in later by cmd_swe_evolve
    })
    logger.info("SWE memory updated for %s", next_version)

    return output_path
