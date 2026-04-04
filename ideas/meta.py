"""Meta-generator: uses an LLM to propose the next idea-generator system.

The meta-LLM reads the full evolution history (all system code, win rates,
what succeeded / failed) and writes the next S{n}.py from scratch.
"""

import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import log as _log

logger = _log.setup("meta")

META_MODEL = "claude-sonnet-4-6"
META_TIMEOUT = 180  # seconds — writing a full system takes longer than a single idea

META_PROMPT = """\
You are designing the next version of an automated research idea generator \
that competes in a Gödel-style self-improvement loop.

## How the system works
Each version is a Python class that generates research ideas. \
Ideas are scored pairwise by an LLM judge on four criteria (0-10 each):
  - Novelty: genuinely new angle, not obvious incremental work
  - Scientific usefulness: addresses real open problems, actionable by researchers
  - Experimental clarity: clear methodology with testable predictions
  - Feasibility: achievable with current or near-future resources

A candidate beats the champion if it wins >60% of pairwise comparisons (75 pairs, 15 topics × 5 ideas).
The generator model is **gpt-4.1-mini** — capable, but benefits from structured prompting.

## Evolution history (accepted versions)
{history}

## All attempted versions and outcomes
{tried_approaches}

## Current champion: {champion_version}
```python
{champion_code}
```

## Your task: design {next_version}

Think carefully about what makes research ideas score high on ALL four criteria simultaneously. \
The judge rewards ideas that are novel but also clearly specified and feasible — \
vague "ambitious" ideas lose to concrete, actionable ones.

**Do NOT** just add more critique rounds or more reviewer personas — \
that's been tried (S4: two-round critique, S5: three-perspective critique). \
Consider fundamentally different strategies, for example:
  - Analogical transfer: find related solved problems and map solutions to the target
  - Constraint-first design: start from what's measurable/testable, build the idea backward
  - Literature gap mining: explicitly reason about what adjacent work has NOT addressed
  - Adversarial stress-testing: generate then actively try to falsify the idea before finalising
  - Structured creativity: use a fixed ideation framework (TRIZ, morphological analysis, etc.)
  - Ensemble voting: generate N independent ideas and pick/synthesise the best
  - Decompose the scoring rubric: explicitly optimise each dimension in a separate pass

Rules for the implementation:
1. Subclass IdeaGenerator, set VERSION = "{next_version}"
2. Override generate_idea() — multi-step is expected
3. Use call_llm() for all LLM calls (already imported from systems.base)
4. End every final prompt with IDEA_FORMAT (already imported) for consistent output
5. Define a module-level GENERATOR = {next_version}Generator() singleton
6. Write a clear docstring at the top of the file explaining the approach

Respond with ONLY the complete Python file — no markdown fences, no explanation outside the code.
"""


def _summarise_system(version: str, systems_dir: Path) -> str:
    """Extract first non-empty docstring line from a system file."""
    path = systems_dir / f"{version}.py"
    if not path.exists():
        return "(code not found)"
    try:
        text = path.read_text()
        for line in text.split("\n")[1:]:
            line = line.strip().strip('"""').strip("'''").strip()
            if line:
                return line
    except Exception:
        pass
    return ""


def generate_next_system(ideas_dir: Path, next_version: str) -> str:
    """Ask the meta-LLM to write the next system file. Returns Python source code."""
    sys.path.insert(0, str(ideas_dir / "systems"))
    from base import make_client, call_llm

    results_dir = ideas_dir / "results"
    systems_dir = ideas_dir / "systems"

    # Evolution log (accepted versions only)
    evo_log = results_dir / "evolution_log.jsonl"
    history_lines = []
    if evo_log.exists():
        with open(evo_log) as f:
            for line in f:
                if line.strip():
                    e = json.loads(line)
                    history_lines.append(
                        f"  {e['timestamp'][:10]}: {e['from_version']} → {e['to_version']}"
                        f"  (win rate: {(e['win_rate'] or 0)*100:.0f}%)"
                    )
    history_str = "\n".join(history_lines) or "  (no accepted versions yet)"

    # All comparison reports
    tried_lines = []
    for path in sorted(results_dir.glob("compare_*.json")):
        try:
            with open(path) as f:
                r = json.load(f)
            cand = r.get("candidate", "?")
            wr = r.get("win_rate_b", 0)
            status = "ACCEPTED" if wr > 0.60 else "REJECTED"
            desc = _summarise_system(cand, systems_dir)
            blind = r.get("blind") or {}
            blind_str = ""
            if blind.get("win_rate_b") is not None:
                blind_str = f", blind={blind['win_rate_b']*100:.0f}%"
            tried_lines.append(
                f"  {cand} [{status}] primary={wr*100:.0f}%{blind_str}: {desc}"
            )
        except Exception:
            pass
    tried_str = "\n".join(tried_lines) or "  (none yet)"

    # Champion code
    current_version = (ideas_dir / "CURRENT_VERSION").read_text().strip()
    champion_path = systems_dir / f"{current_version}.py"
    champion_code = champion_path.read_text() if champion_path.exists() else "(not found)"

    prompt = META_PROMPT.format(
        history=history_str,
        tried_approaches=tried_str,
        champion_version=current_version,
        champion_code=champion_code,
        next_version=next_version,
    )

    client = make_client(META_MODEL)
    logger.info("Meta-generating %s via %s...", next_version, META_MODEL)
    code = call_llm(prompt, META_MODEL, client, temperature=0.8,
                    max_tokens=4096, timeout=META_TIMEOUT)

    # Strip accidental markdown fences
    code = re.sub(r"^```python\s*\n", "", code.strip())
    code = re.sub(r"^```\s*\n", "", code)
    code = re.sub(r"\n```\s*$", "", code)

    return code.strip() + "\n"
