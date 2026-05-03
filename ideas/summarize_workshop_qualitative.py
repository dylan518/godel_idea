#!/usr/bin/env python3
"""LLM-written qualitative summaries for workshop text.

For a pairwise benchmark (primary judge JSON from ``compare_*.json``):

1. **Generator delta** — contrasts the two ``systems/S*.py`` implementations
   (truncated for context limits), unless ``--skip-code``.
2. **Judge reasoning themes** — stratified sample of per-pair ``reasoning``
   strings where the winner was current (A), candidate (B), or tie.

Output is Markdown under ``results/workshop_qualitative/`` for pasting into
discussion / qualitative paragraphs. **Costs one LLM call per invocation**
(plus one per row if you use ``--batch``).

Examples::

    cd ideas
    python3 summarize_workshop_qualitative.py --current S15 --candidate S16
    python3 summarize_workshop_qualitative.py --batch
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys
from pathlib import Path

IDEAS_DIR = Path(__file__).resolve().parent
RESULTS_DIR = IDEAS_DIR / "results"
SYSTEMS_DIR = IDEAS_DIR / "systems"

if str(IDEAS_DIR) not in sys.path:
    sys.path.insert(0, str(IDEAS_DIR))
if str(SYSTEMS_DIR) not in sys.path:
    sys.path.insert(0, str(SYSTEMS_DIR))

import log as _log

_log.load_dotenv()

from base import call_llm, make_client  # noqa: E402

DEFAULT_SUMMARY_MODEL = os.environ.get("IDEAS_WORKSHOP_SUMMARY_MODEL", "claude-sonnet-4-6")

_BATCH_COMPARE_FILES: list[tuple[str, str, str]] = [
    ("S12", "S15", "compare_S12_vs_S15.json"),
    ("S15", "S16", "compare_S15_vs_S16.json"),
    ("S15", "S17", "compare_S15_vs_S17.json"),
    ("S15", "S18", "compare_S15_vs_S18.json"),
    ("S15", "S19", "compare_S15_vs_S19.json"),
]


def _truncate(text: str, max_chars: int) -> str:
    text = text.strip()
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 20] + "\n\n[…truncated…]\n"


def _system_path(version: str) -> Path:
    safe = re.sub(r"[^0-9A-Za-z_]", "", version)
    return SYSTEMS_DIR / f"{safe}.py"


def _load_compare(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(path)
    return json.loads(path.read_text())


def _sample_reasonings(
    verdicts: list[dict],
    *,
    seed: int,
    per_bucket: int,
    max_chars_each: int,
) -> tuple[list[str], list[str], list[str]]:
    rng = random.Random(seed)
    a_wins: list[str] = []
    b_wins: list[str] = []
    ties: list[str] = []
    for v in verdicts:
        r = (v.get("reasoning") or "").strip()
        if not r:
            continue
        r = r[:max_chars_each]
        w = v.get("winner")
        if w == "A":
            a_wins.append(r)
        elif w == "B":
            b_wins.append(r)
        elif w == "tie":
            ties.append(r)

    def pick(xs: list[str]) -> list[str]:
        if len(xs) <= per_bucket:
            return list(xs)
        return rng.sample(xs, per_bucket)

    return pick(a_wins), pick(b_wins), pick(ties)


def _build_prompt(
    current: str,
    candidate: str,
    compare_path: Path,
    data: dict,
    *,
    include_code: bool,
    code_max_chars: int,
    per_bucket: int,
    seed: int,
) -> str:
    verdicts = data.get("verdicts") or []
    wa, wb, wt = _sample_reasonings(verdicts, seed=seed, per_bucket=per_bucket, max_chars_each=500)
    meta = (
        f"Compare file: {compare_path.name}\n"
        f"Primary judge (from rows): {verdicts[0].get('judge_model', '?') if verdicts else '?'}\n"
        f"N verdicts: {len(verdicts)}  |  wins A (current): {data.get('wins_a')}  "
        f"wins B (candidate): {data.get('wins_b')}  ties: {data.get('ties')}\n"
        f"In each verdict, **A** is system **{current}**, **B** is **{candidate}**.\n"
    )
    cur_py = _system_path(current)
    cand_py = _system_path(candidate)
    have_code = include_code and cur_py.is_file() and cand_py.is_file()
    if have_code:
        code_section = (
            "## Generator implementations (Python; may be truncated)\n\n"
            f"### `{cur_py.name}` (current champion)\n```python\n"
            f"{_truncate(cur_py.read_text(), code_max_chars)}\n```\n\n"
            f"### `{cand_py.name}` (candidate)\n```python\n"
            f"{_truncate(cand_py.read_text(), code_max_chars)}\n```\n"
        )
        section1 = (
            "## 1. Generator / strategy change (≈150–250 words)\n"
            "What did the meta-edit likely change in *behavior* (prompting structure, critique rounds, "
            "constraints, rubric emphasis), based only on the two Python files? "
            "Avoid quoting long code; focus on differences a reader would care about.\n"
        )
    elif include_code:
        code_section = (
            "## Generator implementations\n\n"
            f"_(Paths `{cur_py.name}` / `{cand_py.name}` not both readable — "
            "section 1 cannot use code.)_\n"
        )
        section1 = (
            "## 1. Generator / strategy change (≈60 words)\n"
            "State that generator sources were unavailable for this run; defer to repository history.\n"
        )
    else:
        code_section = (
            "## Generator implementations\n\n"
            "_``systems/*.py`` omitted from this prompt to save tokens._\n"
        )
        section1 = (
            "## 1. Generator / strategy change (≤80 words)\n"
            "Note code was omitted; suggest readers inspect ``systems/{0}.py`` vs ``systems/{1}.py`` "
            "for a manual qualitative diff.\n".format(current, candidate)
        )

    def bullets(lines: list[str]) -> str:
        if not lines:
            return "_No samples in this bucket._\n"
        return "\n".join(f"- {s}" for s in lines)

    return f"""You are helping write an ICML workshop paper on LLM-as-judge reliability \
in a self-improving idea generator.

## Benchmark metadata
{meta}

## Stratified samples of the primary judge's short reasoning (2–3 sentences each)

### When the judge chose **A** ({current}, current champion) — {len(wa)} samples
{bullets(wa)}

### When the judge chose **B** ({candidate}, challenger) — {len(wb)} samples
{bullets(wb)}

### When the judge declared a **tie** — {len(wt)} samples
{bullets(wt)}

{code_section}

---

Write **Markdown** with exactly these sections:

{section1}

## 2. Why the judge favored the **current** ({current}) when it won (≈120–200 words)
Synthesize recurring themes from the A-win reasonings only. \
Mention rubric dimensions (novelty, usefulness, clarity, feasibility) when relevant.

## 3. Why the judge favored the **candidate** ({candidate}) when it won (≈120–200 words)
Same, from B-win reasonings only.

## 4. Caveats (≈60–100 words)
Note that reasonings are post-hoc rationales (not independent), may be inconsistent with \
scores in edge cases, and that this is **not** a substitute for human ground truth.

Use neutral academic tone. Do not invent statistics; only qualitative synthesis.
"""


def run_one(
    current: str,
    candidate: str,
    compare_json: Path,
    *,
    model: str,
    temperature: float,
    include_code: bool,
    code_max_chars: int,
    per_bucket: int,
    seed: int,
    out_path: Path,
) -> None:
    data = _load_compare(compare_json)
    prompt = _build_prompt(
        current,
        candidate,
        compare_json,
        data,
        include_code=include_code,
        code_max_chars=code_max_chars,
        per_bucket=per_bucket,
        seed=seed,
    )
    client = make_client(model)
    body = call_llm(prompt, model, client, temperature, max_tokens=4096)
    header = (
        f"<!-- auto: summarize_workshop_qualitative.py | "
        f"{current} vs {candidate} | model={model} | compare={compare_json.name} -->\n\n"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(header + body.strip() + "\n")
    print(f"Wrote {out_path}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--current", help="Champion system id, e.g. S15")
    ap.add_argument("--candidate", help="Challenger system id, e.g. S16")
    ap.add_argument(
        "--compare-json",
        type=Path,
        help="Path to compare_*.json (default: results/compare_{current}_vs_{candidate}.json)",
    )
    ap.add_argument(
        "--out",
        type=Path,
        help="Output .md path (default: results/workshop_qualitative/{current}_vs_{candidate}.md)",
    )
    ap.add_argument("--model", default=DEFAULT_SUMMARY_MODEL, help="LLM for synthesis")
    ap.add_argument("--temperature", type=float, default=0.3)
    ap.add_argument("--code-max-chars", type=int, default=14000, help="Per system .py file")
    ap.add_argument("--per-bucket", type=int, default=14, help="Max reasonings sampled per winner bucket")
    ap.add_argument("--seed", type=int, default=42, help="RNG seed for stratified sampling")
    ap.add_argument(
        "--skip-code",
        action="store_true",
        help="Omit systems/*.py from the prompt (weaker section 1; cheaper tokens)",
    )
    ap.add_argument(
        "--batch",
        action="store_true",
        help=f"Run all workshop primary compares ({len(_BATCH_COMPARE_FILES)} files)",
    )
    args = ap.parse_args()

    include_code = not args.skip_code

    if args.batch:
        for cur, cand, fname in _BATCH_COMPARE_FILES:
            cpath = RESULTS_DIR / fname
            if not cpath.is_file():
                print(f"[skip] missing {cpath}", flush=True)
                continue
            out = RESULTS_DIR / "workshop_qualitative" / f"{cur}_vs_{cand}.md"
            run_one(
                cur,
                cand,
                cpath,
                model=args.model,
                temperature=args.temperature,
                include_code=include_code,
                code_max_chars=args.code_max_chars,
                per_bucket=args.per_bucket,
                seed=args.seed,
                out_path=out,
            )
        return 0

    if not args.current or not args.candidate:
        ap.error("--current and --candidate are required unless --batch")

    compare = args.compare_json
    if compare is None:
        compare = RESULTS_DIR / f"compare_{args.current}_vs_{args.candidate}.json"
    out = args.out
    if out is None:
        out = RESULTS_DIR / "workshop_qualitative" / f"{args.current}_vs_{args.candidate}.md"

    run_one(
        args.current,
        args.candidate,
        compare,
        model=args.model,
        temperature=args.temperature,
        include_code=include_code,
        code_max_chars=args.code_max_chars,
        per_bucket=args.per_bucket,
        seed=args.seed,
        out_path=out,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
