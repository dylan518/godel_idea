#!/usr/bin/env python3
"""Build `ideas/results/paper_darwin_bundle/` for Darwin Gödel loop paper prep.

Run from repo root::

    python3 ideas/build_darwin_paper_bundle.py

Reads: evolution_log.jsonl, compare_*.json (top-level results only), systems docstrings,
optional publish_eval summary. Writes JSON + Markdown artifacts (no API calls).
"""

from __future__ import annotations

import json
import random
import re
import subprocess
import sys
from collections import Counter
from pathlib import Path

IDEAS_DIR = Path(__file__).resolve().parent
RESULTS = IDEAS_DIR / "results"
OUT = RESULTS / "paper_darwin_bundle"
SYSTEMS = IDEAS_DIR / "systems"


def _git_rev() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=str(IDEAS_DIR.parent),
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return None


def _module_doc(path: Path) -> str:
    if not path.exists():
        return ""
    text = path.read_text()
    if not text.startswith('"""'):
        return ""
    end = text.find('"""', 3)
    if end < 0:
        return ""
    return text[3:end].strip()


def _llm_calls_line(doc: str) -> str | None:
    for line in doc.splitlines():
        if "LLM" in line and ("call" in line.lower() or "~" in line):
            return line.strip()
    return None


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    git_rev = _git_rev()
    current = (IDEAS_DIR / "CURRENT_VERSION").read_text().strip()

    # --- Evolution log ---
    log_path = RESULTS / "evolution_log.jsonl"
    events = []
    if log_path.exists():
        for line in log_path.read_text().strip().splitlines():
            if line.strip():
                events.append(json.loads(line))

    trajectory = {
        "git_rev": git_rev,
        "current_version_file": current,
        "events": events,
    }
    (OUT / "trajectory.json").write_text(json.dumps(trajectory, indent=2))

    # --- Compare reports (non-archive) ---
    compares: list[dict] = []
    for p in sorted(RESULTS.glob("compare_*.json")):
        if "archive" in str(p):
            continue
        try:
            d = json.loads(p.read_text())
        except Exception:
            continue
        d["_path"] = str(p.relative_to(IDEAS_DIR))
        compares.append(d)

    (OUT / "compares_index.json").write_text(json.dumps(compares, indent=2))

    # --- Rejection / failed promotion samples (candidate win_rate < 55%) ---
    samples: list[dict] = []
    rng = random.Random(42)
    for d in compares:
        wr = d.get("win_rate_b")
        if wr is None or wr >= 0.55:
            continue
        cand = d.get("candidate") or d.get("current")
        champ = d.get("current")
        verdicts = d.get("verdicts") or []
        losers = [v for v in verdicts if v.get("winner") == "A"]
        rng.shuffle(losers)
        for v in losers[:5]:
            samples.append(
                {
                    "compare": d.get("_path"),
                    "champion": champ,
                    "candidate": d.get("candidate"),
                    "win_rate_b": wr,
                    "topic_id": v.get("topic_id"),
                    "idea_index": v.get("idea_index"),
                    "judge_model": v.get("judge_model"),
                    "reasoning": (v.get("reasoning") or "")[:600],
                }
            )
    (OUT / "rejection_reason_samples.json").write_text(json.dumps(samples, indent=2))

    # --- Mechanism notes from docstrings ---
    # Reboot baseline is logged as S_paper in evolution_log but implemented in S_sota.py;
    # idea-tree paper system is S_paper.py (key S_paper_skills to disambiguate).
    mechanisms: dict[str, dict] = {}
    doc_seed = _module_doc(SYSTEMS / "S_sota.py")
    seed_lines = doc_seed.splitlines()[:20]
    if seed_lines and seed_lines[0].lstrip().startswith("S_sota:"):
        seed_lines[0] = seed_lines[0].replace("S_sota:", "Baseline (module S_sota.py):", 1)
    mechanisms["S_paper"] = {
        "docstring_preview": (
            "S_paper (reboot baseline in results logs): "
            + "\n".join(seed_lines)
            + "\n\n(Implementation: ideas/systems/S_sota.py; cache: ideas/results/S_sota/.)"
        ),
        "llm_calls_hint": _llm_calls_line(doc_seed),
    }
    for v in ["S12", "S15"]:
        doc = _module_doc(SYSTEMS / f"{v}.py")
        mechanisms[v] = {
            "docstring_preview": "\n".join(doc.splitlines()[:22]),
            "llm_calls_hint": _llm_calls_line(doc),
        }
    doc_tree = _module_doc(SYSTEMS / "S_paper.py")
    mechanisms["S_paper_skills"] = {
        "docstring_preview": "\n".join(doc_tree.splitlines()[:22]),
        "llm_calls_hint": _llm_calls_line(doc_tree),
    }
    (OUT / "mechanisms_from_docstrings.json").write_text(
        json.dumps(mechanisms, indent=2)
    )

    # --- publish_eval summary if present ---
    pub_summaries = []
    for p in sorted(RESULTS.glob("publish_eval_*/summary.json")):
        pub_summaries.append(
            {
                "path": str(p.relative_to(IDEAS_DIR)),
                "data": json.loads(p.read_text()),
            }
        )
    if pub_summaries:
        (OUT / "publish_eval_summaries.json").write_text(
            json.dumps(pub_summaries, indent=2)
        )

    # --- Markdown report ---
    lines = [
        "# Darwin Gödel loop — paper prep bundle",
        "",
        f"Generated from repo state `git rev = {git_rev or 'unknown'}`; `CURRENT_VERSION` = `{current}`.",
        "",
        "## 1. Evolution trajectory (accept chain)",
        "",
        "| time | from | to | primary win_rate (B) | blind | notes |",
        "|------|------|-----|----------------------|-------|-------|",
    ]
    for e in events:
        lines.append(
            "| {ts} | {fr} | {to} | {wr} | {bwr} | {note} |".format(
                ts=e.get("timestamp", "")[:19],
                fr=e.get("from_version", ""),
                to=e.get("to_version", ""),
                wr=e.get("win_rate") if e.get("win_rate") is not None else "—",
                bwr=e.get("blind_win_rate")
                if e.get("blind_win_rate") is not None
                else "—",
                note=(e.get("note") or "").replace("|", "/")[:80],
            )
        )
    lines += [
        "",
        "**Reads:** `ideas/results/evolution_log.jsonl` (+ `trajectory.json` here).**",
        "",
        "## 2. Head-to-head compares on disk (non-archive)",
        "",
        "| report | champion (A) | candidate (B) | B win rate | judged |",
        "|--------|----------------|---------------|------------|--------|",
    ]
    for d in compares:
        lines.append(
            "| `{path}` | {a} | {b} | {wr:.4f} | {n} |".format(
                path=d.get("_path", ""),
                a=d.get("current", ""),
                b=d.get("candidate", ""),
                wr=float(d.get("win_rate_b") or 0),
                n=d.get("total_judged", 0),
            )
        )

    lines += [
        "",
        "## 3. Mechanisms (from system module docstrings)",
        "",
    ]
    for v, info in mechanisms.items():
        lines.append(f"### {v}")
        lines.append("")
        if info.get("llm_calls_hint"):
            lines.append(f"- **Call budget hint:** {info['llm_calls_hint']}")
        lines.append("")
        lines.append("```")
        lines.append(info.get("docstring_preview", "")[:1200])
        lines.append("```")
        lines.append("")

    lines += [
        "## 4. Baselines vs champion (supporting eval, not thesis)",
        "",
        "- **Main loop accepts:** see trajectory — `S_paper`→`S12`→`S15` used primary + blind in log (reboot baseline = `S_sota.py`).",
        "- **`S_paper`:** paper-aligned idea search; see `publish_eval_*` / `judge_*.json` if present.",
        "- **Protocol mismatch warning:** some `compare_*.json` files use **75** judged pairs (e.g. `n_ideas=5`); `publish_eval_*` uses **30** pairs (`n_ideas=2`). Do not mix in one table without labeling.",
        "",
        "### Rough LLM budget per idea (from docstrings — verify in code)",
        "",
        "| system | hint (per idea) |",
        "|--------|-----------------|",
    ]
    for v, info in mechanisms.items():
        hint = info.get("llm_calls_hint") or "—"
        lines.append(f"| `{v}` | {hint} |")
    lines += [
        "",
        "Multiply by topics×ideas for a **benchmark generation** cost envelope; add judge pairs × judge calls for evaluation.",
        "",
    ]
    if pub_summaries:
        lines.append("### Latest `publish_eval` summary (embedded)")
        lines.append("")
        lines.append("```json")
        lines.append(json.dumps(pub_summaries[-1]["data"], indent=2)[:4000])
        lines.append("```")
        lines.append("")

    lines += [
        "## 5. Rejection reason samples (candidate lost)",
        "",
        f"See `rejection_reason_samples.json` ({len(samples)} rows).",
        "",
        "## 6. Replication / next runs (manual)",
        "",
        "```bash",
        "# Second multi-judge bundle (new timestamped dir)",
        "python3 ideas/publish_multi_judge.py --max-spend-usd 10 --workers 15",
        "",
        "# Fix missing Gemini pairs + refresh summary (if needed)",
        "python3 ideas/rerun_missing_verdicts.py --publish-dir ideas/results/publish_eval_<STAMP>",
        "```",
        "",
        "## 7. Paper outline (thesis = loop)",
        "",
        "1. **Introduction:** Darwin Gödel loop — self-improving generators under pairwise selection.",
        "2. **Method:** champion/candidate/compare/accept; blind judge rule; SWE/meta optional.",
        "3. **Results — dynamics:** trajectory figure from `trajectory.json` + plateaus.",
        "4. **Results — mechanisms:** per-hop docstrings / diffs (`mechanisms_from_docstrings.json` + git).",
        "5. **Results — baselines:** `S_paper_skills` (`S_paper.py`) vs champion with explicit protocol rows; reboot row uses log label `S_paper` + `S_sota.py`.",
        "6. **Results — failures:** rejection samples + Goodhart episodes from logs.",
        "7. **Cost:** calls per idea from docstrings; $ heuristic from publish harness.",
        "8. **Discussion:** limits of self-improvement; judge coupling.",
        "9. **Appendix:** frozen `compare_*.json`, `publish_eval_*`, judge outputs.",
        "",
    ]
    (OUT / "REPORT.md").write_text("\n".join(lines) + "\n")

    # Simple rejection tag counts (keyword buckets)
    buckets = Counter()
    keywords = [
        ("feasib", "feasibility"),
        ("vague", "vagueness"),
        ("clarity", "experimental_clarity"),
        ("novel", "novelty"),
        ("incremental", "incremental"),
        ("metric", "metrics"),
        ("dataset", "datasets"),
    ]
    for s in samples:
        r = (s.get("reasoning") or "").lower()
        hit = False
        for kw, label in keywords:
            if kw in r:
                buckets[label] += 1
                hit = True
        if not hit:
            buckets["other"] += 1
    (OUT / "rejection_keyword_buckets.json").write_text(
        json.dumps(dict(buckets), indent=2)
    )

    print(f"Wrote bundle under {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
