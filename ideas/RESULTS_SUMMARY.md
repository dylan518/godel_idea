# Gödel loop: results summary and relation to EvoScientist

**This file (`ideas/RESULTS_SUMMARY.md`) is the main narrative results document** for the `ideas/` Gödel loop. For ICML-style tables, CIs, and judge agreement, also use `ideas/results/ci_summary.json`, `ideas/results/judge_agreement_deepseek_blind.json`, and `ideas/results/workshop_paper_bundle/trajectory_table.csv` (regenerate via `ideas/export_workshop_paper_bundle.py`). Workshop checklist: `ideas/ICML_WORKSHOP_CHECKLIST.md`.

This document records **benchmark outcomes to date** for the `ideas/` pipeline and clarifies how it relates to the **main EvoScientist product** (the LangGraph agent package).

---

## Two different things


|                 | **Main EvoScientist** (`EvoScientist/` package)                 | `**ideas/` Gödel loop**                                                                                                         |
| --------------- | --------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------- |
| **What it is**  | Multi-agent research CLI/TUI: tools, MCP, subagents, streaming  | Fixed **IdeaGenerator** strategies in Python: multi-step LLM prompts, optional retrieval + tournament                           |
| **Idea output** | Whatever the agent produces in conversation; no standard rubric | Every system targets the same **IDEA_FORMAT** block (`systems/base.py`)                                                         |
| **Evaluation**  | Not built in                                                    | **Pairwise judging**: primary judge (DeepSeek) + blind judge (Gemini) on novelty, usefulness, experimental clarity, feasibility |
| **Code link**   | `EvoScientist/EvoScientist.py`, `EvoScientist/prompts.py`       | `ideas/godel_loop.py`, `ideas/systems/S*.py`                                                                                    |


There is **no import** from the main package into the loop. The loop is a **separate experiment harness** in the same repository.

---

## “EvoScientist-inspired” idea generation (`S_sota`)

The baseline `**S_sota`** (`ideas/systems/S_sota.py`) is explicitly described there as **EvoScientist-inspired**: it combines **SOTA paper context** (retrieval), an **intra-topic Elo tournament** over multiple candidates, and **multi-perspective critique** (experimentalist / theorist / skeptic), implemented as plain `call_llm` steps—not by running the LangGraph agent.

That baseline **replaced the old S0-style starting point** when the loop was rebooted (see `results/evolution_log.jsonl` bootstrap entry).

## Paper-aligned pipeline (`S_paper`)

`**S_paper`** (`ideas/systems/S_paper.py`) runs `**ideas/idea_tournament/`** (Python) while injecting **repo-root `skills/idea-tournament/`** and `**skills/research-ideation/**` reference Markdown into prompts (same files as Claude Code / the agent). `S_sota` still uses flat candidates + `ideas/tournament.py` on raw text.

- **Benchmark / cache:**  
`python3 ideas/runner.py --system S_paper --output ideas/results/S_paper/ --n-ideas 5 --model deepseek-chat`  
(run from repo root; set keys in `.env` as for other systems.)
- **Compare vs champion:**  
`python3 ideas/godel_loop.py compare --candidate S_paper --n-ideas 5 --workers 3`
- **Reruns:** There is **no** precomputed `compare_*.json` for `S_paper` until you run the above; prior numbers (S12, S15, …) are **not** comparable apples-to-apples until `S_paper` is on the same benchmark settings.

SWE/meta evolution can target `**ideas/idea_tournament/*.py`**, repo `**skills/idea-tournament/references/*.md`**, `**skills/research-ideation/references/*.md**`, or `**S_paper.py**`.

---

## Current champion (active loop)


| Field                 | Value                                                                                                   |
| --------------------- | ------------------------------------------------------------------------------------------------------- |
| **Recorded champion** | `S15` (`ideas/CURRENT_VERSION`)                                                                         |
| **Implementation**    | `ideas/systems/S15.py` — hypothesis-first adversarial loop + multi-perspective critique + `IDEA_FORMAT` |


---

## Accepted evolution (active `results/evolution_log.jsonl`)

Chronological accepts only (each line is a promotion):


| Step      | From     | To       | Primary win rate (B) | Blind win rate (notes)                                                                           |
| --------- | -------- | -------- | -------------------- | ------------------------------------------------------------------------------------------------ |
| Bootstrap | —        | `S_sota` | —                    | Fresh loop seeded with `S_sota`                                                                  |
| 1         | `S_sota` | `S12`    | **89.3%**            | 70%                                                                                              |
| 2         | `S12`    | `S15`    | **55.3%**            | 66.7%; judge agreement metrics logged (high flip/agreement noise on this run—see raw log)        |


So far, **two** head-to-head promotions after bootstrap: **S12** dethroned `S_sota`, then **S15** dethroned **S12**.

---

## Full comparison runs on disk (`ideas/results/compare_*.json`)

These are the **75-pair** (or noted) benchmarks saved as JSON. **Win rate for the candidate** is `win_rate_b` (fraction of pairs where the candidate won).

### Versus **S12** (when S12 was champion)


| Champion (A) | Candidate (B) | B wins / total | Candidate win rate | Notes                                                                 |
| ------------ | ------------- | -------------- | ------------------ | --------------------------------------------------------------------- |
| S12          | S14           | 39 / 75        | **52.0%**          | DeepSeek primary; `compare_S12_vs_S14.json` + blind DeepSeek backfill |
| S12          | S15           | 41 / 75        | **55.3%**          | Accepted → current champion                                           |


### Versus **S15** (current champion)


| Champion (A) | Candidate (B) | B wins / total | Candidate win rate | Notes                               |
| ------------ | ------------- | -------------- | ------------------ | ----------------------------------- |
| S15          | S16           | 36 / 75        | 48.7%              | Rejected                            |
| S15          | S17           | 25 / 75        | 33.3%              | Rejected                            |
| S15          | S18           | 18 / 45        | 43.3%              | **Partial run** (45 judged, not 75) |
| S15          | S19           | 8 / 75         | 10.7%              | Rejected                            |


Interpretation: **no candidate has beaten S15** under the saved full compares; **S16** was closest but still under threshold.

---

## Aggregated tables (workshop / paper)

| Artifact | Path |
| -------- | ---- |
| Topic-block bootstrap CIs on `compare_*.json` | `ideas/results/ci_summary.json` (`ideas/bootstrap_ci_workshop.py`) |
| Primary vs DeepSeek-blind κ / flip | `ideas/results/judge_agreement_deepseek_blind.json` (`ideas/compute_judge_agreement.py --blind-model deepseek-chat`) |
| Merged trajectory CSV | `ideas/results/workshop_paper_bundle/trajectory_table.csv` (`ideas/export_workshop_paper_bundle.py`) |
| Multi-judge S15 vs S_paper | `ideas/results/multi_judge_agreement_S15_vs_Spaper.json` |

`compare_S15_vs_S20.json` is **omitted** from CI / κ scripts (`ideas/workshop_result_excludes.py`).

---

## How this compares to “EvoScientist idea generation” in practice

- **If you mean the shipped agent:** there is **no single `generate_idea()`** or judge pipeline in `EvoScientist/`—quality is emergent from instructions, tools, and user prompts. You cannot line it up with these numbers without building a new adapter that runs the agent on the same 15 topics and the same `IDEA_FORMAT`.
- **If you mean the research loop’s “productized” strategy:** that is `**S_sota`** and its descendants (**S12**, **S15**, etc.)—all under `ideas/systems/`. The **strongest accepted system in the current branch is S15**, not `S_sota`.
- **To reproduce or extend:** `python3 ideas/godel_loop.py status` and `compare --candidate S{n}` from the repo root; see `ideas/README.md` and `ideas/REPRODUCE.md`.

---

## Files of record


| Artifact                 | Path                                                |
| ------------------------ | --------------------------------------------------- |
| Champion pointer         | `ideas/CURRENT_VERSION`                             |
| Accept history           | `ideas/results/evolution_log.jsonl`                 |
| Per-run details          | `ideas/results/compare_*.json`                      |
| Cached ideas per version | `ideas/results/<VERSION>/ideas.json` (when present) |
| Blind judge backfills    | `ideas/results/blind_*__deepseek-chat.json`         |


Last updated **2026-04-17** (aggregated from `evolution_log.jsonl`, `compare_*.json`, and workshop aggregates on disk).
