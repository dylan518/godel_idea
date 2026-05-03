# EvoScientist (Gödel loop) — status brief for faculty

**April 2026**

---

## Summary

We built a **self-improving research-idea generator**: each version is a Python class that maps a scientific topic to a structured idea; versions compete in **pairwise comparisons** on a **fixed set of scientific subjects**; the champion’s code is **edited under Claude’s control** (SWE agent) to produce the next candidate. **Acceptance is driven mainly by an LLM judge (DeepSeek)** on that benchmark; **human blind pairwise ratings** provide a partial check but **need more volume and design rigor** before strong claims.

---

## What the system is

1. **Generator (`systems/S{n}.py`)**  
   The current champion is **`S15`** (see `ideas/CURRENT_VERSION`). For each topic, the pipeline typically includes: retrieve recent papers (Semantic Scholar), generate and stress-test hypotheses, select one, design an experiment, multi-persona critique, and final formatting (IDEA / BACKGROUND / APPROACH / EXPERIMENT / NOVELTY). Details are in `ideas/SYSTEM_OVERVIEW.md` and `ideas/WORKFLOW.md`.

2. **Benchmark: fixed scientific subjects**  
   Automatic evaluations use a **curated list of 15 research topics** in `ideas/benchmark_topics.json`—covering ML, biology, physics, chemistry, genomics, climate, statistics, and related areas (e.g. scaling laws, protein structure, quantum error correction, federated learning). Each system generates **5 ideas per topic**, so one full duel is **15 × 5 = 75** pairwise comparisons (`ideas/runner.py`). This keeps evaluation **comparable across versions** on the same subject matter; it is **not** an exhaustive survey of all of science.

3. **Automatic judge (`ideas/judge.py`)**  
   - **Primary judge** (accept/reject): **`deepseek-chat`** (DeepSeek V3 family).  
   - Scores each idea on **novelty, scientific usefulness, experimental clarity, feasibility** (0–10 each), outputs JSON, and picks **A / B / tie** (tie if total scores within 2 points).  
   - **Order is randomized** to reduce position bias.  
   - **Blind / canary judge**: **`gemini-flash-lite-latest`** — run for diagnostics; **not** used to accept candidates.  
   - If primary and blind **diverge strongly**, we treat the result as a **Goodhart warning** (possible overfitting to the primary judge).

4. **The Gödel loop (how the system improves)**  
   - **Champion:** one version (e.g. `S15`) is the current best generator class.  
   - **Candidate:** we want a new `S{n+1}.py` that might beat it. The usual path is **`swe-evolve`** in `ideas/godel_loop.py`, which runs the **SWE agent** (`ideas/swe_agent.py`).  
   - **Claude’s role:** the agent uses **Claude Sonnet** (configured as `claude-sonnet-4-6`) as the **meta-model**. It reads **why the last candidate lost** (comparison reports, judge reasoning, optional `swe_memory.json` / `swe_context.json`), proposes **targeted edits** to the champion’s Python file—often via **Claude Code–style tooling** (the agent can invoke the Claude CLI to read and patch files)—and **tests** each change on a **small mini-eval** (e.g. a few topics × a few ideas) before keeping it.  
   - **Full evaluation:** after several SWE rounds, the accumulated file becomes the candidate; we run the **full 75-pair** benchmark against the champion. **Rule of thumb: accept if the candidate’s win rate > 55%** (with extra caution if the Goodhart signal is severe). If rejected, the champion stays and the process can repeat.  
   - **Alternative:** `godel_loop.py generate` can ask a meta-LLM to draft a **whole new** `S{n}.py` from scratch; in practice the **surgical SWE path** is the main “self-editing” story.

5. **`S_paper` (comparison arm)**  
   A separate generator aligned with the published EvoScientist paper settings (`ideas/paper_config.py`). Used for **human A/B** against **`S15`** to compare “paper-style pipeline” vs “current evolved champion.”

---

## Results so far (LLM judge)

From `ideas/results/evolution_log.jsonl` and comparison JSON files:

| Transition | Primary win rate (candidate) | Notes |
|------------|------------------------------|--------|
| **`S_paper` → S12** (reboot baseline `S_sota.py`) | **~89%** for S12 | Large win over the reboot baseline; blind judge **~70%** for S12 |
| **S12 → S15** | **~55%** for S15 | Accepted as champion (>55% threshold); **low primary–blind agreement** was logged (~33% agreement, ~33% flip rate) — treat as a warning |
| **S15 vs S16** (full 75 pairs) | **~49%** for S16 | **Rejected** |
| **S15 vs S17, S18, S19** | **~33%, ~43%, ~11%** (full runs in repo) | Rejected |
| **S15 vs S20** | **~75%** in file | **Only 12 pairs** in `compare_S15_vs_S20.json` — **not** a full benchmark; preliminary only |

**Current champion:** **S15**.

**Interpretation:** The loop shows **measurable movement** on the LLM-judged suite, but the **metric is a proxy**. The **S15** acceptance run had **serious disagreement** between primary and blind judges — **human evaluation and/or other validation** matter for scientific claims.

---

## Human blind evaluation (partial; more data needed)

**Tool:** local web UI (`ideas/human_judge_ui.py`), log: `ideas/results/human_blind_scores.jsonl` (workflow: `ideas/HUMAN_JUDGE_COLLAB.md`).  
**Design:** blind side-by-side **A / B / tie** on matched slots; optional rubric scores; shuffling to limit label and position bias.

**Volume so far:** **68** pairwise judgments, all **`S15` vs `S_paper`**.

| Outcome | Count |
|---------|--------|
| **S15 wins** | 45 |
| **`S_paper` wins** | 18 |
| **Tie** | 5 |

- **Decisive pairs only:** S15 **45 / (45 + 18) ≈ 71%**.  
- **All pairs:** **45 / 68 ≈ 66%** preference for S15.

**Raters:** several named participants plus some anonymous entries; topics are **custom and domain-heterogeneous**, not a single fixed protocol.

**Limitations:** small **N**, **ad hoc topics**, **inconsistent use of rubric fields**, **no pre-registered design** or **inter-rater reliability** on identical items. **More human votes and a clearer study design** are needed for publication-grade conclusions.

---

## Methods (short checklist)

1. **Task:** Generate structured research ideas from topic strings.  
2. **Fixed benchmark:** 15 scientific subjects in `benchmark_topics.json`; 5 ideas per topic per system; 75 pairwise judgments per duel.  
3. **Retrieval:** Semantic Scholar (cached), ~5 papers per topic for grounding.  
4. **Generation:** Versioned Python pipelines (`S{n}`).  
5. **Automatic evaluation:** Pairwise LLM judge (DeepSeek primary, Gemini canary), four criteria, JSON; randomized presentation.  
6. **Evolution:** Claude-driven SWE loop (read losses → propose patches → mini-eval → repeat) → full 75-pair compare → threshold / Goodhart checks; optional one-shot `generate` for whole new files.  
7. **Human evaluation:** Blind A/B UI, JSONL log; ongoing (often custom topics, not the fixed 15).

---

## Open questions (useful for feedback)

1. Is pairwise **LLM judging** an acceptable training signal for “better science,” or does it mainly teach systems to **match that judge’s preferences**?  
2. What mix of **human expert rating**, **downstream outcomes**, or **both** would you want to see before treating results as convincing?  
3. Human results favor **S15** over **`S_paper`** in this log — what would **invalidate** or **strengthen** that finding (topic choice, rater expertise, **N**)?  
4. What **sample size and design** (paired design, multiple raters per item) would you consider adequate?

---

*Figures and paths: `ideas/results/evolution_log.jsonl`, `ideas/results/human_blind_scores.jsonl`, `ideas/compare_S15_vs_*.json`.*
