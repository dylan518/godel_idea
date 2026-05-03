# Gödel loop: results summary and relation to EvoScientist

**This file (`ideas/RESULTS_SUMMARY.md`) is the main narrative results document** for the `ideas/` Gödel loop. Workshop checklist: `ideas/ICML_WORKSHOP_CHECKLIST.md`. Commands: `ideas/REPRODUCE.md`.

This document records **benchmark outcomes to date** for the `ideas/` pipeline and clarifies how it relates to the **main EvoScientist product** (the LangGraph agent package).

---

## Two different things


|                 | **Main EvoScientist** (`EvoScientist/` package)                 | `**ideas/` Gödel loop**                                                                                |
| --------------- | --------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------ |
| **What it is**  | Multi-agent research CLI/TUI: tools, MCP, subagents, streaming  | Fixed **IdeaGenerator** strategies in Python: multi-step LLM prompts, optional retrieval + tournament  |
| **Idea output** | Whatever the agent produces in conversation; no standard rubric | Every system targets the same **IDEA_FORMAT** block (`systems/base.py`)                                |
| **Evaluation**  | Not built in                                                    | **Pairwise judging**: primary + blind judges on novelty, usefulness, experimental clarity, feasibility |
| **Code link**   | `EvoScientist/EvoScientist.py`, `EvoScientist/prompts.py`       | `ideas/godel_loop.py`, `ideas/systems/S*.py`                                                           |


There is **no import** from the main package into the loop. The loop is a **separate experiment harness** in the same repository.

---

## Reboot baseline (logged as `S_paper`; implemented as `S_sota.py`)

In **`results/evolution_log.jsonl`** and blind trajectories, the post-bootstrap seed is labeled **`S_paper`**. Those frozen ideas are still produced by **`ideas/systems/S_sota.py`** and cached under **`ideas/results/S_sota/`** (EvoScientist-inspired retrieval, intra-topic tournament, multi-perspective critique—not the LangGraph agent).

**This summary omits the pre-reboot lineage (S0, S1, …)** and tabulates the post-reboot spine using the **log label** `S_paper` for that baseline.

## Paper idea-tree system (`S_paper.py`)

**`S_paper.py`** (`ideas/systems/S_paper.py`) runs **`ideas/idea_tournament/`** while injecting repo-root `skills/idea-tournament/` and `skills/research-ideation/` reference Markdown. This is a **different** generator from the reboot baseline above, though both appear as `S_paper` in prose when meaning the paper-aligned line of work. See `evolution_log.jsonl` for a logged **`S15` → `S_paper`** branch entry (the skills-based system; no primary `win_rate` stored there).

---

## Current champion (active loop)


| Field                 | Value                                                                                                   |
| --------------------- | ------------------------------------------------------------------------------------------------------- |
| **Recorded champion** | `S15` (`ideas/CURRENT_VERSION`)                                                                         |
| **Implementation**    | `ideas/systems/S15.py` — hypothesis-first adversarial loop + multi-perspective critique + `IDEA_FORMAT` |


---

## Full lineage (post-reboot): one table

**Convention:** **B** is always the **candidate** (challenger). **WR_B** is the fraction of judged pairs where **B** wins. **95% CI** on primary scores is **topic-block bootstrap** (`ideas/bootstrap_ci_workshop.py`, 3000 iters, seed 42) from the listed `compare_*.json`. **Blind DeepSeek** columns come from `blind_*__deepseek-chat.json` with the same bootstrap script where the file exists in `ideas/results/workshop_paper_bundle/trajectory_table.csv`. **κ** is Cohen’s κ between **primary** labels and **DeepSeek blind** labels on overlapping pairs (`ideas/compute_judge_agreement.py --blind-model deepseek-chat`, `ideas/results/judge_agreement_deepseek_blind.json`). `compare_S15_vs_S20.json` is excluded from CI/κ automation (`ideas/workshop_result_excludes.py`).

**Naming:** the reboot baseline is logged as **`S_paper`** in `evolution_log.jsonl` and blind JSON metadata; frozen ideas for that side still live under **`ideas/results/S_sota/`** and the generator module is **`ideas/systems/S_sota.py`**. Re-running blind backfill for that row uses **`S_sota S12`** as CLI version paths (not `S_paper S12`, which would load `results/S_paper/` — the skills-based system).


| Transition           | Role           | Primary judge             | Prim N | Prim WR_B | Prim 95% CI    | Blind DeepSeek N | Blind WR_B | Blind 95% CI   | κ (prim vs blind DS) | Notes                                                                                                                                                                                                                                                      |
| -------------------- | -------------- | ------------------------- | ------ | --------- | -------------- | ---------------- | ---------- | -------------- | -------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| bootstrap → `S_paper` | seed           | —                         | —      | —         | —              | —                | —          | —              | —                    | Log label **`S_paper`**; generator **`S_sota.py`**; cache **`results/S_sota/`**. Fresh loop; no pairwise file.                                                                                                                                              |
| `S_paper` → **S12**   | **accepted**   | claude-haiku-4-5-20251001 | 75     | **89.3%** | —              | 75               | **80.7%**  | [72.0%, 88.7%] | —                    | Side A ideas from **`results/S_sota/`** (`S_sota.py`). Primary from `godel.log` / `evolution_log.jsonl` (no retained `compare_*.json`). Logged accept-time blind was **Gemini** on a **5/15-topic** subsample (**70%** WR_B); full **75-pair DeepSeek** backfill: `blind_S_paper_vs_S12__deepseek-chat.json` (regenerate: `blind_backfill.py … S_sota S12`). |
| `S12` → S14          | rejected       | deepseek-chat             | 75     | 52.0%     | [41.3%, 61.3%] | 75               | 50.7%      | [38.0%, 62.7%] | 0.324                | `compare_S12_vs_S14.json`                                                                                                                                                                                                                                  |
| `S12` → **S15**      | **accepted**   | claude-haiku-4-5-20251001 | 75     | **55.3%** | [43.3%, 66.7%] | 75               | **57.3%**  | [49.3%, 66.7%] | 0.171                | `compare_S12_vs_S15.json`; blind file refreshed (`blind_backfill.py`).                                                                                                                                                                                     |
| `S15` → S16          | rejected       | deepseek-chat             | 75     | 48.7%     | [38.0%, 60.0%] | 75               | 54.7%      | [46.7%, 62.7%] | 0.275                |                                                                                                                                                                                                                                                            |
| `S15` → S17          | rejected       | deepseek-chat             | 75     | 33.3%     | [22.7%, 46.7%] | 75               | 33.3%      | [17.3%, 50.7%] | 0.400                |                                                                                                                                                                                                                                                            |
| `S15` → S18          | rejected       | deepseek-chat             | 45     | 43.3%     | [23.3%, 64.4%] | 75               | 40.0%      | [28.0%, 53.3%] | 0.604                | **Primary partial:** 45 judged pairs (9 topics); blind file has 75 pairs; κ on **45** overlap.                                                                                                                                                             |
| `S15` → S19          | rejected       | deepseek-chat             | 75     | 10.7%     | [5.3%, 16.0%]  | 75               | 15.3%      | [9.3%, 21.3%]  | 0.658                |                                                                                                                                                                                                                                                            |
| `S15` → S13          | archival       | deepseek-chat             | 75     | **32.7%** | [22.7%, 43.3%] | 75               | 26.7%      | [17.3%, 37.3%] | 0.083                | **Completion benchmark** (not an evolution gate). `compare_S15_vs_S13.json` + `blind_S15_vs_S13__deepseek-chat.json`. Primary: **`--full-judge`** (75 pairs). |
| `S15` vs `S_paper`   | blind-only     | —                         | —      | —         | —              | 30               | 41.7%      | [23.3%, 60.0%] | —                    | **30** shared pairs; `blind_S15_vs_S_paper__deepseek-chat.json` (not an accept/reject gate).                                                                                                                                                               |


**Accept chain:** bootstrap → **`S_paper`** (reboot baseline; `S_sota.py`) → **S12** → **S15** (champion). All other rows are rejects or auxiliary evals.

---

## Human blind pilot (S15 vs S_paper)

**File:** `ideas/results/human_blind_scores.jsonl` (60 rows).

**Design:** **12** custom **thesis topics**, each aligned with a contributor’s **area of expertise**. For each topic, judges completed **5** blind **pairwise** comparisons (**S15** vs **S_paper**, one idea per side per comparison) → **12 × 5 = 60** comparisons total. Order was randomized (`swap` in the log). **Several human raters** split the work (see `rater_id` per row in the JSONL). **CUSTOM** topics only — not the standard **15-topic** `benchmark_topics.json` benchmark.


| Metric                                        | Value               |
| --------------------------------------------- | ------------------- |
| Thesis-topic areas (custom)                   | 12                  |
| Pairwise comparisons per topic                | 5                   |
| **Total** pairwise comparisons                | **60**              |
| Wins: **S15** / **S_paper** / **tie**         | 40 / 15 / 5         |
| **S15** share (all pairs)                     | **66.7%** (40 / 60) |
| **S15** share on **decisive** (non-tie) pairs | **72.7%** (40 / 55) |
| **S_paper** share (all pairs)                 | 25.0% (15 / 60)     |


### By human rater (who preferred which system?)

**File:** `ideas/results/human_blind_by_rater.json` (regenerate: `python3 ideas/compute_human_blind_by_rater.py`).

Everyone saw **only their own** expertise topics (no two people scored the **same** pair), so this is **descriptive** (“how each contributor labeled their slice”), not a reliability study. **WR_B** matches the automated judges: \((\#\text{`S_paper` wins} + \tfrac12 \#\text{ties}) / n\).


| `rater_id`    | n (pairs) | S15 / S_paper / tie | WR_B (`S_paper`) | S15 share (incl. ½ ties) |
| ------------- | --------: | ------------------- | ---------------- | ------------------------ |
| dylan_pilot   | 20        | 13 / 6 / 1          | 32.5%            | 67.5%                    |
| Matt Stoner   | 10        | 5 / 4 / 1           | 45.0%            | 55.0%                    |
| Rachel Pepin  | 6         | 5 / 1 / 0           | 16.7%            | 83.3%                    |
| Caleb Kendrick| 5         | 4 / 0 / 1           | 10.0%            | 90.0%                    |
| Jack Cui      | 5         | 3 / 1 / 1           | 30.0%            | 70.0%                    |
| Jolene Iseler | 5         | 3 / 1 / 1           | 30.0%            | 70.0%                    |
| josh          | 5         | 3 / 2 / 0           | 40.0%            | 60.0%                    |
| Noor Taher    | 4         | 4 / 0 / 0           | **0%**           | **100%**                 |


**Reading:** **All eight raters give S15 a majority** on their own comparisons (S15 share ≥ 55%). **Heterogeneity is in how much** they reward `S_paper`: **WR_B** ranges from **0%** (Noor Taher, small **n = 4**) to **45%** (Matt Stoner). So people **do not** identical-copy each other’s side preference, but **nobody** flips to an overall preference for `S_paper` on their slice. Treat individual WR_B as **noisy** (several raters have **n ≤ 5**).

**Limitations:** sparse per-topic **n = 5**; **no** inter-human agreement on the same pair (different experts on different topics); not comparable to the standard 75-pair benchmark without regenerating ideas on `benchmark_topics.json`.

---

## CUSTOM slice: human vs automated judges (topic-level)

**File:** `ideas/results/custom_slice_human_vs_judge/human_vs_judge.json`  
**Pair:** S15 vs S_paper · **12** topics with human aggregate · per-judge stats on those topics:


| Judge                    | n topics | Side agreement vs human | Pearson r (human topic WR vs judge topic WR) | Judge overall WR_B |
| ------------------------ | -------- | ----------------------- | -------------------------------------------- | ------------------ |
| deepseek-chat            | 12       | 41.7%                   | −0.060                                       | 58.3%              |
| gemini-flash-lite-latest | 12       | 25.0%                   | −0.041                                       | 81.9%              |
| gpt-5.4                  | 12       | 25.0%                   | −0.050                                       | 54.2%              |
| claude-sonnet-4-6        | 0        | —                       | —                                            | —                  |


**Human** topic-level overall WR_B in that JSON: **27.3%** (encodes side/topic aggregation; compare to **72.7%** S15 wins on raw pair counts above).

---

## Multi-judge publish eval (S15 vs S_paper, N = 64 shared)

**File:** `ideas/results/multi_judge_agreement_S15_vs_Spaper.json` — headline **N = 64** shared pair keys (intersection across judges). The on-disk publish-eval folder is named `publish_eval_n75_*` because `ideas.json` lists **75** pair-slots; Gemini-3 Flash Preview lacked valid JSON on **11** keys, so those are excluded from the intersection. Alias copy: `multi_judge_agreement_S15_vs_Spaper_publish_n75.json` (same 64-row content; filename is legacy). The prior **30-pair** custom-slice panel is archived at `ideas/results/_archive_multi_judge_S15_vs_Spaper_n30_20260419.json`.

**Win rate for B (`S_paper`)** (tie-adjusted WR_B):


| Judge                  | Wins A / B / tie | WR_B      |
| ---------------------- | ---------------- | --------- |
| gemini-3-flash-preview | 32 / 4 / 28      | **28.1%** |
| gpt-5.4                | 39 / 5 / 20      | **23.4%** |
| deepseek-chat          | 43 / 20 / 1      | **32.0%** |


**Pairwise Cohen’s κ** (three-class, n = 64): DeepSeek vs GPT-5.4 **0.17**; DeepSeek vs Gemini-3 **0.07**; GPT-5.4 vs Gemini-3 **0.21**.

---

## Full publish-eval bundle (75 pair-slots on disk): Gemini‑3 vs Flash Lite vs DeepSeek

**Directory:** `ideas/results/publish_eval_n75_gemini3_deepseek_S15_Spaper/`  
**Setup:** standard **15-topic** `benchmark_topics.json`, **5** ideas per topic, **S15** vs **S_paper**, generator **`deepseek-chat`** for both sides (one shared run). Judges are applied to the **same** paired ideas.

**Artifacts:** `protocol.json`, `S15/ideas.json`, `S_paper/ideas.json`, `judge_deepseek-chat.json`, `judge_gemini-3-flash-preview.json`, `judge_gemini-flash-lite-latest.json`, optional reruns **`judge_deepseek-chat_rerun.json`** / **`judge_gemini-flash-lite-latest_rerun.json`**, aggregated **`judge_analysis.json`**.

**Caveats**

- **Gemini‑3** returned **invalid JSON** on **9**/75 pairs during the first pass; only **66** pairs have Gemini‑3 labels. DeepSeek and Flash Lite each have **75**/75.
- **Flash Lite** was run afterward with `ideas/judge_eval_folder.py` (**`--workers 30`**; caps at topic count). This does **not** retroactively fix Gemini‑3 gaps.
- Cohen’s κ is **three-class** (A / B / tie) unless noted. **“Stronger” / larger Gemini** here means **Gemini‑3 preview** vs **Gemini Flash Lite** — not human ground truth.

**Win counts on the 66-pair three-way overlap** (from `judge_analysis.json`)


| Judge | Wins A / B / tie | WR_B (ties ×½) | `winner` vs score total (±2 tie rule) |
| ----- | ---------------- | -------------- | ------------------------------------- |
| gemini-3-flash-preview | 33 / 3 / 30 | **27.3%** | **0**/66 mismatch |
| gemini-flash-lite-latest | 25 / 27 / 14 | **51.5%** | **30**/66 mismatch ( **35**/75 on full file ) |
| deepseek-chat | 37 / 25 / 4 | **40.9%** | **36**/66 mismatch ( **43**/75 on full file ) |

**Pairwise agreement (same overlap)**


| Pair | n | Agreement | κ (3-way) | κ binary (decisive only*) |
| ---- | -- | --------- | --------- | -------------------------- |
| Gemini‑3 vs DeepSeek | 66 | 40.9% | **0.12** | 0.20 (n_dec = 34) |
| **Flash Lite vs DeepSeek** | 66 | 53.0% | **0.24** | 0.32 (n_dec = 50) |
| Gemini‑3 vs Flash Lite | 66 | 43.9% | 0.19 | 0.61 (n_dec = 27) |

\*Decisive = both judges picked A or B (ties excluded).

**Flash Lite vs DeepSeek on all 75 pairs** (Flash completed every pair): agreement **52%**, **κ ≈ 0.22**.

**Reading:** On this slice, **Gemini‑3 is not “more aligned” with DeepSeek than Flash Lite** — the **lighter** Gemini agrees **more** with DeepSeek on labels (higher κ, fewer B-rate extremes vs Gemini‑3’s tie-heavy profile). Higher model tier mainly shifts **tie rate and calibration**, not convergence to the primary judge. Large declared-vs-score mismatches for DeepSeek and Flash Lite on this run reinforce treating **`winner` as unreliable** unless derived from scores.

### Longitudinal caveat: primary judge and “wins” in evolution

The post-reboot table above mixes rows where the **primary** `compare_*.json` judge was **Claude Haiku** (e.g. `S12`→`S15`) and rows where it was **DeepSeek** (most later compares). **Accept/reject is defined on that primary `win_rate_b`**, not on a multi-judge panel. When the primary model changes, **the same candidate can move from clear accept territory to clear reject territory** without any change to the generators — so “we stopped seeing wins” after a judge swap is **consistent with judge-dependent measurement**, not only with candidate quality. Treat headline evolution outcomes as **tied to the primary judge + harness**, not as a judge-free ground truth.

### Test–retest noise (same `ideas.json`, second pass)

To see how much **rerunning** a judge moves the scoreboard, we judged the **same** 75 pairs in `publish_eval_n75_gemini3_deepseek_S15_Spaper` twice for **DeepSeek** and **Gemini Flash Lite** (second files: `judge_*_rerun.json`). Each call still uses **`IDEAS_JUDGE_TEMPERATURE` default 0.2** and **random A/B presentation** per pair (`judge.py`), so this is **not** a pure decoding experiment — it is an **operational** repeatability test under the current protocol.

| Judge (same 75 pairs) | Run 1 WR_B (`S_paper`) | Run 2 WR_B | Exact label agreement | Cohen’s κ (3-way, self vs self) | # labels changed |
| --------------------- | ---------------------- | ---------- | --------------------- | -------------------------------- | ---------------- |
| **deepseek-chat**     | **41.3%**              | **34.0%**  | **73.3%**             | **0.50**                         | 20 / 75          |
| **gemini-flash-lite-latest** | **54.0%**       | **60.7%**  | **72.0%**             | **0.55**                         | 21 / 75          |

**Takeaway — don’t confuse cross-judge spread with intra-judge stability:** **Reruns of the same judge are not the same** (~73% exact agreement here; **WR_B** shifts by ~7 pp on one slice). So a single pass is **not** a stable measurement of “how strong” a system is. By contrast, **cross-judge comparisons** can show a **persistent spread** (e.g. the publish panel cluster vs DeepSeek on tie rate and `S_paper` wins) that **looks** like a consistent story about who is strict or lenient — that spread is **real between-model behavior on one frozen idea set**, but it does **not** imply that **any one** of those judges would repeat itself on a second pass. **Agreement among judges** and **repeatability of one judge** are different axes; strength claims need **test–retest or ensembling**, not one shot. For evolution, prefer **temperature 0**, **fixed or logged A/B order**, **deterministic `winner` from scores**, and/or **multiple judge passes** before strong claims.

**Regenerate / extend**

```bash
# Flash Lite (or any judge) on an existing publish_eval folder — 30 workers
python3 ideas/judge_eval_folder.py \
  --eval-dir ideas/results/publish_eval_n75_gemini3_deepseek_S15_Spaper \
  --judge-model gemini-flash-lite-latest \
  --workers 30

# Second pass without overwriting the first
python3 ideas/judge_eval_folder.py \
  --eval-dir ideas/results/publish_eval_n75_gemini3_deepseek_S15_Spaper \
  --judge-model deepseek-chat \
  --workers 30 \
  --out-name judge_deepseek-chat_rerun.json

python3 ideas/analyze_publish_eval_judges.py \
  --eval-dir ideas/results/publish_eval_n75_gemini3_deepseek_S15_Spaper
```

---

## Rubric means (LLM judges on frozen compares)

Judges fill the same four **0–10** dimensions as the human UI: novelty, scientific usefulness, experimental clarity, feasibility. Totals below are **sums of the four** (max 40). **Absolute levels are not comparable across judges** (different calibration); within a single `compare_*.json`, **A vs B** is on the same scale.

**Regenerate:** `python3 ideas/compute_rubric_summaries.py` → `ideas/results/rubric_summary.json`.

### Pooled by system version (all primary `compare_*.json` cells)

Each system’s ideas contribute one score vector per judged pair where that system is side **A** or **B** (so **S15** pools many compares as champion; **S19** only appears as candidate vs S15).


| Version | Cells (pairs) | Mean novelty | Mean usefulness | Mean clarity | Mean feasibility | Mean total (4 dims) |
| ------- | ------------: | -----------: | --------------: | -----------: | ---------------: | ------------------: |
| S12     | 150           | 7.28         | 7.90            | 8.09         | 7.01             | **30.27**           |
| S13     | 75            | 8.96         | 8.64            | 7.29         | 6.04             | **30.93**           |
| S14     | 75            | 8.21         | 8.64            | 8.28         | 7.11             | **32.24**           |
| S15     | 420           | 7.72         | 8.46            | 8.31         | 7.15             | **31.65**           |
| S16     | 75            | 8.40         | 8.60            | 8.24         | 6.85             | **32.09**           |
| S17     | 75            | 5.07         | 7.16            | 8.72         | 9.23             | **30.17**           |
| S18     | 45            | 6.62         | 8.16            | 8.67         | 8.71             | **32.16**           |
| S19     | 75            | 5.31         | 6.12            | 3.65         | 6.79             | **21.87**           |


**Reading:** **S19** collapses mainly on **experimental clarity** in primary DeepSeek scores; **S17** is high on clarity/feasibility but weak on novelty. **S15** sits mid-pack on pooled means because it is judged against many different opponents.

### Per-compare means (primary judge only)

Side **A** = champion (`current`); side **B** = candidate (`candidate`). Mean total = mean of (novelty + usefulness + clarity + feasibility) over pairs.


| Compare file | Primary judge | N | Mean total A | Mean total B |
| ------------ | ------------- | -: | -----------: | -----------: |
| `compare_S12_vs_S14.json` | deepseek-chat | 75 | 32.01 | 32.24 |
| `compare_S12_vs_S15.json` | claude-haiku-4-5-20251001 | 75 | 28.53 | 28.49 |
| `compare_S15_vs_S13.json` | deepseek-chat | 75 | 32.80 | 30.93 |
| `compare_S15_vs_S16.json` | deepseek-chat | 75 | 32.15 | 32.09 |
| `compare_S15_vs_S17.json` | deepseek-chat | 75 | 31.67 | 30.17 |
| `compare_S15_vs_S18.json` | deepseek-chat | 45 | 31.78 | 32.16 |
| `compare_S15_vs_S19.json` | deepseek-chat | 75 | 33.08 | 21.87 |


Primary and blind for **`S15` vs S13** are both **75 pairs**; regenerate aggregates with `bootstrap_ci_workshop.py`, `compute_judge_agreement.py`, `export_workshop_paper_bundle.py`, and `compute_rubric_summaries.py` after re-running compare.

### Blind DeepSeek sample (same rubric)

Mean totals for side **A** / **B** on selected **75-pair** blind backfills:


| Blind file | A (current) | B (candidate) | Mean total A | Mean total B |
| ---------- | ----------- | ------------- | -----------: | -----------: |
| `blind_S_paper_vs_S12__deepseek-chat.json` | S_paper | S12 | 29.93 | 32.97 |
| `blind_S12_vs_S14__deepseek-chat.json` | S12 | S14 | 31.85 | 31.96 |
| `blind_S12_vs_S15__deepseek-chat.json` | S12 | S15 | 32.01 | 32.13 |


---

## Human pilot rubrics (S15 vs `S_paper`)

From **`ideas/results/human_blind_scores.jsonl`** (same four dimensions, **0–10**). Scores are mapped to **S15** vs **`S_paper`** using `swap` (`swap=false` → **A** = S15, **B** = S_paper; `swap=true` → reversed). **60** rated pairs; **5** ties on winner label (all pairs still have rubric scores).


| Side | Mean novelty | Mean usefulness | Mean clarity | Mean feasibility | Mean total |
| ---- | -----------: | --------------: | -----------: | ---------------: | ---------: |
| **S15** | 4.98 | 5.05 | 5.15 | 4.90 | **20.08** |
| **`S_paper`** | 4.78 | 5.20 | 5.35 | 4.88 | **20.22** |


**Paired mean difference (S15 − S_paper), same 60 pairs:** novelty **+0.20**, usefulness **−0.15**, clarity **−0.20**, feasibility **+0.02** (all small vs a 0–10 span).

**Interpretation:** On average, humans gave **almost identical total rubric mass** to the two systems (~20/40), while side-by-side they still picked **S15** on **40/60** wins and **15/60** for **`S_paper`** (see Human blind pilot table above). So **winners were not driven by higher mean human rubric scores**; they reflect **trade-offs** (e.g. higher novelty on one idea vs higher clarity on the other) and how the rater weighted dimensions. Human scores are also **much lower than typical LLM judge scores** on the same dimensions—different rater calibration, not necessarily different idea quality.

---

## Aggregates and regeneration


| Artifact                           | Path                                                                                                 |
| ---------------------------------- | ---------------------------------------------------------------------------------------------------- |
| CI summaries                       | `ideas/results/ci_summary.json`                                                                      |
| Primary vs DeepSeek-blind κ / flip | `ideas/results/judge_agreement_deepseek_blind.json`                                                  |
| Merged trajectory CSV              | `ideas/results/workshop_paper_bundle/trajectory_table.csv` (`ideas/export_workshop_paper_bundle.py`) |
| Rubric means (LLM + human)         | `ideas/results/rubric_summary.json` (`ideas/compute_rubric_summaries.py`)                            |


---

## How this compares to “EvoScientist idea generation” in practice

- **Shipped agent:** no single `generate_idea()` or judge pipeline in `EvoScientist/`—you would need an adapter on the same topics and `IDEA_FORMAT` to compare.
- **Research loop strategies:** reboot baseline (log label **`S_paper`**, code **`S_sota.py`**) and descendants under `ideas/systems/`. Strongest **accepted** system on this branch: **S15**.
- **Reproduce:** `python3 ideas/godel_loop.py status` and `compare --candidate S{n}` from repo root; `ideas/README.md`, `ideas/REPRODUCE.md`.

---

## Files of record


| Artifact                 | Path                                                |
| ------------------------ | --------------------------------------------------- |
| Champion pointer         | `ideas/CURRENT_VERSION`                             |
| Accept history           | `ideas/results/evolution_log.jsonl`                 |
| Per-run details          | `ideas/results/compare_*.json`                      |
| Cached ideas per version | `ideas/results/<VERSION>/ideas.json` (when present) |
| Blind judge backfills    | `ideas/results/blind_*__deepseek-chat.json`         |


Last updated **2026-04-19** (`evolution_log.jsonl`, `compare_*.json`, `blind_*.json`, `human_blind_scores.jsonl`, `rubric_summary.json`, `custom_slice_human_vs_judge/human_vs_judge.json`, `multi_judge_agreement_S15_vs_Spaper.json` — 64-pair publish-eval intersection, `workshop_paper_bundle/trajectory_table.csv`).