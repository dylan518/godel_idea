# ICML Workshop Checklist (8-Day Sprint)

Last updated: 2026-04-17

## Data / backfill (completed)

- Topic-block bootstrap CIs on all `compare_*.json`
  - Script: `ideas/bootstrap_ci_workshop.py`
  - Output: `ideas/results/ci_summary.json`
- `rater_id` present on every row of `ideas/results/human_blind_scores.jsonl`
- Consistent deepseek-chat blind on every trajectory transition
  - `blind_S_sota_vs_S12__deepseek-chat.json` (N=75, win_rate_b=0.807)
  - `blind_S12_vs_S14__deepseek-chat.json` (N=75, win_rate_b≈0.507)
  - `blind_S12_vs_S15__deepseek-chat.json` (N=75, win_rate_b=0.540)
  - `blind_S15_vs_S16__deepseek-chat.json` (N=75, win_rate_b=0.547)
  - `blind_S15_vs_S17__deepseek-chat.json` (N=75, win_rate_b=0.333)
  - `blind_S15_vs_S18__deepseek-chat.json` (N=75, win_rate_b=0.400)
  - `blind_S15_vs_S19__deepseek-chat.json` (N=75, win_rate_b=0.153)
  - `blind_S15_vs_S_paper__deepseek-chat.json` (N=30, win_rate_b=0.417)
  - **S20** omitted from workshop CI/κ aggregates (`ideas/workshop_result_excludes.py`); raw `compare_S15_vs_S20` / blinds are not kept in-repo unless you regenerate.
- Primary ↔ blind (deepseek) agreement table recomputed
  - Script: `ideas/compute_judge_agreement.py --blind-model deepseek-chat`
  - Output: `ideas/results/judge_agreement_deepseek_blind.json`
- Multi-judge agreement on S15 vs S_paper (benchmark T1..T15)
  - Script: `ideas/compute_multi_judge_agreement.py`
  - Judges: claude-sonnet-4-6, gemini-3-flash-preview, gpt-5.4, deepseek-chat
  - Output: `ideas/results/multi_judge_agreement_S15_vs_Spaper.json`

## Data / backfill (dropped or deferred)

- **`S12 vs S14`:** **`compare_S12_vs_S14.json`** (DeepSeek primary, N=75) + **`blind_S12_vs_S14__deepseek-chat.json`** (second pass; κ reflects same-model stochasticity).
- `compare_S12_vs_S15.json` primary is `claude-haiku` (one-off inconsistency). We report the deepseek blind value as the headline and keep the claude-haiku primary as a cross-model robustness check (→ kappa=0.39).

## Headline numbers (deepseek blind; topic-block bootstrap 95% CI)

Regenerate the merged table anytime: `cd ideas && python3 export_workshop_paper_bundle.py` → `results/workshop_paper_bundle/trajectory_table.csv`.


| Transition     | N_blind | blind WR_B | blind CI_low | blind CI_high | κ (primary vs blind DS) | Notes                                      |
| -------------- | ------- | ---------- | ------------ | ------------- | ----------------------- | ------------------------------------------ |
| S_sota → S12   | 75      | 0.807      | 0.72         | 0.887         | —                       | blind only                                 |
| S12 → S14      | 75      | 0.507      | 0.38         | 0.627         | 0.324                   | rejected candidate; primary+blind both DS   |
| S12 → S15      | 75      | 0.540      | 0.407        | 0.667         | 0.387                   | primary compare used Haiku (see REPRODUCE) |
| S15 → S16      | 75      | 0.547      | 0.467        | 0.627         | 0.275                   |                                            |
| S15 → S17      | 75      | 0.333      | 0.173        | 0.507         | 0.400                   |                                            |
| S15 → S18      | 75      | 0.400      | 0.280        | 0.533         | 0.604                   | primary compare N=45; blind N=75           |
| S15 → S19      | 75      | 0.153      | 0.093        | 0.213         | 0.658                   |                                            |
| S15 vs S_paper | 30      | 0.417      | 0.233        | 0.600         | —                       | blind-only row; see multi-judge table      |


**Version numbering:** the **accepted** spine in `evolution_log.jsonl` is only **S_sota → S12 → S15** (then the S_paper branch). **S14** and **S16–S19** are *rejected challengers* (vs S12 or S15), not sequential replacements. **S13** is a generator-only line in this narrative unless you add compares.

### Where the “missing” `S*` versions are

| Label | `systems/S*.py` | `results/S*/ideas.json` | Role in *this* loop |
| ----- | ----------------- | ------------------------- | -------------------- |
| **S_sota, S12, S15** | yes | S12, S15 yes | Champion path |
| **S_paper** | yes | under `publish_eval_*` + top-level `S_paper` | Paper-aligned comparator |
| **S13, S14** | yes | **yes** (`results/S13`, `S14`; 75 ideas each) | Full benchmark snapshots regenerated 2026-04-17 |
| **S16–S19** | yes | **yes** | Rejected vs S15; full compares + blinds |
| **S20** | yes | optional / excluded from default CI | Not on workshop spine |
| **S0–S11** (gaps) | — | — | Older loop; not kept in this tree |

Numbers skip because each **S*n*** is a **meta-generator attempt**, not “version + 1 of the champion.”

## Multi-judge on S15 vs S_paper (T1..T15, N=30)


| Judge                      | win_rate_b (S_paper) | Verdict                |
| -------------------------- | -------------------- | ---------------------- |
| claude-sonnet-4-6          | 0.267                | S15 preferred          |
| gemini-3-flash-preview     | 0.200                | S15 strongly preferred |
| gpt-5.4                    | 0.200                | S15 strongly preferred |
| deepseek-chat (loop judge) | 0.417                | approximately tied     |


Pairwise Cohen's kappa (judge-judge):


|          | sonnet | gemini-3 | gpt-5 | deepseek |
| -------- | ------ | -------- | ----- | -------- |
| sonnet   | —      | 0.62     | 0.37  | 0.14     |
| gemini-3 | 0.62   | —        | 0.38  | 0.08     |
| gpt-5    | 0.37   | 0.38     | —     | 0.03     |
| deepseek | 0.14   | 0.08     | 0.03  | —        |


Story: the three strong judges agree **moderately** with each other and all **disagree** with the loop judge (deepseek-chat). The loop judge is systematically more generous to S_paper than the strong judges, consistent with a Goodhart signal on the specific judge the loop optimized against.

## Human pilot (60 ratings, 1 rater = dylan_pilot, 12 CUSTOM topics)

- **Fixed T1–T15 benchmark (recommended for judge alignment):** run `ideas/prepare_fixed_benchmark_human_eval.py`, then `human_judge_ui.py` with `?left=fixed_benchmark_S15&right=fixed_benchmark_S_paper` (30 slots, same texts as `publish_eval_20260413T163806Z`). Summarize vs judges with `ideas/compute_human_judge_agreement_fixed.py` → e.g. `results/human_judge_agreement_fixed.json`.
- Existing data: `ideas/results/human_blind_scores.jsonl` (pair = S15 vs S_paper)
- Topic-level human vs judge (CUSTOM slice, regenerated ideas; **not** the same text as the UI ratings):
  - Inputs: `ideas/results/custom_slice_human_vs_judge/summary.json` (from `run_custom_slice_judge.py`)
  - Metrics: `ideas/compute_human_vs_judge.py` → `ideas/results/custom_slice_human_vs_judge/human_vs_judge.json`
  - **Results (12 topics, n_human 4–5 per topic in summary):** Pearson *r* (human topic WR vs judge topic WR) is near **zero** for DeepSeek / Gemini-flash-lite / GPT-5.4 (slightly negative). **Side agreement** (majority-winner side): DeepSeek **42%**, Gemini-flash-lite **25%**, GPT-5.4 **25%**. Judges’ overall `win_rate_b` (S_paper) is **much higher** than humans’ on this slice (e.g. Gemini **0.82** vs human **0.27**), so judges systematically favor S_paper on regenerated ideas while humans favor S15 on the UI session—consistent with different instantiations + single rater.
  - **Claude-sonnet** rows in `summary.json` have **n=0** per topic in that run → κ / *r* not computed; rerun `run_custom_slice_judge.py` if you need Claude in this table.

Caveats (document in paper):

- Single rater → kappa between humans not estimable; pilot is descriptive only.
- Human ratings were on on-the-fly-generated ideas that were not persisted, so judge-human agreement is **topic-level** (different idea instantiations, same systems/topics), not pair-level.

## Remaining before submit (you’re “chilling” on data — this is the short list)

- Confirm **deadline + page limit** for the target workshop.
- **Draft / paste** figures from `workshop_paper_bundle/trajectory_table.csv` + `judge_agreement_deepseek_blind.json` + multi-judge JSON (no new human collection required).
- **Limitations paragraph:** small human *n*, optional “disagreement subsets underpowered,” Haiku primary on `S12 vs S15`, regenerated-ideas caveat for CUSTOM correlation.
- Optional: **`git tag icml-2026-workshop-freeze`** when the PDF matches the frozen results.
- Final proofread and upload.

## Writing (Days 3–8)

- Paper skeleton already exists at `ideas/paper/paper_skeleton.md`
- Fig F1: trajectory win rates with topic-block bootstrap 95% CI
- Fig F2: primary vs blind-deepseek agreement trajectory (kappa)
- Fig F3: multi-judge confusion on S15 vs S_paper + topic-level human-vs-judge scatter
- Results section narrative
  1. Loop plateaus after S15 (Table 1: `judge_agreement_deepseek_blind.json`)
  2. Judge agreement decays on rejected candidates (F2)
  3. Strong judges disagree with loop judge on S_paper (Table 2: multi-judge)
  4. Human pilot: humans vs strong judges vs loop judge (F3 + table)
- Methods
- Limitations (single rater, short horizon, one champion, topic-level human-vs-judge only)
- `REPRODUCE.md` (commands + caveats)
- `export_workshop_paper_bundle.py` → `results/workshop_paper_bundle/*`
- `git tag icml-2026-workshop-freeze`
- Final proofread and submit

## Nice-to-have (skip if behind)

- Additional human raters (already decided: ship with N=60 single-rater pilot)
- Adversarial "tell the agent to cheat" arm (deferred to future work)
- Cross-seed replication (deferred)

