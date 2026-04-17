# Reproducing ICML workshop numbers

All commands assume the **EvoScientist** repo root:

`/Users/dylanwilson/Documents/GitHub/self-play/EvoScientist`

(or your clone path). Run Python scripts from the `ideas/` package directory or prefix paths with `ideas/`.

## Environment

```bash
cd /path/to/EvoScientist
pip install -r ideas/requirements.txt
```

API keys live in `.env` at repo root or under `ideas/` (auto-loaded by the tooling). Re-running judges or backfills **costs tokens**; the committed JSON under `ideas/results/` is the frozen artifact for the paper.

## One-shot export (tables / human summary)

From `ideas/`:

```bash
cd ideas
python3 export_workshop_paper_bundle.py
```

Writes:

- `ideas/results/workshop_paper_bundle/trajectory_table.csv` — primary CIs (from `ci_summary.json`), blind DeepSeek win rates + CIs, κ vs primary where defined
- `ideas/results/workshop_paper_bundle/human_pilot_summary.json` — counts from `human_blind_scores.jsonl`

## Topic-block bootstrap CIs (primary `compare_*.json`)

```bash
cd ideas
python3 bootstrap_ci_workshop.py
```

Output: `ideas/results/ci_summary.json` (default pattern `compare_*.json`).

## Blind judge backfill (DeepSeek on frozen idea pairs)

Requires `ideas/results/<VERSION>/ideas.json` for each side. Regenerate a full 15×5 benchmark (high parallelism) with::

    python3 ideas/godel_loop.py benchmark --version S14 --model gpt-4.1-mini --n-ideas 5 --workers 15

Then e.g. `python3 ideas/blind_backfill.py --model deepseek-chat S12 S14`.

Example (pairwise):

```bash
cd ideas
python3 blind_backfill.py --model deepseek-chat --workers 4 S12 S15
python3 blind_backfill.py --model deepseek-chat --workers 4 S15 S16
```

Outputs: `ideas/results/blind_<A>_vs_<B>__deepseek-chat.json`.

## Primary vs blind agreement (Cohen’s κ, flip rate)

```bash
cd ideas
python3 compute_judge_agreement.py --blind-model deepseek-chat \
  --output results/judge_agreement_deepseek_blind.json
```

(Use `python3 compute_judge_agreement.py --help` if defaults differ in your tree.)

## Multi-judge agreement on S15 vs S_paper

Uses the frozen `publish_eval_*` judge JSON files (no regeneration in this step):

```bash
cd ideas
python3 compute_multi_judge_agreement.py \
  --pairs results/publish_eval_20260413T163806Z/judge_claude-sonnet-4-6.json \
          results/publish_eval_20260413T163806Z/judge_gemini-3-flash-preview.json \
          results/publish_eval_20260413T163806Z/judge_gpt-5.4.json \
          results/publish_eval_20260413T163806Z/judge_deepseek-chat.json \
  --output results/multi_judge_agreement_S15_vs_Spaper.json
```

## Human pilot

- Data: `ideas/results/human_blind_scores.jsonl`
- UI entrypoint (for future collection): `ideas/human_judge_ui.py`

## Fixed benchmark (T1–T15) — same texts as `publish_eval_*`

To collect **human** ratings on the **identical** idea pairs the multi-judge run used (not regenerated CUSTOM text):

```bash
cd ideas
python3 prepare_fixed_benchmark_human_eval.py
# optional: --publish-dir results/publish_eval_<STAMP>
python3 human_judge_ui.py --port 8765
```

Open `http://127.0.0.1:8765/?left=fixed_benchmark_S15&right=fixed_benchmark_S_paper`, enter blind mode, and rate **without** using the custom-topic box. You should see **30** slots (15 topics × 2 `idea_index`).

Pair-level **human vs judge** agreement (after votes exist in `human_blind_scores.jsonl`):

```bash
cd ideas
python3 compute_human_judge_agreement_fixed.py \
  --human-log results/human_blind_scores.jsonl \
  --judge results/publish_eval_20260413T163806Z/judge_claude-sonnet-4-6.json \
          results/publish_eval_20260413T163806Z/judge_gemini-3-flash-preview.json \
          results/publish_eval_20260413T163806Z/judge_gpt-5.4.json \
          results/blind_S15_vs_S_paper__deepseek-chat.json \
  --output results/human_judge_agreement_fixed.json
```

## Human vs judge (topic-level, CUSTOM slice)

Requires `ideas/results/custom_slice_human_vs_judge/summary.json` (produce with `run_custom_slice_judge.py` if missing).

```bash
cd ideas
python3 compute_human_vs_judge.py \
  --summary results/custom_slice_human_vs_judge/summary.json \
  --output results/custom_slice_human_vs_judge/human_vs_judge.json
```

## Release tag (optional)

When the result snapshot is final:

```bash
git tag -a icml-2026-workshop-freeze -m "Freeze workshop results + export scripts"
```

## Caveats called out in the paper

1. `**compare_S12_vs_S15.json**` used **Claude Haiku** as primary judge during the run; later compares use **DeepSeek**. Blind DeepSeek scores are still comparable; κ mixes Haiku vs DeepSeek for that transition only.
2. **S15 vs S18**: primary compare has **45** pairs; blind backfill has **75** — overlap for κ is the smaller protocol; the CSV shows full N per file.
3. `**compare_S12_vs_S14`**: present in `ci_summary.json` but **not** on the champion trajectory; S14 artifacts are incomplete — omit from trajectory figure or label as auxiliary.

**S15 vs S20** is excluded from `ci_summary.json` / `judge_agreement_*.json` (see `ideas/workshop_result_excludes.py`). Regenerate locally if you need that transition.