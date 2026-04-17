# Darwin Gödel loop — paper prep bundle

Generated from repo state `git rev = 24508b3f0c1aeff6b4f14b36b063ec69df9a06f5`; `CURRENT_VERSION` = `S15`.

## 1. Evolution trajectory (accept chain)


| time                | from      | to      | primary win_rate (B) | blind              | notes                                                                          |
| ------------------- | --------- | ------- | -------------------- | ------------------ | ------------------------------------------------------------------------------ |
| 2026-04-06T13:19:00 | bootstrap | S_sota  | —                    | —                  | fresh loop from S_sota                                                         |
| 2026-04-06T19:48:15 | S_sota    | S12     | 0.8933333333333333   | 0.7                |                                                                                |
| 2026-04-06T21:36:36 | S12       | S15     | 0.5533333333333333   | 0.6666666666666666 |                                                                                |
| 2026-04-12T20:52:52 | S15       | S_paper | —                    | —                  |                                                                                |


**Reads:** `ideas/results/evolution_log.jsonl` (+ `trajectory.json` here).**

## 2. Head-to-head compares on disk (non-archive)


| report                            | champion (A) | candidate (B) | B win rate | judged |
| --------------------------------- | ------------ | ------------- | ---------- | ------ |
| `results/compare_S12_vs_S14.json` | S12          | S14           | 0.5400     | 75     |
| `results/compare_S12_vs_S15.json` | S12          | S15           | 0.5533     | 75     |
| `results/compare_S15_vs_S16.json` | S15          | S16           | 0.4867     | 75     |
| `results/compare_S15_vs_S17.json` | S15          | S17           | 0.3333     | 75     |
| `results/compare_S15_vs_S18.json` | S15          | S18           | 0.4333     | 45     |
| `results/compare_S15_vs_S19.json` | S15          | S19           | 0.1067     | 75     |
| `results/compare_S15_vs_S20.json` | S15          | S20           | 0.7500     | 12     |


## 3. Mechanisms (from system module docstrings)

### S_sota

- **Call budget hint:** LLM calls per idea: 3 (candidates) + tournament (2-4 comparisons) + 5 (critique) = ~12

```
S_sota: SOTA-grounded + tournament + multi-perspective critique.

This is the new strong baseline that replaces S0 as the starting point
for the self-improvement loop. It combines three components:

1. SOTA RETRIEVAL: Fetches 5 recent Semantic Scholar papers on the topic
   before generating anything. The generator sees actual recent work and
   must produce ideas that go beyond it — grounding novelty claims.

2. MULTI-PERSPECTIVE CRITIQUE (from S5, current champion): draft → three
   reviewer personas (experimentalist, theorist, skeptic) → synthesise →
   revise. Catches orthogonal failure modes.

3. INTRA-TOPIC TOURNAMENT: Generates 3 independent candidates using the
   SOTA context, runs a 2-round Elo tournament to select the best one,
   then applies the S5-style critique to that winner. Only the tournament
   winner advances to cross-system comparison — raising quality before
   the "finals".

This is the EvoScientist-inspired baseline: tournament + SOTA context
+ multi-perspective critique, all in one pipeline.

```

### S12

- **Call budget hint:** 1. HYPOTHESIS GENERATION: One LLM call produces 5 sharp, testable scientific

```
S12_r3: Hypothesis-First Adversarial Loop.

Replaces the top-down tree/tournament with a FALSIFIABLE-HYPOTHESIS-FIRST loop:

1. HYPOTHESIS GENERATION: One LLM call produces 5 sharp, testable scientific
   hypotheses about the topic (e.g., "scaling laws break under data-constrained
   regimes because X"). Starting from *claimed truths about the world* rather
   than technique names.

2. ADVERSARIAL ATTACK (parallel, 5 calls): Each hypothesis gets independently
   attacked by an adversarial critic that probes assumption violations, dataset
   biases, theoretical gaps, and practical limitations. Each attack also forces
   a refined/alternative variant.

3. HYPOTHESIS SELECTION (1 call): Aggregate attack+revision pairs and pick the
   strongest surviving hypothesis — the one whose refined version is most
   concrete, novel, and falsifiable.

4. IDEA CONSTRUCTION (1 call): Design the full experimental idea around
   proving/disproving the surviving hypothesis. Forces concrete datasets,
   baselines, and metrics because the falsification target is explicit.

```

### S15

- **Call budget hint:** 1. HYPOTHESIS GENERATION: One LLM call produces 5 sharp, testable scientific

```
S12_r3: Hypothesis-First Adversarial Loop.

Replaces the top-down tree/tournament with a FALSIFIABLE-HYPOTHESIS-FIRST loop:

1. HYPOTHESIS GENERATION: One LLM call produces 5 sharp, testable scientific
   hypotheses about the topic (e.g., "scaling laws break under data-constrained
   regimes because X"). Starting from *claimed truths about the world* rather
   than technique names.

2. ADVERSARIAL ATTACK (parallel, 5 calls): Each hypothesis gets independently
   attacked by an adversarial critic that probes assumption violations, dataset
   biases, theoretical gaps, and practical limitations. Each attack also forces
   a refined/alternative variant.

3. HYPOTHESIS SELECTION (1 call): Aggregate attack+revision pairs and pick the
   strongest surviving hypothesis — the one whose refined version is most
   concrete, novel, and falsifiable.

4. IDEA CONSTRUCTION (1 call): Design the full experimental idea around
   proving/disproving the surviving hypothesis. Forces concrete datasets,
   baselines, and metrics because the falsification target is explicit.

```

### S_paper

- **Call budget hint:** 2. `build_idea_tree` — L1→L2→L3 JSON tree + review (4 LLM calls), once per topic

```
S_paper: IdeaTreeSearch + Elo wired to repo-root ``skills/`` (idea-tournament + research-ideation).

Uses ``ideas/idea_tournament/`` Python (tree_search + tournament). Prompts load canonical
Markdown from ``skills/idea-tournament/references/*.md`` and
``skills/research-ideation/references/literature-tree.md`` (same sources as Claude Code).

Per benchmark topic (``generate_idea`` × n_ideas uses thread-local cache):
  1. OpenAlex SOTA context (same retrieval as S_sota / S15)
  2. ``build_idea_tree`` — L1→L2→L3 JSON tree + review (4 LLM calls), once per topic
  3. ``run_tournament_ranked`` — Swiss Elo on leaf dicts (paper-style judge), once per topic
  4. Each idea slot: expand one ranked leaf to ``IDEA_FORMAT`` (1 LLM call each)

``generate_batch`` runs steps 1–3 once, then step 4 for ``n`` leaves (fresh-eval path).

This is **not** the LangGraph EvoScientist agent from the main package; it is the
paper's *idea-search* subroutine, runnable under ``godel_loop.py`` / ``runner.py``.

LLM calls (typical): ~4 (tree) + tournament pairs + n (expand). Re-benchmark vs S15
after editing ``skills/…`` rubrics or ``idea_tournament/prompts.py``.
```

## 4. Baselines vs champion (supporting eval, not thesis)

- **Main loop accepts:** see trajectory — `S_sota`→`S12`→`S15` used primary + blind in log.
- `**S_paper`:** paper-aligned idea search; see `publish_eval_`* / `judge_*.json` if present.
- **Protocol mismatch warning:** some `compare_*.json` files use **75** judged pairs (e.g. `n_ideas=5`); `publish_eval_`* uses **30** pairs (`n_ideas=2`). Do not mix in one table without labeling.

### Rough LLM budget per idea (from docstrings — verify in code)


| system    | hint (per idea)                                                                        |
| --------- | -------------------------------------------------------------------------------------- |
| `S_sota`  | LLM calls per idea: 3 (candidates) + tournament (2-4 comparisons) + 5 (critique) = ~12 |
| `S12`     | 1. HYPOTHESIS GENERATION: One LLM call produces 5 sharp, testable scientific           |
| `S15`     | 1. HYPOTHESIS GENERATION: One LLM call produces 5 sharp, testable scientific           |
| `S_paper` | 2. `build_idea_tree` — L1→L2→L3 JSON tree + review (4 LLM calls), once per topic       |


Multiply by topics×ideas for a **benchmark generation** cost envelope; add judge pairs × judge calls for evaluation.

### Latest `publish_eval` summary (embedded)

```json
{
  "out_dir": "results/publish_eval_20260413T163806Z",
  "heuristic_spend_usd_total": 0.8382,
  "heuristic_spend_note": "includes generation guess + per-judge blocks",
  "judges": {
    "gemini-3-flash-preview": {
      "skipped": false,
      "report_path": "results/publish_eval_20260413T163806Z/judge_gemini-3-flash-preview.json",
      "wins_a": 19,
      "wins_b": 1,
      "ties": 10,
      "total_judged": 30,
      "win_rate_b": 0.2,
      "heuristic_usd_block": 0.0093,
      "wilson_95_win_rate_b": [
        0.09504978102401235,
        0.37306047381027446
      ]
    },
    "claude-sonnet-4-6": {
      "skipped": false,
      "report_path": "results/publish_eval_20260413T163806Z/judge_claude-sonnet-4-6.json",
      "wins_a": 16,
      "wins_b": 2,
      "ties": 12,
      "total_judged": 30,
      "win_rate_b": 0.26666666666666666,
      "heuristic_usd_block": 0.3375,
      "wilson_95_win_rate_b": [
        0.14182495553910618,
        0.4444830204431169
      ]
    },
    "gpt-5.4": {
      "skipped": false,
      "report_path": "results/publish_eval_20260413T163806Z/judge_gpt-5.4.json",
      "wins_a": 20,
      "wins_b": 2,
      "ties": 8,
      "total_judged": 30,
      "win_rate_b": 0.2,
      "heuristic_usd_block": 0.4725,
      "wilson_95_win_rate_b": [
        0.09504978102401235,
        0.37306047381027446
      ]
    }
  },
  "judge_pair_agreement": {
    "gemini-3-flash-preview": {
      "claude-sonnet-4-6": 0.8,
      "gpt-5.4": 0.7
    },
    "claude-sonnet-4-6": {
      "gpt-5.4": 0.6666666666666666
    },
    "gpt-5.4": {}
  },
  "default_primary_judge_env": "gpt-5.4"
}
```

## 5. Rejection reason samples (candidate lost)

See `rejection_reason_samples.json` (25 rows).

## 6. Replication / next runs (manual)

```bash
# Second multi-judge bundle (new timestamped dir)
python3 ideas/publish_multi_judge.py --max-spend-usd 10 --workers 15

# Fix missing Gemini pairs + refresh summary (if needed)
python3 ideas/rerun_missing_verdicts.py --publish-dir ideas/results/publish_eval_<STAMP>
```

## 7. Paper outline (thesis = loop)

1. **Introduction:** Darwin Gödel loop — self-improving generators under pairwise selection.
2. **Method:** champion/candidate/compare/accept; blind judge rule; SWE/meta optional.
3. **Results — dynamics:** trajectory figure from `trajectory.json` + plateaus.
4. **Results — mechanisms:** per-hop docstrings / diffs (`mechanisms_from_docstrings.json` + git).
5. **Results — baselines:** `S_paper` / `S_sota` vs champion with explicit protocol rows.
6. **Results — failures:** rejection samples + Goodhart episodes from logs.
7. **Cost:** calls per idea from docstrings; $ heuristic from publish harness.
8. **Discussion:** limits of self-improvement; judge coupling.
9. **Appendix:** frozen `compare_*.json`, `publish_eval_`*, judge outputs.

