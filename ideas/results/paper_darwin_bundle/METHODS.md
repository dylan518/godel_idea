# Methods (draft): Darwin Gödel loop for research idea generation

This section describes the **open `ideas/` harness** used in this work. It is **not** the full multi-agent EvoScientist product runtime; it is a **self-improving loop over explicit Python generator strategies** with pairwise LLM judging. Terminology: **champion** = current accepted system; **candidate** = proposed successor.

---

## Suggested split for multi-agent drafting

If you generate this document in pieces, assign **one agent call per block** below (copy only that heading + bullets into the prompt). Merge in order **M1 → M6**.


| Call ID | Section below                               |
| ------- | ------------------------------------------- |
| **M1**  | §1 System overview                          |
| **M2**  | §2 Benchmark tasks and outputs              |
| **M3**  | §3 Pairwise evaluation (primary + blind)    |
| **M4**  | §4 Selection rule, promotion, and logging   |
| **M5**  | §5 Optional evolution modes (SWE / meta)    |
| **M6**  | §6 Multi-judge publish harness + statistics |


---

## 1. System overview (M1)

**Goal.** Maintain a versioned **idea generator** `S{k}` implemented as a Python class in `ideas/systems/S{k}.py`. Each generator exposes the same interface: given a natural-language **topic** string, it returns one or more research ideas in a shared **IDEA_FORMAT** (summary, background, approach, experiment, novelty).

**Champion state.** The active champion name is stored in `ideas/CURRENT_VERSION` (plain text, one line). Available strategies are discovered as `ideas/systems/S*.py` modules exporting a `GENERATOR` singleton.

**Environment.** API keys and optional model overrides load from a repo `.env` (via `ideas/log.py`); shell-exported variables take precedence.

**Relation to external “paper protocol” systems.** Some strategies (e.g. `S_paper`) intentionally mirror **paper-specified idea-search components** (tree search, tournament ranking, fixed rubric); see `ideas/paper_config.py` for pinned hyperparameter names. That alignment supports **evaluation comparability**, not a claim of shipping the closed-source agent stack.

---

## 2. Benchmark tasks and outputs (M2)

**Topic list.** Unless otherwise noted, experiments use `ideas/benchmark_topics.json`: a fixed list of **15** research topics (each with `id`, `topic` string, and optional `domain` metadata).

**Generation run.** For each system version `V`, the runner (`ideas/runner.py`) executes `V` on every topic. For each topic it produces `n_ideas` ideas (default in many runs: **5** for full-loop compares; other studies may use **2**—always report `n_ideas` explicitly).

**Parallelism.** Topic-level parallelism is controlled by `--workers` / `workers=` in `run_system`: at most one concurrent call per topic for idea slots inside a topic (ideas within a topic remain sequential unless a system implements batch generation).

**Artifacts.** Each run writes:

- `ideas/results/<V>/ideas.json` — list of records: `topic_id`, `topic`, `idea_index`, `text`, `system_version`, `model`, timestamp.
- `ideas/results/<V>/run_config.json` — `system_version`, generator `model`, `n_ideas`, `n_topics`, timestamp.

**Caching.** Before re-running, `cache_is_valid()` checks that cached `ideas.json` matches the requested generator `model`, `n_ideas`, and (when provided) topic count; otherwise the run is regenerated.

**Retrieval.** Several strategies call literature / SOTA retrieval (e.g. OpenAlex-backed context) before generation; details are strategy-specific and documented in each `S{k}.py` module docstring.

---

## 3. Pairwise evaluation (primary + blind) (M3)

**Design.** Ideas from champion **A** and candidate **B** are compared **in pairs matched by `(topic_id, idea_index)`**. For each pair, an LLM **judge** scores both ideas and picks a winner.

**Primary judge model.** Default identifier is `deepseek-chat` (`ideas/judge.py`, `JUDGE_MODEL`). Override with environment variable `IDEAS_JUDGE_MODEL` before import or before launching CLI so all modules agree.

**Prompt and rubric.** The judge uses a fixed template (`JUDGE_PROMPT_TEMPLATE` in `ideas/judge.py`): four criteria scored **0–10** each:

1. Novelty
2. Scientific usefulness
3. Experimental clarity
4. Feasibility

The judge returns strict JSON: `scores_a`, `scores_b`, `winner` ∈ {`A`,`B`,`tie`}, `reasoning`. **Positional bias** is mitigated by random **A/B presentation swap** before judging, then mapping scores back to the original (A,B) assignment.

**Tie rule.** The template instructs: declare **tie** if total scores differ by **≤ 2** points.

**Aggregate win rate for B.** Let `wins_a`, `wins_b`, `ties`, `total = wins_a + wins_b + ties`. The reported candidate rate is:

\text{winrateb} = \frac{\text{winsb} + 0.5 \cdot \text{ties}}{\text{total}}.

**Parallel judging.** `compare_systems(..., workers=k)` parallelizes **by topic** (each topic’s pairs are still evaluated sequentially within that topic worker).

**Blind judge (canary).** Independently of the primary verdict used for promotion, an optional **blind** judge (`BLIND_JUDGE_MODEL`, default `gemini-flash-lite-latest`, overridable via `IDEAS_BLIND_JUDGE_MODEL`) samples a subset of topics (`blind_n`) and returns a second set of verdicts. It is **not** used to flip the primary accept/reject bit in the `compare` CLI path; it is used for **agreement diagnostics** and logging.

**Confusion matrix.** `compute_judge_confusion_matrix` aligns primary vs blind verdicts on overlapping `(topic_id, idea_index)` keys and reports agreement and **flip rate**.

**Goodhart-style alerts (logged).** The harness **logs** warnings when:

- `|win_rate_primary − win_rate_blind| > 0.30`, or  
- blind–primary **flip rate** `> 0.30` on compared pairs,

interpreted as “possible gaming of the primary judge.” **In the `evolve` loop implementation we inspected, acceptance is still gated on primary `win_rate_b` alone** (see §4); alerts are **diagnostic**, not automatically enforced as hard rejects unless a separate policy is added. If the paper claims automatic Goodhart rejection, either (a) implement that branch and re-run, or (b) describe the alerts honestly as **monitoring**.

**Judge robustness (implementation).** `judge_pair` retries up to **three** times if JSON parsing fails, with a stricter follow-up instruction. `ideas/systems/base.py` maps model families to providers (OpenAI-compatible for `gpt-*` / `deepseek-*`, Google GenAI for `gemini-*`, Anthropic otherwise); GPT-5.x uses `max_completion_tokens` instead of `max_tokens` where required.

---

## 4. Selection rule, promotion, and logging (M4)

**Head-to-head compare.** `ideas/godel_loop.py compare --candidate <S{k}> ...` benchmarks both systems (respecting cache), runs the **primary** judge on all pairs (with optional **early stopping** tied to the acceptance threshold during compare), then runs the blind judge sample and writes:

`ideas/results/compare_<champion>_vs_<candidate>.json`

**Acceptance threshold (primary).** `ACCEPTANCE_THRESHOLD = 0.55` in `godel_loop.py`. A candidate is considered to **beat** the champion on the benchmark if `win_rate_b > 0.55`.

**Promotion.** `accept <S{k}> [--force]` updates `ideas/CURRENT_VERSION` to `<S{k}>` and appends a JSON line to `ideas/results/evolution_log.jsonl` with timestamp, `from_version`, `to_version`, primary `win_rate`, optional `blind_win_rate`, and optional judge-agreement fields when present. `**--force` skips the guard that requires a prior compare report**; such promotions should be labeled explicitly in the paper as **non-standard** if used.

**Evolution loop (`evolve`).** Autonomous iterations benchmark champion and candidate, judge, optionally blind-judge, save compare JSON, and **if `win_rate_b > 0.55`** promote and log; otherwise reject and keep the champion.

**Important scope statement for the paper.** Describe precisely **which commands** produced each result (manual `compare`+`accept` vs `evolve` vs SWE loop) and **never mix** runs that used different `n_ideas`, judge models, or topic files without a clear **protocol row** in a table.

---

## 5. Optional evolution modes (M5)

**SWE-style evolution (`swe-evolve`).** An alternate loop uses a stronger **meta-model** (configured in `ideas/swe_agent.py`, typically Claude Sonnet-class) to propose **code edits** to the strategy file or related assets, followed by mini-evaluations and full benchmarks. This is **engineering-time expensive**; methods should state model IDs and round counts used.

**Meta-generation (`generate`).** A separate path can author a new `S{n}.py` from scratch; treat as optional background if not central to the Darwin story in this submission.

---

## 6. Multi-judge publish harness and statistics (M6)

**Purpose.** For **supporting** claims (not the core Darwin thesis unless framed carefully), we batch **multiple strong judge models** on the **same** `ideas.json` pairs.

**Script.** `ideas/publish_multi_judge.py`:

1. Loads topics (default `benchmark_topics.json` or `--topics`).
2. Runs generator **A** and **B** into a timestamped directory (or `--out`).
3. For each judge model in `--judges` (comma list), runs `compare_systems` on the **same** idea records with per-judge topic parallelism (`--workers` / auto `min(32, n_topics)`).
4. By default, **judge models run in parallel** (thread per judge) unless `--sequential-judges`.
5. Writes `judge_<model>.json` per judge and `summary.json` with per-judge win counts, `win_rate_b`, **Wilson 95% intervals** on the tie-adjusted rate, and pairwise **agreement** between judges on overlapping verdict keys.

**Repair / sync.** `ideas/rerun_missing_verdicts.py` re-judges only missing `(topic_id, idea_index)` keys for one judge file and can `--sync-summary-only` to refresh aggregates.

**Heuristic cost.** The harness may record **rough USD estimates** from token-rate heuristics; **true billing** should come from provider dashboards.

---

## Reproducibility checklist (for appendix)

1. Git commit hash (`git rev-parse HEAD`) at analysis time.
2. `ideas/CURRENT_VERSION` and champion/candidate versions.
3. Exact `benchmark_topics.json` path (or custom JSON) and `n_ideas`.
4. Generator `model` from each `run_config.json`.
5. Primary judge: `IDEAS_JUDGE_MODEL` or default; blind: `IDEAS_BLIND_JUDGE_MODEL` / `blind_n`.
6. Paths: `ideas/results/compare_*.json`, `ideas/results/evolution_log.jsonl`, `ideas/results/publish_eval_*/`.
7. Note any `**--force` accepts** or **partial compares** (fewer than `n_topics × n_ideas` pairs).

---

## Limitations (methods-level, one paragraph)

All automatic quality signals are **LLM-judge-dependent**; multi-judge agreement reduces but does not remove that dependence. The loop **selects for performance under the judge and topic distribution**; generalization to other domains or human panels is not implied. **Goodhart alerts** flag disagreement but (in the code paths above) **do not automatically veto** a promotion that clears the **primary** 55% rule—state this clearly or change the code before claiming stricter governance.