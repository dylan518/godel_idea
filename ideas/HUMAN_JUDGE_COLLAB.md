# Human Blind Judge Collaboration Notes

## Purpose

This mini UI enables **blind side-by-side human scoring** of idea quality across generator versions (e.g. `S_paper` vs `S15`) with optional custom-topic generation.

Primary goal: get trustworthy human preferences on idea quality while minimizing version-label bias.

## Where the tool lives

- Server: `ideas/human_judge_ui.py`
- Frontend: `ideas/human_judge_static/index.html`
- Vote log: `ideas/results/human_blind_scores.jsonl`

## How to run

From repo root:

```bash
python3 ideas/human_judge_ui.py --port 8765
```

Open:

- `http://127.0.0.1:8765/`
- Optional preselect versions: `http://127.0.0.1:8765/?left=S_paper&right=S15`

## Current workflow (important)

1. Pick two generator runs.
2. (Optional) Enter custom topic text and set `# ideas / slots` (3/5/7).
3. Enter blind mode.
4. Click **Next pair** to load one blind A/B comparison.
5. Vote: `A`, `B`, or `Tie` (+ optional rubric sliders).
6. Repeat `Next pair` for more slots.

## Blindness model

- Setup labels are hidden after entering blind mode.
- Mapping of selected runs to internal left/right is shuffled per session.
- A/B side assignment is randomized per pair (`swap`).
- Version identities are revealed only after vote submission.

## Custom topic behavior

- If custom topic text is filled, `Next pair` requests `custom_topic` and auto-generates on demand if not cached.
- Generation creates **N matched slots per run** (same `idea_index` matched across both runs).
- UI shows one slot at a time; click `Next pair` to see another slot.

## Data format for votes (`human_blind_scores.jsonl`)

Each line includes:

- `ts`
- `topic_id`, `topic`, `domain`, `idea_index`
- `left_version`, `right_version`, `swap`
- `winner_label` (`A`/`B`/`tie`)
- `winner_version`, `loser_version`
- `scores_a`, `scores_b` (novelty/usefulness/clarity/feasibility)

## First relevant recent result

Latest observed custom-topic vote:

- `topic`: "self play RL and dual play with llms"
- `topic_id`: `CUSTOM:self play rl and dual play with llms`
- compared `S15` vs `S12` (`swap=false`)
- winner label `B` => winner version `S12`

## Known caveats

- First custom generation call may take minutes (LLM latency).
- Browser hard refresh may be needed after UI updates.
- Blindness is UX-level; avoid inspecting network payloads during judging if strict blinding is required.

## Good collaboration asks from here

- Aggregate and summarize human vote logs by version pair.
- Add dedup/coverage tracking so each slot is judged once before repeats.
- Add export command (`csv`/`json`) for downstream analysis.
- Add inter-rater support (multiple human judges + agreement metrics).

