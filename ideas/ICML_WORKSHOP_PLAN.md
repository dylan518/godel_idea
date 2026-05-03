# ICML 2026 Workshop plan — EvoScientist / Gödel loop

*Last revised: 2026-04-16. **Deadline: Apr 24, 2026 (confirm on workshop site).** 8-day sprint.*

---

## Core claim

We characterize how LLM-judge reliability evolves under self-improvement pressure. Across the evolutionary trajectory of a self-editing idea generator (S_paper → S12 → S15 → {S16, S17, S19}; reboot baseline `S_sota.py`), we measure agreement between a primary judge (DeepSeek), a blind judge (Gemini), and human raters, and report what the trajectory looks like.

The paper is **descriptive and methodological**. No adversarial experiments, no seed reruns, no large-scale human study. We use evidence already in hand plus a small set of cheap extensions.

---

## Why this scope works for a workshop

- Existing data: S_paper→S12, S12→S15, S15-vs-{S16, S17, S19} all at 75 pairs
- S12→S15 already shows ~33% primary/blind agreement — enough signal to build a trajectory around
- Blind-judge backfill + small human pilot are the only new evidence needed
- 4–6 page workshop paper, not a full main-track submission
- Acceptance probability: **~70–80%** at most ICML 2026 workshops with thematic fit (LLM-as-judge, foundation models, self-improvement)

---

## Cuts from prior plan

- ❌ Adversarial agent run (R2/R3)
- ❌ Seed reruns
- ❌ 240-rating human study (reduce to ~75–100)
- ❌ Cheat ablations, graduated attacks
- ❌ Loop ablations (memory, components)
- ❌ SWE narrative table / case studies

---

## Evidence (minimum to submit)

### E1. Trajectory win rates with CIs

Win rate vs `S_paper` for {S12, S15, S16, S17, S19} under primary judge, with block-bootstrap 95% CI by topic. Existing compares + one new script.

### E2. Blind-judge agreement trajectory

Backfill Gemini-flash-lite on S15-vs-{S16, S17, S19} (S_paper→S12 and S12→S15 already done). Report primary/blind κ + flip rate per transition.

### E3. Three-way agreement pilot

Human scoring on ~20 unique pairs × 3–5 raters each ≈ 60–100 ratings. Allocation:

- 10 pairs S15-vs-S_paper (plateau-vs-baseline — main comparison)
- 5 pairs where primary and blind disagree (highest info per vote)
- 5 pairs where primary and blind strongly agree (control)

Report per judge: win rate on same items; pairwise κ (primary-blind, primary-human-majority, blind-human-majority); Fleiss's κ across raters if ≥3 per pair.

### E4. Reproducibility

Short section + `REPRODUCE.md`: tag, pinned model IDs, one command to regenerate headline numbers.

---

## 8-day sprint

### Day 1 (Apr 16, today)

- Confirm ICML workshop deadline + page limit; update this doc
- Clean tree, rebase, `git tag icml-2026-workshop-freeze`
- Write `ideas/bootstrap_ci.py`; apply to all `compare_*.json` → `results/ci_summary.json`
- Add `rater_id` to `human_judge_ui.py`; migrate existing 68 rows to `rater_id="dylan_pilot"`
- Kick off Gemini blind-judge backfill on S15-vs-{S16, S17, S19} (background, hours)

### Day 2

- Verify backfill output; compute κ + flip rate → `results/judge_agreement.json`
- Select 20 eval pairs (10+5+5 allocation); generate per-rater assignment
- First human sessions: target 5 raters × 5 = 25 ratings

### Day 3

- Continue human sessions: cumulative 10 raters × 5 = 50 ratings
- Draft paper skeleton (abstract/intro/method/results/discussion/limits)
- Draft Figure F1 (trajectory + CIs)

### Day 4

- Continue human sessions: cumulative 15 raters × 5 = 75 ratings
- Draft Figure F2 (judge-agreement trajectory)
- Write methods section
- Short related work (LLM-as-judge: MT-Bench / Arena-Hard; self-improvement: AI Scientist, Darwin-Gödel Machine)

### Day 5

- Final human session if behind target: cumulative ~20 raters × 5 = 100 ratings
- Compute three-way agreement matrix; draft Figure F3
- Write results section
- Draft abstract + intro

### Day 6

- First full draft complete
- Limitations paragraph (generator-model family, judge-model family, human N, single domain, no seed reruns)
- Polish figures
- Self or collaborator red-team cold-read

### Day 7

- Address red-team feedback
- Final numbers pass — every point estimate has CI or explicit N
- `REPRODUCE.md`: tag + commands + expected runtime
- Format check against workshop template

### Day 8 (Apr 24, submit)

- Final proofread
- Submit

---

## "Done enough to submit" gate

- Claim stated in one sentence, appears in abstract + intro
- Every headline number has CI or explicit N
- Blind-judge backfill complete on all cited transitions
- Human N honestly reported with rater count and per-stratum breakdown
- 68 prior votes clearly labeled "Pilot A, custom topics, single rater"
- Limitations covers: generator-model family, judge-model family, human N, scope (scientific idea generation), no adversarial validation
- Reproducibility section has frozen tag + one-line command

---

## Risks (short list)

1. **Blind backfill cost/time overrun.** Fallback: backfill S15-vs-S19 only (most-rejected transition); report others at current N.
2. **Human recruitment slower than planned.** Fallback: 15 raters × 5 = 75 ratings is sufficient for a pilot-labeled section.
3. **Deadline slip.** Fallback: aim for a different ICML workshop or NeurIPS workshop with later deadline.

---

*Supersedes `NEURIPS_PLAN.md` for this submission cycle. Pair with `PROF_STATUS_REPORT.md`.*