# Research Agenda — Living Document

Last updated: 2026-04-05
Current champion: S5 (multi-perspective critique, 58% vs S1)

## Core insight so far
The judge rewards **specificity and measurability** as much as novelty.
Vague ambitious ideas consistently lose to concrete actionable ones.
Adding more critique rounds helps marginally but hits diminishing returns fast.

## Active hypotheses (to test)

### H1: Constraint-first design (S6 — IN PROGRESS)
Work backward from what's measurable before generating any idea.
Prediction: >60% vs S5. Addresses the vagueness failure mode directly.

### H2: Literature gap mining
Explicitly enumerate what adjacent work has NOT addressed, then target that gap.
Could be combined with constraint-first.
Status: not yet tested as standalone

### H3: Adversarial stress-testing
Generate idea, then actively try to falsify/kill it before finalizing.
Forces the generator to pre-empt the judge's objections.
Status: not yet tested

### H4: Decompose the scoring rubric
Separate optimisation passes for each of the 4 judge dimensions,
then synthesise. (S6 tests a version of this.)
Status: partially tested in S6

### H5: Analogical transfer
Find a well-solved problem in a different domain, map the solution structure.
Might help with novelty specifically.
Status: not yet tested

### H6: SWE-agent style self-editing
Instead of generating new standalone files, have the meta-agent
iteratively edit the current champion's prompts and pipeline.
Multiple edit-test loops per iteration.
Status: PLANNED — major redesign

## Rejected hypotheses
- More critique rounds (S4): diminishing returns, 40%
- More reviewer personas (S5): helpful but not transformative, 58%
- Structured decomposition (S3): Goodhart alert, judge gaming suspected

## Open questions
1. Is S5's 58% a ceiling for critique-based approaches?
2. Does rubric-aware revision (S6) generalize or just overfit to judge?
3. What's the right number of LLM calls per idea? (S5=5, S6=7)
4. Should we use a stronger generator model for hard topics?

## Experiment log
| Date | Experiment | Result | Notes |
|------|-----------|--------|-------|
| 2026-04-02 | S0→S1 | 100% | Self-critique is a massive win |
| 2026-04-02 | S1 vs S2 | 8% | Tournament approach badly overfit |
| 2026-04-03 | S1 vs S3 | 54% (Goodhart) | Decomposition games Claude |
| 2026-04-03 | S1 vs S4 | 40% | More critique rounds don't help |
| 2026-04-04 | S1 vs S5 | 58% | Multi-perspective: best yet, narrow win |
| 2026-04-05 | S5 vs S6 | IN PROGRESS | Constraint-first approach |

## Next actions (for autonomous sessions)
1. Wait for S6–S20 evolve run to complete
2. Analyse which iteration's approach generalises best
3. If S6 wins: explore combining constraint-first + adversarial stress-test
4. If S6 loses: test H3 (adversarial) as S7 explicitly
5. After S10: reflect on pattern — update this document
6. Consider SWE-agent redesign (H6) after S20 baseline
