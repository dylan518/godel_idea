# Noisy Judges, Consistent System-Level Preference in Self-Improving Scientific Idea Generators: A Trajectory Study

*ICML 2026 Workshop submission draft — target deadline Apr 24, 2026. 4–6 pages.*

# Abstract

Self-evolving agent systems rely on evaluation signals to guide optimization, but in subjective domains—where judgments are noisy and inconsistent—it remains unclear whether reliable improvement is possible at all. Individual evaluations in such settings frequently disagree, raising the concern that optimization pressure may amplify noise rather than produce meaningful progress.
We investigate this question in a scientific idea-generation loop where generator programs iteratively compete against a champion system under pairwise judging. Starting from S_sota, the EvoScientist idea-generation pipeline (Lyu et al., 2026), we evolve successive generator variants using only judge-based feedback. Across both automated judges and expert human raters, individual pairwise judgments exhibit substantial disagreement and low consistency at the item level. However, despite this instability, aggregate system-level preferences remain directionally consistent across independently constructed evaluators. The final evolved system (S15) is preferred over the baseline across three independent automated judges on a frozen evaluation slice and in a blind human pilot (40 of 60 comparisons).
These results demonstrate that reliable optimization is possible in subjective domains even when individual evaluations are unreliable. The key mechanism is that system-level preference signals remain stable under aggregation, allowing self-evolving agents to extract usable optimization pressure from noisy judgments. This finding establishes a practical foundation for applying self-evolving agent methods to domains previously considered too subjective for reliable automated improvement.
# Introduction

Large language models are increasingly used to automate parts of scientific discovery, including hypothesis generation, experimental planning, and manuscript drafting. Recent systems such as VirSci, The AI Scientist, Gemini Co-Scientist, and EvoScientist suggest that multi-agent research pipelines can generate plausible scientific ideas and iteratively refine them (Su et al., 2024; Lu et al., 2024; Gottweis et al., 2025; Lyu et al., 2026). The open question is whether these systems can improve subjective idea quality in a way that survives evaluation stronger than the same judge signal that guided optimization.

This question matters because scientific idea generation remains difficult even when model outputs are fluent and well structured. Prior work finds that language models can propose novel-looking ideas, but reliably producing expert-level contributions is still unresolved (Si et al., 2024). In practice, improving agentic research systems still depends heavily on manual prompt and workflow tuning.

Self-improving systems offer an alternative. Rather than hand-editing a workflow indefinitely, one can evolve the workflow itself and retain only variants that outperform the current champion. The Darwin Gödel Machine provides a concrete template for this style of optimization over code and agent structure (Zhang et al., 2025).

We therefore study a narrower question: what survives stronger evaluation once judge noise is made explicit? Our answer has two parts. At the system level, every evaluator we ran — three independently-built automated judges on a frozen slice, and eight human raters on a disjoint topic pilot — prefers the evolved generator over the reboot baseline. At the item level, those same evaluators disagree substantially with each other and with humans on which specific ideas win any given topic, and single-judge single-pass labels are not stable under rerun. We argue this combination is informative rather than contradictory: it shows that aggregate system-level preference can be extracted reliably from a subjective domain even when per-item judgments are noisy, provided the evaluation protocol averages across enough pairs and enough independent evaluators.

**Contributions.**

- We present a trajectory study of self-improving scientific idea generators trained entirely with pairwise LLM-judge feedback, including a human blind pilot and a multi-judge frozen-slice comparison.
- We show that system-level preference for the evolved champion is consistent across three independent automated judges and an eight-rater human pilot, even though pair-level agreement between those same evaluators is weak.
- We argue that this combination — stable aggregate signal with noisy per-item labels — is the operating regime self-improvement loops need in subjective domains, and we identify the protocol practices (multiple judges, repeated passes, topic-block uncertainty) that make aggregate signal legible.

## 2. System and benchmark

### 2.1 Evolution loop (Figure 3)

We study the open `**ideas/*`* self-improvement harness: versioned **generator programs** compete on a fixed topic suite under an LLM judge. The loop maintains a current **champion** system A and a proposed **candidate** B. A candidate may be produced through the SWE-style meta-edit loop, but promotion is always decided by the same outer evaluation protocol.

Figure 3 summarizes the control flow. The benchmark contract fixes the topics and generator model. The current champion and the proposed candidate are both run on the same topic set to produce matched idea pairs. A primary pairwise judge scores those pairs and computes the tie-adjusted candidate win rate \widehat{\mathrm{WR}}_B. The candidate is promoted only if \widehat{\mathrm{WR}}_B > 0.55; otherwise the champion is retained. In the trajectory reported here, the champion ends at `**S15`**, the reboot baseline is `**S_sota**` (generator module `S_sota.py`), and the generator model is held fixed at `**gpt-4.1-mini**`.

Figure 3 — Gödel/Darwin harness: benchmark, SWE edit loop, full eval, judge, promotion gate.

## 3. Methods

### 3.1 Study design and evolution loop

We study a self-improvement harness for scientific idea generation. Each system version is a generator program that maps a topic to one or more research ideas in a shared output format. Section 2 introduced the outer loop; this section specifies the benchmark, judging, uncertainty estimates, and human study used in the analysis.

Each evolution step follows the same procedure:

1. Fix a benchmark topic set.
2. Generate matched idea sets from the champion and candidate.
3. Run pairwise judging on matched idea pairs.
4. Promote the candidate only if its tie-adjusted win rate exceeds a fixed threshold; otherwise retain the champion.

This paper studies the `ideas/` harness rather than the full EvoScientist application runtime.

### 3.2 Benchmark and generator outputs

The fixed evolution benchmark is `benchmark_topics.json`, a 15-topic suite used throughout the trajectory analysis. For each topic, a generator produces multiple ideas in a common **IDEA_FORMAT** consisting of a one-sentence idea summary plus background, approach, experiment, and novelty fields. Different system versions vary in prompting, retrieval, critique, and tournament structure, but all are judged on the same final text output.

### 3.3 Pairwise judging and promotion

For each matched (\texttt{topic_id}, \texttt{idea_index}) pair, the judge scores both sides on four 0-10 dimensions: novelty, scientific usefulness, experimental clarity, and feasibility. The prompt asks the judge to return structured scores for both ideas and a `winner` label in \texttt{A}, \texttt{B}, \texttt{tie}, with `tie` instructed when the total scores differ by at most two points. To reduce position bias, the harness randomizes which system appears as side A versus B before each call and maps outputs back to the true assignment afterward.

Let W_B, W_A, and T denote candidate wins, champion wins, and ties over N judged pairs. We report the tie-adjusted candidate win rate

\widehat{\mathrm{WR}}_B = \frac{W_B + \tfrac{1}{2} T}{N}.

Promotion depends only on the **primary** judge: the candidate is accepted iff \widehat{\mathrm{WR}}_B > 0.55.

### 3.4 Uncertainty and diagnostic analyses

Because verdicts are clustered by topic, confidence intervals for \widehat{\mathrm{WR}}_B use a topic-block bootstrap: resample topics with replacement, pool all pairs attached to sampled topics, and recompute the tie-adjusted win rate. We report the 2.5 and 97.5 percentiles of the bootstrap distribution.

Our post-hoc diagnostics focus on protocol behavior rather than additional trajectory evidence. In particular, we analyze winner-versus-score consistency and cross-judge variation on frozen idea pairs, and we discuss repeated judging as a protocol-level concern (§4.7).

### 3.5 Human blind evaluation

Human evaluation is separate from the fixed 15-topic evolution benchmark. We collected blind pairwise judgments on 12 custom thesis-style topics aligned with raters' areas of expertise. Each topic contributes five comparisons between `S15` and the reboot baseline (`S_sota`), giving 60 total human judgments. Raters score the same four rubric dimensions used by the LLM judge and also record an overall preferred side.

The human pilot is designed as an external probe, not as a definitive validation study. Eight raters split disjoint topic slices, so no two raters evaluate the same matched pair and the study does not measure inter-rater reliability on identical items.

### 3.6 Scope of the study

The study is descriptive: one realized post-reboot trajectory under a fixed harness. We do not claim adversarial robustness to judge gaming, broad transfer beyond scientific idea generation, or multi-seed replication of the full meta-edit process.

## 4. Results

### 4.1 Headline: system-level preference is consistent across evaluators even where item-level agreement is weak

Before working through the trajectory, we state the main empirical finding: on the `S15` vs. `S_sota` comparison — the question of whether the evolved system improves on the reboot baseline — every evaluator we ran prefers `S15` at the system level. Three independent automated judges on a frozen 64-pair slice all assign $\widehat{\mathrm{WR}}_B$ for `S_sota` well below 0.5 (Figure 2, left), and eight human raters on a separate 12-topic blind pilot prefer `S15` in 40 of 60 pairwise comparisons. No judge and no rater group in our study produces an aggregate verdict in favor of `S_sota`.

This system-level consistency coexists with substantial disagreement at the item level, which §4.3–§4.5 document in detail: judges disagree with each other on individual pair labels, automated-judge and human topic-level win rates are near-zero-correlated, and rerunning the same judge flips a non-trivial share of verdicts. We treat this combination as the substantive finding rather than as a tension. Three independently-built automated judges converging on the same directional verdict — and a human pilot landing on the same side on disjoint topics — is a stronger claim than any one evaluator alone, precisely because the per-item signals each evaluator produces are noisy. System-level preference is recoverable even when item-level labels are not reliable.

### 4.2 Trajectory under the promotion gate (Figure 1)

Figure 1 plots the tie-adjusted candidate win rate $\widehat{\mathrm{WR}}_B$ for every promotion decision in the post-reboot trajectory, under both the training judge (DeepSeek compare session, filled markers) and a blind re-evaluation judge (`gemini-flash-lite-latest`, open markers), with 95% topic-block bootstrap intervals. Two transitions clear the 0.55 promotion gate and are accepted: `S_sota→S12` ($\widehat{\mathrm{WR}}_B \approx 0.89$ train, $\approx 0.70$ blind eval) and `S12→S15` ($\widehat{\mathrm{WR}}_B \approx 0.55$ train, $\approx 0.49$ blind eval). Subsequent candidates — `S12→S14`, `S15→S13`, `S15→S16`, `S15→S17`, `S15→S18`, `S15→S19` — are all rejected, with training-judge point estimates clustered between roughly 0.10 and 0.55.

Two features are worth flagging. First, the large initial jump at `S_sota→S12` is the dominant source of the gap between `S15` and the reboot baseline; the later `S12→S15` promotion is marginal on the training judge and sits just below 0.50 on the blind eval, with a confidence interval that spans both sides of the line. Second, the blind-eval and train-judge points disagree in magnitude on several transitions, even when they agree in direction. Both observations are consistent with the §4.1 headline — `S15` is preferred over `S_sota` at the system level — while showing that the within-chain evidence is not uniformly strong step-by-step.

### 4.3 Batch margins on the frozen publish-eval slice (Figure 2, left)

On the 64-pair publish-eval intersection, all three automated judges assign $\widehat{\mathrm{WR}}_B$ for `S_sota` below 0.5: approximately 0.28 for Gemini 3 Flash Preview, 0.23 for GPT-5.4, and 0.32 for DeepSeek Chat. Equivalently, each judge's batch win rate for `S15` sits between roughly 0.68 and 0.77. The batch margins are clearly not ties, and the three judges — despite being built by different organizations and differing substantially at the pair level — converge on the same direction and on comparable magnitudes. This is the strongest single readout in favor of the evolved system: three independent judges, one frozen slice, same verdict.

### 4.4 Judge–judge agreement (Figure 2, right)

The unanimity of batch direction does not extend to pair-level agreement. Cohen's $\kappa$ between judges on the same 64 pairs is low across the board: $\kappa \approx 0.16$ for DeepSeek↔GPT-5.4 (53% exact label match), $\kappa \approx 0.07$ for DeepSeek↔Gemini 3 Flash (41% match), and $\kappa \approx 0.21$ for GPT-5.4↔Gemini 3 Flash (56% match). The judges agree on which system wins on average but frequently disagree on which side wins any particular pair — the pattern expected if each judge is a noisy but roughly unbiased estimator of a genuine underlying preference, rather than a reliable per-item oracle. This is the mechanism behind the §4.1 headline: the batch-level preference for `S15` is robust to which of these three judges one picks precisely because the agreement shows up in aggregates, and three independent noisy estimators landing on the same side is stronger evidence about the underlying system ranking than any one estimator alone.

### 4.5 Judge–human alignment (Figure 2, right)

Alignment with expert human judgments is weaker still at the pair level. On the 12-topic custom pilot, Pearson correlations between each automated judge's topic-level win rates and human topic win rates are near zero and slightly negative: $r \approx -0.06$ for DeepSeek, $r \approx -0.05$ for GPT-5.4, and $r \approx -0.10$ for Gemini 3 Flash Preview (the same model ID as the publish-eval multi-judge panel). Side-match rates — the fraction of topics where the automated judge's majority side matches the human majority side — are 42%, 25%, and 17% respectively. The pattern mirrors §4.4: automated judges and humans agree on the aggregate direction (`S15` preferred) while disagreeing substantially on which particular topics go which way. The two layers of evidence are not in conflict — they are measurements of different quantities. A stable aggregate preference does not require that every evaluator agree on every item; it requires that the per-item noise be unbiased enough for the aggregate signal to survive averaging, and that is what we observe.

### 4.6 Human blind pilot

The 12-topic, 60-comparison human pilot yields a clear directional signal: raters prefer `S15` in 40 of 60 pairwise comparisons (≈ 67%). Two features qualify the magnitude rather than the direction. First, rubric means across the four dimensions are close between `S15` and `S_sota`, so the preferred-side count is not mirrored by a large rubric-score gap. Second, the pilot uses disjoint topic slices across eight raters and therefore does not measure inter-rater reliability on shared items. The 40/60 preference is best read as a directional external probe that agrees with the automated judges in §4.3; it is not a calibrated estimate of effect size.

### 4.7 Test–retest and winner–score consistency

Post-hoc diagnostics on the pairwise protocol itself echo the cross-judge picture. Rerunning the same judge on the same pairs changes a non-trivial fraction of `winner` labels, and the judge's declared `winner` is not always consistent with the sign of its own total rubric score. These are protocol-level noise estimates rather than evidence about any particular system version, and they set a floor on how reliably any single-pass pairwise result can be interpreted. Aggregates survive this noise; individual pair labels do not.

On the frozen publish-eval intersection ($N = 64$ pair keys shared across judges), we compared a **second API pass** to the stored primary verdicts for **GPT-5.4** and **DeepSeek Chat** on the same `S15` vs `S_sota` ideas. Artifacts live under `ideas/results/publish_eval_n75_gemini3_deepseek_S15_Spaper/` (on-disk bundle path; name retains a legacy `Spaper` suffix). Directory contents use the full **75** pairwise slots in the frozen `ideas.json` files; statistics here use the **64**-pair intersection loaded by `judge_analysis.json`. GPT-5.4 agrees with itself on about **77%** of pair labels (Cohen \kappa \approx 0.55 on the three-class A,B,\text{tie} space); DeepSeek agrees on about **69%** (\kappa \approx 0.38). Tie-adjusted batch \widehat{\mathrm{WR}}_B for `S_sota` moves by only about **0.8** percentage points between GPT passes on this slice, versus about **5.5** pp between DeepSeek passes—so **batch direction is stable** even when item-level labels fluctuate. Figure 4 summarizes the label-stability readout; raw files are `judge_*.json` / `judge_*_rerun.json` and `judge_analysis.json`.

Figure 4 — Intra-judge test–retest on the publish-eval intersection: exact match rate and Cohen 

\kappa

 for GPT-5.4 vs GPT-5.4 rerun and DeepSeek vs DeepSeek rerun.

### 4.8 Summary

Across every evaluator we ran — three automated judges on a 64-pair frozen slice, and eight human raters on a disjoint 12-topic pilot — the aggregate verdict favors `S15` over the reboot baseline `S_sota`. This system-level agreement holds even though pair-level agreement between judges is low (κ 0.07–0.21), automated-judge-to-human topic correlations are near zero, and repeated judging flips a non-trivial share of labels.

The coexistence is the finding. A subjective domain in which evaluators disagree substantially on which specific idea wins any given topic can still produce a stable, consistent system-level preference when aggregate counts are taken across many pairs and multiple independently-built evaluators. This is the regime self-improvement loops actually need: per-item labels can be noisy as long as system-level signal integrates through the noise. The methodological implication — multiple judges, repeated passes, topic-block uncertainty, and external human validation — is not a hedge against the positive result but the protocol under which the positive result is legible.

## 6. Limitations

- We study one generator-model family (`gpt-4.1-mini`) and a small set of judge models. The system-level preference we report is stable across the three independently-built judges and the human pilot we tested, but generalization to other generator families and other judge panels remains open.
- The human evidence is a small, non-benchmark pilot: 60 pairwise judgments across 12 custom expert topics, scored by eight raters on disjoint slices. It is useful as an external probe that agrees with the automated judges on system-level direction, but the disjoint-slice design does not measure inter-rater reliability on shared items and the pilot is not a calibrated estimate of effect size.
- We study one domain, scientific idea generation. Transfer to coding, theorem proving, or other agentic tasks remains open.
- We do not test adversarial cheating against the judge, so the paper should not be read as evidence that the loop is Goodhart-proof. Additionally, we do not run extensive ablations on the improvement agent itself.
- The full meta-trajectory is a single realized run. Our cross-judge readout on a frozen slice is Figure 2 (left: N = 64 publish-eval intersection), not a rerun of the entire evolution process from scratch.

## 7. Reproducibility

`REPRODUCE.md` documents the regeneration commands for all headline numbers. The key frozen artifacts used here are `ideas/results/evolution_log.jsonl`, `ideas/results/compare_*.json`, `ideas/results/human_blind_scores.jsonl`, and the publish-eval directories under `ideas/results/publish_eval_`*. Model IDs are pinned in `ideas/paper_config.py`. Wall-clock time depends strongly on API concurrency and judge-model latency, so we report the exact commands and artifact paths rather than a single fixed runtime estimate.

## Acknowledgements

Human raters (anonymous at submission time; named in camera-ready if accepted).

## References

- Su, H., Chen, R., Tang, S., Zheng, X., Li, J., Yin, Z., Ouyang, W., and Dong, N. (2024). *Two Heads Are Better Than One: A Multi-Agent System Has the Potential to Improve Scientific Idea Generation*. arXiv:2410.09403.
- Lu, C., Zhang, J., Hu, S., Lange, R. T., and Clune, J. (2024). *The AI Scientist: Towards Fully Automated Open-Ended Scientific Discovery*. arXiv:2408.06292.
- Gottweis, A., et al. (2025). *Towards an AI Co-Scientist*. arXiv:2502.18864.
- Lyu, Y., et al. (2026). *EvoScientist: Towards Multi-Agent Evolving AI Scientists for End-to-End Scientific Discovery*. arXiv:2603.08127.
- Si, C., et al. (2024). *Can Large Language Models Generate Novel Research Ideas?* arXiv:2409.04109.
- Zhang, J., Hu, S., Lu, C., Lange, R. T., and Clune, J. (2025). *Darwin Gödel Machine: Open-Ended Evolution of Self-Improving Agents*. arXiv:2505.22954.

## Appendix A — Supplementary table

### A.1 `S15` vs `S_sota`: automated judges (publish-eval intersection, $N = 64$) and human pilot ($n = 60$)

Side **A** is the champion `S15` and side **B** is `S_sota`. Tie-adjusted $\widehat{\mathrm{WR}}_B$ for `S_sota` assigns mass $1$ to $B$ wins, $0$ to $A$ wins, and $0.5$ to ties. **95% intervals** use topic-block bootstrap: resample topics with replacement, pool attached pairs, recompute $\widehat{\mathrm{WR}}_B$, take 2.5 and 97.5 percentiles ($3000$ draws, seed $42$). Automated rows come from the frozen publish-eval bundle `ideas/results/publish_eval_n75_gemini3_deepseek_S15_Spaper/` (75 pairwise slots on disk; table above uses the **$N=64$** judge intersection). The human row uses blind ratings in `ideas/results/human_blind_scores.jsonl` (eight raters, twelve disjoint topics, five comparisons per topic).


| Judge / source | Wins A / B / tie | $\widehat{\mathrm{WR}}_B$ for `S_sota` | 95% CI (topic-block) |
| -------------- | ---------------- | ---------------------------------------- | -------------------- |
| Gemini 3 Flash Preview | 32 / 4 / 28 | 28.1% | 16.4–40.6% |
| GPT-5.4 | 39 / 5 / 20 | 23.4% | 15.7–32.3% |
| DeepSeek Chat | 43 / 20 / 1 | 32.0% | 19.9–46.4% |
| Human experts | 40 / 15 / 5 | 29.2% | 18.3–40.0% |


Pairwise three-class κ on the **64-pair** automated overlap only: GPT-5.4 vs Gemini 3 = 0.21, DeepSeek vs Gemini 3 = 0.07, DeepSeek vs GPT-5.4 = 0.17.
