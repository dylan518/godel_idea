# Benchmark topics and example ideas (concrete illustrations)

This file is for readers and reviewers who want **specific examples** of what the Gödel-loop / EvoScientist `ideas/` harness actually generates and compares. All idea excerpts below are taken from real cached outputs (`ideas/results/S15/ideas.json`, generator `**gpt-4.1-mini`**, `**n_ideas = 5` per topic**) unless noted.

---

## 1. What gets compared (pairing)

- **Topic list:** fixed JSON `ideas/benchmark_topics.json` (15 topics).
- **Per topic:** each system produces ideas with `idea_index` 0 \dots n_{\text{ideas}}-1.
- **Head-to-head:** for champion **A** vs candidate **B**, the judge sees pairs matched on `**(topic_id, idea_index)`** only (same slot, two different generators).

Example: **T1 / idea_index 2** means “second idea for topic T1 from each system.”

---

## 2. Required idea format (`IDEA_FORMAT`)

Every generator is steered to the same section structure (from `ideas/systems/base.py`):

```text
IDEA: [One sentence summary of the core proposal]

BACKGROUND: [The specific open problem this addresses, 2-3 sentences]

APPROACH: [The proposed method or technique, 2-3 sentences]

EXPERIMENT: [How to test it — datasets, metrics, or methodology, 1-2 sentences]

NOVELTY: [What makes this meaningfully different from prior work, 1-2 sentences]
```

In practice, length and tone vary by system version; the judge still receives the full `text` field for each side.

---

## 3. Full benchmark topic list (15)


| ID  | Domain (tag) | Topic string                                           |
| --- | ------------ | ------------------------------------------------------ |
| T1  | ML           | Scaling laws for Large Language Models                 |
| T2  | Biology      | Protein structure prediction beyond AlphaFold          |
| T3  | Physics      | Quantum error correction in NISQ devices               |
| T4  | Statistics   | Causal discovery from observational data               |
| T5  | Hardware     | Energy-efficient neuromorphic computing                |
| T6  | Chemistry    | Generative AI for drug discovery and molecular design  |
| T7  | ML           | Federated learning with heterogeneous and non-IID data |
| T8  | Climate      | Deep learning emulators for climate model acceleration |
| T9  | ML Theory    | Uncertainty quantification in deep neural networks     |
| T10 | Genomics     | Foundation models for genomics and single-cell biology |
| T11 | Robotics     | Emergent communication in multi-agent systems          |
| T12 | ML Safety    | Mechanistic interpretability of transformer models     |
| T13 | ML           | Zero-shot generalization in reinforcement learning     |
| T14 | Mathematics  | Topological methods for high-dimensional data analysis |
| T15 | Quantum      | Quantum advantage in machine learning tasks            |


---

## 4. Example A — T1, `idea_index` 0 (ML scaling laws)

**Topic:** Scaling laws for Large Language Models  

**Provenance:** `S15`, `ideas/results/S15/ideas.json`, 2026-04-17.

**Full generated text (as stored):**

```text
IDEA:  
Investigate how intrinsic training data diversity, measured by a novel, multi-faceted diversity metric, critically modulates the scaling efficiency of large language models (LLMs), independent of dataset size, by training fixed-size models on equally large but carefully diversity-controlled corpora and evaluating their performance and scaling behavior.

BACKGROUND:  
While scaling laws have characterized how model and dataset size affect LLM performance, the role of data diversity in these laws remains poorly understood and often conflated with dataset size or domain effects. Prior work has not rigorously isolated diversity from size or complexity, leaving open whether there exists a critical diversity threshold below which scaling efficiency degrades substantially. Clarifying this relationship is essential for optimizing data curation and understanding fundamental limits of LLM scaling.

APPROACH:  
We develop and validate a theoretically grounded, multi-dimensional data diversity metric capturing linguistic, semantic, stylistic, and structural variation, enabling construction of training corpora matched in token count and effective information content but spanning a continuum from low to high diversity. Fixed-parameter transformer LLMs (1B and 3B) will be trained on three datasets of equal size but varying diversity: (1) high diversity (The Pile), (2) medium diversity (OpenWebText2), and (3) low diversity (narrow-domain astrophysics ArXiv subset). Controlled hyperparameters ensure only diversity varies. Performance will be evaluated intrinsically (cross-entropy loss, perplexity) and extrinsically (zero-shot accuracy on diverse benchmarks like LAMBADA and CodeXGLUE, plus domain-matched probing tasks).

EXPERIMENT:  
We will compare scaling curves across the three datasets by training models at fixed compute budgets and measuring statistically significant differences in perplexity and zero-shot performance; a >10% relative perplexity increase or >5% accuracy drop for low-diversity data compared to high-diversity data at equal size will confirm the hypothesis. We will also analyze interaction effects between model size and diversity, and verify the diversity metric’s predictive power over scaling efficiency.

NOVELTY:  
This work uniquely isolates and quantifies data diversity’s impact on LLM scaling laws using a rigorously developed diversity metric and controlled dataset construction, advancing beyond prior studies that consider dataset size or domain but not intrinsic diversity. It provides the first theoretical framing and empirical evidence for diversity thresholds that fundamentally constrain scaling efficiency, offering actionable insights for data-centric model optimization.
```

**Why this is useful as an example:** it names **concrete model sizes (1B, 3B)**, **named corpora (The Pile, OpenWebText2)**, **benchmarks (LAMBADA, CodeXGLUE)**, and **numeric success thresholds (>10% perplexity, >5% accuracy)**—the kind of specificity the pairwise judge rubric rewards under “experimental clarity” and “feasibility.”

---

## 5. Example B — T2, `idea_index` 0 (protein structure / IDPs)

**Topic:** Protein structure prediction beyond AlphaFold  

**Provenance:** `S15`, same file as above.

**Full generated text (as stored):**

```text
IDEA: Integrating enhanced sampling molecular dynamics simulations initialized from AlphaFold2 structures with Bayesian reweighting produces hybrid conformational ensembles that significantly improve intrinsic disorder region predictions by better capturing IDP structural heterogeneity compared to AlphaFold2 alone.

BACKGROUND: Although AlphaFold2 excels at predicting folded protein structures, it struggles to accurately model intrinsically disordered protein (IDP) regions, which exist as dynamic ensembles rather than single conformations. Current static predictions fail to represent the conformational heterogeneity revealed by NMR ensemble data, limiting functional insights for IDPs. Improved methods that combine data-driven predictions with biophysical simulations are needed to better recapitulate IDP ensembles.

APPROACH: We propose generating initial IDP conformers using AlphaFold2, then applying enhanced sampling molecular dynamics (e.g., temperature replica exchange MD) with IDP-optimized force fields (CHARMM36m or AMBER ff99SB-ILDN with TIP4P-D water) to produce diverse structural ensembles. These ensembles will be integrated with AlphaFold outputs through Bayesian ensemble reweighting to form hybrid ensembles reflecting both learned structural propensities and physical dynamics. Controls will include MD ensembles initiated from extended chains and comparisons to disorder-specific predictors and flexible-meccano ensembles.

EXPERIMENT: Using a curated DisProt-derived dataset of intrinsically disordered proteins with high-quality NMR ensembles from the Protein Ensemble Database (PED), we will evaluate models by ensemble-averaged RMSD and ensemble RMSD coverage metrics against NMR conformers, supplemented with ensemble RMSD distribution and principal component overlap analyses. Statistical significance will be assessed via paired t-tests comparing hybrid ensembles to vanilla AlphaFold2 and disorder predictors, with a ≥10% RMSD reduction and ≥15% coverage increase required to support the hypothesis.

NOVELTY: This work uniquely combines AlphaFold2 predictions with physics-based enhanced sampling simulations and principled Bayesian reweighting to generate IDP ensembles, explicitly addressing limitations of prior hybrid modeling by carefully validating against experimentally observed conformational heterogeneity and employing IDP-optimized force fields and rigorous ensemble metrics, thus advancing beyond static or purely sequence-based disorder predictions.
```

**Pointers for readers:** **AlphaFold2**, **replica-exchange MD**, **CHARMM36m / AMBER**, **DisProt**, **PED**, explicit **RMSD / coverage** targets—again, concrete enough that a judge can argue about whether the proposal is testable.

---

## 6. Example C — T12, `idea_index` 0 (mechanistic interpretability)

**Topic:** Mechanistic interpretability of transformer models  

**Provenance:** `S15`, `ideas/results/S15/ideas.json`.

**Full generated text (as stored):**

```text
IDEA:  
Investigate whether specific attention heads in multilingual transformer models mechanistically encode universal compositional syntactic rules by demonstrating consistent, causally grounded head-function mappings that generalize across typologically diverse languages.

BACKGROUND:  
While attention patterns in transformer models have been correlated with syntactic structures, it remains unclear if these patterns reflect mechanistic, compositional syntactic rules that are language-universal or merely statistical associations. Prior monolingual probing studies and correlational analyses do not establish causal roles of attention heads or their cross-lingual consistency, leaving an open problem in mechanistic interpretability and the universality of syntactic representations in multilingual models.

APPROACH:  
We will identify attention heads in pretrained multilingual transformers (mBERT, XLM-R) that robustly predict syntactic dependencies within individual languages by combining attention alignment metrics with causal interventions such as targeted head ablations and synthetic attention rewiring. We then assess cross-lingual generalization by comparing head-syntactic function mappings across English, Finnish, and Japanese UD treebanks using canonical correlation analysis (CCA/SVCCA) on aligned attention patterns with careful preprocessing to normalize tokenization and morphology effects.

EXPERIMENT:  
Using Universal Dependencies datasets for English, Finnish, and Japanese, we measure Attention Head Accuracy (AHA) as the fraction of tokens whose highest-attention corresponds to their syntactic head, requiring cross-lingual AHA >70% and CCA similarity >0.75 to confirm shared head roles. Controlled baselines include random head assignments, monolingual syntactic probes, and finetuned syntactic classifiers. Ablation studies will test causal necessity of identified heads, while failure to find consistent cross-lingual alignments or causal effects (AHA <50%, CCA <0.4) falsifies the hypothesis.

NOVELTY:  
This approach advances mechanistic interpretability by integrating causal intervention methods with cross-lingual syntactic probing to rigorously test universality of attention-head syntactic roles, moving beyond purely correlational monolingual studies. It provides the first comprehensive, causally grounded evaluation of whether transformer attention heads encode truly compositional, language-general syntactic rules across typologically diverse languages.
```

**Pointers for readers:** names **mBERT**, **XLM-R**, **Universal Dependencies**, **head ablations**, and **numeric gates (AHA, CCA)**—typical of the benchmark’s emphasis on falsifiable claims.

---

## 7. What the judge sees (rubric reminder)

For each pair, the judge scores **both** ideas on four **0–10** criteria:

1. Novelty
2. Scientific usefulness
3. Experimental clarity
4. Feasibility

then assigns `**A` / `B` / `tie`** (tie if total scores differ by **≤ 2**). Presentation order is **randomly swapped** per call to reduce position bias.

*Illustrative (synthetic) micro-example, not from logs:* if Idea A totals 32 and Idea B totals 30, the template asks for a **tie** even if A is slightly ahead—so the label distribution is not a pure argmax on every call.

---

## 8. How to regenerate or verify

```bash
# Re-run the benchmark for a system version (example)
python3 ideas/godel_loop.py benchmark --version S15 --n-ideas 5 --workers 3

# Cached ideas used above
cat ideas/results/S15/ideas.json | python3 -m json.tool | less
```

If you cite this document in a paper, cite the **generator model** and **date/commit** alongside the examples; model drift will change wording even for the same `S15` code.

---

## 9. Human ratings: **not** these 15 benchmark topics

The **human blind pilot** described in `**ideas/RESULTS_SUMMARY.md`** uses **twelve custom expert topics** (thesis-style prompts), **not** the T1–T15 list above. Raters completed **five** blind pairwise comparisons per topic (**S15** vs `**S_paper`**) → 60 judgments in `ideas/results/human_blind_scores.jsonl`. Each expert rated only their own topics, so there is no multi-rater overlap on identical pairs. Automated evolution and LLM-judge trajectories in the paper use `**benchmark_topics.json`**; human numbers are a **separate slice** and should not be described as “the benchmark with humans in the loop” unless you run a new study on T1–T15.