# S15 Idea Generator — Full Prompt Pipeline

Topic: *Uncertainty quantification in deep neural networks*

**Total: ~13 LLM calls** (1 + 5 parallel attacks + 1 select + 1 construct + 4 critique + 1 revise)

======================================================================
## Step 0 — SOTA Retrieval (Semantic Scholar)
**Size: 147 chars**

### INPUT PROMPT:
```
Query: 'Uncertainty quantification in deep neural networks'
→ Returns 5 paper titles+abstracts (~2000 chars)
(Used in Steps 1 & 4 as context_block)
```

### OUTPUT:
## Recent related work (for context — your idea must go beyond these)

1. **Leveraging the Bhattacharyya coefficient for uncertainty quantification in deep neural networks** (2021)
   Abstract Modern deep learning models achieve state-of-the-art results for many tasks in computer vision, such as image classification and segmentation. However, its adoption into high-risk applications, e.g. automate...
[5 papers total]

======================================================================
## Step 1 — Hypothesis Generation (1 LLM call)
**Size: 4,271 chars**

### INPUT PROMPT:
```
Research topic: Uncertainty quantification in deep neural networks

## Recent related work (for context — your idea must go beyond these)

1. **Leveraging the Bhattacharyya coefficient for uncertainty quantification in deep neural networks** (2021)
   Abstract Modern deep learning models achieve state-of-the-art results for many tasks in computer vision, such as image classification and segmentation. However, its adoption into high-risk applications, e.g. automated medical diagnosis systems, happens at a slow pace. One of the main reasons for this is that regular neural networks do not capture uncertainty. To assess uncertainty in classification, several techniques have been proposed casting neural network approaches in a Bayesian setting. Amongst these techniques, Monte Carlo dropout is by far the most popular. This particular technique es

2. **Uncertainty quantification for deep neural networks: An empirical comparison and usage guidelines** (2023)
   Summary Deep neural networks (DNN) are increasingly used as components of larger software systems that need to process complex data, such as images, written texts, audio/video signals. DNN predictions cannot be assumed to be always correct for several reasons, amongst which the huge input space that is dealt with, the ambiguity of some inputs data, as well as the intrinsic properties of learning algorithms, which can provide only statistical warranties. Hence, developers have to cope with some residual error probability. An architectural pattern commonly adopted to manage failure prone compone

3. **Uncertainty quantification in deep neural networks: Techniques and applications in autonomous decision-making systems** (2021)
   Uncertainty quantification (UQ) in deep neural networks (DNNs) is an essential area of research, particularly for enhancing the reliability and safety of autonomous decision-making systems deployed in high-stakes environments such as autonomous vehicles, healthcare, and robotics. This article provides a comprehensive overview of the key techniques for UQ in DNNs, including Bayesian Neural Networks, Monte Carlo Dropout, ensemble methods, and Gaussian Processes, highlighting their respective strengths and limitations. The applications of UQ in critical domains are examined, demonstrating how the

4. **Uncertainty Quantification of Deep Neural Network-Based Turbulence Model for Reactor Transient Analysis** (2021)
   Abstract Deep neural networks (DNNs) have demonstrated good performance in learning highly non-linear relationships in large datasets, thus have been considered as a promising surrogate modeling tool for parametric partial differential equations (PDEs). On the other hand, quantifying the predictive uncertainty in DNNs is still a challenging problem. The Bayesian neural network (BNN), a sophisticated method assuming the weights of the DNNs follow certain uncertainty distributions, is considered as a state-of-the-art method for the UQ of DNNs. However, the method is too computationally expensive

5. **Integrating deep neural networks with full-waveform inversion: Reparameterization, regularization, and uncertainty quantification** (2021)
   ABSTRACT Full-waveform inversion (FWI) is an accurate imaging approach for modeling the velocity structure by minimizing the misfit between recorded and predicted seismic waveforms. However, the strong nonlinearity of FWI resulting from fitting oscillatory waveforms can trap the optimization in local minima. We have adopted a neural-network-based full-waveform inversion (NNFWI) method that integrates deep neural networks with FWI by representing the velocity model with a generative neural network. Neural networks can naturally introduce spatial correlations as regularization to the generated v


Generate exactly 5 sharp, falsifiable scientific hypotheses about this topic.
Each hypothesis must:
- Make a specific claim about the world (not just 'we can improve X')
- Be testable: name what experiment would prove i...
```

### OUTPUT:
H1: [hypothesis]
H2: [hypothesis]
...
H5: [hypothesis]

======================================================================
## Step 2a-e — Adversarial Attack × 5 (parallel LLM calls)
**Size: 929 chars**

### INPUT PROMPT:
```
Research topic: Uncertainty quantification in deep neural networks

Hypothesis: Bayesian deep ensembles fail to calibrate uncertainty in distribution-shifted inputs because the ensemble members share implicit inductive biases from identical architectures, producing overconfident predictions that a simple temperature-scaled single model outperforms.

You are an adversarial critic. Attack this hypothesis on EACH of these dimensions:
1. Assumption violation: What unstated assumption does this rely on?
2. Dataset bias: What dataset artifact could make this appear true without actually being true?
3. Theoretical gap: What known result from the literature contradicts or undermines this?
4. Practical limitation: What makes this hypothesis untestable or too expensive to test?

Then, given these attacks, write a REVISED hypothesis that survives them:
Revised: <1-2 sentence refined hypothesis that addresses the above attacks>
```

### OUTPUT:
1. Assumption: ...
2. Dataset bias: ...
3. Theoretical gap: ...
4. Practical: ...

Revised: [refined hypothesis]

======================================================================
## Step 3 — Hypothesis Selection (1 LLM call)
**Size: 1,756 chars**

### INPUT PROMPT:
```
Research topic: Uncertainty quantification in deep neural networks

Below are 5 original hypotheses, each attacked by an adversarial critic and refined:

--- Hypothesis 1 ---
Original: Bayesian deep ensembles fail to calibrate uncertainty in distribution-shifted inputs because the ensemble members share implicit inductive biases from identical architectures, producing overconfident predictions that a simple temperature-scaled single model outperforms.
Critique+Revision:
1. Assumption violation: assumes ensemble diversity comes only from architecture, ignoring random init diversity.
2. Dataset bias: benchmarks like CIFAR-10-C may show this because shift types correlate with training noise, not architecture.
3. Theoretical gap: Lakshminarayanan 2017 shows diverse random inits alone create substantial diversity even in identical architectures.
4. Practical limitation: requires many ensemble members (5+) to test properly; expensive.

Revised: Bayesian deep ensembles trained with identical architectures but different random seeds exhibit no better uncertainty calibration under covariate shift than single-model temperature scaling when measured on ECE, because functional diversity — not parameter diversity — drives calibration gains, and identical architectures produce functionally similar predictors.

[... 4 more hypothesis+attack pairs ...]

Select the ONE hypothesis (by number) whose REVISED form is:
- Most concrete and specific (names mechanisms, not just outcomes)
- Most falsifiable (clearest path to a disproof experiment)
- Most novel (least covered by standard literature)

Respond with:
SELECTED: <number 1-5>
REVISED HYPOTHESIS: <copy the revised hypothesis text exactly>
REASONING: <1-2 sentences on why this is the strongest>
```

### OUTPUT:
SELECTED: 3
REVISED HYPOTHESIS: ...
REASONING: ...

======================================================================
## Step 4 — Experimental Construction (1 LLM call)
**Size: 1,440 chars**

### INPUT PROMPT:
```
Research topic: Uncertainty quantification in deep neural networks

Existing work to differentiate from:
## Recent related work (for context — your idea must go beyond these)

1. **Leveraging the Bhattacharyya coefficient for uncertainty quantification in deep neural networks** (2021)
   Abstract Modern deep learning models achieve state-of-the-art results for many tasks in computer vision, such as image classification and segmentation. However, its adoption into high-risk applications, e.g. automated medical diagnosis systems, happens at a slow pace. One of the main reasons for this is that regular neural networks do not capture uncertainty. To assess uncertainty in classification, several techn
...[truncated]

Core hypothesis to test: Bayesian deep ensembles trained with identical architectures but different random seeds exhibit no better uncertainty calibration under covariate shift than single-model temperature scaling when measured on ECE.

Design a concrete research experiment to PROVE OR DISPROVE this hypothesis.
The experiment must:
- Name specific datasets you will use (not just 'standard benchmarks')
- Name specific baselines you will compare against
- Define quantitative success metrics (what number proves it, what number disproves it)
- Describe the key technical method in enough detail to implement
- Explain precisely what result would falsify the hypothesis

Write 3-4 paragraphs. Be direct and specific.
```

### OUTPUT:
[3-4 paragraphs naming datasets, baselines, metrics, falsification criteria]

======================================================================
## Step 5a — Experimentalist Critique (1 LLM call)
**Size: 525 chars**

### INPUT PROMPT:
```
You are a hard-nosed experimentalist reviewing a research proposal about 'Uncertainty quantification in deep neural networks'.

Hypothesis being tested: Bayesian deep ensembles trained with identical architectures but different random seeds exhibit no better uncertainty calibration under covariate shift than single-model temperature scaling when measured on ECE.

Proposed experiment:
[3-4 paragraph experimental design would appear here — ~600 words]

Give 2-3 sharp criticisms focusing purely on experimental feasibility.
```

### OUTPUT:
1. [feasibility criticism]
2. [feasibility criticism]

======================================================================
## Step 5b — Theorist Critique (1 LLM call)
**Size: 513 chars**

### INPUT PROMPT:
```
You are a rigorous theorist reviewing a research proposal about 'Uncertainty quantification in deep neural networks'.

Hypothesis being tested: Bayesian deep ensembles trained with identical architectures but different random seeds exhibit no better uncertainty calibration under covariate shift than single-model temperature scaling when measured on ECE.

Proposed experiment:
[3-4 paragraph experimental design would appear here — ~600 words]

Give 2-3 sharp criticisms focusing purely on theoretical grounding.
```

### OUTPUT:
1. [theory criticism]
2. [theory criticism]

======================================================================
## Step 5c — Skeptic Critique (1 LLM call)
**Size: 604 chars**

### INPUT PROMPT:
```
You are a skeptical reviewer who has seen many overhyped proposals about 'Uncertainty quantification in deep neural networks'.

Hypothesis being tested: Bayesian deep ensembles trained with identical architectures but different random seeds exhibit no better uncertainty calibration under covariate shift than single-model temperature scaling when measured on ECE.

Proposed experiment:
[3-4 paragraph experimental design would appear here — ~600 words]

Give 2-3 sharp criticisms focusing on: why this probably won't work, what the likely negative result is, and whether the payoff justifies the effort.
```

### OUTPUT:
1. [skeptic criticism]
2. [skeptic criticism]

======================================================================
## Step 5d — Synthesis (1 LLM call)
**Size: 522 chars**

### INPUT PROMPT:
```
Three reviewers critiqued a research idea about 'Uncertainty quantification in deep neural networks' testing the hypothesis:
'Bayesian deep ensembles trained with identical architectures but different random seeds exhibit no better uncertainty calibration under covariate shift than single-model temperature scaling when measured on ECE.'

Experimentalist:
[critique here]

Theorist:
[critique here]

Skeptic:
[critique here]

Synthesize these into the 3 most important actionable improvements. Be concise and prioritized.
```

### OUTPUT:
1. [top actionable improvement]
2. [second improvement]
3. [third improvement]

======================================================================
## Step 6 — Final Revision (1 LLM call)
**Size: 1,751 chars**

### INPUT PROMPT:
```
Research topic: Uncertainty quantification in deep neural networks

Core hypothesis: Bayesian deep ensembles trained with identical architectures but different random seeds exhibit no better uncertainty calibration under covariate shift than single-model temperature scaling when measured on ECE.

Experimental design:
[3-4 paragraph experimental design would appear here — ~600 words]

Key improvements required:
1. Specify the covariate shift severity axis (mild/moderate/severe). 2. Include ECE confidence intervals, not just point estimates. 3. Test on NLP domain too, not just vision.

Existing work to differentiate from:
## Recent related work (for context — your idea must go beyond these)

1. **Leveraging the Bhattacharyya coefficient for uncertainty quantification in deep neural networks** (2021)
   Abstract Modern deep learning models achieve state-of-the-art results for many tasks in computer vision, such as image classification and segmentation. However, its adoption into high-risk applications, e.g. automated medical diagnosis systems, happens at a slow pace. One of the main reasons for this is that regular neural networks do not capture uncertainty. To assess uncertainty in classification, several techn
...[truncated]

Write the final, improved version of the research idea. Ensure the hypothesis is clearly stated, the falsification experiment is concrete, and all datasets/baselines/metrics are named explicitly.

Present your final research idea in exactly this format:

IDEA: [One sentence summary]

BACKGROUND: [The specific open problem, 2-3 sentences]

APPROACH: [The proposed method, 2-3 sentences]

EXPERIMENT: [How to test it — datasets, metrics, 1-2 sentences]

NOVELTY: [What makes this different, 1-2 sentences]

```

### OUTPUT:

---
**REAL OUTPUT from ideas.json:**
IDEA:  
Neural architecture search can discover layer-wise, epoch-segmented adaptive dropout schedules that, when combined with Monte Carlo dropout, significantly improve uncertainty calibration on clean, balanced image classification datasets relative to fixed dropout schedules, without compromising accuracy and while maintaining feasible computational cost.

BACKGROUND:  
Accurate uncertainty quantification in deep neural networks remains a critical challenge, especially for safety-sensitive applications. While Monte Carlo dropout offers a practical Bayesian approximation, fixed dropout rates limit flexibility in modeling uncertainty across training dynamics and network depth. Prior adaptive dropout methods either lack targeted optimization for calibration or incur prohibitive computational overhead, leaving open whether efficient NAS can tailor dropout schedules to produce better calibrated uncertainty estimates.

APPROACH:  
We employ differentiable neural architecture search (DARTS) to optimize dropout schedules defined as piecewise-constant dropout rates per layer across discrete training epoch segments within a ResNet-34 architecture. The search jointly optimizes for classification accuracy and Expected Calibration Error (ECE) on a clean validation set, constraining the search space to dropout rates from 0 to 0.7 in 0.1 increments and limiting epoch segments to ensure tractability. The optimized schedules are evaluated with 50 Monte Carlo forward passes at test time to assess uncertainty calibration.

EXPERIMENT:  
We evaluate on CIFAR-10, CIFAR-100, and STL-10 datasets, measuring calibration via ECE, Negative Log Likelihood (NLL), and Brier Score, while monitoring classification accuracy. Baselines include fixed uniform dropout with MC dropout, heuristic adaptive dropout schedules, NAS-optimized schedules targeting accuracy alone, and deep ensembles. Statistical significance is assessed over multiple seeds with corrections for multiple comparisons. Computational overhead is quantified to ensure practical feasibility.

NOVELTY:  
This work uniquely integrates efficient differentiable NAS to discover adaptive dropout schedules explicitly optimized for uncertainty calibration, differentiating from prior adaptive dropout strategies that focus on variational formulations or accuracy alone. By constraining datasets to low-noise, balanced scenarios, we isolate calibration improvements attributable to adaptive dropout scheduling, thereby advancing principled and computationally viable uncertainty quantification in deep image classifiers.