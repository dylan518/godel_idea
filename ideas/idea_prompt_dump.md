# S12 Idea Generator — Full Prompt Chain

Topic: *Scaling laws for Large Language Models*

---

## Step 1 — Hypothesis Generation

**Size: 3,989 chars**

Research topic: Scaling laws for Large Language Models

## Recent related work (for context — your idea must go beyond these)

1. **D-CPT Law: Domain-specific Continual Pre-Training Scaling Law for Large Language Models** (2024)
  Continual Pre-Training (CPT) on Large Language Models (LLMs) has been widely used to expand the model's fundamental understanding of specific downstream domains (e.g., math and code). For the CPT on domain-specific LLMs, one important question is how to choose the optimal mixture ratio between the general-corpus (e.g., Dolma, Slim-pajama) and the downstream domain-corpus. Existing methods usually adopt laborious human efforts by grid-searching on a set of mixture ratios, which require high GPU training consumption costs. Besides, we cannot guarantee the selected ratio is optimal for the specif
2. **Scaling Laws for Neural Language Models** (2020)
  We study empirical scaling laws for language model performance on the cross-entropy loss. The loss scales as a power-law with model size, dataset size, and the amount of compute used for training, with some trends spanning more than seven orders of magnitude. Other architectural details such as network width or depth have minimal effects within a wide range. Simple equations govern the dependence of overfitting on model/dataset size and the dependence of training speed on model size. These relationships allow us to determine the optimal allocation of a fixed compute budget. Larger models are s
3. **Predictable Scale: Part I, Step Law -- Optimal Hyperparameter Scaling Law in Large Language Model Pretraining** (2025)
  The impressive capabilities of Large Language Models (LLMs) across diverse tasks are now well\text{-}established, yet their effective deployment necessitates careful hyperparameter optimization. Although existing methods have explored the influence of hyperparameters on model performance, a principled and generalizable framework across model architectures and data recipes remains absent. In this study, we conduct an unprecedented empirical investigation\text{-} training over 3,700 LLMs from scratch across 100 trillion tokens, consuming nearly one million NVIDIA H800 GPU hours to establish a un
4. **Sparks of Artificial General Intelligence: Early experiments with GPT-4** (2023)
  Artificial intelligence (AI) researchers have been developing and refining large language models (LLMs) that exhibit remarkable capabilities across a variety of domains and tasks, challenging our understanding of learning and cognition. The latest model developed by OpenAI, GPT-4, was trained using an unprecedented scale of compute and data. In this paper, we report on our investigation of an early version of GPT-4, when it was still in active development by OpenAI. We contend that (this early version of) GPT-4 is part of a new cohort of LLMs (along with ChatGPT and Google's PaLM for example)
5. **K-BERT: Enabling Language Representation with Knowledge Graph** (2020)
  Pre-trained language representation models, such as BERT, capture a general language representation from large-scale corpora, but lack domain-specific knowledge. When reading a domain text, experts make inferences with relevant knowledge. For machines to achieve this capability, we propose a knowledge-enabled language representation model (K-BERT) with knowledge graphs (KGs), in which triples are injected into the sentences as domain knowledge. However, too much knowledge incorporation may divert the sentence from its correct meaning, which is called knowledge noise (KN) issue. To overcome KN,

Generate exactly 5 sharp, falsifiable scientific hypotheses about this topic.
Each hypothesis must:

- Make a specific claim about the world (not just "we can improve X")
- Be testable: name what experiment would prove it false
- Be non-obvious: it should NOT be directly supported by the related work above
- Be concise: 1-2 sentences max

Format each as:
H1: ...
H2: ...
H3: ...
H4: ...
H5: ...

---

## Step 2 — Adversarial Attack (per hypothesis, run x5 parallel)

**Size: 771 chars**

Research topic: Scaling laws for Large Language Models

Hypothesis: Scaling laws break under data-constrained regimes because token repetition above a threshold induces memorization rather than generalization, creating a phase transition in loss curves not predicted by current power-law models.

You are an adversarial critic. Attack this hypothesis on EACH of these dimensions:

1. Assumption violation: What unstated assumption does this rely on?
2. Dataset bias: What dataset artifact could make this appear true without being true?
3. Theoretical gap: What known result contradicts or undermines this?
4. Practical limitation: What makes this untestable or too expensive?

Then write a REVISED hypothesis that survives them:
Revised: <1-2 sentence refined hypothesis>

---

## Step 3 — Hypothesis Selection

**Size: 264 chars**

Research topic: Scaling laws for Large Language Models

[5 hypothesis + attack pairs...]

Select the ONE hypothesis whose REVISED form is most concrete, falsifiable, and novel.

Respond with:
SELECTED: 
REVISED HYPOTHESIS: 
REASONING: <1-2 sentences>

---

## Step 4 — Idea Construction

**Size: 4,084 chars**

Research topic: Scaling laws for Large Language Models

Existing work to differentiate from:

## Recent related work (for context — your idea must go beyond these)

1. **D-CPT Law: Domain-specific Continual Pre-Training Scaling Law for Large Language Models** (2024)
  Continual Pre-Training (CPT) on Large Language Models (LLMs) has been widely used to expand the model's fundamental understanding of specific downstream domains (e.g., math and code). For the CPT on domain-specific LLMs, one important question is how to choose the optimal mixture ratio between the general-corpus (e.g., Dolma, Slim-pajama) and the downstream domain-corpus. Existing methods usually adopt laborious human efforts by grid-searching on a set of mixture ratios, which require high GPU training consumption costs. Besides, we cannot guarantee the selected ratio is optimal for the specif
2. **Scaling Laws for Neural Language Models** (2020)
  We study empirical scaling laws for language model performance on the cross-entropy loss. The loss scales as a power-law with model size, dataset size, and the amount of compute used for training, with some trends spanning more than seven orders of magnitude. Other architectural details such as network width or depth have minimal effects within a wide range. Simple equations govern the dependence of overfitting on model/dataset size and the dependence of training speed on model size. These relationships allow us to determine the optimal allocation of a fixed compute budget. Larger models are s
3. **Predictable Scale: Part I, Step Law -- Optimal Hyperparameter Scaling Law in Large Language Model Pretraining** (2025)
  The impressive capabilities of Large Language Models (LLMs) across diverse tasks are now well\text{-}established, yet their effective deployment necessitates careful hyperparameter optimization. Although existing methods have explored the influence of hyperparameters on model performance, a principled and generalizable framework across model architectures and data recipes remains absent. In this study, we conduct an unprecedented empirical investigation\text{-} training over 3,700 LLMs from scratch across 100 trillion tokens, consuming nearly one million NVIDIA H800 GPU hours to establish a un
4. **Sparks of Artificial General Intelligence: Early experiments with GPT-4** (2023)
  Artificial intelligence (AI) researchers have been developing and refining large language models (LLMs) that exhibit remarkable capabilities across a variety of domains and tasks, challenging our understanding of learning and cognition. The latest model developed by OpenAI, GPT-4, was trained using an unprecedented scale of compute and data. In this paper, we report on our investigation of an early version of GPT-4, when it was still in active development by OpenAI. We contend that (this early version of) GPT-4 is part of a new cohort of LLMs (along with ChatGPT and Google's PaLM for example)
5. **K-BERT: Enabling Language Representation with Knowledge Graph** (2020)
  Pre-trained language representation models, such as BERT, capture a general language representation from large-scale corpora, but lack domain-specific knowledge. When reading a domain text, experts make inferences with relevant knowledge. For machines to achieve this capability, we propose a knowledge-enabled language representation model (K-BERT) with knowledge graphs (KGs), in which triples are injected into the sentences as domain knowledge. However, too much knowledge incorporation may divert the sentence from its correct meaning, which is called knowledge noise (KN) issue. To overcome KN,

Core hypothesis to test: Scaling laws break under data-constrained regimes because token repetition above a threshold induces memorization rather than generalization, creating a phase transition in loss curves not predicted by current power-law models.

Design a concrete research experiment to PROVE OR DISPROVE this hypothesis.
Must name: specific datasets, specific baselines, quantitative success metrics, key technical method.
Write 3-4 paragraphs.

---

## Step 5a — Experimentalist Critique

**Size: 478 chars**

You are a hard-nosed experimentalist reviewing a research proposal about "Scaling laws for Large Language Models".

Hypothesis: Scaling laws break under data-constrained regimes because token repetition above a threshold induces memorization rather than generalization, creating a phase transition in loss curves not predicted by current power-law models.

Proposed experiment:
[experimental design draft — 3-4 paragraphs]

Give 2-3 sharp criticisms on experimental feasibility.

---

## Step 5b — Theorist Critique

**Size: 466 chars**

You are a rigorous theorist reviewing a research proposal about "Scaling laws for Large Language Models".

Hypothesis: Scaling laws break under data-constrained regimes because token repetition above a threshold induces memorization rather than generalization, creating a phase transition in loss curves not predicted by current power-law models.

Proposed experiment:
[experimental design draft — 3-4 paragraphs]

Give 2-3 sharp criticisms on theoretical grounding.

---

## Step 5c — Skeptic Critique

**Size: 506 chars**

You are a skeptical reviewer who has seen many overhyped proposals about "Scaling laws for Large Language Models".

Hypothesis: Scaling laws break under data-constrained regimes because token repetition above a threshold induces memorization rather than generalization, creating a phase transition in loss curves not predicted by current power-law models.

Proposed experiment:
[experimental design draft — 3-4 paragraphs]

Give 2-3 criticisms: why this probably won't work, likely negative result, payoff.

---

## Step 5d — Synthesis

**Size: 469 chars**

Three reviewers critiqued a research idea about "Scaling laws for Large Language Models" testing:
"Scaling laws break under data-constrained regimes because token repetition above a threshold induces memorization rather than generalization, creating a phase transition in loss curves not predicted by current power-law models."

[Experimentalist critique]
[Theorist critique]
[Skeptic critique]

Synthesize into the 3 most important actionable improvements. Be concise.

---

## Step 6 — Final Revision

**Size: 4,547 chars**

Research topic: Scaling laws for Large Language Models

Core hypothesis: Scaling laws break under data-constrained regimes because token repetition above a threshold induces memorization rather than generalization, creating a phase transition in loss curves not predicted by current power-law models.

Experimental design:
[experimental design draft — 3-4 paragraphs]

Key improvements required:
[synthesis of 3 critic perspectives]

Existing work to differentiate from:

## Recent related work (for context — your idea must go beyond these)

1. **D-CPT Law: Domain-specific Continual Pre-Training Scaling Law for Large Language Models** (2024)
  Continual Pre-Training (CPT) on Large Language Models (LLMs) has been widely used to expand the model's fundamental understanding of specific downstream domains (e.g., math and code). For the CPT on domain-specific LLMs, one important question is how to choose the optimal mixture ratio between the general-corpus (e.g., Dolma, Slim-pajama) and the downstream domain-corpus. Existing methods usually adopt laborious human efforts by grid-searching on a set of mixture ratios, which require high GPU training consumption costs. Besides, we cannot guarantee the selected ratio is optimal for the specif
2. **Scaling Laws for Neural Language Models** (2020)
  We study empirical scaling laws for language model performance on the cross-entropy loss. The loss scales as a power-law with model size, dataset size, and the amount of compute used for training, with some trends spanning more than seven orders of magnitude. Other architectural details such as network width or depth have minimal effects within a wide range. Simple equations govern the dependence of overfitting on model/dataset size and the dependence of training speed on model size. These relationships allow us to determine the optimal allocation of a fixed compute budget. Larger models are s
3. **Predictable Scale: Part I, Step Law -- Optimal Hyperparameter Scaling Law in Large Language Model Pretraining** (2025)
  The impressive capabilities of Large Language Models (LLMs) across diverse tasks are now well\text{-}established, yet their effective deployment necessitates careful hyperparameter optimization. Although existing methods have explored the influence of hyperparameters on model performance, a principled and generalizable framework across model architectures and data recipes remains absent. In this study, we conduct an unprecedented empirical investigation\text{-} training over 3,700 LLMs from scratch across 100 trillion tokens, consuming nearly one million NVIDIA H800 GPU hours to establish a un
4. **Sparks of Artificial General Intelligence: Early experiments with GPT-4** (2023)
  Artificial intelligence (AI) researchers have been developing and refining large language models (LLMs) that exhibit remarkable capabilities across a variety of domains and tasks, challenging our understanding of learning and cognition. The latest model developed by OpenAI, GPT-4, was trained using an unprecedented scale of compute and data. In this paper, we report on our investigation of an early version of GPT-4, when it was still in active development by OpenAI. We contend that (this early version of) GPT-4 is part of a new cohort of LLMs (along with ChatGPT and Google's PaLM for example)
5. **K-BERT: Enabling Language Representation with Knowledge Graph** (2020)
  Pre-trained language representation models, such as BERT, capture a general language representation from large-scale corpora, but lack domain-specific knowledge. When reading a domain text, experts make inferences with relevant knowledge. For machines to achieve this capability, we propose a knowledge-enabled language representation model (K-BERT) with knowledge graphs (KGs), in which triples are injected into the sentences as domain knowledge. However, too much knowledge incorporation may divert the sentence from its correct meaning, which is called knowledge noise (KN) issue. To overcome KN,

Write the final improved research idea. Hypothesis clearly stated, falsification concrete, datasets/baselines/metrics named explicitly.
Present your final research idea in exactly this format:

IDEA: [One sentence summary of the core proposal]

BACKGROUND: [The specific open problem this addresses, 2-3 sentences]

APPROACH: [The proposed method or technique, 2-3 sentences]

EXPERIMENT: [How to test it — datasets, metrics, or methodology, 1-2 sentences]

NOVELTY: [What makes this meaningfully different from prior work, 1-2 sentences]

---