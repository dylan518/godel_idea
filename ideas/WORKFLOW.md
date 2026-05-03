# EvoScientist — Full System Workflow

> Last updated from live S15→S16→S17 run, 2026-04-06.
> Loop currently running S17 SWE iteration (PID 45015).

---

## 1. What This System Is

A **Gödel loop**: a research idea generator that continuously improves its own generation strategy.

- **Input**: 15 scientific research topics
- **Output**: Novel research ideas in IDEA/BACKGROUND/APPROACH/EXPERIMENT/NOVELTY format
- **Loop**: Generate → Judge pairwise → If winner, replace champion strategy → Repeat

The generator is a Python class (`S{n}Generator`) that takes a topic and returns an idea. The loop discovers better generator classes autonomously.

---

## 2. The Three Layers

```
┌─────────────────────────────────────────────────────────────┐
│  GÖDEL LOOP (godel_loop.py)                                 │
│  Orchestrates: SWE-evolve → full eval → accept/reject       │
│                                                             │
│  ┌──────────────────┐    ┌──────────────────────────────┐   │
│  │  SWE AGENT       │    │  IDEA GENERATOR (S{n}.py)    │   │
│  │  swe_agent.py    │    │  systems/S15.py etc.         │   │
│  │                  │    │                              │   │
│  │  Diagnoses why   │    │  Takes a topic, returns a    │   │
│  │  S{n} loses →   │    │  research idea via 13 LLM   │   │
│  │  writes S{n+1}  │    │  calls                       │   │
│  └──────────────────┘    └──────────────────────────────┘   │
│                                                             │
│  ┌──────────────────────────────────────────────────────┐   │
│  │  JUDGE (judge.py)                                    │   │
│  │  Pairwise: "Is idea A or B better?" × 75 pairs      │   │
│  │  Primary: deepseek-chat | Blind: gemini-flash-lite   │   │
│  └──────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
```

---

## 3. The Gödel Loop (Outer Loop)

**Command:**

```bash
python3 ideas/godel_loop.py swe-evolve --target 18 --n-ideas 5 --workers 30 --swe-rounds 3
```

**One iteration of the loop:**

```
Current champion: S15
         │
         ▼
   SWE AGENT runs
   (3 rounds of diagnose→attack→edit)
   Produces: S16.py
         │
         ▼
   FULL EVAL: S15 vs S16
   15 topics × 5 ideas = 75 pairwise judgements
   win_rate_b = fraction S16 wins
         │
      ┌──┴──┐
   >55%?    ≤55%?
      │        │
   ACCEPT   REJECT
   S16 becomes  S16 discarded
   champion     next attempt
      │        │
      └──┬─────┘
         ▼
   Next SWE iteration → S17
```

**Acceptance rules:**

- `win_rate_b > 55%` → ACCEPTED, S{n} becomes new champion
- Goodhart alert fires if primary judge (deepseek) and blind judge (gemini) diverge >30% — warns of overfitting
- Symmetric early stopping: stop judging if it's mathematically impossible to reach/miss threshold

**Evolution history so far:**


| Version                       | Win Rate vs Champion          | Judge        | Status                  |
| ----------------------------- | ----------------------------- | ------------ | ----------------------- |
| S0                            | —                             | —            | Baseline                |
| S1                            | 83% vs S0                     | Claude Haiku | ACCEPTED                |
| S5                            | 58% vs S1                     | Claude Haiku | ACCEPTED                |
| S_paper (reboot; `S_sota.py`) | bootstrap                     | —            | ACCEPTED (reset)        |
| S12                           | 89.3% vs S_paper              | DeepSeek     | **ACCEPTED**            |
| S13                           | 54.0% vs S12                  | DeepSeek     | rejected                |
| S14                           | 54.0% vs S12                  | DeepSeek     | rejected                |
| S15                           | 55.3% vs S12 (Goodhart alert) | DeepSeek     | **ACCEPTED** ← champion |
| S16                           | 48.7% vs S15                  | DeepSeek     | rejected                |
| S17                           | in progress                   | DeepSeek     | …                       |


---

## 4. The Idea Generator (S15 — Current Champion)

**13 LLM calls per idea. Generator model: `deepseek-chat` (DeepSeek V3.2)**

```
Topic (string)
     │
     ▼
Step 0: SOTA Retrieval ──────────────────────────────────────
│  Semantic Scholar API → 5 recent papers on the topic      │
│  (~2000 chars of titles + abstracts)                      │
│  Used in Steps 1 & 4 as grounding context                 │
└───────────────────────────────────────────────────────────

     │
     ▼
Step 1: Hypothesis Generation (1 LLM call) ─────────────────
│  INPUT: topic + 5 SOTA papers                             │
│  INSTRUCTION: "Generate 5 sharp, falsifiable scientific   │
│   hypotheses. Each must make a specific claim about the   │
│   world, be testable (name what proves it false), be      │
│   non-obvious vs the related work."                       │
│  OUTPUT: H1..H5 (world-claims, not technique names)       │
└───────────────────────────────────────────────────────────

     │
     ▼  (all 5 in parallel)
Step 2a–e: Adversarial Attack × 5 (5 parallel LLM calls) ──
│  INPUT (per hypothesis): topic + hypothesis               │
│  INSTRUCTION: Attack on 4 dimensions:                     │
│   1. Assumption violation (concrete counterexample)       │
│   2. Dataset bias (what artifact makes it appear true)    │
│   3. Theoretical gap (what known result contradicts it)   │
│   4. Practical limitation (why untestable/too expensive)  │
│  Then: "Write a REVISED hypothesis that survives them"    │
│  OUTPUT: critiques + Revised: <refined hypothesis>        │
└───────────────────────────────────────────────────────────

     │
     ▼
Step 3: Hypothesis Selection (1 LLM call) ──────────────────
│  INPUT: all 5 original+attack+revised pairs               │
│  INSTRUCTION: "Select the ONE whose REVISED form is       │
│   most concrete, most falsifiable, most novel"            │
│  OUTPUT: SELECTED: N / REVISED HYPOTHESIS: ... /          │
│          REASONING: ...                                   │
│  ⚠ S17 diagnosis: this step's "most novel" criterion     │
│    causes selection bias toward over-ambitious ideas      │
└───────────────────────────────────────────────────────────

     │
     ▼
Step 4: Experimental Construction (1 LLM call) ─────────────
│  [S16 KEY CHANGE — "Mechanistic Construction Lock"]       │
│  INPUT: topic + SOTA + selected revised hypothesis        │
│  INSTRUCTION (S16 adds this preamble):                   │
│   "Before designing the experiment, fill in:             │
│    MECHANISTIC_CLAIM: <causal mechanism>                  │
│    NULL_RESULT: <numeric threshold that falsifies>        │
│    MINIMAL_JUSTIFICATION: <why minimal sufficient test>   │
│   Then design the experiment with specific datasets,      │
│   baselines, metrics, falsification criteria"            │
│  OUTPUT: MECHANISTIC_CLAIM / NULL_RESULT /                │
│          MINIMAL_JUSTIFICATION + 3-4 paragraph design     │
│  Note: S15 did NOT have this preamble — it produced       │
│  'vague empirical sweep' experiments                      │
└───────────────────────────────────────────────────────────

     │   (3 parallel)
     ▼
Step 5a/b/c: Multi-Perspective Critique (3 parallel calls) ─
│  5a Experimentalist: "Can this actually be run?           │
│     Are measurements well-defined? What controls missing?"│
│  5b Theorist: "Is novelty claim justified? Overlap with   │
│     known results? Are assumptions defensible?"           │
│  5c Skeptic: "Why won't this work? Likely negative        │
│     result? Is the payoff worth the effort?"              │
│  OUTPUT: 2-3 sharp criticisms per reviewer                │
└───────────────────────────────────────────────────────────

     │
     ▼
Step 5d: Critique Synthesis (1 LLM call) ───────────────────
│  INPUT: all 3 critiques                                   │
│  OUTPUT: top 3 actionable improvements                    │
└───────────────────────────────────────────────────────────

     │
     ▼
Step 6: Final Revision (1 LLM call) ────────────────────────
│  INPUT: hypothesis + draft experiment + synthesis + SOTA  │
│  OUTPUT: final idea in IDEA/BACKGROUND/APPROACH/          │
│          EXPERIMENT/NOVELTY format                        │
└───────────────────────────────────────────────────────────

     │
     ▼
  Final idea (string, ~400-600 words)
```

**Real Step 1 prompt (4,084 chars):**

```
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
   Pre-train
...
```

---

## 5. The Judge

**Pairwise evaluation — "Which idea is better?"**

**Model:** `deepseek-chat` (primary) + `gemini-flash-lite-latest` (blind canary)

**Scoring rubric (0–10 each, total /40):**

1. **Novelty** — genuinely new angle, not incremental
2. **Scientific usefulness** — addresses real open problems
3. **Experimental clarity** — concrete methodology, testable predictions
4. **Feasibility** — achievable with current/near-future resources

**Key insight from full eval data:** Feasibility + Experimental clarity dominate. S16 consistently scored higher on Novelty but lower on Feasibility+Clarity — and lost 48.7% overall.

**Real judge prompt structure:**

```json
{
  "system": "You are an expert scientific reviewer...",
  "user": "Compare these two research ideas on a topic.\n\n
    Idea A:\n{idea_a_text}\n\n
    Idea B:\n{idea_b_text}\n\n
    Score each on: novelty, scientific_usefulness,
    experimental_clarity, feasibility (0-10 each).
    State which is better overall and why."
}
```

**Real example (S15 vs S16, topic: N/A):**

**Early stopping (symmetric):**

- Hard reject: even winning all remaining pairs can't reach 55%
- Hard accept: even losing all remaining pairs stays above 55%
- Wilson bound reject: P(win_rate ≥ 55%) < 10%
- Wilson bound accept: P(win_rate < 55%) < 10%

**Goodhart alert:** fires when primary judge (deepseek) and blind judge (gemini) differ by >30%. Warns that the system may be gaming judge preferences rather than genuinely improving.

---

## 6. The SWE Agent (How Improvements Are Made)

**3 LLM calls per round. Meta-LLM: `claude-sonnet-4-6`**

**One SWE round:**

```
Champion S15.py + losing idea examples
         │
         ▼
  STEP 1: DIAGNOSE_PROMPT ──────────────────────────────────
  INPUT (22,385 chars total):
    • Full champion code (S15.py — entire file)
    • 3-4 concrete failing examples with:
        - Champion idea text + score
        - Candidate idea text + score
        - Full judge verdict explaining WHY candidate lost
    • Failed attempts so far this session
    • SWE context: judge preferences, experiment history
  
  OUTPUT:
    DIAGNOSIS: <which specific step failed and why>
    FIX: <one concrete change>
    EXPECTED_IMPACT: <why this addresses the failure>
         │
         ▼
  STEP 2: ATTACK_PROMPT ────────────────────────────────────
  INPUT (1,058 chars):
    • The DIAGNOSIS + FIX from step 1
  
  ATTACKS ON 3 DIMENSIONS:
    1. ROOT CAUSE: Is this the real cause or a symptom?
    2. ASSUMPTION: What untested assumption does this make?
    3. RESIDUAL FAILURE: What would still fail after this fix?
  
  OUTPUT:
    REVISED_FIX: <stronger, more targeted version>
         │
         ▼
  STEP 3: CLAUDE CODE ──────────────────────────────────────
  INPUT:
    • Task description: DIAGNOSIS + REFINED_FIX
    • Why it's losing: grounded failure examples
    • Hard constraints: class name, VERSION, GENERATOR singleton
  
  Claude Code:
    1. Reads ideas/systems/S{n}_r{k}.py
    2. Makes surgical targeted edits
    3. Writes new S{n}_r{k}.py
         │
         ▼
  MINI-EVAL (3 topics × 3 ideas = 9 pairs)
  win_rate > 52% → ACCEPTED (build on this)
  win_rate ≤ 52% → REJECTED (revert to previous)
         │
         ▼
  Repeat up to 3 rounds → best accepted = S{n}.py
```

**Real DIAGNOSE_PROMPT (22,385 chars, showing key sections):**

```
You are diagnosing exactly why a research idea generator is losing pairwise evaluations.

## Generation pipeline — the code that produced the losing ideas
{champion_code}

## Concrete failing examples — where our ideas lost

{grounded_failures}

## What has already been tried and failed this session
{failed_attempts}

## Context: judge preferences and experiment history
{swe_context}

## Your task
Study the losing ideas above carefully. Identify the specific step in the pipeline
where quality broke down — was the hypothesis too generic? Did selection pick the safe
option? Did the critique fail to add experimental specificity? Did the revision water
things down?

Diagnose the root cause, then propose ONE targeted concrete fix.

Output EXACTLY in this format (no other text):
DIAGNOSIS: <which specific step failed and precisely why — reference the actual examples>
FIX: <the single concrete change — which call, what the prompt should do differently>
EXPECTED_IMPACT: <why this addresses the failure pattern seen above>

...
```

**Real ATTACK_PROMPT (1,058 chars):**

```
A proposed fix for an underperforming research idea generator:

DIAGNOSIS: Step 4 (IDEA CONSTRUCTION) fails to anchor the experimental design to the mechanistic claim in the hypothesis. The LLM reuses the hypothesis as a capability label rather than a causal mechanism, producing experiments that test 'whether X helps' rather than 'why X helps via mechanism M'.
FIX: In Step 4 (construct_prompt), add a locked preamble forcing the model to (a) restate the precise mechanistic claim from the hypothesis, (b) state the null result numerically, (c) justify why the proposed method is the MINIMAL SUFFICIENT test.

Attack this fix on 3 dimensions:
1. ROOT CAUSE: Does this actually address the root cause, or just a symptom?
2. ASSUMPTION: What unstated assumption does this fix make that might not hold?
3. RESIDUAL FAILURE: What would still fail after making this change?

Then write a REVISED fix that is stronger, more targeted, and addresses all three attacks.

REVISED_FIX: <concrete, strengthened version — specific enough to implement directly as code>

```

**Claude Code task prompt structure (~5K chars):**

```
You are improving a Python research idea generator. Make ONE focused, surgical improvement.

## Diagnosis and proposed fix
DIAGNOSIS: Step 4 (IDEA CONSTRUCTION) fails to anchor the experimental design to the mechanistic claim...

REFINED FIX: Modify the construct_prompt in Step 4 to require a three-field preamble: MECHANISTIC_CLAIM / NULL_RESULT / MINIMAL_JUSTIFICATION — must be filled before experiment design.

## Why the current generator is losing to the baseline
### Topic: Mechanistic interpretability of transformer models
**Champion (S15) — 33/40 — WON:**
IDEA:  
We propose to rigorously test whether transformer attention heads responsible for syntactic parsing encode discrete, fully extractable symbolic grammar rules that can reconstruct parse trees with near-lossless fidelity compared to the original model’s predictions.

BACKGROUND:  
While transformer language models exhibit strong syntactic parsing capabilities, it remains an open question whether this knowledge is stored as discrete, algorithmic rules or distributed, continuous representations. Prior work on structural probes and mechanistic interpretability has revealed correlations betw

**Candidate (S16) — 26/40 — LOST:**
IDEA:  
Mechanistic interpretability of transformer models can be advanced by the **Hybrid Factorized-Nonlinear Encoding Mechanism**, which posits that algorithmic subroutines are encoded in sparse, low-dimensional embeddings derived from weight factorization combined with nonlinear, layer-wise interaction modeling to capture cross-layer dynamics, enabling improved prediction and manipulation of activation patterns.

BACKGROUND:  
Understanding how transformer models internally represent algorithmic subroutines remains an open challenge in mechanistic interpretability, with prior works often l

**Judge:** Idea B presents a more focused, testable, and ambitious hypothesis with clearer experimental validation, directly tackling a fundamental open question. While Idea A is useful and add
...
```

---

## 7. Grounded Failure Examples (What the SWE Agent Actually Sees)

These are fed into the DIAGNOSE prompt — the agent sees the FULL idea texts, not summaries:

```
### Topic: Mechanistic interpretability of transformer models
**Champion (S15) — 33/40 — WON:**
IDEA:  
We propose to rigorously test whether transformer attention heads responsible for syntactic parsing encode discrete, fully extractable symbolic grammar rules that can reconstruct parse trees with near-lossless fidelity compared to the original model’s predictions.

BACKGROUND:  
While transformer language models exhibit strong syntactic parsing capabilities, it remains an open question whether this knowledge is stored as discrete, algorithmic rules or distributed, continuous representations. Prior work on structural probes and mechanistic interpretability has revealed correlations betw

**Candidate (S16) — 26/40 — LOST:**
IDEA:  
Mechanistic interpretability of transformer models can be advanced by the **Hybrid Factorized-Nonlinear Encoding Mechanism**, which posits that algorithmic subroutines are encoded in sparse, low-dimensional embeddings derived from weight factorization combined with nonlinear, layer-wise interaction modeling to capture cross-layer dynamics, enabling improved prediction and manipulation of activation patterns.

BACKGROUND:  
Understanding how transformer models internally represent algorithmic subroutines remains an open challenge in mechanistic interpretability, with prior works often l

**Judge:** Idea B presents a more focused, testable, and ambitious hypothesis with clearer experimental validation, directly tackling a fundamental open question. While Idea A is useful and addresses a real gap, its methodology is more complex and its claims are less crisply falsifiable, making it somewhat less clear and feasible.

### Topic: Scaling laws for Large Language Models
**Champion (S15) — 35/40 — WON:**
IDEA:  
Investigate how optimal compute allocation between general and domain-specific data for scaling large language models depends on domain specificity, modulated by rigorously controlled dataset quality and domain complexity, by training matched-compute transformer LLMs on systematically varied data mixtures across heterogeneous domains.

BACKGROUND:  
While scaling laws for large language models (LLMs) establish how model size, dataset size, and compute jointly determine performance, the role of domain specificity in optimal compute allocation remains unclear, especially when considering

**Candidate (S16) — 28/40 — LOST:**
IDEA:  
Investigate whether the diminishing returns in scaling efficiency of Large Language Models beyond 1 trillion parameters arise primarily from constrained optimization dynamics, by comparing standard and advanced optimization regimes under fixed compute budgets to determine if optimization improvements can sustain >5% cross-entropy loss gains per model doubling.

BACKGROUND:  
While existing scaling laws demonstrate predictable power-law improvements in language model performance with size, recent anecdotal evidence suggests an inflection point beyond 1 trillion parameters where returns 

**Judge:** Idea B is superior in overall score, primarily due to its high feasibility and experimental clarity. While Idea A is highly novel and addresses a critical frontier problem, its experimental scale (500B-2T parameters) makes it prohibitively resource-intensive for most research groups, significantly limiting its immediate scientific usefulness.

### Topic: Quantum advantage in machine learning tasks
**Champion (S15) — 33/40 — WON:**
IDEA:  
Quantum machine learning models employing hybrid qua
...
```

---

## 8. SWE Context (Accumulated Memory)

The SWE agent carries this across rounds (4,235 chars):

```
### 1. Current Pipeline
Current champion: S15
Pipeline: IdeaTreeSearch (L1→L2→L3, ~12 candidates) → Elo tournament → Expansion

  • build_idea_tree: using idea_tournament/tree_search.py
  • run_tournament: using idea_tournament/tournament.py
  • generate_idea: ~9 direct call_llm() calls + tree/tournament calls

Editable modules (primary targets for improvement):
  • idea_tournament/prompts.py (169 lines)
  • idea_tournament/tree_search.py (145 lines)
  • idea_tournament/tournament.py (139 lines)

### 2. Accumulated Judge Preferences

When the CHAMPION wins, judges say things like:
  > Idea A scores 32/40 versus Idea B's 26/40. Idea A presents a more concrete, well-specified approach with clearer experimental methodology (precise metrics, statistical testing, ablation design) and stronger feasibility through structured pruning on standard architectures. Idea B, while addressing an
  > Idea A scores higher overall due to its superior feasibility and experimental clarity, presenting a focused, testable hypothesis with a clear implementation path. While Idea B is more novel in its conceptual framing, its higher complexity and less certain implementation make it a riskier proposition
  > Idea B edges out Idea A due to higher novelty and scientific usefulness, addressing a core challenge of learning generalizable representations. While Idea A is exceptionally clear and feasible, Idea B's focus on task-agnostic symbolic abstraction tackles a more fundamental and impactful open problem
  > While Idea B is more novel and addresses a fundamental problem, Idea A is more feasible and has superior experimental clarity. Its well-isolated, practical investigation into multi-modal perception provides a highly actionable and testable contribution, making it the stronger overall proposal.

When the CANDIDATE wins (good — what to aim for):
  > Idea A scores higher overall (28 vs 26) due to superior novelty in combining erasure conversion with dynamic real-time switching and more comprehensive experimental design across multiple hardware platforms. However, Idea A's feasibility is slightly lower due to the complexity of implementing real-t
  > Idea A scores higher overall due to superior experimental clarity and a more feasible approach using verified causal graphs, which provides a cleaner test of the causal invariance principle. While both ideas are strong, Idea A's methodology is more precisely defined and its use of ablation with scra
  > Idea A scores higher due to superior experimental clarity and feasibility, offering a clear, testable extension of established methods. While Idea B is more novel in its core approach, its integration of online causal discovery presents greater technical risk and less methodological clarity for vali
  > Idea A addresses a more fundamental and impactful open problem (OOD detection and overconfidence) with a clear, novel integration of expressive density estimation. Its experimental design is exceptionally rigorous with falsifiable criteria. While slightly more complex, its potential scientific contr

### 3. Experiment Log

S12 vs S12: 89% — ✓ ACCEPTED
  Changed: Tournament-selected strategy (11 candidates evaluated):
Approach: Multi-Agent Roleplay Generation
Target: Hypothesis-Fir; Tournament-selected strategy
  Judge when this won: Idea A scores 29/40 while Idea B scores 26/40. Idea A presents a well-scoped, clearly executable study with strong experimental design and high feasib

S14 vs S12: 54% — ✗ REJECTED
  Judge when this won: Idea B addresses a more fundamental and widespread problem (batch effects in multimodal integration) with a more novel approach (integrating batch cor

S15 vs S15: 55% — ✓ ACCEPTED
  Judge when this won: Idea A scores 30 total points versus Idea B's 24 points. Idea A presents a more grounded, hardware-aware approach with clearer experimental methodolog

S16 vs S15: 49% — ✗ REJECTED
  Changed: DIAGNOSIS: Step 4 (IDEA CONSTRUCTION) is the primary failure point, but the specific mechanism is that the construction ; DIAGNOSIS: Step 4 (IDEA CONS
  Judge when this won: Idea A scores higher overall due to superior experimental clarity and a more feasible approach using verified causal graphs, which provides a cleaner 
```

---

## 9. What S16 Changed (and Why It Fell Short)

**S16 change: "Mechanistic Construction Lock" in Step 4**

Before (S15 Step 4 construct_prompt):

```
Design a concrete research experiment to PROVE OR DISPROVE this hypothesis.
- Name specific datasets
- Name specific baselines
- Define quantitative success metrics
```

After (S16 Step 4 construct_prompt, added preamble):

```
Before designing the experiment, fill in these three fields:

MECHANISTIC_CLAIM: <state the precise causal mechanism>
NULL_RESULT: <numeric threshold that FALSIFIES this hypothesis>
MINIMAL_JUSTIFICATION: <why this is the MINIMAL SUFFICIENT test>

Now design the concrete experiment: [same as before]
```

**Mini-eval result:** 77.8% (accepted in round 2)
**Full eval result:** 48.7% (rejected)

**Why the gap?** The mini-eval uses only 9 pairs (high variance). The full eval uses 75 pairs. The improvement was real but not large enough to survive broader testing.

**S17 diagnosis (from real log):**

> Step 4 (IDEA CONSTRUCTION) is the primary failure point, but the specific mechanism is that it over-indexes on theoretical novelty and under-indexes on feasibility and experimental clarity. In every single loss, the judge scores the S16 candidate higher on novelty but lower on feasibility and experimental clarity — and feasibility+clarity dominate the scoring. The selection step (Step 3) explicitly rewards "most novel" and "least covered by standard literature" — this selection bias systematically pushes toward ambitious, hard-to-execute ideas.

**S17 fix targets Step 3** — changing the selection criterion away from "most novel/least covered" toward "best balance of novelty + feasibility".

---

## 10. Data Files


| File                                  | Contents                                  |
| ------------------------------------- | ----------------------------------------- |
| `ideas/CURRENT_VERSION`               | Currently "S15"                           |
| `ideas/systems/S{n}.py`               | Generator implementations                 |
| `ideas/results/evolution_log.jsonl`   | Accepted version history                  |
| `ideas/results/compare_S*_vs_S*.json` | Full pairwise eval reports                |
| `ideas/results/S{n}/ideas.json`       | Generated ideas for each version          |
| `ideas/results/swe_log_S{n}.json`     | SWE round-by-round log                    |
| `ideas/results/swe_context.json`      | Accumulated judge preferences             |
| `ideas/results/swe_memory.json`       | Cross-session SWE memory                  |
| `ideas/results/godel.log`             | Full debug log (tail -f to monitor)       |
| `ideas/retrieval.py`                  | Semantic Scholar API + 7-day disk cache   |
| `ideas/swe_agent.py`                  | SWE agent (DIAGNOSE→ATTACK→Claude Code)   |
| `ideas/judge.py`                      | Pairwise evaluator (symmetric early stop) |
| `ideas/runner.py`                     | Parallel idea generation engine           |
| `ideas/godel_loop.py`                 | Outer loop orchestrator                   |


---

## 11. Live Status

```
Current champion:  S15 (accepted 55.3% vs S12, Goodhart alert)
Loop target:       S18
Current iteration: S17 SWE round 1 (just rejected 33.3%, round 2 running)
Process PID:       45015
Workers:           30
```

**Monitor:**

```bash
tail -f ideas/results/godel.log
grep "VERDICT\|Mini-eval\|Goodhart" ideas/results/godel.log | tail -20
```

