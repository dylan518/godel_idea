# EvoScientist System Map

*One document: how the system works + what every agent actually sees.*

Real data from: S15 champion, S15→S16 SWE run, S15→S17 proposal (today).

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────┐
│  GÖDEL LOOP  (godel_loop.py)                                        │
│                                                                     │
│  Champion (e.g. S15.py)  ──────────────────────────────────────┐   │
│                                                                 │   │
│  ┌──────────────────────────────────────────────────────────┐  │   │
│  │  SWE AGENT  (swe_agent.py)                               │  │   │
│  │                                                          │  │   │
│  │  per round:                                              │  │   │
│  │  1. _build_self_improvement_topic()                      │  │   │
│  │       └─► champion docstring + failure examples          │  │   │
│  │  2. _call_generate_idea_proposal()  ◄── PRIMARY PATH     │  │   │
│  │       └─► calls champion.generate_idea(topic) ──────┐   │  │   │
│  │           [13 DeepSeek calls — same pipeline as      │   │  │   │
│  │            generating research ideas, but topic is   │   │  │   │
│  │            "improve this pipeline"]                  │   │  │   │
│  │           └─► returns IDEA/BACKGROUND/APPROACH/...  ◄┘   │  │   │
│  │  2b. DIAGNOSE+ATTACK  ◄────────────────── FALLBACK        │  │   │
│  │  3. _call_claude_code_edit()                              │  │   │
│  │       └─► Claude Code (claude-sonnet-4-6)                │  │   │
│  │           reads S15.py, makes surgical edit              │  │   │
│  │           writes S17_r{n}.py                             │  │   │
│  │  4. _run_mini_eval()  (3 topics × 3 ideas = 9 pairs)     │  │   │
│  │       └─► accept (>52%) or revert                        │  │   │
│  └──────────────────────────────────────────────────────────┘  │   │
│            │ final candidate S17.py                            │   │
│            ▼                                                   │   │
│  FULL EVAL  (75 pairs, 15 topics × 5 ideas)                    │   │
│  champion.generate_idea() vs candidate.generate_idea()         │   │
│            │ per pair, both run the 13-call pipeline           │   │
│            ▼                                                   │   │
│  JUDGE  (deepseek-chat)  ──────────────────────────────────────┘   │
│  scores A vs B on 4 criteria → win_rate_b > 55% → accept           │
│                                                                     │
│  BLIND JUDGE  (gemini-flash-lite-latest)  — canary only             │
│  divergence > 30% → Goodhart alert                                  │
└─────────────────────────────────────────────────────────────────────┘
```

**Model assignments:**


| Role              | Model                                 | Why                                                 |
| ----------------- | ------------------------------------- | --------------------------------------------------- |
| Idea generator    | `deepseek-chat`                       | Cheap, fast, good at structured output              |
| SWE meta-proposer | `deepseek-chat` via `generate_idea()` | Self-referential — same model proposes improvements |
| Code editor       | `claude-sonnet-4-6`                   | Best at surgical file editing with tools            |
| Primary judge     | `deepseek-chat`                       | Independent from generator; accept/reject decisions |
| Blind judge       | `gemini-flash-lite-latest`            | Different family; canary for Goodhart detection     |


---

## Part 1 — SWE Agent Outer Loop

The SWE agent runs up to 6 rounds per session. Each round proposes one targeted code
change, tests it with a 9-pair mini-eval, and either keeps or reverts it.

### Step 1 — Build self-improvement topic

**What:** Constructs the "research topic" string that gets fed into `generate_idea()`.
The key design choice: treat pipeline improvement as a research problem. Use the same
hypothesis-first adversarial loop that generates science ideas to generate improvement proposals.

**Why not send the full source code?**
5K chars × 4 downstream prompts = ~20K tokens of bloat. The model can't reason about
Python syntax in hypothesis mode anyway. The docstring (400 chars) conveys the architecture.

**Code:** `_build_self_improvement_topic()` in `swe_agent.py`

**Real input — what it reads:**

```
Champion: S15
Docstring (first 400 chars of S15.py):
  S12_r3: Hypothesis-First Adversarial Loop.
  Replaces the top-down tree/tournament with a FALSIFIABLE-HYPOTHESIS-FIRST loop:
  1. HYPOTHESIS GENERATION: One LLM call produces 5 sharp, testable scientific
     hypotheses about the topic (e.g., "scaling laws break under data-constrained
     regimes because X"). Starting from *claimed truths about the world* rather
     than technique names.
  2. ADVERSARIAL

Grounded failures (from compare_S15_vs_S16.json, top 1 shown):
  Topic: Mechanistic interpretability of transformer models
  S15 lost 26/40 vs S16 33/40
  S16 idea used "Hybrid Factorized-Nonlinear Encoding Mechanism" — branded name

Failed attempts: (none yet — first round)
```

**Real output — the topic string (2,368 chars):**

```
Self-improvement of a research idea generation pipeline (S15).

CURRENT PIPELINE OVERVIEW:
S12_r3: Hypothesis-First Adversarial Loop.
[...400 chars of docstring...]

RECENT FAILURES (ideas where the current system lost to a weaker baseline):
### Topic: Mechanistic interpretability of transformer models
**Champion (S15) — 33/40 — WON:**
IDEA: We propose to rigorously test whether transformer attention heads responsible
for syntactic parsing encode discrete, fully extractable symbolic grammar rules...

**Candidate (S16) — 26/40 — LOST:**
IDEA: Mechanistic interpretability of transformer models can be advanced by the
**Hybrid Factorized-Nonlinear Encoding Mechanism**, which posits that algorithmic
subroutines are encoded in sparse, low-dimensional embeddings...

**Judge:** Idea B presents a more focused, testable, and ambitious hypothesis...

PREVIOUSLY ATTEMPTED FIXES THAT DID NOT WORK:
(none yet — first round)

Your task: propose ONE targeted change to the generation pipeline that would
address the failure patterns above. Focus on the APPROACH section — describe
a concrete, implementable modification to a prompt, algorithm, or step.
```

---

### Step 2a — generate_idea() as meta-proposer (PRIMARY PATH)

**What:** Calls the champion's own 13-call pipeline on the self-improvement topic.
The pipeline that generates research hypotheses now generates *improvement hypotheses*.
The APPROACH field (which normally describes a method) describes a prompt/code change.

**Why self-referential?** The same adversarial loop that stress-tests scientific claims
stress-tests improvement proposals — it attacks assumptions, finds residual failures,
selects the strongest surviving fix, then critiques it from 3 angles. This is strictly
more rigorous than a single DIAGNOSE call.

**Code:** `_call_generate_idea_proposal()` in `swe_agent.py`

**What happens inside (13 DeepSeek calls):**
See Part 2 for full detail. The topic string from Step 1 is substituted for `{topic}`
throughout the S15 pipeline. SOTA retrieval runs on "Self-improvement of a research
idea generation pipeline" — usually returns no papers (expected), falls back to `""`.

**Real output from today's run (1,751 chars):**

```
IDEA: We hypothesize that the S15 idea-generation pipeline's adversarial judge is
biased by novel, branded mechanism names (e.g., "Hybrid Factorized-Nonlinear
Encoding"), and we will test this by measuring the win-rate drop when such names
are replaced with concise generic labels in otherwise regenerated candidate ideas.

BACKGROUND: The pipeline's recent failures show losing candidate ideas often embed
their core claim within a branded mechanism name, which may create an illusory
specificity that sways the LLM judge. This addresses the open problem of lexical
framing effects corrupting automated research-idea evaluation, where the form of
an idea unduly influences its perceived quality over its testable substance.

APPROACH: We will modify the hypothesis-generation prompt to explicitly forbid
novel mechanism names and instead require a one-sentence empirical prediction,
a concrete experimental intervention, and a quantifiable success metric.

EXPERIMENT: Using the last 50 lost candidates from S12_r3, we regenerate two
variants per idea (Branded vs. Generic-Label) and re-evaluate each against its
original champion using the standard S15 judge; we measure the win-rate
differential with a one-sided binomial test (threshold p<0.05).

NOVELTY: This directly intervenes on the generator — the root cause — not just
the judge, and isolates terminology effects by comparing branded names to concise
generic labels within regenerated, fluency-controlled text.
```

**Parsed fields → task_description for Claude Code (3,264 chars):**

```
DIAGNOSIS: The pipeline's recent failures show losing candidate ideas often embed
their core claim within a branded mechanism name, which may create an illusory
specificity that sways the LLM judge...

PROPOSED FIX: We will modify the hypothesis-generation prompt to explicitly forbid
novel mechanism names and instead require a one-sentence empirical prediction...

EXPECTED IMPACT: Using the last 50 lost candidates from S12_r3, regenerate two
variants (Branded vs. Generic-Label)... measure win-rate differential...

DIFFERENTIATION FROM PRIOR FAILED ATTEMPTS: This directly intervenes on the
generator — the root cause — not just the judge...

FULL PROPOSAL:
[full raw idea text]

Edit history:
(none yet)
```

---

### Step 2b — DIAGNOSE + ATTACK (FALLBACK)

Only runs if `generate_idea()` raises an exception (API failure, output too short, etc.).
Preserved exactly — no changes from the S15→S16 run.

**DIAGNOSE prompt (~996 chars, truncated champion code version):**

```
You are diagnosing exactly why a research idea generator is losing pairwise evaluations.

## Generation pipeline — the code that produced the losing ideas
[champion docstring]

## Concrete failing examples — where our ideas lost
[top 4 losses with idea text + judge reasoning]

## What has already been tried and failed this session
[failed_attempts_str]

## Context: judge preferences and experiment history
[swe_context_str]

Output EXACTLY:
DIAGNOSIS: <which specific step failed and precisely why>
FIX: <the single concrete change>
EXPECTED_IMPACT: <why this addresses the failure pattern>
```

**ATTACK prompt (~535 chars):**

```
A proposed fix for an underperforming research idea generator:

DIAGNOSIS: [from above]
FIX: [from above]

Attack this fix on 3 dimensions:
1. ROOT CAUSE: Does this actually address the root cause, or just a symptom?
2. ASSUMPTION: What unstated assumption does this fix make that might not hold?
3. RESIDUAL FAILURE: What would still fail after making this change?

Then write a REVISED fix that is stronger, more targeted, and addresses all three attacks.

REVISED_FIX: <concrete, strengthened version>
```

**Real DIAGNOSE output (S15→S16 round 2):**

```
DIAGNOSIS: Step 4 (IDEA CONSTRUCTION) loses because the hypothesis is stated at
the capability level ('multi-modal fusion improves zero-shot transfer') rather than
the mechanism level, so the experiment becomes a vague empirical sweep.

FIX: In Step 3 (hypothesis selection), add a 'MECHANISM' constraint: selected
hypothesis must explicitly name the causal mechanism linking intervention to outcome.

EXPECTED_IMPACT: Forces mechanistic grounding before construction, eliminating
the vague-capability-sweep failure pattern.
```

---

### Step 3 — Claude Code edit

**What:** `claude-sonnet-4-6` receives the task_description + failure examples,
reads the champion file with its own tools, and makes ONE surgical edit.
Output is written to `systems/S17_r{n}.py`.

**Why Claude Code (not raw API)?** Raw API rewrites the whole file from scratch.
Claude Code reads the file first, understands context, then diffs — producing
targeted changes like "add 3 lines to construct_prompt" vs a 250-line rewrite.

**Prompt structure (~5K chars):**

```
You are improving a Python research idea generator. Make ONE focused, surgical improvement.

## Diagnosis and proposed fix
DIAGNOSIS: [background from generate_idea/DIAGNOSE]
PROPOSED FIX: [approach from generate_idea/ATTACK]
EXPECTED IMPACT: [experiment from generate_idea]
[...]

## Why the current generator is losing to the baseline
[grounded_failures_str — real idea texts side by side with scores]

## What to do
1. Read `ideas/systems/S17_r1.py` — already a copy of the champion, your starting point
2. Read the relevant `ideas/idea_tournament/` modules for full context
3. Implement the improvement strategy above — make TARGETED edits to `ideas/systems/S17_r1.py`

## Hard constraints
- Class name MUST be `S17_r1Generator(IdeaGenerator)`
- VERSION = "S17_r1"
- File MUST end with `GENERATOR = S17_r1Generator()`
- Any code you MODIFY from `idea_tournament/` must be inlined
- Every `call_llm()` call must be wrapped in try/except
- `generate_idea()` must always return a non-empty string
```

**Real output (S15→S16, round 2 — the edit that worked):**
Claude Code added a 3-field preamble to Step 4's construct_prompt:

```python
construct_prompt = (
    f"Research topic: {topic}\n"
    f"{context_reminder}\n"
    f"Core hypothesis to test: {selected_hyp}\n\n"
    # ─── NEW: mechanistic lock ───────────────────────────────────────
    "Before designing the experiment, fill in these three fields:\n\n"
    "MECHANISTIC_CLAIM: <state the precise causal mechanism — what property "
    "causes what effect via what process>\n"
    "NULL_RESULT: <numeric threshold — what specific measurement would "
    "FALSIFY this hypothesis>\n"
    "MINIMAL_JUSTIFICATION: <why your proposed method is the MINIMAL "
    "SUFFICIENT test of this mechanism>\n\n"
    # ────────────────────────────────────────────────────────────────
    "Now design the concrete experiment:\n"
    ...
)
```

Mini-eval result: **77.8%** (accepted → became S16_r2, but full eval rejected at 48.7%)

---

### Step 4 — Mini-eval

**What:** Runs the edited system against the current champion on 3 fresh topics × 3 ideas = 9 pairs.
Topics are generated fresh each round (prevents overfitting to a fixed set).
Accept threshold: **>52% win rate**.

**Why 52%, not 55%?** Mini-eval is high-variance (9 pairs). 52% is just enough signal
to not regress while still making progress. The 55% threshold is for the final full eval.

**Warning:** Mini-eval is *very* noisy. S16 hit 77.8% mini but 48.7% full eval.
Never over-read mini results.

---

## Part 2 — generate_idea() Pipeline (S15)

This is the 13-call pipeline that runs for *every single idea* — both when generating
research ideas in the full eval, AND when the SWE agent calls it on the self-improvement topic.

The topic below is real: "Zero-shot generalization in reinforcement learning"
from the S15→S16 comparison. All prompts and outputs are real.

---

### Step 0 — SOTA Retrieval (not an LLM call)

**What:** OpenAlex API query for 5 recent papers on the topic. 7-day disk cache.
Returns formatted `## Recent related work` block injected into Steps 1 and 4.

**Why:** Forces the model to differentiate from existing work. Without this,
ideas drift toward well-known techniques the model already knows.

**Real output:**

```
## Recent related work (for context — your idea must go beyond these)

1. **Learning quadrupedal locomotion over challenging terrain** (2020)
   Legged locomotion can extend the operational domain of robots to some of the
   most challenging environments on Earth...

2. **A survey on Image Data Augmentation for Deep Learning** (2019)
   Deep convolutional neural networks have performed remarkably well...

3. **Improving Zero-shot Generalization in Offline Reinforcement Learning
   using Generalized Similarity Functions** (2021)
   Reinforcement learning agents are widely used for solving complex sequential
   decision making tasks...

[5 papers total, ~2000 chars]
```

*(Note: OpenAlex sometimes returns loosely-related papers. The model still uses
them as differentiation context even when imperfect.)*

---

### Step 1 — Hypothesis Generation (1 call)

**What:** Generates 5 sharp, falsifiable scientific hypotheses — claims about
the world, not technique names. Starting from *claimed truths* forces the model
to commit to a mechanistic claim before thinking about methods.

**Real prompt (4,148 chars, including SOTA block):**

```
Research topic: Zero-shot generalization in reinforcement learning

## Recent related work (for context — your idea must go beyond these)
[5 papers]

Generate exactly 5 sharp, falsifiable scientific hypotheses about this topic.
Each hypothesis must:
- Make a specific claim about the world (not just 'we can improve X')
- Be testable: name what experiment would prove it false
- Be non-obvious: it should NOT be directly supported by the related work above
- Be concise: 1-2 sentences max

Format each as:
H1: <hypothesis statement>
H2: <hypothesis statement>
H3: <hypothesis statement>
H4: <hypothesis statement>
H5: <hypothesis statement>
```

**Real output:**

```
H1: Zero-shot RL generalization fails because policy networks learn task-specific
reward heuristics that don't transfer, not because of observation distribution
shift — models trained with reward-randomization but fixed observations will
generalize no better than standard baselines.

H2: Hierarchical latent variable models that separately encode environment dynamics
and reward-conditioned task representations improve zero-shot RL transfer across
tasks with shared dynamics but varying reward functions, because disentangling
these factors prevents policy overfitting to reward-specific features.

H3: [...]
H4: [...]
H5: [...]
```

---

### Step 2 — Adversarial Attack ×5 (parallel calls)

**What:** Each hypothesis is independently attacked on 4 dimensions by an adversarial critic,
then forced to produce a stronger revised form. All 5 run in parallel (ThreadPoolExecutor).

**Why adversarial?** Naive hypothesis generation produces safe, obvious claims.
Forcing the model to attack its own hypotheses surfaces hidden assumptions and
pushes the revised form toward something genuinely falsifiable.

**Real prompt (1,001 chars, one of 5):**

```
Research topic: Zero-shot generalization in reinforcement learning

Hypothesis: Hierarchical latent variable models that separately encode environment
dynamics and reward-conditioned task representations improve zero-shot RL transfer
across tasks with shared dynamics but varying reward functions, because disentangling
these factors prevents policy overfitting to reward-specific features.

You are an adversarial critic. Attack this hypothesis on EACH of these dimensions:
1. Assumption violation: What unstated assumption does this rely on? Give a concrete counterexample.
2. Dataset bias: What dataset artifact could make this appear true without actually being true?
3. Theoretical gap: What known result from the literature contradicts or undermines this?
4. Practical limitation: What makes this hypothesis untestable or too expensive to test?

Then, given these attacks, write a REVISED hypothesis that survives them:
Revised: <1-2 sentence refined hypothesis that addresses the above attacks>
```

**Real output:**

```
1. Assumption violation: Assumes dynamics and reward are independently variable,
   but most RL environments couple them by design...
2. Dataset bias: MiniGrid/Procgen results may show this because their task families
   happen to share dynamics, not because of latent disentanglement...
3. Theoretical gap: Modular RL (Aljalbout et al. 2021) already attempts this
   without explicit latent variable separation...
4. Practical: Requires constructing matched task pairs with controlled
   dynamics-vs-reward variation — not available in standard benchmarks.

Revised: Hierarchical latent variable models improve zero-shot RL transfer
specifically when task distribution has shared dynamics but varying reward
structures — a condition most benchmarks conflate, making this testable by
constructing paired task sets with independently controlled dynamics-vs-reward
variation.
```

---

### Step 3 — Hypothesis Selection (1 call)

**What:** Aggregates all 5 (original + attack + revised) triplets, selects the ONE
whose revised form is most concrete, falsifiable, and novel.

**Real prompt (1,637 chars):**

```
Research topic: Zero-shot generalization in reinforcement learning

Below are 5 original hypotheses, each attacked by an adversarial critic and
refined into a stronger revised form:

--- Hypothesis 1 ---
Original: Zero-shot RL generalization fails because policy networks learn
task-specific reward heuristics...
Critique+Revision: [4 attacks + Revised: ...]

--- Hypothesis 2 ---
Original: Hierarchical latent variable models...
Critique+Revision: [4 attacks]
Revised: Hierarchical latent variable models improve zero-shot RL transfer
specifically when task distribution has shared dynamics but varying reward
structures...

[... 3 more pairs ...]

Select the ONE hypothesis (by number) whose REVISED form is:
- Most concrete and specific (names mechanisms, not just outcomes)
- Most falsifiable (clearest path to a disproof experiment)
- Most novel (least covered by standard literature)

Respond with:
SELECTED: <number 1-5>
REVISED HYPOTHESIS: <copy the revised hypothesis text exactly>
REASONING: <1-2 sentences on why this is the strongest>
```

**Real output:**

```
SELECTED: 2
REVISED HYPOTHESIS: Hierarchical latent variable models improve zero-shot RL
transfer specifically when task distribution has shared dynamics but varying
reward structures — a condition most benchmarks conflate, making this testable
by constructing paired task sets with independently controlled dynamics-vs-reward
variation.
REASONING: This is the most falsifiable — it makes a specific claim about when
hierarchical models help and names a concrete experimental design that could
disprove it.
```

---

### Step 4 — Experimental Construction (1 call)

**What:** Designs a full experiment around the selected hypothesis.
In S15 (current champion), this is a plain design prompt.
In S16 (rejected), a 3-field mechanistic lock was added here — that's what the
SWE agent changed and what drove mini-eval to 77.8% before the full eval rejection.

**Real prompt (S16 version, ~1,869 chars):**

```
Research topic: Zero-shot generalization in reinforcement learning

Existing work to differentiate from:
[5 papers from SOTA retrieval]

Core hypothesis to test: Hierarchical latent variable models improve zero-shot RL
transfer specifically when task distribution has shared dynamics but varying reward
structures...

Before designing the experiment, fill in these three fields:

MECHANISTIC_CLAIM: <state the precise causal mechanism — what property causes
what effect via what process>
NULL_RESULT: <numeric threshold — what specific measurement would FALSIFY this
hypothesis>
MINIMAL_JUSTIFICATION: <why your proposed method is the MINIMAL SUFFICIENT test
of this mechanism>

Now design the concrete experiment:
- Name specific datasets
- Name specific baselines
- Define quantitative success metrics
- Describe the key technical method
- Explain what result falsifies the hypothesis

Write 3-4 paragraphs. Be direct and specific.
```

**Real output (S16 version):**

```
MECHANISTIC_CLAIM: Disentangled latent representations prevent policy overfitting
to reward-specific features by encoding dynamics and reward-conditioned task
separately, enabling the dynamics encoder to transfer to new task reward structures
without relearning.

NULL_RESULT: If the hierarchical model achieves ≤2% higher zero-shot transfer
accuracy than a flat baseline on held-out reward-varied tasks (shared dynamics),
the disentanglement mechanism does not provide transfer benefit.

MINIMAL_JUSTIFICATION: We need only one architecture change (separate
dynamics/reward encoders with a disentanglement loss) tested on paired task sets;
additional components would confound whether disentanglement — not extra capacity —
drives the improvement.

[3-4 paragraphs: MiniGrid-4rooms with reward variants, baselines DIAYN/PPO/RNN-PPO,
metrics zero-shot success rate, falsification criteria...]
```

---

### Steps 5a–d — Multi-Perspective Critique (4 parallel calls + 1 synthesis)

**What:** Three reviewers independently critique the experiment, then a fourth
synthesizes their notes into 3 actionable improvements.

**Why three personas?** Each has a blind spot the others cover:

- Experimentalist: "can this actually run?" — catches missing controls, undefined measurements
- Theorist: "is the novelty claim justified?" — catches overlap with known results
- Skeptic: "why this probably won't work" — forces realistic payoff assessment

**Real prompts (~530–610 chars each):**

```
[Experimentalist]
You are a hard-nosed experimentalist reviewing a research proposal about
'Zero-shot generalization in reinforcement learning'.

Hypothesis: Hierarchical latent variable models improve zero-shot RL transfer
specifically when task distribution has shared dynamics but varying reward structures...

Proposed experiment:
[3-4 paragraph experimental design from Step 4]

Give 2-3 sharp criticisms focusing purely on experimental feasibility:
Can these experiments actually be run? Are the measurements well-defined?
What controls are missing? What will fail in practice?
```

```
[Theorist]
...Give 2-3 sharp criticisms focusing purely on theoretical grounding:
Is the novelty claim justified? Does it overlap with known results?
Are the underlying assumptions stated and defensible?
```

```
[Skeptic]
...Give 2-3 sharp criticisms focusing on: why this probably won't work,
what the likely negative result is, and whether the scientific payoff
justifies the effort even if it succeeds.
```

```
[Synthesis]
Three reviewers critiqued a research idea about 'Zero-shot generalization in RL'
testing: 'Hierarchical latent variable models...'

Experimentalist: [2-3 criticisms]
Theorist: [2-3 criticisms]
Skeptic: [2-3 criticisms]

Synthesize into the 3 most important actionable improvements. Be concise.
```

---

### Step 6 — Final Revision (1 call)

**What:** Incorporates the synthesis + SOTA context into the final structured output.
This is what gets stored in `ideas.json` and compared by the judge.

**Real prompt (~1,894 chars):**

```
Research topic: Zero-shot generalization in reinforcement learning

Core hypothesis: Hierarchical latent variable models improve zero-shot RL
transfer specifically when task distribution has shared dynamics but varying
reward structures...

Experimental design:
[3-4 paragraphs from Step 4]

Key improvements required:
[synthesis from Step 5d]

Existing work to differentiate from:
[5 papers from SOTA retrieval]

Write the final, improved version of the research idea. Ensure the hypothesis
is clearly stated, the falsification experiment is concrete, and all
datasets/baselines/metrics are named explicitly.

Present your final research idea in exactly this format:

IDEA: [One sentence summary]
BACKGROUND: [The specific open problem, 2-3 sentences]
APPROACH: [The proposed method, 2-3 sentences]
EXPERIMENT: [How to test it — datasets, metrics, 1-2 sentences]
NOVELTY: [What makes this different, 1-2 sentences]
```

**Real output (S15 on this topic, from ideas.json):**

```
IDEA: Incorporating dynamically updated task embeddings derived from online
environment interactions during reinforcement learning training significantly
improves zero-shot generalization to unseen tasks compared to static or
contrastively learned embeddings.

BACKGROUND: Reinforcement learning agents typically struggle to generalize
zero-shot to new tasks or environment variations due to overfitting to training
distributions and insufficiently adaptive task representations. Existing approaches
like meta-RL and contrastive representation learning rely on fixed or offline-learned
embeddings that may not capture evolving task dynamics during deployment.

APPROACH: We propose an online embedding framework where a recurrent encoder
(LSTM or GRU) continuously integrates recent state-action-reward trajectories into
a latent task embedding vector that conditions the policy and value networks;
the encoder is jointly optimized end-to-end with the RL objective via truncated
backpropagation through time.

EXPERIMENT: We evaluate on Meta-World ML1/ML10 and DeepMind Control Suite's
Walker and Cheetah with randomized physical parameters, measuring zero-shot
generalization as normalized average episode return on held-out tasks, with
significance tested by paired t-tests over 10+ seeds. Baselines: static task
embeddings, no task embeddings, contrastive embedding agents from PEARL and VariBAD.

NOVELTY: Unlike PEARL and VariBAD that learn fixed or slowly updated context
embeddings, our approach introduces explicitly online-updated recurrent embeddings
trained jointly with the policy to capture fine-grained, evolving task characteristics,
with theoretical analysis grounding why online adaptation better approximates
underlying task distributions under partial observability.
```

---

## Part 3 — Judge

The judge sees *only* the final IDEA/BACKGROUND/APPROACH/EXPERIMENT/NOVELTY text.
It does not know which system produced which idea. A/B positions are randomly shuffled
per comparison to eliminate positional bias.

### Prompt (exact, from `judge.py`)

```
You are an expert scientific reviewer evaluating two research ideas on the same topic.

Topic: Zero-shot generalization in reinforcement learning

---
IDEA A:
[idea text — either champion or candidate, randomly assigned]

---
IDEA B:
[idea text]

---
Score each idea on the following criteria (0-10 each):
1. Novelty: Is this a genuinely new angle, not obvious incremental work?
2. Scientific usefulness: Does it address real open problems, actionable by researchers?
3. Experimental clarity: Is the methodology clear with testable predictions?
4. Feasibility: Is it achievable with current or near-future resources?

Provide your evaluation in the following JSON format exactly:
{
  "scores_a": {"novelty": <0-10>, "scientific_usefulness": <0-10>,
               "experimental_clarity": <0-10>, "feasibility": <0-10>},
  "scores_b": {"novelty": <0-10>, "scientific_usefulness": <0-10>,
               "experimental_clarity": <0-10>, "feasibility": <0-10>},
  "winner": "<A|B|tie>",
  "reasoning": "<2-3 sentence explanation>"
}

Base the winner on total score across all criteria. Declare a tie if totals
differ by 2 or fewer points.
Respond with only the JSON object, no other text.
```

### Real verdict (S15 vs S16, topic: Zero-shot generalization in RL)

S15 won this pair:

```json
{
  "scores_a": {"novelty": 7, "scientific_usefulness": 8,
               "experimental_clarity": 9, "feasibility": 8},
  "scores_b": {"novelty": 9, "scientific_usefulness": 8,
               "experimental_clarity": 8, "feasibility": 6},
  "winner": "A",
  "reasoning": "Idea A scores higher overall due to its superior feasibility and
  experimental clarity, presenting a focused, testable hypothesis with a clear
  implementation path. While Idea B is more novel in its conceptual framing, its
  higher complexity and less certain implementation make it a riskier proposition
  with a lower feasibility score."
}
```

**Total score: A=32, B=31. Winner: A (S15)**

The tie threshold (≤2 points) matters: with 4 criteria × 0–10, many pairs are
genuinely close. This prevents the judge from being decisive on noise.

### What the judge rewards

Pattern from 75 pairs in compare_S15_vs_S16.json:

> "...superior experimental clarity (explicit falsification criteria, rigorous
> statistical design with 50+ runs)..."

> "...clearer experimental methodology with well-defined baselines, and higher
> feasibility since it avoids the substantial engineering overhead..."

> "...more focused and theoretically grounded contribution by systematically
> investigating how dynamic, empirically-motivated noise shapes emergent
> communication protocols with explicit architectural inductive biases..."

**The judge consistently rewards:**

1. Named datasets + baselines + metrics (not "standard benchmarks")
2. Explicit falsification criteria (what number would disprove it)
3. Theoretical grounding for *why* the intervention should work
4. Feasibility — ambitious but achievable

**The judge consistently penalizes:**

1. Branded mechanism names that sound specific but aren't
2. Vague capability claims ("X improves Y") without mechanism
3. Ambitious scope that requires resources beyond current reach
4. Missing controls or undefined measurements

---

## Part 4 — Stopping Criteria and Acceptance

### SWE loop stops when:

- `max_rounds=6` accepted edits reached (configurable)
- `max_failures=3` consecutive mini-eval rejections
- 2-round stall (win rate hasn't improved in 2 rounds)

### SWE round accepts an edit when:

- Mini-eval win rate > **52%** (9 pairs)

### Final eval accepts a version when:

- Full eval win rate > **55%** (75 pairs, 15 topics × 5 ideas)
- Blind judge (Gemini) does NOT diverge from primary judge by >30% flip rate

### Version history:


| Version | Win rate vs champion      | Status                                    |
| ------- | ------------------------- | ----------------------------------------- |
| S12     | 89.3% vs S_paper (reboot) | ACCEPTED                                  |
| S13     | 54.0% vs S12              | REJECTED                                  |
| S14     | 54.0% vs S12              | REJECTED (0 SWE rounds — credits)         |
| S15     | 55.3% vs S12              | ACCEPTED — current champion               |
| S16     | 48.7% vs S15              | REJECTED (mini was 77.8% — high variance) |
| S17     | in progress               | —                                         |


---

## Part 5 — Data Flow Summary

```
godel_loop.py swe-evolve --target 18
│
├── find champion: S15
├── find compare report: compare_S15_vs_S16.json
│
└── run_swe_loop(champion="S15", next="S17", max_rounds=3)
    │
    ├── _extract_grounded_failures()
    │    reads compare_S15_vs_S16.json
    │    → grounded_failures_str (real losing idea texts + judge quotes)
    │
    └── for round in 1..3:
        │
        ├── _build_self_improvement_topic()
        │    champion docstring[:400] + grounded_failures_str[:1500]
        │    + failed_attempts_str[:600]
        │    → topic string (~2,400 chars)
        │
        ├── _call_generate_idea_proposal(topic)
        │    loads S15.py dynamically
        │    calls S15.generate_idea(topic, client, temperature=0.7)
        │    → 13 DeepSeek calls (SOTA + 1 + 5 + 1 + 1 + 4 + 1)
        │    → raw idea string (~1,800 chars)
        │    → parsed: {background, approach, experiment, novelty}
        │    → task_description (~3,300 chars)
        │
        ├── _call_claude_code_edit(task_description)
        │    claude-sonnet-4-6 reads S15.py with tools
        │    writes S17_r{n}.py (surgical edit)
        │    → new Python file
        │
        ├── _smoke_test(S17_r{n}.py)
        │    loads module, calls generate_idea("test") → must not crash
        │
        └── _run_mini_eval(S17_r{n}.py vs S15)
             3 topics × 3 ideas = 9 pairs
             → win_rate
             if > 52%: accept edit, champion_path = S17_r{n}.py
             else: revert, failures += 1
    │
    └── output: S17.py (best accumulated edits)

full eval: compare S17 vs S15 (75 pairs)
→ win_rate_b > 55% AND blind judge divergence < 30%
→ accept: update CURRENT_VERSION to S17
```

