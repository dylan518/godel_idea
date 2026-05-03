# EvoScientist — Complete System Overview

> **What is this?** A Gödel loop: a system that generates research ideas, evaluates them, and then *rewrites its own idea-generation code* to generate better ideas. It self-improves.

---

## 1. The Big Picture

```
┌─────────────────────────────────────────────────────────────────┐
│                        GÖDEL LOOP                               │
│                                                                 │
│   ┌──────────┐    generates    ┌──────────────┐                 │
│   │ S{n}.py  │ ─────────────▶ │  75 ideas    │                 │
│   │ (current │                │  (15 topics  │                 │
│   │ champion)│                │   × 5 ideas) │                 │
│   └──────────┘                └──────┬───────┘                 │
│        ▲                             │                         │
│        │ if win >55%                 │ pairwise judging        │
│        │ accept as                   ▼                         │
│        │ new champion         ┌──────────────┐                 │
│        │                      │   DeepSeek   │                 │
│        │                      │   judge      │                 │
│        │                      │  (primary)   │                 │
│        │                      └──────┬───────┘                 │
│        │                             │ win_rate_b              │
│        │                             ▼                         │
│   ┌────┴─────────────────────────────────────┐                 │
│   │            SWE AGENT                     │                 │
│   │  reads failures → proposes edits →       │                 │
│   │  Claude Code implements → mini-eval      │                 │
│   │  → writes S{n+1}.py                      │                 │
│   └──────────────────────────────────────────┘                 │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

**The key insight:** The system doesn't just generate better ideas — it rewrites the *code that generates ideas*. Each version (S15, S16, ...) is a Python class with a `generate_idea()` method. The SWE agent reads why the current version is losing and surgically edits that code.

---

## 2. Key Components


| Component          | File                       | What it does                                               |
| ------------------ | -------------------------- | ---------------------------------------------------------- |
| **Gödel Loop**     | `godel_loop.py`            | Orchestrates everything: run generator, judge, SWE agent   |
| **Idea Generator** | `systems/S{n}.py`          | Generates research ideas (the thing being evolved)         |
| **Runner**         | `runner.py`                | Runs S{n}.py in parallel across 15 topics × 5 ideas        |
| **Judge**          | `judge.py`                 | DeepSeek pairwise evaluation with early stopping           |
| **Blind Judge**    | `judge.py`                 | Gemini canary (never used for accept/reject)               |
| **SWE Agent**      | `swe_agent.py`             | Diagnoses failures, proposes edits, runs mini-evals        |
| **Retrieval**      | `retrieval.py`             | Semantic Scholar API — fetches recent papers per topic     |
| **SWE Memory**     | `results/swe_memory.json`  | Cross-iteration memory: what worked/failed across versions |
| **SWE Context**    | `results/swe_context.json` | Judge preferences, experiment history, pipeline state      |


---

## 3. The Idea Generator (S15 / S16)

Each `S{n}.py` is a standalone Python class. The current champion is **S15**. Here is its complete pipeline for **one idea on one topic**:

```
Topic: "Zero-shot generalization in reinforcement learning"
         │
         ▼
┌────────────────────────────────────────────────────────────┐
│ STEP 0: SOTA Retrieval (not LLM — Semantic Scholar API)    │
│                                                            │
│ Input:  topic string                                       │
│ Output: 5 recent paper titles + abstracts (~2000 chars)    │
│         cached 7 days on disk                              │
│                                                            │
│ Example output:                                            │
│   1. "Improving Zero-Shot Generalization in Offline RL     │
│       using Generalized Similarity Functions" (2021)...    │
│   2. "Zero-Shot Generalization through Abstract            │
│       Representations" (2025)...                           │
└────────────────────┬───────────────────────────────────────┘
                     │ sota_context string
                     ▼
┌────────────────────────────────────────────────────────────┐
│ STEP 1: Hypothesis Generation  [1 LLM call]               │
│                                                            │
│ Input:  topic + 5 SOTA papers                             │
│                                                            │
│ Prompt: "Generate exactly 5 sharp, falsifiable scientific  │
│          hypotheses. Each must: make a specific claim      │
│          about the world (not 'we can improve X'),         │
│          be testable, be non-obvious vs related work,      │
│          be concise (1-2 sentences max).                   │
│          Format: H1: ... H2: ... H3: ... H4: ... H5: ..."  │
│                                                            │
│ Output: 5 hypotheses like:                                 │
│   H1: "Zero-shot RL generalization fails because policy    │
│        networks learn task-specific reward heuristics,     │
│        not observation shift — reward-randomization        │
│        alone won't improve generalization."                │
│   H2: "Hierarchical latent variable models that encode     │
│        dynamics and reward separately improve zero-shot    │
│        transfer when dynamics are shared but rewards vary."│
│   H3-H5: ...                                              │
└────────────────────┬───────────────────────────────────────┘
                     │ [H1, H2, H3, H4, H5]
                     ▼
┌────────────────────────────────────────────────────────────┐
│ STEP 2: Adversarial Attack × 5  [5 PARALLEL LLM calls]    │
│                                                            │
│ Each hypothesis gets its own adversarial critic call:      │
│                                                            │
│ Input per call:  topic + one hypothesis                    │
│                                                            │
│ Prompt: "You are an adversarial critic. Attack this        │
│          hypothesis on EACH dimension:                     │
│          1. Assumption violation: what counterexample?     │
│          2. Dataset bias: what artifact makes this look    │
│             true when it isn't?                            │
│          3. Theoretical gap: what known result contradicts?│
│          4. Practical limitation: too expensive to test?   │
│          Then write a REVISED hypothesis that survives."   │
│                                                            │
│ Output per call:                                           │
│   "1. Assumption: assumes dynamics/reward are              │
│       independently variable, but most envs couple them.  │
│    2. Dataset bias: MiniGrid task families share dynamics  │
│       by design, not because of latent disentanglement.   │
│    3. Theoretical gap: Modular RL (Aljalbout 2021)...     │
│    4. Practical: requires paired task sets with controlled │
│       dynamics-vs-reward variation — not in benchmarks.   │
│    Revised: Hierarchical LVMs improve zero-shot RL         │
│    specifically when dynamics are shared but rewards vary  │
│    — testable by constructing paired task sets with        │
│    independently controlled dynamics-vs-reward variation." │
└────────────────────┬───────────────────────────────────────┘
                     │ [(H1, attack1), (H2, attack2), ...]
                     ▼
┌────────────────────────────────────────────────────────────┐
│ STEP 3: Hypothesis Selection  [1 LLM call]                │
│                                                            │
│ Input:  all 5 (original, attack, revised) triples         │
│                                                            │
│ Prompt: "Select the ONE hypothesis whose REVISED form is:  │
│          - Most concrete and specific (names mechanisms)   │
│          - Most falsifiable (clearest disproof path)       │
│          - Most novel (least covered by standard lit)      │
│          Respond: SELECTED: N                              │
│                   REVISED HYPOTHESIS: <text>              │
│                   REASONING: <1-2 sentences>"             │
│                                                            │
│ Output: "SELECTED: 2                                       │
│          REVISED HYPOTHESIS: Hierarchical LVMs improve     │
│          zero-shot RL transfer specifically when task      │
│          distribution has shared dynamics but varying      │
│          rewards — testable with paired task sets.         │
│          REASONING: Most falsifiable — names the exact     │
│          condition under which it holds."                  │
└────────────────────┬───────────────────────────────────────┘
                     │ selected_hyp (string)
                     ▼
┌────────────────────────────────────────────────────────────┐
│ STEP 4: Experimental Construction  [1 LLM call]            │
│                                                            │
│ === S15 PROMPT (old) ===                                   │
│ "Core hypothesis: {selected_hyp}                           │
│  Design a concrete research experiment to PROVE OR         │
│  DISPROVE this hypothesis. Must name: specific datasets,   │
│  specific baselines, quantitative success metrics,         │
│  key technical method. Write 3-4 paragraphs."             │
│                                                            │
│ → Problem: model treats hypothesis as a capability label,  │
│   produces vague empirical sweep ("does X help?")         │
│                                                            │
│ === S16 PROMPT (new — the winning change) ===              │
│ "Core hypothesis: {selected_hyp}                           │
│                                                            │
│  Before designing the experiment, fill in:                 │
│  MECHANISTIC_CLAIM: <the causal mechanism — what           │
│    property causes what effect via what process>           │
│  NULL_RESULT: <numeric threshold that FALSIFIES this>      │
│  MINIMAL_JUSTIFICATION: <why this is the MINIMAL          │
│    SUFFICIENT test of this specific mechanism>             │
│                                                            │
│  Now design the concrete experiment:..."                   │
│                                                            │
│ → Forces mechanistic grounding BEFORE experiment design.   │
│   Mini-eval result: 33% → 77.8% after this change.        │
│                                                            │
│ Example output (S16):                                      │
│  "MECHANISTIC_CLAIM: Disentangled representations prevent  │
│   policy overfitting by encoding dynamics and reward       │
│   separately, enabling dynamics encoder to transfer        │
│   without relearning reward-specific features.             │
│   NULL_RESULT: ≤2% higher zero-shot accuracy vs flat       │
│   baseline falsifies disentanglement benefit.              │
│   MINIMAL_JUSTIFICATION: Only one arch change needed       │
│   (separate encoders + disentanglement loss); additional   │
│   components would confound attribution.                   │
│   [3-4 paragraphs: MiniGrid-4rooms, baselines DIAYN/PPO,  │
│   metrics: zero-shot success rate, statistical testing]"   │
└────────────────────┬───────────────────────────────────────┘
                     │ draft (string)
                     ▼
┌────────────────────────────────────────────────────────────┐
│ STEP 5: Multi-Perspective Critique  [4 PARALLEL LLM calls] │
│                                                            │
│ 5a. EXPERIMENTALIST:                                       │
│     "You are a hard-nosed experimentalist. Give 2-3 sharp  │
│      criticisms on experimental feasibility: can these     │
│      experiments actually be run? Are measurements         │
│      well-defined? What controls are missing?"             │
│                                                            │
│ 5b. THEORIST:                                              │
│     "You are a rigorous theorist. Give 2-3 sharp           │
│      criticisms on theoretical grounding: is the novelty   │
│      claim justified? Does it overlap with known results?" │
│                                                            │
│ 5c. SKEPTIC:                                               │
│     "You are a skeptical reviewer. Give 2-3 criticisms:    │
│      why this probably won't work, likely negative result, │
│      whether payoff justifies the effort."                 │
│                                                            │
│ 5d. SYNTHESIS:                                             │
│     "Three reviewers critiqued this proposal. Synthesize   │
│      into the 3 most important actionable improvements."   │
└────────────────────┬───────────────────────────────────────┘
                     │ synthesis (3 improvements)
                     ▼
┌────────────────────────────────────────────────────────────┐
│ STEP 6: Final Revision  [1 LLM call]                       │
│                                                            │
│ Input:  topic + hypothesis + draft + synthesis + SOTA      │
│                                                            │
│ Prompt: "Write the final improved research idea. Ensure    │
│          hypothesis is clearly stated, falsification is    │
│          concrete, datasets/baselines/metrics named.       │
│                                                            │
│          Format EXACTLY:                                   │
│          IDEA: [one sentence]                              │
│          BACKGROUND: [open problem, 2-3 sentences]        │
│          APPROACH: [proposed method, 2-3 sentences]        │
│          EXPERIMENT: [datasets, metrics, 1-2 sentences]    │
│          NOVELTY: [what makes this different, 1-2 sent]"   │
│                                                            │
│ Output (real S15 example):                                 │
│  "IDEA: Incorporating dynamically updated task embeddings  │
│   derived from online interactions improves zero-shot      │
│   generalization vs static contrastive embeddings.         │
│   BACKGROUND: RL agents overfit to training distributions  │
│   due to insufficiently adaptive task representations...   │
│   APPROACH: Recurrent encoder (LSTM) integrates recent     │
│   trajectories into task embedding conditioning policy...  │
│   EXPERIMENT: Meta-World ML1/ML10, DeepMind Control,       │
│   zero-shot return on held-out tasks, 10+ seeds...         │
│   NOVELTY: Unlike PEARL/VariBAD with fixed context,        │
│   online-updated recurrent embedding trained jointly..."   │
└────────────────────┘

Total per idea: ~13 LLM calls (1 + 5 parallel + 1 + 1 + 3 parallel + 1 + 1)
```

---

## 4. The Judge

Every generated idea is compared **head-to-head** against the current champion's idea on the same topic.

```
Champion idea (A)  vs  Candidate idea (B)
         │                    │
         └────────┬───────────┘
                  ▼
         ┌─────────────────┐
         │  DeepSeek judge │
         │                 │
         │ Scores 0-10 on: │
         │  • Novelty      │
         │  • Scientific   │
         │    usefulness   │
         │  • Experimental │
         │    clarity      │
         │  • Feasibility  │
         └────────┬────────┘
                  │
         ┌────────▼────────┐
         │  A/B randomized │  ← prevents positional bias
         │  each pair      │
         └────────┬────────┘
                  │
         ┌────────▼──────────────────────────┐
         │  Wilson score early stopping       │
         │                                    │
         │  REJECT early: if max achievable   │
         │  rate < 55% threshold              │
         │                                    │
         │  ACCEPT early: if min achievable   │
         │  rate > 55% threshold              │
         └────────────────────────────────────┘

Full eval:  15 topics × 5 ideas = 75 pairs
Mini-eval:  3 topics × 3 ideas = 9 pairs
Threshold:  >55% to accept full eval
            >52% to accept mini-eval (SWE round)
Blind judge: Gemini checks for Goodhart (diverge >30% = alert)
```

**Real judge output example:**

> "Idea A presents a principled hierarchical decomposition of latent representations that directly addresses a core challenge in zero-shot RL (disentangling dynamics from rewards), with clear experimental design and well-defined success criteria. Idea B, while methodologically sound, tackles a less fundamental problem (multi-modal perception benefits) that is somewhat orthogonal to zero-shot generalization. A: 30/40, B: 22/40."

---

## 5. The SWE Agent

When a candidate is rejected (or even accepted but we want more), the SWE agent analyzes *why* the ideas lost and rewrites `S{n}.py` to fix it.

```
INPUT: champion S{n}.py + compare_S{n-1}_vs_S{n}.json
         │
         ▼
┌────────────────────────────────────────────────────────────┐
│ SWE ROUND (runs up to 3 times)                             │
│                                                            │
│ ── CALL 1: DIAGNOSE ──────────────────────────────────     │
│                                                            │
│ Input:                                                     │
│  • Full S{n}.py source code (~12K chars)                   │
│  • 4 concrete losing examples with FULL idea text:        │
│      "Topic: Zero-shot RL                                  │
│       Champion (S12, 30/40, WON):                          │
│       IDEA: Integrating a two-level hierarchical latent    │
│       variable model that separately encodes dynamics...   │
│       Candidate (S15, 22/40, LOST):                        │
│       IDEA: Zero-shot RL improves with multi-modal         │
│       sensory inputs (proprioception, vision, audio)...    │
│       Judge: Idea A addresses the core challenge (latent   │
│       disentanglement) with clear experimental design.     │
│       Idea B tackles a less fundamental problem..."        │
│  • Failed attempts this session                            │
│  • SWE context (judge preferences, experiment history)    │
│                                                            │
│ Prompt: "Diagnose exactly why the generator is losing.     │
│  Study the losing ideas carefully. Which step failed —     │
│  was the hypothesis too generic? Did selection pick        │
│  the safe option? Did construction produce a vague         │
│  empirical sweep?                                          │
│  Output EXACTLY:                                           │
│  DIAGNOSIS: <which step failed and precisely why>          │
│  FIX: <the single concrete change>                         │
│  EXPECTED_IMPACT: <why this addresses the pattern>"        │
│                                                            │
│ Example output (round 2 that worked):                      │
│  "DIAGNOSIS: Step 4 (IDEA CONSTRUCTION) fails to anchor    │
│   experimental design to the mechanistic claim. LLM        │
│   reuses hypothesis as capability label not causal         │
│   mechanism — produces 'does X help?' not 'why X helps     │
│   via mechanism M'. Losing ideas like 'multi-modal fusion  │
│   improves zero-shot RL' describe intervention+outcome     │
│   without naming mechanism.                                │
│   FIX: In Step 4 construct_prompt, add locked preamble     │
│   forcing (a) restate mechanistic claim, (b) state null    │
│   result numerically, (c) justify minimal sufficient test. │
│   EXPECTED_IMPACT: Forces mechanistic grounding before     │
│   experiment design, eliminating vague-sweep pattern."     │
│                                                            │
│ ── CALL 2: ATTACK ────────────────────────────────────     │
│                                                            │
│ Input:  DIAGNOSIS + FIX from call 1                        │
│                                                            │
│ Prompt: "Attack this fix on 3 dimensions:                  │
│  1. ROOT CAUSE: does it address root cause or symptom?     │
│  2. ASSUMPTION: what unstated assumption might not hold?   │
│  3. RESIDUAL FAILURE: what would still fail after?         │
│  Then: REVISED_FIX: <strengthened version>"               │
│                                                            │
│ Example attack output:                                     │
│  "1. ROOT CAUSE: Fix is correct about Step 4 but the       │
│     hypothesis string from Step 3 is already compressed    │
│     — model will confabulate a mechanism not actually      │
│     in the hypothesis.                                     │
│   2. ASSUMPTION: Assumes hypothesis contains extractable   │
│     mechanistic content. 'Multi-modal fusion' is an        │
│     architectural proposal, not a mechanism.               │
│   3. RESIDUAL FAILURE: Even with preamble, thresholds      │
│     will be confabulated to fit design, not derived from   │
│     actual hypothesis predictions.                         │
│   REVISED_FIX: Modify Step 4 to require three-field        │
│   preamble: MECHANISTIC_CLAIM (causal mechanism not        │
│   capability), NULL_RESULT (numeric falsification),        │
│   MINIMAL_JUSTIFICATION (why minimal sufficient test)."   │
│                                                            │
│ ── CALL 3: CLAUDE CODE ───────────────────────────────     │
│                                                            │
│ This is a NESTED Claude Code session (subprocess).         │
│                                                            │
│ Claude Code gets:                                          │
│  • Task: "Implement this ONE targeted improvement"         │
│  • Refined fix (from ATTACK output)                        │
│  • Concrete failing examples (why it's losing)            │
│  • Hard constraints (class name, VERSION, GENERATOR)      │
│                                                            │
│ Claude Code then:                                          │
│  1. Reads S{n}.py (the champion file)                      │
│  2. Reads relevant idea_tournament/ modules               │
│  3. Makes ONE targeted surgical edit                       │
│  4. Writes S{n+1}_r{round}.py                             │
│                                                            │
│ ── MINI-EVAL ─────────────────────────────────────────     │
│                                                            │
│ Run S{n+1}_r{round} on 3 topics × 3 ideas = 9 pairs       │
│ Judge against champion.                                    │
│                                                            │
│ If win_rate > 52%: ACCEPT → use as base for next round    │
│ If win_rate ≤ 52%: REJECT → try different diagnosis       │
└────────────────────────────────────────────────────────────┘
         │
         │ after max 3 rounds
         ▼
Best accepted round becomes S{n+1}.py
         │
         ▼
FULL EVAL: 15 topics × 5 ideas = 75 pairs vs champion
         │
         ├─ win_rate > 55%: ACCEPT → new champion
         └─ win_rate ≤ 55%: REJECT → try again next SWE iteration
```

---

## 6. Persistent Memory — What the Agent Remembers

The SWE agent builds up two persistent stores across ALL iterations:

### `results/swe_memory.json` — iteration-level memory

```json
{
  "iterations": [
    {
      "version": "S16",
      "champion": "S15",
      "mini_eval_best": 0.778,
      "accepted_edits": ["Mechanistic Construction Lock in Step 4"],
      "failed_edits": ["scope minimization (round 1)"],
      "full_eval_win_rate": 0.487,
      "accepted": false
    }
  ]
}
```

### `results/swe_context.json` — cumulative judge signal

Stores:

- Every winning and losing judge quote (what the judge says when each wins)
- Experiment log: S12 vs S14 = 54%, S12 vs S15 = 55%, etc.
- Pipeline description (what code S{n} uses)

This context is injected into every DIAGNOSE prompt so the SWE agent knows:

- What the judge rewards (judge quotes from wins)
- What has been tried before (failed edits)
- How much each previous version improved

---

## 7. Evolution History


| Version                    | Strategy                                           | SWE Rounds | Mini-eval best | Full eval | Status                         |
| -------------------------- | -------------------------------------------------- | ---------- | -------------- | --------- | ------------------------------ |
| S0                         | Direct prompting                                   | —          | —              | —         | baseline                       |
| S1                         | Self-critique                                      | —          | —              | —         | old champion                   |
| S_paper (log; `S_sota.py`) | SOTA + Elo tournament + multi-perspective critique | —          | —              | —         | reset point                    |
| **S12**                    | Hypothesis-first + adversarial attacks + selection | 3 rounds   | 88.9%          | **89.3%** | ✓ ACCEPTED                     |
| S13                        | (SWE attempt from S12)                             | 3 rounds   | —              | —         | old loop                       |
| S14                        | (SWE attempt)                                      | 3 rounds   | —              | 54.0%     | ✗ rejected                     |
| **S15**                    | + mechanism extraction in Step 3 selection         | 3 rounds   | 55.6%          | **55.3%** | ✓ ACCEPTED (Goodhart alert)    |
| S16                        | + mechanistic preamble lock in Step 4 construction | 3 rounds   | 77.8%          | 48.7%     | ✗ rejected (mini-eval overfit) |
| S17                        | SWE running now →                                  | ...        | ...            | ...       | in progress                    |


**Key insight from S16:** Mini-eval 77.8% → full eval 48.7% = significant overfit. The mechanistic preamble worked on 3 mini-eval topics but hurt on the full 15 topics. S17 can now diagnose from S16's full-eval failures.

---

## 8. The Command Interface

```bash
# See current state
python3 ideas/godel_loop.py status

# Run the full self-improving loop
python3 ideas/godel_loop.py swe-evolve \
  --target 18 \      # evolve up to S18
  --n-ideas 5 \      # 5 ideas per topic (75 pairs)
  --workers 30 \     # parallel workers
  --swe-rounds 3     # SWE rounds per iteration

# Compare a specific candidate manually
python3 ideas/godel_loop.py compare --candidate S16 --n-ideas 5 --workers 30

# Accept a candidate manually
python3 ideas/godel_loop.py accept S16

# Watch live
tail -f ideas/results/godel.log
```

---

## 9. Key Design Decisions

**Why pairwise judging, not absolute scores?**
Absolute scores drift and are hard to calibrate. Pairwise comparison is more robust: "is this idea better than that idea?" is a question humans and LLMs answer more reliably.

**Why DeepSeek as judge?**
Fast, cheap, strong at structured evaluation. Gemini runs in parallel as a canary — if primary and blind judge disagree >30%, it's a Goodhart alert (the system may be gaming the judge rather than actually improving).

**Why early stopping in the judge?**
Wilson score bounds: if after 8+ pairs the upper bound is below threshold, we know it can't pass — stop wasting API calls. Saves ~40% of judging cost.

**Why SWE agent instead of meta-LLM writing whole new file?**
Earlier versions (S7-S11) had the meta-LLM write entire new generator files. They all failed — the LLM can't reason about a whole pipeline at once. Surgical edits (one targeted change per round) with empirical validation (mini-eval) converge much faster.

**Why DIAGNOSE → ATTACK → implement?**

- DIAGNOSE from concrete failures: prevents the LLM from proposing generic improvements ("add more critique rounds") instead of targeted fixes
- ATTACK adversarially refines: prevents implementing a fix that addresses a symptom not the root cause
- Claude Code implements: uses full tool-use context (reads the actual file) rather than writing from memory

