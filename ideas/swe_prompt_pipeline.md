# SWE Agent — Live Prompt Pipeline

Real data from S15→S16 session.

======================================================================

## SWE STEP 1 — DIAGNOSE_PROMPT

**Size: 996 chars**

You are diagnosing exactly why a research idea generator is losing pairwise evaluations.

## Generation pipeline — the code that produced the losing ideas

"""S12_r3: Hypothesis-First Adversarial Loop.

Replaces the top-down tree/tournament with a FALSIFIABLE-HYPOTHESIS-FIRST loop:

1. HYPOTHESIS GENERATION: One LLM call produces 5 sharp, testable scientific
  hypotheses about the topic (e.g., "scaling laws break under data-constrained
   regimes because X"). Starting from *claimed truths about the world* rather
   than technique names.
2. ADVERSARIAL ATTACK (parallel, 5 calls): Each hypothesis gets independently
  attacked by an adversarial critic that probes assumption violations, dataset
   biases, theoretical gaps, and practical limitations. Each attack also forces
   a refined/alternative variant.
3. HYPOTHESIS SELECTION (1 call): Aggregate attack+revision pairs and pick the
  strongest surviving hypothesis — the one whose refined version is most
   concrete, novel, and falsifiable.
4. IDEA CONSTRUCTION (1 call): Design the full experimental idea around
  proving/disproving the surviving hypothesis. Forces concrete datasets,
   baselines, and metrics because the falsification target is explicit.
5. MULTI-PERSPECTIVE CRITIQUE (4 calls): Experimentalist, theorist, skeptic
  critique + synthesis.
6. FINAL REVISION (1 call): Incorporate critique.

LLM calls: 1 (hyp) + 5 (attacks) + 1 (select) + 1 (construct) + 4 (critique) + 1 (revise) = ~13
"""

import sys
import os
import concurrent.futures

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(**file**))))
from systems.base import IdeaGenerator, DEFAULT_MODEL, call_llm, IDEA_FORMAT

class S15Generator(IdeaGenerator):
    VERSION = "S15"
    DESCRIPTION = (
        "Hypothesis-first adversarial loop: generate falsifiable hypotheses, "
        "attack each in parallel, select strongest survivor, build experimental "
        "idea around falsification, then multi-perspective critique."
    )

```
def get_prompt(self, topic: str) -> str:
    return f"Generate a novel research idea about: {topic}"

def generate_idea(
    self,
    topic: str,
    client,
    model: str = DEFAULT_MODEL,
    temperature: float = 0.8,
) -> str:
    # ── Step 0: retrieve SOTA context ───────────────────────────────────
    try:
        import os as _os
        sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
        from retrieval import get_topic_context
        sota_context = get_topic_context(topic, n=5)
    except Exception:
        sota_context = ""

    context_block = f"\n\n{sota_context}\n" if sota_context else ""

    # ── Step 1: generate 5 falsifiable hypotheses ───────────────────────
    hyp_prompt = (
        f"Research topic: {topic}{context_block}\n"
        "Generate exactly 5 sharp, falsifiable scientific hypotheses about this topic.\n"
        "Each hypothesis must:\n"
        "- Make a specific claim about the world (not just 'we can improve X')\n"
        "- Be testable: name what experiment would p
```

... [truncated for display]

## Concrete failing examples — where our ideas lost

### Topic: Zero-shot generalization in reinforcement learning

**Champion (S12) — 30/40 — WON:**
IDEA: Integrating a two-level hierarchical latent variable model that separately encodes task-agnostic environmental dynamics and abstract reward-conditioned task representations can significantly improve zero-shot generalization across tasks with shared dynamics but structurally varying reward functions.

BACKGROUND: Zero-shot generalization in reinforcement learning remains an open challenge, particularly when tasks share underlying environment dynamics but differ in reward structures and objectives. Existing context-based or modular methods often entangle dynamics and reward signals, limiti

**Candidate (S15) — 22/40 — LOST:**
IDEA:  
Zero-shot generalization in reinforcement learning improves significantly when policies are trained using multi-modal sensory inputs (proprioception, vision, and audio) with modality-specific encoders and attention-based fusion, even without data augmentation or explicit regularization.

BACKGROUND:  
Reinforcement learning (RL) agents often fail to generalize zero-shot to out-of-distribution environments, limiting their deployment in complex real-world tasks. Prior approaches focus heavily on data augmentation or representation learning to improve generalization but rarely isolate the

**Judge:** Idea A presents a principled hierarchical decomposition of latent representations that directly addresses a core challenge in zero-shot RL (disentangling dynamics from rewards), with clear experimental design and well-defined success criteria. Idea B, while methodologically sound, tackles a less fundamental problem (multi-modal perception benefits) that is somewhat orthogonal to zero-shot generalization and lacks strong theoretical motivation for why audio would substantially improve transfer without augmentation—the core claim feels incremental and the audio simulation pipeline adds complexity without clear necessity for the core research quest

## What has already been tried and failed this session

(none yet — first round)

## Context: judge preferences and experiment history

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

> Idea A presents a more focused and theoretically grounded contribution by systematically investigating how dynamic, empirically-motivated noise shapes emergent communication protocols with explicit architectural inductive biases (noise-aware VAE, communication bottlenecks). It has clearer falsifiabl
> Idea A scores higher across all dimensions with a total of 29 vs. 23 for Idea B. Idea A presents a more grounded approach using RL within realistic hardware constraints, clearer experimental methodology with well-defined baselines, and higher feasibility since it avoids the substantial engineering o
> Idea A scores higher overall (29 vs 26) with stronger novelty in demonstrating real-time adaptive feedback within coherence constraints and superior experimental clarity with well-defined latency budgets and rigorous controls. While Idea B addresses a complementary problem with slightly better fea

## Your task

Study the losing ideas above carefully. Identify the specific step in the pipeline
where quality broke down — was the hypothesis too generic? Did selection pick the safe
option? Did the critique fail to add experimental specificity? Did the revision water
things down?

Diagnose the root cause, then propose ONE targeted concrete fix.

Output EXACTLY in this format (no other text):
DIAGNOSIS: <which specific step failed and precisely why — reference the actual examples>
FIX: <the single concrete change — which call, what the prompt should do differently>
EXPECTED_IMPACT: 

======================================================================

## SWE STEP 2 — ATTACK_PROMPT

**Size: 535 chars**

A proposed fix for an underperforming research idea generator:

DIAGNOSIS: Step 4 (IDEA CONSTRUCTION) loses because the hypothesis is stated at the capability level (e.g., 'multi-modal fusion improves zero-shot transfer') rather than the mechanism level, so the experiment becomes a vague empirical sweep.
FIX: In Step 3 (hypothesis selection), add a 'MECHANISM' constraint: selected hypothesis must explicitly name the causal mechanism linking intervention to outcome. Reject hypotheses that only state what will improve, not why.

Attack this fix on 3 dimensions:

1. ROOT CAUSE: Does this actually address the root cause, or just a symptom?
2. ASSUMPTION: What unstated assumption does this fix make that might not hold?
3. RESIDUAL FAILURE: What would still fail after making this change?

Then write a REVISED fix that is stronger, more targeted, and addresses all three attacks.

REVISED_FIX: <concrete, strengthened version — specific enough to implement directly as code>

======================================================================

## SWE STEP 3 — PROPOSE_EDIT_PROMPT (fallback path)

**Size: 2,042 chars**

You are a software engineer implementing a targeted improvement to a research idea generator.

## Full generator codebase (S15)

### FILE: systems/S15.py

```python
"""S12_r3: Hypothesis-First Adversarial Loop.

Replaces the top-down tree/tournament with a FALSIFIABLE-HYPOTHESIS-FIRST loop:

1. HYPOTHESIS GENERATION: One LLM call produces 5 sharp, testable scientific
   hypotheses about the topic (e.g., "scaling laws break under data-constrained
   regimes because X"). Starting from *claimed truths about the world* rather
   than technique names.

2. ADVERSARIAL ATTACK (parallel, 5 calls): Each hypothesis gets independently
   attacked by an adversarial critic that probes assumption violations, dataset
   biases, theoretical gaps, and practical limitations. Each attack also forces
   a refined/alternative variant.

3. HYPOTHESIS SELECTION (1 call): Aggregate attack+revision pairs and pick the
   strongest surviving hypothesis — the one whose refined version is most
   concrete, novel, and falsifiable.

4. IDEA CONSTRUCTION (1 call): Design the full experimental idea around
   proving/disproving the surviving hypothesis. Forces concrete datasets,
   baselines, and metrics because the falsification target is explicit.

5. MULTI-PERSPECTIVE CRITIQUE (4 calls): Experimentalist, theorist, skeptic
   critique + synthesis.

6. FINAL REVISION (1 call): Incorporate critique.

LLM calls: 1 (hyp) + 5 (attacks) + 1 (select) + 1 (construct) + 4 (critique) + 1 (revise) = ~13
"""

import sys
import os
import concurrent.futures

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from systems.base import IdeaGenerator, DEFAULT_MODEL, call_llm, IDEA_FORMAT


class S15Generator(IdeaGenerator):
    VERSION = "S15"
    DESCRIPTION = (
        "Hypothesis-first adversarial loop: generate falsifiable hypotheses, "
        "attack each in parallel, select strongest survivor, build experimental "
        "idea around falsification, then multi-perspective critique."
    )

    def get_prompt(self, topic: str) -> str:
        return f"Generate a novel research idea about: {topic}"

    def generate_idea(
        self,
        topic: str,
        client,
        model: str = DEFAULT_MODEL,
        temperature: float = 0.8,
    ) -> str:
        # ── Step 0: retrieve SOTA context ───────────────────────────────────
        try:
            import os as _os
            sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
            from retrieval import get_topic_context
            sota_context = get_topic_context(topic, n=5)
        except Exception:
            sota_context = ""

        context_block = f"\n\n{sota_context}\n" if sota_context else ""

        # ── Step 1: generate 5 falsifiable hypotheses ───────────────────────
        hyp_prompt = (
            f"Research topic: {topic}{context_block}\n"
            "Generate exactly 5 sharp, falsifiable scientific hypotheses about this topic.\n"
            "Each
...[truncated]
```

