# Complete Prompt Dump — SWE Agent + Idea Generator

Generated from live S15→S16 run. All prompts shown with REAL data.

---

# PART 1: SWE AGENT

The SWE agent runs 3 rounds. Each round = 3 LLM calls:

1. DIAGNOSE — reads champion code + losing examples → diagnosis + fix
2. ATTACK — adversarially refines the fix
3. CLAUDE CODE — implements the refined fix

---

## SWE STEP 1 — DIAGNOSE_PROMPT

**Input size: 25,226 chars**

**Purpose:** Reads the full champion code + actual losing idea texts + judge quotes → outputs structured DIAGNOSIS/FIX/EXPECTED_IMPACT

```
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
            "Each hypothesis must:\n"
            "- Make a specific claim about the world (not just 'we can improve X')\n"
            "- Be testable: name what experiment would prove it false\n"
            "- Be non-obvious: it should NOT be directly supported by the related work above\n"
            "- Be concise: 1-2 sentences max\n\n"
            "Format each as:\n"
            "H1: <hypothesis statement>\n"
            "H2: <hypothesis statement>\n"
            "H3: <hypothesis statement>\n"
            "H4: <hypothesis statement>\n"
            "H5: <hypothesis statement>"
        )
        try:
            hyp_raw = call_llm(hyp_prompt, model, client, temperature)
        except Exception as e:
            hyp_raw = f"H1: Standard approaches to {topic} fail because of distribution shift."

        # Parse hypotheses
        hypotheses = []
        for line in hyp_raw.strip().split("\n"):
            line = line.strip()
            if line and (line.startswith("H") and ":" in line[:4]):
                hyp_text = line.split(":", 1)[1].strip()
                if hyp_text:
                    hypotheses.append(hyp_text)
        if not hypotheses:
            hypotheses = [hyp_raw.strip()]
        hypotheses = hypotheses[:5]  # cap at 5

        # ── Step 2: parallel adversarial attacks on each hypothesis ─────────
        def attack_hypothesis(hyp: str) -> str:
            attack_prompt = (
                f"Research topic: {topic}\n\n"
                f"Hypothesis: {hyp}\n\n"
                "You are an adversarial critic. Attack this hypothesis on EACH of these dimensions:\n"
                "1. Assumption violation: What unstated assumption does this rely on? Give a concrete counterexample.\n"
                "2. Dataset bias: What dataset artifact could make this appear true without actually being true?\n"
                "3. Theoretical gap: What known result from the literature contradicts or undermines this?\n"
                "4. Practical limitation: What makes this hypothesis untestable or too expensive to test?\n\n"
                "Then, given these attacks, write a REVISED hypothesis that survives them:\n"
                "Revised: <1-2 sentence refined hypothesis that addresses the above attacks>"
            )
            try:
                return call_llm(attack_prompt, model, client, temperature=0.6)
            except Exception as e:
                return f"Revised: {hyp} (attack failed: {e})"

        attacks = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            futures = [executor.submit(attack_hypothesis, h) for h in hypotheses]
            for f in futures:
                try:
                    attacks.append(f.result(timeout=120))
                except Exception as e:
                    attacks.append(f"Revised: (timeout) {e}")

        # ── Step 3: select the strongest surviving hypothesis ────────────────
        pairs_str = ""
        for i, (hyp, attack) in enumerate(zip(hypotheses, attacks)):
            pairs_str += f"\n--- Hypothesis {i+1} ---\nOriginal: {hyp}\nCritique+Revision:\n{attack}\n"

        select_prompt = (
            f"Research topic: {topic}\n\n"
            f"Below are {len(hypotheses)} original hypotheses, each attacked by an adversarial critic "
            f"and refined into a stronger revised form:\n{pairs_str}\n"
            "Select the ONE hypothesis (by number) whose REVISED form is:\n"
            "- Most concrete and specific (names mechanisms, not just outcomes)\n"
            "- Most falsifiable (clearest path to a disproof experiment)\n"
            "- Most novel (least covered by standard literature)\n\n"
            "Respond with:\n"
            "SELECTED: <number 1-5>\n"
            "REVISED HYPOTHESIS: <copy the revised hypothesis text exactly>\n"
            "REASONING: <1-2 sentences on why this is the strongest>"
        )
        try:
            selection_raw = call_llm(select_prompt, model, client, temperature=0.3)
        except Exception as e:
            selection_raw = f"SELECTED: 1\nREVISED HYPOTHESIS: {hypotheses[0]}\nREASONING: fallback"

        # Extract the selected revised hypothesis
        selected_hyp = hypotheses[0]  # fallback
        for line in selection_raw.strip().split("\n"):
            if line.strip().upper().startswith("REVISED HYPOTHESIS:"):
                candidate = line.split(":", 1)[1].strip()
                if candidate:
                    selected_hyp = candidate
                    break

        # ── Step 4: construct experimental idea around the hypothesis ────────
        context_reminder = (
            f"\nExisting work to differentiate from:\n{sota_context}\n"
            if sota_context else ""
        )
        construct_prompt = (
            f"Research topic: {topic}\n"
            f"{context_reminder}\n"
            f"Core hypothesis to test: {selected_hyp}\n\n"
            "Design a concrete research experiment to PROVE OR DISPROVE this hypothesis.\n"
            "The experiment must:\n"
            "- Name specific datasets you will use (not just 'standard benchmarks')\n"
            "- Name specific baselines you will compare against\n"
            "- Define quantitative success metrics (what number proves it, what number disproves it)\n"
            "- Describe the key technical method in enough detail to implement\n"
            "- Explain precisely what result would falsify the hypothesis\n\n"
            "Write 3-4 paragraphs. Be direct and specific."
        )
        try:
            draft = call_llm(construct_prompt, model, client, temperature)
        except Exception as e:
            draft = f"Research idea for '{topic}' based on hypothesis: {selected_hyp}"

        # ── Step 5: multi-perspective critique ───────────────────────────────
        exp_prompt = (
            f"You are a hard-nosed experimentalist reviewing a research proposal about '{topic}'.\n\n"
            f"Hypothesis being tested: {selected_hyp}\n\n"
            f"Proposed experiment:\n{draft}\n\n"
            "Give 2-3 sharp criticisms focusing purely on experimental feasibility: "
            "Can these experiments actually be run? Are the measurements well-defined? "
            "What controls are missing? What will fail in practice?"
        )
        try:
            critique_exp = call_llm(exp_prompt, model, client, temperature=0.5)
        except Exception:
            critique_exp = "No experimental critique available."

        theory_prompt = (
            f"You are a rigorous theorist reviewing a research proposal about '{topic}'.\n\n"
            f"Hypothesis being tested: {selected_hyp}\n\n"
            f"Proposed experiment:\n{draft}\n\n"
            "Give 2-3 sharp criticisms focusing purely on theoretical grounding: "
            "Is the novelty claim justified? Does it overlap with known results? "
            "Are the underlying assumptions stated and defensible?"
        )
        try:
            critique_theory = call_llm(theory_prompt, model, client, temperature=0.5)
        except Exception:
            critique_theory = "No theoretical critique available."

        skeptic_prompt = (
            f"You are a skeptical reviewer who has seen many overhyped proposals about '{topic}'.\n\n"
            f"Hypothesis being tested: {selected_hyp}\n\n"
            f"Proposed experiment:\n{draft}\n\n"
            "Give 2-3 sharp criticisms focusing on: why this probably won't work, "
            "what the likely negative result is, and whether the scientific payoff "
            "justifies the effort even if it succeeds."
        )
        try:
            critique_skeptic = call_llm(skeptic_prompt, model, client, temperature=0.5)
        except Exception:
            critique_skeptic = "No skeptic critique available."

        synthesis_prompt = (
            f"Three reviewers critiqued a research idea about '{topic}' testing the hypothesis:\n"
            f"'{selected_hyp}'\n\n"
            f"Experimentalist:\n{critique_exp}\n\n"
            f"Theorist:\n{critique_theory}\n\n"
            f"Skeptic:\n{critique_skeptic}\n\n"
            "Synthesize these into the 3 most important actionable improvements "
            "the author must make. Be concise and prioritized."
        )
        try:
            synthesis = call_llm(synthesis_prompt, model, client, temperature=0.3)
        except Exception:
            synthesis = "Improve specificity, add baselines, clarify falsification criteria."

        # ── Step 6: final revision ────────────────────────────────────────────
        revise_prompt = (
            f"Research topic: {topic}\n\n"
            f"Core hypothesis: {selected_hyp}\n\n"
            f"Experimental design:\n{draft}\n\n"
            f"Key improvements required:\n{synthesis}\n"
            f"{context_reminder}\n"
            "Write the final, improved version of the research idea. "
            "Ensure the hypothesis is clearly stated, the falsification experiment is concrete, "
            "and all datasets/baselines/metrics are named explicitly."
            + IDEA_FORMAT
        )
        try:
            return call_llm(revise_prompt, model, client, temperature)
        except Exception:
            return draft if draft else f"Research idea about {topic}: {selected_hyp}"


GENERATOR = S15Generator()


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

**Judge:** Idea A presents a principled hierarchical decomposition of latent representations that directly addresses a core challenge in zero-shot RL (disentangling dynamics from rewards), with clear experimental design and well-defined success criteria. Idea B, while methodologically sound, tackles a less fundamental problem (multi-modal perception benefits) that is somewhat orthogonal to zero-shot generalization and lacks strong theoretical motivation for why audio would substantially improve transfer without augmentation—the core claim feels incremental and the audio simulation pipeline adds complexity without clear necessity for the core research question.

### Topic: Emergent communication in multi-agent systems
**Champion (S12) — 32/40 — WON:**
IDEA:  
Multi-agent systems trained with communication protocols evolving under realistic, temporally correlated wireless channel noise develop more robust, generalizable, and efficient emergent languages than those trained in noise-free environments.

BACKGROUND:  
Robust and generalizable emergent communication protocols remain a central challenge in multi-agent systems, especially under environmental noise that degrades message fidelity. While prior work explores inductive biases and simplified noise models, there is limited understanding of how realistic, temporally correlated wireless noi

**Candidate (S15) — 25/40 — LOST:**
IDEA:  
Multi-agent systems that integrate a hierarchical reinforcement learning meta-controller to dynamically evolve communication protocols based on local environmental feedback develop more efficient, robust, and adaptive emergent languages under noisy conditions than systems using fixed or statically learned protocols.

BACKGROUND:  
Current multi-agent communication frameworks often rely on either fixed protocols or learned but static communication structures, limiting adaptability and robustness in realistic, noisy environments. The challenge remains to design multi-agent systems capabl

**Judge:** Idea A scores 32/40 versus Idea B's 25/40. Idea A excels with concrete experimental design (validated wireless noise models, clear falsification criteria, 50+ runs with statistical rigor), superior clarity on methodology, and realistic grounding in actual wireless channel data. Idea B proposes an interesting hierarchical meta-learning approach but lacks specificity in how the meta-controller operates, provides vague feedback mechanisms, and offers less rigorous experimental design with undefined theoretical framework. Idea A's integration of realistic temporally-correlated noise is more novel than Idea B's general adaptive protocol concept, and its feasibility is higher given established benchmarks and noise models.

### Topic: Topological methods for high-dimensional data analysis
**Champion (S12) — 30/40 — WON:**
IDEA:  
Applying rigorously parameterized persistent homology to PHATE embeddings of time-series single-cell RNA-seq data can robustly capture dynamic topological features indicative of biological state transitions, complementing Mapper and Ball Mapper by revealing stable multi-scale loops and branches that correlate with annotated lineage progression.

BACKGROUND:  
Current topological data analysis (TDA) methods like Mapper and Ball Mapper effectively characterize cellular heterogeneity and continuous trajectories in high-dimensional single-cell data, but may miss certain persistent multisca

**Candidate (S15) — 23/40 — LOST:**
IDEA:  
Incorporating multi-parameter persistent homology that jointly analyzes gene expression similarity and functional annotation similarity improves cell type classification accuracy in scRNA-seq data beyond state-of-the-art Mapper and UMAP-based methods.

BACKGROUND:  
Single-cell RNA sequencing (scRNA-seq) data are intrinsically high-dimensional and heterogeneous, posing challenges for accurate cell type classification. Existing topological approaches such as Mapper capture shape but ignore functional gene annotations, while manifold learning methods like UMAP reduce dimensionality but m

**Judge:** Idea A presents a more focused and rigorous approach with clear theoretical grounding in PH stability theorems, comprehensive negative controls, and well-defined failure criteria (>10% performance drop, <50% sensitivity). Idea B attempts a more ambitious multi-parameter framework but lacks specificity in how bi-filtration stability is theoretically justified, makes vague claims about "scalable" implementations without detail, and the 3% accuracy improvement threshold is modest and potentially achievable through simpler methods. Idea A's total score (30) substantially exceeds Idea B's (23), with particular weaknesses in B's feasibility (multi-parameter persistence at million-cell scale remains computationally challenging) and experimental clarity (unclear how functional annotations are aggregated and validated).

### Topic: Quantum error correction in NISQ devices
**Champion (S12) — 29/40 — WON:**
IDEA: Implementing an adaptive quantum error correction (AQEC) scheme that dynamically tunes syndrome extraction frequency and decoding thresholds based on real-time noise characterization from IBM Falcon and Eagle processor datasets will reduce logical error rates in NISQ devices by at least 20% compared to static low-overhead codes under matched hardware noise conditions.

BACKGROUND: Quantum error correction remains a critical bottleneck for practical quantum computing on noisy intermediate-scale quantum (NISQ) devices, as existing static QEC codes either incur prohibitive overhead or fail 

**Candidate (S15) — 23/40 — LOST:**
IDEA:  
Implementing a supervised recurrent neural network (RNN)-based adaptive, real-time syndrome extraction protocol with low-latency FPGA feedback on IBM Falcon NISQ devices can reduce logical error rates in Surface-17 quantum error correction codes by at least 20% compared to static and heuristic adaptive syndrome measurement schedules.

BACKGROUND:  
Current NISQ quantum error correction (QEC) demonstrations predominantly rely on static syndrome measurement schedules, limiting their ability to respond dynamically to evolving error patterns and noise fluctuations. Adaptive syndrome extrac

**Judge:** Idea A scores higher across all dimensions with a total of 29 vs. 23 for Idea B. Idea A presents a more grounded approach using RL within realistic hardware constraints, clearer experimental methodology with well-defined baselines, and higher feasibility since it avoids the substantial engineering overhead of FPGA integration. Idea B's reliance on FPGA co-processor integration, offline RNN training on limited datasets, and real-time inference introduces significant technical and practical challenges that are less well-justified given the modest novelty gain over heuristic adaptive methods.


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
  > Idea A scores higher overall (29 vs 26) with stronger novelty in demonstrating real-time adaptive feedback within coherence constraints and superior experimental clarity with well-defined latency budgets and rigorous controls. While Idea B addresses a complementary problem with slightly better feasi
  > Idea A scores 32/40 versus Idea B's 26/40. Idea A presents a more concrete, well-specified approach with clearer experimental methodology (precise metrics, statistical testing, ablation design) and stronger feasibility through structured pruning on standard architectures. Idea B, while addressing an

When the CANDIDATE wins (good — what to aim for):
  > Idea A scores higher overall (31 vs 29) with superior novelty in its rigorous multi-omic harmonization and GRN-specific application, greater scientific usefulness by addressing a more fundamental regulatory biology problem with clear biological validation (perturbation screens, motif enrichment), an
  > Idea A scores 30 total points versus Idea B's 24 points. Idea A presents a more grounded, hardware-aware approach with clearer experimental methodology, concrete pulse-level optimizations, and stronger feasibility on current NISQ devices. Idea B's adaptive RNN decoder, while conceptually interesting
  > Idea A scores higher overall (29 vs 24) with stronger experimental clarity, more concrete methodology (Bayesian/Kalman filtering with specific decoding algorithms), and better feasibility grounding on established platforms. Idea B references a future 2026 publication and proposes less clearly specif
  > Idea A scores higher overall (28 vs 26) due to superior novelty in combining erasure conversion with dynamic real-time switching and more comprehensive experimental design across multiple hardware platforms. However, Idea A's feasibility is slightly lower due to the complexity of implementing real-t

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

```

---

### → DIAGNOSE OUTPUT (sample from round 2):

```
DIAGNOSIS: Step 4 (IDEA CONSTRUCTION) fails to anchor the experimental design to the mechanistic claim in the hypothesis. The LLM reuses the hypothesis as a capability label rather than a causal mechanism, producing experiments that test 'whether X helps' rather than 'why X helps via mechanism M'. Losing ideas like 'multi-modal sensory fusion improves zero-shot RL' describe an intervention+outcome without naming a mechanism, making the experiment a vague empirical sweep.

FIX: In Step 4 (construct_prompt), add a locked preamble forcing the model to (a) restate the precise mechanistic claim from the hypothesis, (b) state the null result numerically before designing the experiment, (c) justify why the proposed method is the MINIMAL SUFFICIENT test of that mechanism — not a general improvement.

EXPECTED_IMPACT: Forces mechanistic grounding before experimental design, eliminating the vague-capability-sweep failure pattern.
```

---

## SWE STEP 2 — ATTACK_PROMPT

**Input size: 1,313 chars**

**Purpose:** Takes diagnosis+fix, attacks it on 3 dimensions (root cause / assumption / residual failure) → strengthened REVISED_FIX

```
A proposed fix for an underperforming research idea generator:

DIAGNOSIS: Step 4 (IDEA CONSTRUCTION) fails to anchor the experimental design to the mechanistic claim in the hypothesis. The LLM reuses the hypothesis as a capability label rather than a causal mechanism, producing experiments that test 'whether X helps' rather than 'why X helps via mechanism M'. Losing ideas like 'multi-modal sensory fusion improves zero-shot RL' describe an intervention+outcome without naming a mechanism, making the experiment a vague empirical sweep.
FIX: In Step 4 (construct_prompt), add a locked preamble forcing the model to (a) restate the precise mechanistic claim from the hypothesis, (b) state the null result numerically before designing the experiment, (c) justify why the proposed method is the MINIMAL SUFFICIENT test of that mechanism — not a general improvement.

Attack this fix on 3 dimensions:
1. ROOT CAUSE: Does this actually address the root cause, or just a symptom?
2. ASSUMPTION: What unstated assumption does this fix make that might not hold?
3. RESIDUAL FAILURE: What would still fail after making this change?

Then write a REVISED fix that is stronger, more targeted, and addresses all three attacks.

REVISED_FIX: <concrete, strengthened version — specific enough to implement directly as code>

```

---

### → ATTACK OUTPUT (sample):

```
1. ROOT CAUSE: The fix is correct about Step 4 being the failure point but may not be sufficient...
2. ASSUMPTION: Assumes the hypothesis string passed to Step 4 already contains an extractable mechanism...
3. RESIDUAL FAILURE: Even with a preamble, the model will generate threshold confabulation...

REVISED_FIX: Modify the construct_prompt in Step 4 to require a three-field preamble that the model must fill BEFORE writing the experiment: MECHANISTIC_CLAIM (the causal mechanism, not the capability), NULL_RESULT (numeric threshold that falsifies), MINIMAL_JUSTIFICATION (why this specific design isolates the mechanism). This forces mechanistic grounding upstream of experimental design, preventing the 'vague empirical sweep' pattern seen in losing ideas.
```

---

## SWE STEP 3 — CLAUDE CODE TASK PROMPT

**Input size: 5,211 chars**

**Purpose:** Sends refined fix + failure context to Claude Code, which reads S15.py and makes surgical edits → outputs S16_r2.py

```
You are improving a Python research idea generator. Make ONE focused, surgical improvement.

## Diagnosis and proposed fix
DIAGNOSIS: Step 4 (IDEA CONSTRUCTION) fails to anchor the experimental design to the mechanistic claim in the hypothesis. The LLM reuses the hypothesis as a capability label rather than a causal mechanism, producing experiments that test 'whether X helps' rather than 'why X helps via mechanism M'. Losing ideas like 'multi-modal sensory fusion improves zero-shot RL' describe an intervention+outcome without naming a mechanism, making the experiment a vague empirical sweep.

REFINED FIX: Modify the construct_prompt in Step 4 to require a three-field preamble that the model must fill BEFORE writing the experiment: MECHANISTIC_CLAIM (the causal mechanism, not the capability), NULL_RESULT (numeric threshold that falsifies), MINIMAL_JUSTIFICATION (why this specific design isolates the mechanism). This forces mechanistic grounding upstream of experimental design, preventing the 'vague empirical sweep' pattern seen in losing ideas.

EXPECTED IMPACT: Forces mechanistic grounding before experimental design, eliminating the vague-capability-sweep failure pattern.

Edit history this session:
(none yet)

## Why the current generator is losing to the baseline
### Topic: Zero-shot generalization in reinforcement learning
**Champion (S12) — 30/40 — WON:**
IDEA: Integrating a two-level hierarchical latent variable model that separately encodes task-agnostic environmental dynamics and abstract reward-conditioned task representations can significantly improve zero-shot generalization across tasks with shared dynamics but structurally varying reward functions.

BACKGROUND: Zero-shot generalization in reinforcement learning remains an open challenge, particularly when tasks share underlying environment dynamics but differ in reward structures and objectives. Existing context-based or modular methods often entangle dynamics and reward signals, limiti

**Candidate (S15) — 22/40 — LOST:**
IDEA:  
Zero-shot generalization in reinforcement learning improves significantly when policies are trained using multi-modal sensory inputs (proprioception, vision, and audio) with modality-specific encoders and attention-based fusion, even without data augmentation or explicit regularization.

BACKGROUND:  
Reinforcement learning (RL) agents often fail to generalize zero-shot to out-of-distribution environments, limiting their deployment in complex real-world tasks. Prior approaches focus heavily on data augmentation or representation learning to improve generalization but rarely isolate the

**Judge:** Idea A presents a principled hierarchical decomposition of latent representations that directly addresses a core challenge in zero-shot RL (disentangling dynamics from rewards), with clear experimental design and well-defined success criteria. Idea B, while methodologically sound, tackles a less fundamental problem (multi-modal perception benefits) that is somewhat orthogonal to zero-shot generalization and lacks strong theoretical motivation for why audio would substantially improve transfer without augmentation—the core claim feels incremental and the audio simulation pipeline adds complexity without clear necessity for the core research question.

### Topic: Emergent communication in multi-agent systems
**Champion (S12) — 32/40 — WON:**
IDEA:  
Multi-agent systems trained with communication protocols evolving under realistic, temporally correlated wireless channel noise develop more robust, generalizable, and efficient emergent languages than those trained in noise-free environments.

BACKGROUND:  
Robust and generalizable emergent communication protocols remain a central challenge in multi-agent systems, especially under environmental noise that degrades message fidelity. While prior work explores inductive biases and simplified noise models, there is limited understanding of how realistic, temporally correlated wireless noi

**Candidate (S15) — 25/40 — LOST:**
IDEA:  
Multi-agent systems that integrate a hierarchical reinforcement learning meta-controller to dynamically evolve communication protocols based on local environmental feedback develop more efficient, robust, and adaptive emergent languages under noisy conditio

## What to do
1. Read `ideas/systems/S16_r2.py` — this is already a copy of the champion, your starting point
2. Read the relevant `ideas/idea_tournament/` modules for full context on what the pipeline does
3. Implement the improvement strategy above — make TARGETED edits to `ideas/systems/S16_r2.py`

## Hard constraints (failure to meet these causes the system to crash or be disqualified)
- Class name MUST be `S16_r2Generator(IdeaGenerator)`
- `VERSION = "S16_r2"` (update from whatever the champion has)
- File MUST end with `GENERATOR = S16_r2Generator()`
- Any code you MODIFY from `idea_tournament/` must be inlined in the file (not imported)
- Every `call_llm()` call must be wrapped in try/except with a sensible fallback
- `generate_idea(self, topic, client, model, temperature)` must always return a non-empty string
- Keep total LLM calls per idea: ~10-15 (same budget as champion, just use them differently)

```

---

### → CLAUDE CODE OUTPUT:

Claude Code reads `ideas/systems/S15.py`, edits Step 4 construct_prompt, writes `S16_r2.py`.

Key change: Step 4 now has a 3-field preamble (MECHANISTIC_CLAIM / NULL_RESULT / MINIMAL_JUSTIFICATION)

before the LLM can design the experiment.

Result: mini-eval S15 vs S16_r2 → **77.8% win rate** (accepted)

---

# PART 2: IDEA GENERATOR (S16)

13 LLM calls per idea. Topic example: 'Zero-shot generalization in reinforcement learning'

---

## IDEA STEP 0 — SOTA Retrieval (Semantic Scholar, not LLM)

**Size: 87 chars**
**Purpose:** Fetches 5 recent papers on topic — used in Steps 1 & 4 as grounding context

### → INPUT TO LLM:

```
Query: 'Zero-shot generalization in reinforcement learning'
→ Returns 5 paper abstracts
```

### → LLM OUTPUT:

## Recent related work (for context — your idea must go beyond these)

1. **Learning quadrupedal locomotion over challenging terrain** (2020)
  Legged locomotion can extend the operational domain of robots to some of the most challenging environments on Earth. However, conventional controllers for legged locomotion are based on elaborate state machines that explicitly trigger the execution of motion primitives and reflexes. These designs have increased in complexity but fallen short of the generality and robustness of animal locomotion. Here, we present a robust controller for blind quadrupedal locomotion in challenging natural environments. Our approach incorporates proprioceptive feedback in locomotion control and demonstrates zero-
2. **A survey on Image Data Augmentation for Deep Learning** (2019)
  Deep convolutional neural networks have performed remarkably well on many Computer Vision tasks. However, these networks are heavily reliant on big data to avoid overfitting. Overfitting refers to the phenomenon when a network learns a function with very high variance such as to perfectly model the training data. Unfortunately, many application domains do not have access to big d

## ...[5 papers total]

## IDEA STEP 1 — Hypothesis Generation

**Size: 4,148 chars**
**Purpose:** Generates 5 falsifiable scientific hypotheses — claims about the world, not techniques

### → INPUT TO LLM:

```
Research topic: Zero-shot generalization in reinforcement learning

## Recent related work (for context — your idea must go beyond these)

1. **Learning quadrupedal locomotion over challenging terrain** (2020)
   Legged locomotion can extend the operational domain of robots to some of the most challenging environments on Earth. However, conventional controllers for legged locomotion are based on elaborate state machines that explicitly trigger the execution of motion primitives and reflexes. These designs have increased in complexity but fallen short of the generality and robustness of animal locomotion. Here, we present a robust controller for blind quadrupedal locomotion in challenging natural environments. Our approach incorporates proprioceptive feedback in locomotion control and demonstrates zero-

2. **A survey on Image Data Augmentation for Deep Learning** (2019)
   Deep convolutional neural networks have performed remarkably well on many Computer Vision tasks. However, these networks are heavily reliant on big data to avoid overfitting. Overfitting refers to the phenomenon when a network learns a function with very high variance such as to perfectly model the training data. Unfortunately, many application domains do not have access to big data, such as medical image analysis. This survey focuses on Data Augmentation, a data-space solution to the problem of limited data. Data Augmentation encompasses a suite of techniques that enhance the size and quality

3. **Improving Zero-shot Generalization in Offline Reinforcement Learning using Generalized Similarity Functions** (2021)
   Reinforcement learning (RL) agents are widely used for solving complex sequential decision making tasks, but still exhibit difficulty in generalizing to scenarios not seen during training. While prior online approaches demonstrated that using additional signals beyond the reward function can lead to better generalization capabilities in RL agents, i.e. using self-supervised learning (SSL), they struggle in the offline RL setting, i.e. learning from a static dataset. We show that performance of online algorithms for generalization in RL can be hindered in the offline setting due to poor estimat

4. **Improving Zero-shot Generalization in Offline Reinforcement Learning\n using Generalized Similarity Functions** (2021)
   Reinforcement learning (RL) agents are widely used for solving complex\nsequential decision making tasks, but still exhibit difficulty in generalizing\nto scenarios not seen during training. While prior online approaches\ndemonstrated that using additional signals beyond the reward function can lead\nto better generalization capabilities in RL agents, i.e. using self-supervised\nlearning (SSL), they struggle in the offline RL setting, i.e. learning from a\nstatic dataset. We show that performance of online algorithms for\ngeneralization in RL can be hindered in the offline setting due to poor\

5. **Improving Zero-Shot Generalization in Reinforcement Learning Through Abstract Representations** (2025)
   Reinforcement learning algorithms suffer from zero-shot generalization, where a learned policy is tested in new, out-of-distribution environments. One of the key reasons is the overfitting of the representation learning model. Previous studies have used data augmentation or contrastive learning to address this overfitting. Yet these methods often rely on task-specific augmentations or hand-picked negative samples, which are not robust to various tasks. We hypothesize that learning more abstract representations by ignoring pixel-level details allows the representation learning model to generali


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

### → LLM OUTPUT:

## H1: Zero-shot RL generalization fails because policy networks learn task-specific reward heuristics that don't transfer, not because of observation distribution shift — and models trained with reward-randomization but fixed observations will generalize no better than standard baselines.

H2: Hierarchical latent variable models that separately encode environment dynamics and reward-conditioned task representations improve zero-shot RL transfer across tasks with shared dynamics but varying reward functions, because disentangling these factors prevents policy overfitting to reward-specific features.
H3: ...
H4: ...
H5: ...

## IDEA STEP 2a–e — Adversarial Attack × 5 (PARALLEL)

**Size: 1,001 chars**
**Purpose:** Each hypothesis independently attacked on 4 dimensions + forced revised form — runs 5 calls in parallel

### → INPUT TO LLM:

```
Research topic: Zero-shot generalization in reinforcement learning

Hypothesis: Hierarchical latent variable models that separately encode environment dynamics and reward-conditioned task representations improve zero-shot RL transfer across tasks with shared dynamics but varying reward functions, because disentangling these factors prevents policy overfitting to reward-specific features.

You are an adversarial critic. Attack this hypothesis on EACH of these dimensions:
1. Assumption violation: What unstated assumption does this rely on? Give a concrete counterexample.
2. Dataset bias: What dataset artifact could make this appear true without actually being true?
3. Theoretical gap: What known result from the literature contradicts or undermines this?
4. Practical limitation: What makes this hypothesis untestable or too expensive to test?

Then, given these attacks, write a REVISED hypothesis that survives them:
Revised: <1-2 sentence refined hypothesis that addresses the above attacks>
```

### → LLM OUTPUT:

1. Assumption violation: Assumes dynamics and reward are independently variable, but most RL envs couple them by design...
2. Dataset bias: MiniGrid/Procgen results may show this because their task families happen to share dynamics, not because of latent disentanglement...
3. Theoretical gap: Modular RL (Aljalbout et al. 2021) already attempts this without explicit latent variable separation...
4. Practical: Requires constructing matched task pairs with controlled dynamics/reward variation — not available in standard benchmarks.

## Revised: Hierarchical latent variable models improve zero-shot RL transfer specifically when task distribution has shared dynamics but varying reward structures — a condition most benchmarks conflate, making this testable by constructing paired task sets with independently controlled dynamics-vs-reward variation.

## IDEA STEP 3 — Hypothesis Selection

**Size: 1,637 chars**
**Purpose:** Aggregates all 5 original+attack+revised pairs, selects the strongest surviving hypothesis

### → INPUT TO LLM:

```
Research topic: Zero-shot generalization in reinforcement learning

Below are 5 original hypotheses, each attacked and refined:

--- Hypothesis 1 ---
Original: Zero-shot RL generalization fails because policy networks learn task-specific reward heuristics that don't transfer, not because of observation distribution shift — and models trained with reward-randomization but fixed observations will generalize no better than standard baselines.
Critique+Revision:
[adversarial critique...]
Revised: [revised form]

--- Hypothesis 2 ---
Original: Hierarchical latent variable models that separately encode environment dynamics and reward-conditioned task representations improve zero-shot RL transfer across tasks with shared dynamics but varying reward functions, because disentangling these factors prevents policy overfitting to reward-specific features.
Critique+Revision:
[adversarial critique...]
Revised: Hierarchical latent variable models improve zero-shot RL transfer specifically when task distribution has shared dynamics but varying reward structures — a condition most benchmarks conflate, making this testable by constructing paired task sets with independently controlled dynamics-vs-reward variation.

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

### → LLM OUTPUT:

## SELECTED: 2

REVISED HYPOTHESIS: Hierarchical latent variable models improve zero-shot RL transfer specifically when task distribution has shared dynamics but varying reward structures — a condition most benchmarks conflate, making this testable by constructing paired task sets with independently controlled dynamics-vs-reward variation.
REASONING: This is the most falsifiable — it makes a specific claim about when hierarchical models help (shared dynamics, varying rewards) and names a concrete experimental design that could disprove it.

## IDEA STEP 4 (S16 — NEW) — Experimental Construction with Mechanistic Lock

**Size: 1,869 chars**
**Purpose:** S16 KEY CHANGE: Forces 3-field preamble (MECHANISTIC_CLAIM / NULL_RESULT / MINIMAL_JUSTIFICATION) before experiment design — this was the fix that drove mini-eval from 33% to 77.8%

### → INPUT TO LLM:

```
Research topic: Zero-shot generalization in reinforcement learning

Existing work to differentiate from:
## Recent related work (for context — your idea must go beyond these)

1. **Learning quadrupedal locomotion over challenging terrain** (2020)
   Legged locomotion can extend the operational domain of robots to some of the most challenging environments on Earth. However, conventional controllers for legged locomotion are based on elaborate state machines that explicitly trigger the execution of motion primitives and reflexes. These designs have increased in complexity but fallen short of the generality and robustness of animal locomotion. Here, we present a robust controller for blind quadrupedal locomotion in challenging natural environments. Our approach incorporates proprioceptive feedback in locomotion control and demonstrates zero-

2. **A survey on Image Data Augmentation for Deep Lea
...[truncated]

Core hypothesis to test: Hierarchical latent variable models improve zero-shot RL transfer specifically when task distribution has shared dynamics but varying reward structures — a condition most benchmarks conflate, making this testable by constructing paired task sets with independently controlled dynamics-vs-reward variation.

Before designing the experiment, fill in these three fields:

MECHANISTIC_CLAIM: <state the precise causal mechanism — what property causes what effect via what process>
NULL_RESULT: <numeric threshold — what specific measurement would FALSIFY this hypothesis>
MINIMAL_JUSTIFICATION: <why your proposed method is the MINIMAL SUFFICIENT test of this mechanism>

Now design the concrete experiment:
- Name specific datasets
- Name specific baselines
- Define quantitative success metrics
- Describe the key technical method
- Explain what result falsifies the hypothesis

Write 3-4 paragraphs. Be direct and specific.
```

### → LLM OUTPUT:

MECHANISTIC_CLAIM: Disentangled latent representations prevent policy overfitting to reward-specific features by encoding dynamics and reward-conditioned task separately, enabling the dynamics encoder to transfer to new task reward structures without relearning.

NULL_RESULT: If the hierarchical model achieves ≤2% higher zero-shot transfer accuracy than a flat baseline on held-out reward-varied tasks (shared dynamics), the disentanglement mechanism does not provide transfer benefit beyond standard feature learning.

MINIMAL_JUSTIFICATION: We need only one architecture change (separate dynamics/reward encoders with a disentanglement loss) tested on paired task sets; any additional components (attention, auxiliary tasks) would confound whether disentanglement — not extra capacity — drives the improvement.

## [3-4 paragraphs: specific datasets (MiniGrid-4rooms with reward variants), baselines (DIAYN, standard PPO, RNN-PPO), metrics (zero-shot success rate on held-out reward structures), falsification criteria...]

## IDEA STEP 5a — Experimentalist Critique

**Size: 644 chars**
**Purpose:** Hard-nosed experimentalist: can this actually be run? are measurements well-defined?

### → INPUT TO LLM:

```
You are a hard-nosed experimentalist reviewing a research proposal about 'Zero-shot generalization in reinforcement learning'.

Hypothesis: Hierarchical latent variable models improve zero-shot RL transfer specifically when task distribution has shared dynamics but varying reward structures — a condition most benchmarks conflate, making this testable by constructing paired task sets with independently controlled dynamics-vs-reward variation.

Proposed experiment:
[3-4 paragraph experimental design with MECHANISTIC_CLAIM/NULL_RESULT/MINIMAL_JUSTIFICATION preamble...]

Give 2-3 sharp criticisms focusing purely on experimental feasibility.
```

### → LLM OUTPUT:

## *2-3 feasibility criticisms*

## IDEA STEP 5b — Theorist Critique

**Size: 552 chars**
**Purpose:** Rigorous theorist: is the novelty claim justified? does it overlap with known results?

### → INPUT TO LLM:

```
You are a rigorous theorist reviewing a research proposal about 'Zero-shot generalization in reinforcement learning'.

Hypothesis: Hierarchical latent variable models improve zero-shot RL transfer specifically when task distribution has shared dynamics but varying reward structures — a condition most benchmarks conflate, making this testable by constructing paired task sets with independently controlled dynamics-vs-reward variation.

Proposed experiment:
[experimental design...]

Give 2-3 sharp criticisms focusing purely on theoretical grounding.
```

### → LLM OUTPUT:

## *2-3 theory criticisms*

## IDEA STEP 5c — Skeptic Critique

**Size: 576 chars**
**Purpose:** Skeptic: why this won't work, likely negative result, is the payoff worth it?

### → INPUT TO LLM:

```
You are a skeptical reviewer who has seen many overhyped proposals about 'Zero-shot generalization in reinforcement learning'.

Hypothesis: Hierarchical latent variable models improve zero-shot RL transfer specifically when task distribution has shared dynamics but varying reward structures — a condition most benchmarks conflate, making this testable by constructing paired task sets with independently controlled dynamics-vs-reward variation.

Proposed experiment:
[experimental design...]

Give 2-3 criticisms: why this probably won't work, likely negative result, payoff.
```

### → LLM OUTPUT:

## *2-3 skeptic criticisms*

## IDEA STEP 5d — Critique Synthesis

**Size: 609 chars**
**Purpose:** Synthesizes all 3 critiques into top 3 actionable improvements

### → INPUT TO LLM:

```
Three reviewers critiqued a research idea about 'Zero-shot generalization in reinforcement learning' testing:
'Hierarchical latent variable models improve zero-shot RL transfer specifically when task distribution has shared dynamics but varying reward structures — a condition most benchmarks conflate, making this testable by constructing paired task sets with independently controlled dynamics-vs-reward variation.'

Experimentalist:
[2-3 feasibility criticisms]

Theorist:
[2-3 theory criticisms]

Skeptic:
[2-3 skeptic criticisms]

Synthesize into the 3 most important actionable improvements. Be concise.
```

### → LLM OUTPUT:

- 

1. [most important improvement]
2. [second]
3. [third]*

---

## IDEA STEP 6 — Final Revision

**Size: 1,894 chars**
**Purpose:** Incorporates synthesis + SOTA grounding → final IDEA/BACKGROUND/APPROACH/EXPERIMENT/NOVELTY format

### → INPUT TO LLM:

```
Research topic: Zero-shot generalization in reinforcement learning

Core hypothesis: Hierarchical latent variable models improve zero-shot RL transfer specifically when task distribution has shared dynamics but varying reward structures — a condition most benchmarks conflate, making this testable by constructing paired task sets with independently controlled dynamics-vs-reward variation.

Experimental design:
[3-4 paragraphs from Step 4, including mechanistic preamble...]

Key improvements required:
[synthesis of 3 critics...]

Existing work to differentiate from:
## Recent related work (for context — your idea must go beyond these)

1. **Learning quadrupedal locomotion over challenging terrain** (2020)
   Legged locomotion can extend the operational domain of robots to some of the most challenging environments on Earth. However, conventional controllers for legged locomotion are based on elaborate state machines that explicitly trigger the execution of motion primitives and reflexes. These designs have increased in complexity but fallen short of the generality and robustness of animal locomotion. Here, we present a robust controller for blind quadrupedal locomotion in challenging natural environments. Our approach incorporates proprioceptive feedback in locomotion control and demonstrates zero-

2. **A survey on Image Data Augmentation for Deep Lea
...[truncated]

Write the final, improved version of the research idea. Ensure the hypothesis is clearly stated, the falsification experiment is concrete, and all datasets/baselines/metrics are named explicitly.

Present your final research idea in exactly this format:

IDEA: [One sentence summary]

BACKGROUND: [The specific open problem, 2-3 sentences]

APPROACH: [The proposed method, 2-3 sentences]

EXPERIMENT: [How to test it — datasets, metrics, 1-2 sentences]

NOVELTY: [What makes this different, 1-2 sentences]

```

### → LLM OUTPUT:

**REAL OUTPUT from S15 ideas.json (same topic):**
IDEA: Incorporating dynamically updated task embeddings derived from online environment interactions during reinforcement learning training significantly improves zero-shot generalization to unseen tasks compared to static or contrastively learned embeddings.

BACKGROUND: Reinforcement learning agents typically struggle to generalize zero-shot to new tasks or environment variations due to overfitting to training distributions and insufficiently adaptive task representations. Existing approaches like meta-RL and contrastive representation learning rely on fixed or offline-learned embeddings that may not capture evolving task dynamics during deployment, limiting adaptability and robustness.

APPROACH: We propose an online embedding framework where a recurrent encoder (e.g., LSTM or GRU) continuously integrates recent state-action-reward trajectories into a latent task embedding vector that conditions the policy and value networks; the encoder parameters are jointly optimized end-to-end with the RL objective via truncated backpropagation through time. This dynamic embedding contrasts with static embeddings precomputed from offline environment descriptors and contrastive embeddings learned via prior methods, enabling continual adaptation to task variations encountered during training.

EXPERIMENT: We evaluate on challenging continuous control benchmarks including Meta-World ML1 and ML10 robotic manipulation suites and DeepMind Control Suite’s Walker and Cheetah with randomized physical parameters, measuring zero-shot generalization performance as normalized average episode return on held-out unseen tasks, with significance tested by paired t-tests over 10+ random seeds. Baselines include agents with static task embeddings, agents without task embeddings, and contrastive embedding agents from state-of-the-art methods, controlling embedding capacities and hyperparameters for fair comparison.

NOVELTY: Unlike prior meta-RL and context-based methods such as PEARL and VariBAD that learn fixed or slowly updated context embeddings, our approach introduces an explicitly online-updated recurrent embedding trained jointly with the policy to capture fine-grained, evolving task characteristics. Theoretical analysis grounds why online adaptation of embeddings better approximates underlying task distributions under assumptions of partial observability and trajectory sufficiency, providing formal justification missing in earlier work.