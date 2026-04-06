"""S13_r2: Constraint-Inversion Seeding + Adversarial Refinement.

Replaces hypothesis-from-technique generation with CONSTRAINT-INVERSION SEEDING:

1. IMPOSSIBILITY MINING (1 call): Name the 3 hardest fundamental blockers/
   impossibilities that prevent progress on this topic today — not "we need
   better methods" but concrete, named failure modes (e.g., "dataset X lacks
   Y property required for Z claim").

2. DISSOLUTION CANDIDATES (3 parallel calls): For each impossibility, generate
   a candidate idea as the mechanism that would DISSOLVE that specific blocker.
   Starting from *what's broken* rather than *what techniques exist*.

3. ADVERSARIAL ATTACK (3 parallel calls): Each dissolution candidate is attacked
   on assumption violations, dataset bias, theoretical gaps, practical limits.
   Each attack produces a refined version.

4. CANDIDATE SELECTION (1 call): Pick the strongest surviving dissolution +
   refined idea pair — most concrete, most falsifiable, most novel.

5. IDEA CONSTRUCTION (1 call): Build full experimental design around the
   selected dissolution mechanism. Forces concrete datasets, baselines, metrics.

6. MULTI-PERSPECTIVE CRITIQUE (4 calls): Experimentalist, theorist, skeptic
   critique + synthesis.

7. FINAL REVISION (1 call): Incorporate critique.

LLM calls: 1 (impossibilities) + 3 (dissolve) + 3 (attacks) + 1 (select) +
           1 (construct) + 4 (critique) + 1 (revise) = 14
"""

import sys
import os
import concurrent.futures

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from systems.base import IdeaGenerator, DEFAULT_MODEL, call_llm, IDEA_FORMAT


class S13Generator(IdeaGenerator):
    VERSION = "S13"
    DESCRIPTION = (
        "Constraint-inversion seeding: mine fundamental blockers, generate ideas "
        "as dissolution mechanisms, adversarial refinement, then multi-perspective critique."
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

        # ── Step 1: mine 3 fundamental impossibilities/blockers ──────────────
        impossible_prompt = (
            f"Research topic: {topic}{context_block}\n"
            "You are a harsh critic who has seen this field fail repeatedly.\n"
            "Name exactly 3 FUNDAMENTAL BLOCKERS that prevent real progress on this topic today.\n\n"
            "Each blocker must:\n"
            "- Be a CONCRETE, NAMED failure mode (not vague like 'we need better data')\n"
            "- Explain WHY it is structurally hard, not just that it is hard\n"
            "- NOT be something that standard techniques in the related work above can solve\n"
            "- Be specific enough that someone could design an experiment targeting only it\n\n"
            "Format each as:\n"
            "BLOCKER1: <name> — <1-2 sentence explanation of the structural reason it is hard>\n"
            "BLOCKER2: <name> — <1-2 sentence explanation>\n"
            "BLOCKER3: <name> — <1-2 sentence explanation>"
        )
        try:
            blockers_raw = call_llm(impossible_prompt, model, client, temperature)
        except Exception:
            blockers_raw = (
                f"BLOCKER1: Distribution shift — standard {topic} methods fail when "
                f"test distributions differ structurally from training.\n"
                f"BLOCKER2: Evaluation gap — benchmarks do not measure the properties "
                f"that matter in deployment.\n"
                f"BLOCKER3: Scalability wall — methods that work at small scale break "
                f"under realistic data volume."
            )

        # Parse blockers
        blockers = []
        for line in blockers_raw.strip().split("\n"):
            line = line.strip()
            if line and line.upper().startswith("BLOCKER") and ":" in line[:10]:
                blocker_text = line.split(":", 1)[1].strip()
                if blocker_text:
                    blockers.append(blocker_text)
        if not blockers:
            blockers = [blockers_raw.strip()]
        blockers = blockers[:3]

        # ── Step 2: generate dissolution candidates (parallel) ───────────────
        def dissolve_blocker(blocker: str) -> str:
            dissolve_prompt = (
                f"Research topic: {topic}\n\n"
                f"Fundamental blocker: {blocker}\n\n"
                "Design a research idea whose CORE MECHANISM directly dissolves this blocker.\n"
                "The idea should:\n"
                "- Attack the blocker structurally, not work around it\n"
                "- Be grounded in a specific technical mechanism (not a research direction)\n"
                "- Name what data, model, or system component it changes and why that change matters\n"
                "- Be surprising: it should NOT be the obvious next step from existing work\n\n"
                "Write 2-3 sentences describing the dissolution mechanism."
            )
            try:
                return call_llm(dissolve_prompt, model, client, temperature)
            except Exception as e:
                return f"Approach to dissolve '{blocker}': investigate root cause and build targeted intervention. (fallback: {e})"

        dissolution_candidates = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
            futures = [executor.submit(dissolve_blocker, b) for b in blockers]
            for f in futures:
                try:
                    dissolution_candidates.append(f.result(timeout=120))
                except Exception as e:
                    dissolution_candidates.append(f"Dissolution mechanism unavailable. ({e})")

        # ── Step 3: adversarial attacks on each dissolution candidate ────────
        def attack_candidate(pair) -> str:
            blocker, candidate = pair
            attack_prompt = (
                f"Research topic: {topic}\n\n"
                f"Blocker being dissolved: {blocker}\n"
                f"Proposed dissolution mechanism: {candidate}\n\n"
                "You are an adversarial critic. Attack this proposal on EACH dimension:\n"
                "1. Assumption violation: What does this silently assume that is probably false?\n"
                "2. Dataset bias: What artifact would make this look like it works without working?\n"
                "3. Theoretical gap: What known result contradicts or limits this approach?\n"
                "4. Practical failure: What will break when you actually try to implement this?\n\n"
                "Then write a REVISED version that survives these attacks:\n"
                "Revised: <2-3 sentence tightened version addressing the above>"
            )
            try:
                return call_llm(attack_prompt, model, client, temperature=0.6)
            except Exception as e:
                return f"Revised: {candidate} (attack failed: {e})"

        attacks = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
            futures = [executor.submit(attack_candidate, (b, c))
                       for b, c in zip(blockers, dissolution_candidates)]
            for f in futures:
                try:
                    attacks.append(f.result(timeout=120))
                except Exception as e:
                    attacks.append(f"Revised: (timeout) {e}")

        # ── Step 4: select the strongest dissolution candidate ───────────────
        triples_str = ""
        for i, (blocker, candidate, attack) in enumerate(zip(blockers, dissolution_candidates, attacks)):
            triples_str += (
                f"\n--- Candidate {i+1} ---\n"
                f"Blocker targeted: {blocker}\n"
                f"Dissolution mechanism: {candidate}\n"
                f"Critique+Revision:\n{attack}\n"
            )

        select_prompt = (
            f"Research topic: {topic}\n\n"
            f"Three candidate ideas, each targeting a different fundamental blocker:\n"
            f"{triples_str}\n"
            "Select the ONE candidate (by number) whose REVISED form is:\n"
            "- Most structurally novel (attacks the blocker at its root, not the surface)\n"
            "- Most concrete (names specific mechanisms, datasets, or model components)\n"
            "- Most falsifiable (clearest path to an experiment that could prove it wrong)\n\n"
            "Respond with:\n"
            "SELECTED: <number 1-3>\n"
            "BLOCKER: <copy the blocker this candidate targets>\n"
            "REVISED IDEA: <copy the revised dissolution mechanism text exactly>\n"
            "REASONING: <1-2 sentences on why this is strongest>"
        )
        try:
            selection_raw = call_llm(select_prompt, model, client, temperature=0.3)
        except Exception as e:
            selection_raw = (
                f"SELECTED: 1\n"
                f"BLOCKER: {blockers[0] if blockers else 'unknown blocker'}\n"
                f"REVISED IDEA: {dissolution_candidates[0] if dissolution_candidates else topic}\n"
                f"REASONING: fallback"
            )

        # Extract selected blocker and idea
        selected_blocker = blockers[0] if blockers else ""
        selected_idea = dissolution_candidates[0] if dissolution_candidates else topic
        for line in selection_raw.strip().split("\n"):
            stripped = line.strip()
            if stripped.upper().startswith("BLOCKER:"):
                candidate = stripped.split(":", 1)[1].strip()
                if candidate:
                    selected_blocker = candidate
            elif stripped.upper().startswith("REVISED IDEA:"):
                candidate = stripped.split(":", 1)[1].strip()
                if candidate:
                    selected_idea = candidate

        # ── Step 5: construct full experimental idea ──────────────────────────
        context_reminder = (
            f"\nExisting work to differentiate from:\n{sota_context}\n"
            if sota_context else ""
        )
        construct_prompt = (
            f"Research topic: {topic}\n"
            f"{context_reminder}\n"
            f"Fundamental blocker being dissolved: {selected_blocker}\n"
            f"Core dissolution mechanism: {selected_idea}\n\n"
            "Design a concrete research experiment to demonstrate that this mechanism "
            "genuinely dissolves the stated blocker.\n"
            "The experiment must:\n"
            "- Name specific datasets (not 'standard benchmarks')\n"
            "- Name specific baselines to compare against\n"
            "- Define quantitative success metrics (what number proves it, what number disproves it)\n"
            "- Describe the key technical implementation in enough detail to replicate\n"
            "- Explain precisely what result would show the mechanism does NOT dissolve the blocker\n\n"
            "Write 3-4 paragraphs. Be direct and specific."
        )
        try:
            draft = call_llm(construct_prompt, model, client, temperature)
        except Exception as e:
            draft = (
                f"Research idea for '{topic}': {selected_idea} "
                f"(targeting blocker: {selected_blocker})"
            )

        # ── Step 6: multi-perspective critique ───────────────────────────────
        exp_prompt = (
            f"You are a hard-nosed experimentalist reviewing a research proposal about '{topic}'.\n\n"
            f"Blocker being dissolved: {selected_blocker}\n"
            f"Dissolution mechanism: {selected_idea}\n\n"
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
            f"Blocker being dissolved: {selected_blocker}\n"
            f"Dissolution mechanism: {selected_idea}\n\n"
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
            f"Blocker being dissolved: {selected_blocker}\n"
            f"Dissolution mechanism: {selected_idea}\n\n"
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
            f"Three reviewers critiqued a research idea about '{topic}' targeting the blocker:\n"
            f"'{selected_blocker}'\n\n"
            f"Dissolution mechanism: '{selected_idea}'\n\n"
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

        # ── Step 7: final revision ────────────────────────────────────────────
        revise_prompt = (
            f"Research topic: {topic}\n\n"
            f"Fundamental blocker dissolved: {selected_blocker}\n"
            f"Core dissolution mechanism: {selected_idea}\n\n"
            f"Experimental design:\n{draft}\n\n"
            f"Key improvements required:\n{synthesis}\n"
            f"{context_reminder}\n"
            "Write the final, improved version of the research idea. "
            "Ensure the blocker being dissolved is clearly stated, the dissolution mechanism "
            "is concrete and surprising, and all datasets/baselines/metrics are named explicitly."
            + IDEA_FORMAT
        )
        try:
            return call_llm(revise_prompt, model, client, temperature)
        except Exception:
            return draft if draft else f"Research idea about {topic}: {selected_idea}"


GENERATOR = S13Generator()
