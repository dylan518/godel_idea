# EvoScientist — Gödel Loop for Scientific Idea Generation

A self-improving research idea generator that uses pairwise LLM evaluation to evolve its own generation strategy. Each "system" version (S0, S1, ...) competes head-to-head against the current champion; winners become the new champion.

> **Gödel-style**: the system writes and evaluates its own successor, iterating toward better idea quality without human intervention.

---

## How It Works

```
┌─────────────────────────────────────────────────────┐
│                  Gödel Loop                         │
│                                                     │
│  meta-LLM (Claude Sonnet)                          │
│     │  writes new S{n}.py                          │
│     ▼                                              │
│  Generator S{n} ──generates──► Ideas               │
│     │                             │                │
│     │              Judge (Claude Haiku)             │
│     │              Blind judge (Gemini)             │
│     │                             │                │
│     └──── win_rate > 55%? ────► Accept / Reject    │
└─────────────────────────────────────────────────────┘
```

1. **Generate** — Current champion system generates N ideas per topic (15 topics × 5 ideas)
2. **Judge** — Pairwise LLM evaluation scores each idea on novelty, usefulness, clarity, feasibility
3. **Compare** — Candidate system vs. champion; blind judge (Gemini) used as Goodhart canary
4. **Evolve** — If win_rate > 55% (and no Goodhart alert), accept as new champion and repeat

---

## Current Architecture

| Component | Model | Role |
|-----------|-------|------|
| Generator | `gpt-4.1-mini` | Cheap, fast idea generation |
| Meta-LLM / SWE agent | `claude-sonnet-4-6` | Writes new S{n}.py strategy files |
| Primary judge | `claude-haiku-4-5-20251001` | Accept/reject decisions |
| Blind judge | `gemini-flash-lite-latest` | Independent canary (never used for accept/reject) |

**Current champion**: `S_sota` — SOTA paper retrieval + IdeaTreeSearch + Swiss Elo tournament

---

## Evolution History

| Version | Strategy | Win Rate | Status |
|---------|----------|----------|--------|
| S0 | Direct prompting (baseline) | — | baseline |
| S1 | Self-critique: draft → critique → revise | 100% / 83% vs S0 | champion |
| S2 | Multi-candidate tournament (3→synthesize) | 8% vs S1 | rejected |
| S3 | Structured decomposition (subproblems→idea) | 54% vs S1 (Goodhart alert) | rejected |
| S4 | Two-round self-critique | 40% vs S1 | rejected |
| S5 | Multi-perspective critique (experimentalist/theorist/skeptic) | 58% vs S1 | champion |
| S_sota | SOTA retrieval + IdeaTreeSearch + Elo tournament | current | **champion** |

**Key finding**: the judge rewards *specificity and measurability* as much as novelty. Vague ambitious ideas consistently lose to concrete, actionable ones.

---

## Idea Quality Criteria

Each idea is scored 0–10 on four dimensions:

1. **Novelty** — genuinely new angle, not incremental
2. **Scientific usefulness** — addresses real open problems
3. **Experimental clarity** — concrete methodology, testable predictions
4. **Feasibility** — achievable with current or near-future resources

---

## S_sota Pipeline (~14 LLM calls per idea)

```
SOTA papers (Semantic Scholar, cached 7 days)
         │
         ▼
IdeaTreeSearch (3-level tree)
  L0: Technique candidates
  L1: Domain applications
  L2: Concrete formulations
  L3: ~12 leaf candidates
         │
         ▼
Swiss Elo Tournament (4 rounds, 4 dimensions)
         │
         ▼
Winner expanded → IDEA_FORMAT
```

---

## Setup

```bash
git clone https://github.com/YOUR_USERNAME/evoscientist.git
cd evoscientist

pip install openai anthropic google-genai
```

Create a `.env` file:

```
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...
GEMINI_API_KEY=AIza...
SEMANTIC_SCHOLAR_API_KEY=...   # optional — increases rate limit
```

---

## Usage

```bash
# Check current state
python3 ideas/godel_loop.py status

# Run autonomous evolution loop (generates → compares → accepts → repeat)
python3 ideas/godel_loop.py swe-evolve --target 16 --n-ideas 5 --workers 3 --swe-rounds 6

# Manually compare a candidate against champion
python3 ideas/godel_loop.py compare --candidate S6 --n-ideas 5 --workers 3

# Accept a candidate as new champion
python3 ideas/godel_loop.py accept S6

# Watch live logs
tail -f ideas/results/godel.log
```

---

## Repository Structure

```
ideas/
├── godel_loop.py          # Main CLI entry point
├── runner.py              # Parallel benchmarking engine
├── judge.py               # Pairwise evaluation engine
├── swe_agent.py           # SWE agent: writes and tests new S{n}.py edits
├── meta.py                # Meta-generator: prompts claude-sonnet to write new systems
├── retrieval.py           # Semantic Scholar API with 7-day disk cache
├── tournament.py          # Swiss-system Elo tournament (K=32, start=1500)
├── log.py                 # Logging + dotenv loader
├── benchmark_topics.json  # 15 research topic categories
├── CURRENT_VERSION        # Points to current champion (e.g. "S_sota")
├── RESEARCH_AGENDA.md     # Living document: hypotheses, experiment log
├── systems/
│   ├── base.py            # Abstract IdeaGenerator + call_llm (OpenAI/Anthropic/Gemini)
│   ├── S0.py              # Baseline direct prompting
│   ├── S1.py              # Self-critique
│   ├── S5.py              # Multi-perspective critique
│   └── S_sota.py          # Current champion (tree search + tournament)
├── idea_tournament/
│   ├── prompts.py         # All LLM prompts (primary edit target for SWE agent)
│   ├── tree_search.py     # IdeaTreeSearch algorithm
│   └── tournament.py      # Elo tournament implementation
└── results/
    ├── godel.log          # Full debug log
    ├── evolution_log.jsonl # History of accepted versions
    ├── compare_*.json     # Per-comparison detailed reports
    └── S*/ideas.json      # Cached idea outputs per version
```

---

## Goodhart Alert

A Goodhart alert fires when the primary judge (Claude) and blind judge (Gemini) diverge by more than 30%. This indicates the system may be overfitting to Claude's preferences rather than genuinely improving idea quality.

- **S3 example**: Claude preferred S3 (54%) but Gemini preferred S1 (100%) — classic Goodhart failure

When the alert fires, the candidate is rejected even if win_rate > 55%.

---

## Adding a New Strategy

1. Create `ideas/systems/S{n}.py` implementing `IdeaGenerator`:

```python
from systems.base import IdeaGenerator, IDEA_FORMAT

class S7Generator(IdeaGenerator):
    VERSION = "S7"
    DESCRIPTION = "Your strategy description"

    def generate_idea(self, topic, client, model, temperature=0.8):
        # Your multi-step logic here
        ...
```

2. Compare against champion:

```bash
python3 ideas/godel_loop.py compare --candidate S7 --n-ideas 5 --workers 3
```

3. If win_rate > 55% with no Goodhart alert, accept:

```bash
python3 ideas/godel_loop.py accept S7
```

---

## License

MIT
