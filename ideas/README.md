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
│     │  writes new S{n}.py via SWE agent             │
│     ▼                                              │
│  Generator S{n} ──generates──► Ideas               │
│     │                             │                │
│     │              Primary judge (DeepSeek V3)      │
│     │              Blind judge (Gemini Flash)        │
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
| Generator | `deepseek-chat` (DeepSeek V3) | Cheap, fast idea generation |
| Meta-LLM / SWE agent | `claude-sonnet-4-6` | Writes new S{n}.py strategy files |
| Primary judge | `deepseek-chat` (DeepSeek V3) | Accept/reject decisions |
| Blind judge | `gemini-flash-lite-latest` | Independent canary (never used for accept/reject) |

**Current champion**: `S15` — Hypothesis-First Adversarial Loop (55.3% vs S12)

---

## Evolution History

Loop started fresh from `S_sota` as the bootstrap champion (prior S0–S6 experiments archived).

| Version | Strategy | Win Rate vs Prev Champion | Status |
|---------|----------|--------------------------|--------|
| S_sota | SOTA retrieval + IdeaTreeSearch + Elo tournament | — | bootstrap champion |
| S12 | Hypothesis-First Adversarial Loop (v1) | **89.3%** vs S_sota | champion |
| S13 | SWE refinement of S12 (round 1) | 54% vs S12 | rejected |
| S14 | SWE refinement of S12 (round 2) | 54% vs S12 | rejected |
| S15 | Hypothesis-First Adversarial Loop (SWE-refined) | **55.3%** vs S12 | **champion** |

**Key finding**: the judge rewards *specificity and measurability* as much as novelty. Vague ambitious ideas consistently lose to concrete, actionable ones. Starting from falsifiable hypotheses rather than technique names produces significantly stronger ideas.

---

## Idea Quality Criteria

Each idea is scored 0–10 on four dimensions:

1. **Novelty** — genuinely new angle, not incremental
2. **Scientific usefulness** — addresses real open problems
3. **Experimental clarity** — concrete methodology, testable predictions
4. **Feasibility** — achievable with current or near-future resources

---

## Current Champion: S15 Pipeline (~9 LLM calls per idea)

```
Topic
  │
  ▼
Hypothesis Generation (1 call)
  5 sharp, falsifiable scientific hypotheses
  │
  ▼
Adversarial Attack (5 parallel calls)
  Each hypothesis attacked: assumption violations,
  dataset biases, theoretical gaps → refined variant
  │
  ▼
Hypothesis Selection (1 call)
  Picks strongest surviving hypothesis
  │
  ▼
Idea Construction (1 call)
  Full experimental design around proving/disproving
  the hypothesis — forces concrete datasets + metrics
  │
  ▼
Multi-Perspective Critique (4 calls)
  Experimentalist / Theorist / Skeptic / Synthesis
  │
  ▼
IDEA_FORMAT output
```

---

## SWE Agent

The SWE agent autonomously improves the current champion by making surgical edits to the generator code:

1. **Analyze** — inspects recent losing verdicts to identify failure patterns
2. **Propose** — drafts a targeted code edit to the generator
3. **Mini-eval** — tests the edit on 3 topics × 3 ideas (9 pairs); accepts if win_rate > 52%
4. **Commit or revert** — accepted edits become permanent; rejected edits are discarded

Stopping criteria: max 6 rounds, 3 consecutive failures, or 2-round stall with no improvement.

---

## Paper Retrieval (OpenAlex)

SOTA paper context is fetched via the **OpenAlex API** (switched from Semantic Scholar due to rate limiting):

- **Semantic search**: relevance-ranked via ML + citation networks (not boolean keyword matching)
- **Abstract reconstruction**: OpenAlex stores abstracts as inverted index; reconstructed automatically
- **Filters**: `from_publication_date:2019-01-01`, `has_abstract:true`
- **Cache**: 7-day disk cache at `results/retrieval_cache/`
- **Rate limit**: 0.15s between requests (vs 3s for Semantic Scholar)

---

## Goodhart Alert & Confusion Matrix

A Goodhart alert fires when primary judge (DeepSeek) and blind judge (Gemini) diverge significantly. The system tracks a **per-pair confusion matrix** across every comparison run:

| | Gemini says B wins | Gemini says A wins |
|---|---|---|
| **DeepSeek says B wins** | agreement | flip |
| **DeepSeek says A wins** | flip | agreement |

- Alert threshold: `flip_rate > 30%`
- Blind judge runs on 3 randomly sampled topics (not all 15) to save cost
- When alert fires, candidate is rejected even if `win_rate > 55%`
- **S3 example**: DeepSeek preferred S3 (54%) but Gemini preferred S1 (100%) — classic Goodhart failure

---

## Setup

```bash
git clone https://github.com/YOUR_USERNAME/evoscientist.git
cd evoscientist

pip install openai anthropic google-genai
```

Create a `.env` file:

```
OPENAI_API_KEY=sk-...          # used for gpt-* models (optional)
ANTHROPIC_API_KEY=sk-ant-...   # Claude Sonnet (SWE meta-agent)
GEMINI_API_KEY=AIza...         # Gemini blind judge
DEEPSEEK_API_KEY=sk-...        # Primary judge + generator
OPENALEX_API_KEY=...           # optional — increases rate limit for paper retrieval
```

---

## Usage

```bash
# Check current state
python3 ideas/godel_loop.py status

# Run autonomous evolution loop (generates → compares → accepts → repeat)
python3 ideas/godel_loop.py swe-evolve --target 20 --n-ideas 5 --workers 3 --swe-rounds 6

# Manually compare a candidate against champion
python3 ideas/godel_loop.py compare --candidate S16 --n-ideas 5 --workers 3

# Accept a candidate as new champion
python3 ideas/godel_loop.py accept S16

# Watch live logs
tail -f ideas/results/godel.log
```

---

## Repository Structure

```
ideas/
├── godel_loop.py          # Main CLI entry point
├── runner.py              # Parallel benchmarking engine
├── judge.py               # Pairwise evaluation engine (DeepSeek primary, Gemini blind)
├── swe_agent.py           # SWE agent: writes and tests new S{n}.py edits
├── meta.py                # Meta-generator: prompts claude-sonnet to write new systems
├── retrieval.py           # OpenAlex API with 7-day disk cache
├── tournament.py          # Swiss-system Elo tournament (K=32, start=1500)
├── log.py                 # Logging + dotenv loader
├── benchmark_topics.json  # 15 research topic categories
├── CURRENT_VERSION        # Points to current champion (e.g. "S15")
├── RESEARCH_AGENDA.md     # Living document: hypotheses, experiment log
├── systems/
│   ├── base.py            # Abstract IdeaGenerator + call_llm (OpenAI/Anthropic/Gemini/DeepSeek)
│   ├── S0.py              # Baseline direct prompting
│   ├── S1.py              # Self-critique
│   ├── S5.py              # Multi-perspective critique
│   ├── S_sota.py          # SOTA retrieval + IdeaTreeSearch + Elo tournament
│   ├── S12.py             # Hypothesis-First Adversarial Loop (v1)
│   └── S15.py             # Current champion (SWE-refined adversarial loop)
└── results/
    ├── godel.log          # Full debug log
    ├── evolution_log.jsonl # History of accepted versions
    ├── compare_*.json     # Per-comparison detailed reports
    ├── retrieval_cache/   # OpenAlex paper cache (7-day TTL)
    └── S*/ideas.json      # Cached idea outputs per version
```

---

## Adding a New Strategy

1. Create `ideas/systems/S{n}.py` implementing `IdeaGenerator`:

```python
from systems.base import IdeaGenerator, IDEA_FORMAT, call_llm

class S16Generator(IdeaGenerator):
    VERSION = "S16"
    DESCRIPTION = "Your strategy description"

    def generate_idea(self, topic, client, model, temperature=0.8):
        # Your multi-step logic here
        ...
```

2. Compare against champion:

```bash
python3 ideas/godel_loop.py compare --candidate S16 --n-ideas 5 --workers 3
```

3. If win_rate > 55% with no Goodhart alert, accept:

```bash
python3 ideas/godel_loop.py accept S16
```

---

## License

MIT
