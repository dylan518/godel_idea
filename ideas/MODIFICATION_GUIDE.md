# Modification Guide for Idea Generator Systems

## Purpose

This guide defines the constraints and allowed changes for creating new versions
(`S{n+1}.py`) of the idea generator. The goal is structured improvement: each new
version makes one targeted change, enabling rigorous A/B comparison via `godel_loop.py`.

---

## The Interface Contract (Do Not Modify)

All versions must implement `IdeaGenerator` from `systems/base.py`:

```python
class IdeaGenerator(ABC):
    VERSION: str       # e.g. "S3"
    DESCRIPTION: str   # one sentence on what changed vs. prior version

    @abstractmethod
    def get_prompt(self, topic: str) -> str: ...

    def generate_idea(self, topic: str, client, model=DEFAULT_MODEL, temperature=0.8) -> str:
        # Default: single LLM call via get_prompt(); override for multi-step
        ...

GENERATOR = ConcreteGenerator()  # module-level singleton — required
```

`runner.py` imports `GENERATOR` via `importlib`, so the singleton must exist at module level.

---

## What You MAY Change

- **Prompt templates** — the text inside `get_prompt()`
- **Reasoning steps** — add chain-of-thought, step-by-step decomposition
- **Multi-step generation** — override `generate_idea()` to make multiple LLM calls
  (e.g., draft → critique → revise)
- **Self-critique loops** — ask the model to evaluate and improve its own output
- **Sampling parameters** — temperature, max_tokens passed to `generate_idea()`
- **Helper prompts** — add private methods (prefixed `_`) on the class
- **System messages** — use system-role messages in multi-turn variants

## What You MAY Add

- New methods on the class (private `_methods` or additional public helpers)
- Additional refinement calls within `generate_idea()`
- Structured output parsing (extract the best idea from multiple candidates)

## What You MAY NOT Change

- The `IdeaGenerator` abstract interface in `base.py`
- `runner.py`, `judge.py`, or `godel_loop.py`
- Other existing version files (S0.py, S1.py, S2.py, ...)
- The `benchmark_topics.json` or `CURRENT_VERSION` file (use `accept` subcommand)
- Cross-imports between version files (each version is self-contained)

---

## How to Create S{n+1}

1. Copy the current version file:
   ```bash
   cp ideas/systems/S2.py ideas/systems/S3.py
   ```

2. Update the class name (e.g., `S2Generator` → `S3Generator`)

3. Update `VERSION = "S3"` and write a clear `DESCRIPTION` of the single change made

4. Make **one targeted improvement** — keep the diff minimal and focused

5. Update the module-level singleton:
   ```python
   GENERATOR = S3Generator()
   ```

6. Test it loads:
   ```bash
   python -c "import importlib.util, sys; sys.path.insert(0,'ideas'); \
       spec = importlib.util.spec_from_file_location('S3','ideas/systems/S3.py'); \
       m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); \
       print(m.GENERATOR.VERSION)"
   ```

7. Run comparison:
   ```bash
   python ideas/godel_loop.py compare --candidate S3
   ```

8. If win rate > 60%, accept:
   ```bash
   python ideas/godel_loop.py accept S3
   ```

---

## Improvement Ideas (Ordered by Expected Impact)

| Idea | Strategy |
|------|----------|
| S3 | Add a self-critique step: generate idea, then ask model to identify weaknesses and revise |
| S4 | Generate 3 candidates, ask model to select and synthesize the best elements |
| S5 | Add domain-specific framing: prepend field-specific context before the main prompt |
| S6 | Use structured output: ask for explicit Hypothesis / Method / Expected Result sections |
| S7 | Add analogical reasoning: prompt model to find analogies from other fields first |
| S8 | Chain-of-thought decomposition: break problem into subproblems before ideating |

---

## Evaluation Criteria

Ideas are judged on four axes (0–10 each):
- **Novelty** — genuinely new angle, not obvious incremental work
- **Scientific usefulness** — addresses real open problems, actionable by researchers
- **Experimental clarity** — clear methodology, testable predictions
- **Feasibility** — achievable with current or near-future resources

Focus improvements on whichever axis the current version scores weakest.
