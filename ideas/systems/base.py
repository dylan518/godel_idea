"""Abstract base class for idea generator systems."""

from abc import ABC, abstractmethod

DEFAULT_MODEL = "deepseek-chat"
STEP_TIMEOUT = 90  # seconds per LLM call before raising TimeoutError

# All systems must produce ideas in this format.
# Append to every final-step prompt to ensure consistency across S0, S1, S2, ...
IDEA_FORMAT = """
Present your final research idea in exactly this format:

IDEA: [One sentence summary of the core proposal]

BACKGROUND: [The specific open problem this addresses, 2-3 sentences]

APPROACH: [The proposed method or technique, 2-3 sentences]

EXPERIMENT: [How to test it — datasets, metrics, or methodology, 1-2 sentences]

NOVELTY: [What makes this meaningfully different from prior work, 1-2 sentences]"""


def make_client(model: str):
    """Instantiate the right API client based on model name prefix.
    Returns None for Gemini (client created inline in call_llm)."""
    if model.startswith(("gpt-", "o1-", "o3-", "o4-")):
        from openai import OpenAI
        return OpenAI()
    elif model.startswith("deepseek-"):
        import os
        from openai import OpenAI
        return OpenAI(api_key=os.environ["DEEPSEEK_API_KEY"],
                      base_url="https://api.deepseek.com")
    elif model.startswith("gemini-"):
        return None
    else:
        import anthropic
        return anthropic.Anthropic()


def _classify_api_error(e: Exception, model: str) -> str:
    """Return a human-readable label for common API errors to aid debugging."""
    cls = type(e).__name__
    msg = str(e).lower()
    # Check quota/billing before generic RateLimitError — OAI uses RateLimitError for both
    if "InsufficientQuotaError" in cls or "insufficient_quota" in msg or "quota" in msg:
        return f"QUOTA_EXCEEDED ({cls}) — out of credits"
    # OpenAI / Anthropic rate limiting (transient — back off and retry)
    if "RateLimitError" in cls or "rate_limit" in msg or "rate limit" in msg:
        return f"RATE_LIMIT ({cls})"
    if "AuthenticationError" in cls:
        return f"AUTH_ERROR ({cls}) — check API key"
    if "APIStatusError" in cls or "APIError" in cls:
        status = getattr(e, "status_code", "?")
        return f"API_ERROR/{status} ({cls})"
    # Anthropic
    if "OverloadedError" in cls:
        return f"OVERLOADED ({cls})"
    # Generic
    if "connection" in msg or "ConnectError" in cls or "ConnectionError" in cls:
        return f"CONNECTION_ERROR ({cls})"
    return f"{cls}"


RETRY_DELAYS = [5, 15, 30]              # seconds between retries: connection/overload
RATE_LIMIT_DELAYS = [60, 120, 300]      # seconds between retries: rate limits (longer)
RETRYABLE = ("CONNECTION_ERROR", "OVERLOADED", "RATE_LIMIT")  # QUOTA_EXCEEDED not retried


def call_llm(prompt: str, model: str, client, temperature: float,
             max_tokens: int = 1024, timeout: int = STEP_TIMEOUT) -> str:
    """Single LLM call supporting OpenAI, Anthropic, and Gemini with per-step timeout."""
    import concurrent.futures
    import time as _time
    import logging as _logging
    _logger = _logging.getLogger("base")

    def _call():
        if model.startswith(("gpt-", "o1-", "o3-", "o4-", "deepseek-")):
            response = client.chat.completions.create(
                model=model,
                max_tokens=max_tokens,
                temperature=temperature,
                messages=[{"role": "user", "content": prompt}],
                timeout=timeout,
            )
            return response.choices[0].message.content.strip()
        elif model.startswith("gemini-"):
            import os
            from google import genai
            from google.genai import types
            gemini = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
            response = gemini.models.generate_content(
                model=model,
                contents=prompt,
                config=types.GenerateContentConfig(
                    temperature=temperature,
                    max_output_tokens=max_tokens,
                ),
            )
            return response.text.strip()
        else:
            response = client.messages.create(
                model=model,
                max_tokens=max_tokens,
                temperature=temperature,
                messages=[{"role": "user", "content": prompt}],
                timeout=timeout,
            )
            return response.content[0].text.strip()

    last_err = None
    # Build attempt schedule: first attempt (no pre-sleep) + retries with delays
    # We track retry index separately so we can pick the right delay list per error type
    max_retries = max(len(RETRY_DELAYS), len(RATE_LIMIT_DELAYS))
    for attempt in range(max_retries + 1):
        executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
        future = executor.submit(_call)
        try:
            return future.result(timeout=timeout)
        except concurrent.futures.TimeoutError:
            future.cancel()
            executor.shutdown(wait=False)
            raise TimeoutError(f"LLM call exceeded {timeout}s timeout (model={model})")
        except Exception as e:
            executor.shutdown(wait=False)
            label = _classify_api_error(e, model)
            err = RuntimeError(f"[{label}] {e}")
            last_err = err

            if not any(label.startswith(r) for r in RETRYABLE):
                raise err from e  # non-retryable — fail immediately

            # Pick delay schedule based on error type
            if label.startswith("RATE_LIMIT"):
                delays = RATE_LIMIT_DELAYS
            else:
                delays = RETRY_DELAYS

            if attempt >= len(delays):
                raise err from e  # exhausted retries

            wait = delays[attempt]
            _logger.warning("API error [%s] — pausing %ds before retry %d/%d (model=%s)",
                            label, wait, attempt + 1, len(delays), model)
            _time.sleep(wait)
        finally:
            executor.shutdown(wait=False)

    raise last_err
    raise last_err  # unreachable but satisfies type checker


BATCH_PROMPT_TEMPLATE = """\
Research topic: {topic}

Generate {n} distinct, concrete research ideas on this topic.
Number each idea 1 through {n}.

For each idea use this exact format:

## Idea {{i}}
IDEA: [One sentence summary of the core proposal]
BACKGROUND: [The specific open problem this addresses, 2-3 sentences]
APPROACH: [The proposed method or technique, 2-3 sentences]
EXPERIMENT: [How to test it — datasets, metrics, or methodology, 1-2 sentences]
NOVELTY: [What makes this meaningfully different from prior work, 1-2 sentences]
---
"""


def _parse_batch_ideas(text: str, n: int) -> list[str]:
    """Split a batched LLM response into n individual idea strings."""
    import re
    # Split on ## Idea N headers
    parts = re.split(r'(?:^|\n)##\s*Idea\s+\d+\s*\n', text)
    ideas = [p.strip() for p in parts if p.strip()]
    if len(ideas) >= n:
        return ideas[:n]
    # Fallback: split on --- separator
    ideas = [p.strip() for p in re.split(r'\n---+\n', text) if p.strip()]
    if len(ideas) >= n:
        return ideas[:n]
    # Last resort: pad
    while len(ideas) < n:
        ideas.append(ideas[0] if ideas else "ERROR: batch generation failed")
    return ideas[:n]


class IdeaGenerator(ABC):
    """Abstract interface that all idea generator versions must implement."""

    VERSION: str
    DESCRIPTION: str

    @abstractmethod
    def get_prompt(self, topic: str) -> str:
        """Return the LLM prompt for generating an idea on the given topic.
        Should include IDEA_FORMAT at the end to enforce consistent output."""
        ...

    def generate_idea(
        self,
        topic: str,
        client,
        model: str = DEFAULT_MODEL,
        temperature: float = 0.8,
    ) -> str:
        """Generate a single research idea for the given topic.

        Default implementation: one LLM call using get_prompt().
        Override this method for multi-step variants (draft → critique → revise, etc.).
        """
        return call_llm(self.get_prompt(topic), model, client, temperature)

    def generate_batch(
        self,
        topic: str,
        client,
        model: str = DEFAULT_MODEL,
        n: int = 5,
        temperature: float = 0.9,
    ) -> list[str]:
        """Generate n ideas in a SINGLE LLM call.

        Used for fresh evaluation runs (judge comparison) to prevent overfitting
        to cached outputs. One call per topic regardless of n.

        Override this in systems that can produce richer batches (e.g. with
        retrieval context), but keep it to a single LLM call.
        """
        prompt = BATCH_PROMPT_TEMPLATE.format(topic=topic, n=n)
        response = call_llm(prompt, model, client, temperature,
                            max_tokens=n * 500, timeout=STEP_TIMEOUT * 2)
        return _parse_batch_ideas(response, n)
