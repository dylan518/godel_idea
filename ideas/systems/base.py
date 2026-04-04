"""Abstract base class for idea generator systems."""

from abc import ABC, abstractmethod

DEFAULT_MODEL = "gpt-4.1-mini"
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
    elif model.startswith("gemini-"):
        return None
    else:
        import anthropic
        return anthropic.Anthropic()


def _classify_api_error(e: Exception, model: str) -> str:
    """Return a human-readable label for common API errors to aid debugging."""
    cls = type(e).__name__
    msg = str(e).lower()
    # OpenAI
    if "RateLimitError" in cls:
        return f"RATE_LIMIT ({cls})"
    if "AuthenticationError" in cls:
        return f"AUTH_ERROR ({cls}) — check API key"
    if "InsufficientQuotaError" in cls or "quota" in msg or "insufficient_quota" in msg:
        return f"QUOTA_EXCEEDED ({cls}) — out of credits"
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


def call_llm(prompt: str, model: str, client, temperature: float,
             max_tokens: int = 1024, timeout: int = STEP_TIMEOUT) -> str:
    """Single LLM call supporting OpenAI, Anthropic, and Gemini with per-step timeout."""
    import concurrent.futures

    def _call():
        if model.startswith(("gpt-", "o1-", "o3-", "o4-")):
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
        raise RuntimeError(f"[{label}] {e}") from e
    finally:
        executor.shutdown(wait=False)


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
