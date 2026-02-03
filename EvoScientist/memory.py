"""EvoScientist Memory Middleware.

Automatically extracts and persists long-term memory (user profile, research
preferences, experiment conclusions) from conversations.

Two mechanisms:
1. **Injection** (every LLM call): Reads ``/memory/MEMORY.md`` and appends it
   to the system prompt so the agent always has context.
2. **Extraction** (threshold-triggered): When the conversation exceeds a
   configurable message count, uses an LLM call to pull out structured facts
   and merges them into the appropriate MEMORY.md sections.

## Usage

```python
from EvoScientist.memory import EvoMemoryMiddleware

middleware = EvoMemoryMiddleware(
    backend=my_backend,          # or backend factory
    memory_path="/memory/MEMORY.md",
    extraction_model=chat_model,
    trigger=("messages", 20),
)
agent = create_deep_agent(middleware=[middleware, ...])
```
"""

from __future__ import annotations

import logging
import re
from collections.abc import Awaitable, Callable
from typing import TYPE_CHECKING, Any, cast

from langchain.agents.middleware.types import (
    AgentMiddleware,
    AgentState,
    ModelRequest,
    ModelResponse,
)
from langchain.tools import ToolRuntime
from langchain_core.messages import AnyMessage, HumanMessage, AIMessage
from langchain_core.runnables.config import RunnableConfig
from langgraph.runtime import Runtime

if TYPE_CHECKING:
    from langchain.chat_models import BaseChatModel
    from deepagents.backends.protocol import BACKEND_TYPES, BackendProtocol

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Extraction prompt – sent to a (cheap) LLM to pull structured facts
# ---------------------------------------------------------------------------

EXTRACTION_PROMPT = """\
You are a memory extraction assistant for a scientific experiment agent called EvoScientist.

Analyze the following conversation and extract any NEW information that should be
remembered long-term. Only extract facts that are **not already present** in the
current memory shown below.

<current_memory>
{current_memory}
</current_memory>

<conversation>
{conversation}
</conversation>

Return a JSON object with ONLY the keys that have new information to add.
Omit keys where there is nothing new. Use `null` for unknown values.

```json
{{
  "user_profile": {{
    "name": "string or null",
    "role": "string or null",
    "institution": "string or null",
    "language": "string or null"
  }},
  "research_preferences": {{
    "primary_domain": "string or null",
    "sub_fields": "string or null",
    "preferred_frameworks": "string or null",
    "preferred_models": "string or null",
    "hardware": "string or null",
    "constraints": "string or null"
  }},
  "experiment_conclusion": {{
    "title": "string – experiment name",
    "question": "string – research question",
    "method": "string – method summary",
    "key_result": "string – primary metric/outcome",
    "conclusion": "string – one-line conclusion",
    "artifacts": "string – report path if any"
  }},
  "learned_preferences": [
    "string – each new preference or habit observed"
  ]
}}
```

Rules:
- Only return keys with genuinely new information.
- If nothing new was found, return an empty JSON object: `{{}}`
- Do NOT repeat information already in <current_memory>.
- For experiment_conclusion, only include if a complete experiment was actually run.
- Be concise. Each value should be a short phrase, not a paragraph.
"""

# ---------------------------------------------------------------------------
# System-prompt snippet injected every turn
# ---------------------------------------------------------------------------

MEMORY_INJECTION_TEMPLATE = """<evo_memory>
{memory_content}
</evo_memory>

<memory_instructions>
The above <evo_memory> contains your long-term memory about the user and past experiments.
Use this to personalize your responses and avoid re-asking known information.

**When to update memory:**
- User shares their name, role, institution, or language
- User mentions their research domain, preferred frameworks, models, or hardware
- User explicitly asks you to remember something
- An experiment completes with notable conclusions

**How to update memory:**
- If /memory/MEMORY.md does not exist yet, use `write_file` to create it
- If it already exists, use `edit_file` to update specific sections
- Use this markdown structure:

```markdown
# EvoScientist Memory

## User Profile
- **Name**: ...
- **Role**: ...
- **Institution**: ...
- **Language**: ...

## Research Preferences
- **Primary Domain**: ...
- **Sub-fields**: ...
- **Preferred Frameworks**: ...
- **Preferred Models**: ...
- **Hardware**: ...
- **Constraints**: ...

## Experiment History
### [YYYY-MM-DD] Experiment Title
- **Question**: ...
- **Key Result**: ...
- **Conclusion**: ...

## Learned Preferences
- ...
```

**Priority:** Update memory IMMEDIATELY when the user provides personal or research
information — before composing your main response.
</memory_instructions>"""


# ---------------------------------------------------------------------------
# Helper: merge extracted JSON into MEMORY.md markdown
# ---------------------------------------------------------------------------

def _merge_memory(existing_md: str, extracted: dict[str, Any]) -> str:
    """Merge extracted fields into the existing MEMORY.md content.

    Performs targeted replacements within the known sections.  Unknown
    sections or empty extractions are left untouched.
    """
    if not extracted:
        return existing_md

    result = existing_md

    # --- User Profile ---
    profile = extracted.get("user_profile")
    if profile and isinstance(profile, dict):
        field_map = {
            "name": "Name",
            "role": "Role",
            "institution": "Institution",
            "language": "Language",
        }
        for key, label in field_map.items():
            value = profile.get(key)
            if value and value != "null":
                # Replace the line  "- **Label**: ..." with new value
                pattern = rf"(- \*\*{label}\*\*: ).*"
                replacement = rf"\g<1>{value}"
                result = re.sub(pattern, replacement, result)

    # --- Research Preferences ---
    prefs = extracted.get("research_preferences")
    if prefs and isinstance(prefs, dict):
        field_map = {
            "primary_domain": "Primary Domain",
            "sub_fields": "Sub-fields",
            "preferred_frameworks": "Preferred Frameworks",
            "preferred_models": "Preferred Models",
            "hardware": "Hardware",
            "constraints": "Constraints",
        }
        for key, label in field_map.items():
            value = prefs.get(key)
            if value and value != "null":
                pattern = rf"(- \*\*{label}\*\*: ).*"
                replacement = rf"\g<1>{value}"
                result = re.sub(pattern, replacement, result)

    # --- Experiment History (append) ---
    exp = extracted.get("experiment_conclusion")
    if exp and isinstance(exp, dict) and exp.get("title"):
        from datetime import datetime
        date_str = datetime.now().strftime("%Y-%m-%d")
        entry = f"\n### [{date_str}] {exp.get('title', 'Untitled')}\n"
        entry += f"- **Question**: {exp.get('question', 'N/A')}\n"
        entry += f"- **Method**: {exp.get('method', 'N/A')}\n"
        entry += f"- **Key Result**: {exp.get('key_result', 'N/A')}\n"
        entry += f"- **Conclusion**: {exp.get('conclusion', 'N/A')}\n"
        if exp.get("artifacts"):
            entry += f"- **Artifacts**: {exp['artifacts']}\n"

        # Insert before "## Learned Preferences"
        marker = "## Learned Preferences"
        if marker in result:
            result = result.replace(marker, entry + "\n" + marker)
        else:
            # Fallback: append at end
            result = result.rstrip() + "\n" + entry

    # --- Learned Preferences (append) ---
    learned = extracted.get("learned_preferences")
    if learned and isinstance(learned, list):
        new_items = "\n".join(f"- {item}" for item in learned if item)
        if new_items:
            marker = "## Learned Preferences"
            # Find the section and append after existing items
            if marker in result:
                # Find end of section (next ## or end of file)
                idx = result.index(marker) + len(marker)
                result = result[:idx] + "\n" + new_items + result[idx:]
            else:
                result = result.rstrip() + f"\n\n{marker}\n{new_items}\n"

    return result


# ---------------------------------------------------------------------------
# Middleware
# ---------------------------------------------------------------------------

class EvoMemoryMiddleware(AgentMiddleware):
    """Middleware that injects and auto-extracts long-term memory.

    Args:
        backend: Backend instance or factory for reading/writing memory files.
        memory_path: Virtual path to MEMORY.md (default ``/memory/MEMORY.md``).
        extraction_model: Chat model used for extraction (can be a cheap/fast
            model like ``claude-haiku``). If ``None``, automatic extraction is
            disabled and only prompt injection + manual ``edit_file`` works.
        trigger: When to run automatic extraction.  Supports
            ``("messages", N)`` to trigger every *N* human messages.
            Defaults to ``("messages", 20)``.
    """

    def __init__(
        self,
        *,
        backend: BACKEND_TYPES,
        memory_path: str = "/memory/MEMORY.md",
        extraction_model: BaseChatModel | None = None,
        trigger: tuple[str, int] = ("messages", 20),
    ) -> None:
        self._backend = backend
        self._memory_path = memory_path
        self._extraction_model = extraction_model
        self._trigger = trigger
        self._last_extraction_at: int = 0  # message count at last extraction

    # -- backend resolution --------------------------------------------------

    def _get_backend(
        self,
        state: AgentState[Any],
        runtime: Runtime,
    ) -> BackendProtocol:
        if callable(self._backend):
            config = cast("RunnableConfig", getattr(runtime, "config", {}))
            tool_runtime = ToolRuntime(
                state=state,
                context=runtime.context,
                stream_writer=runtime.stream_writer,
                store=runtime.store,
                config=config,
                tool_call_id=None,
            )
            return self._backend(tool_runtime)
        return self._backend

    # -- read / write helpers ------------------------------------------------

    def _read_memory(self, backend: BackendProtocol) -> str:
        """Read MEMORY.md content (raw bytes → str)."""
        try:
            responses = backend.download_files([self._memory_path])
            if responses and responses[0].content is not None and responses[0].error is None:
                return responses[0].content.decode("utf-8")
        except Exception as e:  # noqa: BLE001
            logger.debug("Failed to read memory at %s: %s", self._memory_path, e)
        return ""

    async def _aread_memory(self, backend: BackendProtocol) -> str:
        try:
            responses = await backend.adownload_files([self._memory_path])
            if responses and responses[0].content is not None and responses[0].error is None:
                return responses[0].content.decode("utf-8")
        except Exception as e:  # noqa: BLE001
            logger.debug("Failed to read memory at %s: %s", self._memory_path, e)
        return ""

    def _write_memory(self, backend: BackendProtocol, old_content: str, new_content: str) -> None:
        """Write updated MEMORY.md (edit if exists, write if new)."""
        try:
            if old_content:
                result = backend.edit(self._memory_path, old_content, new_content)
            else:
                result = backend.write(self._memory_path, new_content)
            if result and result.error:
                logger.warning("Failed to write memory: %s", result.error)
        except Exception as e:  # noqa: BLE001
            logger.warning("Exception writing memory: %s", e)

    async def _awrite_memory(self, backend: BackendProtocol, old_content: str, new_content: str) -> None:
        try:
            if old_content:
                result = await backend.aedit(self._memory_path, old_content, new_content)
            else:
                result = await backend.awrite(self._memory_path, new_content)
            if result and result.error:
                logger.warning("Failed to write memory: %s", result.error)
        except Exception as e:  # noqa: BLE001
            logger.warning("Exception writing memory: %s", e)

    # -- threshold check -----------------------------------------------------

    def _should_extract(self, messages: list[AnyMessage]) -> bool:
        """Check if we should run automatic extraction."""
        if self._extraction_model is None:
            return False

        trigger_type, trigger_value = self._trigger
        if trigger_type == "messages":
            human_count = sum(1 for m in messages if isinstance(m, HumanMessage))
            return (human_count - self._last_extraction_at) >= trigger_value
        return False

    # -- extraction ----------------------------------------------------------

    def _extract(self, model: BaseChatModel, memory: str, messages: list[AnyMessage]) -> dict[str, Any]:
        """Run LLM extraction on recent messages."""
        import json

        # Build conversation string from recent messages (last 30)
        recent = messages[-30:]
        conv_parts = []
        for msg in recent:
            role = "user" if isinstance(msg, HumanMessage) else "assistant"
            content = msg.content if isinstance(msg.content, str) else str(msg.content)
            conv_parts.append(f"[{role}]: {content}")
        conversation = "\n".join(conv_parts)

        prompt = EXTRACTION_PROMPT.format(
            current_memory=memory,
            conversation=conversation,
        )

        try:
            response = model.invoke(prompt)
            text = response.content if isinstance(response.content, str) else str(response.content)
            # Extract JSON from response (may be wrapped in ```json ... ```)
            json_match = re.search(r"```(?:json)?\s*([\s\S]*?)```", text)
            if json_match:
                text = json_match.group(1)
            return json.loads(text.strip())
        except Exception as e:  # noqa: BLE001
            logger.warning("Memory extraction failed: %s", e)
            return {}

    async def _aextract(self, model: BaseChatModel, memory: str, messages: list[AnyMessage]) -> dict[str, Any]:
        import json

        recent = messages[-30:]
        conv_parts = []
        for msg in recent:
            role = "user" if isinstance(msg, HumanMessage) else "assistant"
            content = msg.content if isinstance(msg.content, str) else str(msg.content)
            conv_parts.append(f"[{role}]: {content}")
        conversation = "\n".join(conv_parts)

        prompt = EXTRACTION_PROMPT.format(
            current_memory=memory,
            conversation=conversation,
        )

        try:
            response = await model.ainvoke(prompt)
            text = response.content if isinstance(response.content, str) else str(response.content)
            json_match = re.search(r"```(?:json)?\s*([\s\S]*?)```", text)
            if json_match:
                text = json_match.group(1)
            return json.loads(text.strip())
        except Exception as e:  # noqa: BLE001
            logger.warning("Memory extraction failed: %s", e)
            return {}

    # -- middleware hooks -----------------------------------------------------

    def modify_request(self, request: ModelRequest) -> ModelRequest:
        """Inject memory content and instructions into the system message.

        Always injects ``<memory_instructions>`` so the agent knows it can
        save memories, even when MEMORY.md does not exist yet.
        """
        memory_content = getattr(self, "_current_memory", "")
        # Use placeholder when memory file doesn't exist yet
        if not memory_content:
            memory_content = "(No memory saved yet. Create /memory/MEMORY.md when you learn important information.)"

        from deepagents.middleware._utils import append_to_system_message
        injection = MEMORY_INJECTION_TEMPLATE.format(memory_content=memory_content)
        new_system = append_to_system_message(request.system_message, injection)
        return request.override(system_message=new_system)

    def wrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], ModelResponse],
    ) -> ModelResponse:
        """Inject memory into system prompt before every LLM call."""
        modified = self.modify_request(request)
        return handler(modified)

    async def awrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], Awaitable[ModelResponse]],
    ) -> ModelResponse:
        modified = self.modify_request(request)
        return await handler(modified)

    def before_model(
        self,
        state: AgentState[Any],
        runtime: Runtime,
    ) -> dict[str, Any] | None:
        """Read memory and optionally run extraction before each LLM call."""
        backend = self._get_backend(state, runtime)
        messages = state["messages"]

        # Always read memory for injection
        memory = self._read_memory(backend)
        self._current_memory = memory

        # Check extraction threshold
        if self._should_extract(messages):
            human_count = sum(1 for m in messages if isinstance(m, HumanMessage))
            extracted = self._extract(self._extraction_model, memory, messages)
            if extracted:
                new_memory = _merge_memory(memory, extracted)
                if new_memory != memory:
                    self._write_memory(backend, memory, new_memory)
                    self._current_memory = new_memory
                    logger.info("Auto-extracted and updated memory")
            self._last_extraction_at = human_count

        return None

    async def abefore_model(
        self,
        state: AgentState[Any],
        runtime: Runtime,
    ) -> dict[str, Any] | None:
        """Async: Read memory and optionally run extraction."""
        backend = self._get_backend(state, runtime)
        messages = state["messages"]

        memory = await self._aread_memory(backend)
        self._current_memory = memory

        if self._should_extract(messages):
            human_count = sum(1 for m in messages if isinstance(m, HumanMessage))
            extracted = await self._aextract(self._extraction_model, memory, messages)
            if extracted:
                new_memory = _merge_memory(memory, extracted)
                if new_memory != memory:
                    await self._awrite_memory(backend, memory, new_memory)
                    self._current_memory = new_memory
                    logger.info("Auto-extracted and updated memory")
            self._last_extraction_at = human_count

        return None
