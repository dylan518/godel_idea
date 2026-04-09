"""Resolve repo-root ``skills/`` (Claude Code / EvoScientist skills, not ``EvoScientist/skills``).

The Gödel harness loads markdown protocols from here so there is a single source of truth
alongside what the interactive agent reads (``skills/idea-tournament``, ``skills/research-ideation``).
"""

from __future__ import annotations

from pathlib import Path

# ideas/canonical_skills.py → parents[1] == repo root (EvoScientist/)
_REPO_ROOT = Path(__file__).resolve().parents[1]


def repo_root() -> Path:
    return _REPO_ROOT


def skills_dir() -> Path:
    return _REPO_ROOT / "skills"


def load_skill_document(relative_under_skills: str) -> str:
    """Load UTF-8 text from ``skills/<relative>``. Returns "" if missing."""
    p = skills_dir() / relative_under_skills.replace("\\", "/")
    if not p.is_file():
        return ""
    return p.read_text(encoding="utf-8", errors="replace")
