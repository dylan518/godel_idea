"""Resolve optional repo-root ``skills/`` (Markdown protocols for idea-tournament / research-ideation).

Works when this package lives inside a larger repo (``…/ideas/`` next to ``skills/``) or when
the Gödel harness is the repository root (``godel_loop.py`` next to ``skills/``). If ``skills/``
is missing, :func:`load_skill_document` returns an empty string.
"""

from __future__ import annotations

from pathlib import Path


def workspace_root(ideas_dir: Path | None = None) -> Path:
    """Directory used as cwd for subprocess tools (e.g. Claude Code).

    If this tree is a subfolder named ``ideas`` (with ``godel_loop.py`` inside it), returns the
    parent directory so paths like ``ideas/systems/…`` resolve. Otherwise returns the package
    directory (standalone clone with ``godel_loop.py`` at repo root).
    """
    d = (
        ideas_dir.resolve()
        if ideas_dir is not None
        else Path(__file__).resolve().parent
    )
    if d.name == "ideas" and (d / "godel_loop.py").exists():
        return d.parent
    return d


def ideas_path_prefix(ideas_dir: Path, workspace: Path | None = None) -> str:
    """Prefix for paths in prompts: ``\"\"`` when the package is the workspace root, else ``\"ideas/\"`` etc."""
    ide = ideas_dir.resolve()
    ws = workspace.resolve() if workspace is not None else workspace_root(ide)
    try:
        sub = ide.relative_to(ws)
    except ValueError:
        return ""
    if sub == Path("."):
        return ""
    return sub.as_posix() + "/"


def repo_root() -> Path:
    """Backward-compatible alias for :func:`workspace_root` (no explicit ``ideas_dir``)."""
    return workspace_root(None)


def skills_dir() -> Path:
    return workspace_root(None) / "skills"


def load_skill_document(relative_under_skills: str) -> str:
    """Load UTF-8 text from ``skills/<relative>``. Returns "" if missing."""
    p = skills_dir() / relative_under_skills.replace("\\", "/")
    if not p.is_file():
        return ""
    return p.read_text(encoding="utf-8", errors="replace")
