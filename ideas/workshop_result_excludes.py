"""Filenames under ``results/`` omitted from workshop aggregates (bootstrap CI, κ table)."""

from __future__ import annotations

# Thin protocol / off-spine candidate; keep raw copies in archive if needed.
PRIMARY_COMPARE_EXCLUDE: frozenset[str] = frozenset({"compare_S15_vs_S20.json"})


def skip_compare_path(path_name: str) -> bool:
    return path_name in PRIMARY_COMPARE_EXCLUDE
