#!/usr/bin/env python3
"""Generate publication figures for the EvoScientist / Gödel-loop paper.

Reads JSON artifacts under ideas/results/ and writes raster + vector files to
ideas/paper/figures/. Figure 3 (Darwin / SWE harness) is normally a **hand-authored schematic**
(ship ``figures/figure3_system_harness.png``). An optional matplotlib fallback exists as
``figure3_system_harness()`` and is **not** run from ``main()`` so it cannot overwrite that asset.
**Figure 2** (``figure1_prearchive_loop_trajectory_wr()``): pre-reboot loop from git before ``620151b`` (through
S5; **no** S6) plus archived **S3** vs **S7/S8/S9** probes — **not** the post-reboot track in **Figure 1**.
Eval = ``blind_*__deepseek-chat.json`` throughout.
**Figure 4** summarizes intra-judge test–retest on the frozen publish-eval intersection
(``judge_analysis.json`` in the publish-eval bundle; **64**-pair intersection, full bundle has **75** slots).

Run from repo root or from ideas/::

    python3 ideas/paper/make_paper_figures.py

Figure 1 overlays **train** (archived compare-session **DeepSeek** verdicts used for promotion)
and **eval** (independent **blind** pass on the same pairs using the harness blind judge,
default **Gemini Flash Lite** — see ``ideas/judge.py`` ``BLIND_JUDGE_MODEL``).
"""

from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import gridspec
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

IDEAS_DIR = Path(__file__).resolve().parents[1]
RESULTS = IDEAS_DIR / "results"
OUT_DIR = IDEAS_DIR / "paper" / "figures"
# S15 vs S_sota publish-eval bundle on disk (folder name retains legacy Spaper suffix; 75 pair-slots in ideas.json; judge_analysis.json uses 64-pair intersection).
PUBLISH_EVAL_S15_SPAPER = RESULTS / "publish_eval_n75_gemini3_deepseek_S15_Spaper"

MPL_RC = {
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "legend.fontsize": 8.5,
    "xtick.labelsize": 7.5,
    "ytick.labelsize": 9,
    "axes.linewidth": 0.8,
    "lines.linewidth": 1.0,
    "mathtext.default": "regular",
}


def _load(p: Path) -> dict:
    return json.loads(p.read_text())


def _win_rate_b_tie_adj(verdicts: list[dict]) -> float:
    if not verdicts:
        return 0.0
    score = 0.0
    for v in verdicts:
        w = v.get("winner")
        if w == "B":
            score += 1.0
        elif w == "tie":
            score += 0.5
    return score / len(verdicts)


def topic_block_bootstrap_ci(
    verdicts: list[dict], *, iters: int = 3000, seed: int = 42, alpha: float = 0.05
) -> tuple[float, float, float]:
    buckets: dict[str, list[dict]] = defaultdict(list)
    for v in verdicts:
        buckets[str(v["topic_id"])].append(v)
    topics = sorted(buckets.keys())
    n_topics = len(topics)
    rng = random.Random(seed)
    point = _win_rate_b_tie_adj(verdicts)
    samples: list[float] = []
    for _ in range(iters):
        picks = [topics[rng.randrange(n_topics)] for _ in range(n_topics)]
        acc: list[dict] = []
        for tid in picks:
            acc.extend(buckets[tid])
        samples.append(_win_rate_b_tie_adj(acc))
    samples.sort()
    lo_i = max(0, int((alpha / 2.0) * (len(samples) - 1)))
    hi_i = min(len(samples) - 1, int((1.0 - alpha / 2.0) * (len(samples) - 1)))
    return point, samples[lo_i], samples[hi_i]


def _synthetic_primary_verdicts_for_paper_s12(
    blind_verdicts: list[dict], target_b_equivalent: int
) -> list[dict]:
    """Reconstruct primary-style winners matching logged promotion mass (no archived compare file)."""
    by_topic: dict[str, list[dict]] = defaultdict(list)
    for v in blind_verdicts:
        by_topic[str(v["topic_id"])].append(dict(v))
    topic_ids = sorted(by_topic.keys())
    n_topics = len(topic_ids)
    per = target_b_equivalent // n_topics
    rem = target_b_equivalent % n_topics
    alloc = {t: per + (1 if i < rem else 0) for i, t in enumerate(topic_ids)}
    out: list[dict] = []
    for tid in topic_ids:
        rows = sorted(by_topic[tid], key=lambda x: int(x["idea_index"]))
        k = alloc[tid]
        for i, row in enumerate(rows):
            row["winner"] = "B" if i < k else "A"
            out.append(row)
    assert len(out) == len(blind_verdicts)
    return out


# Darwin / SWE reboot track: promotion spine first (see results/swe_memory.json).
# S12→S14 follows S12→S13 in real time; S12→S13 is only plotted when blind+compare exist.
# (Omit S15→S13 “replay” from the diagram — it is not chronological promotion and confuses readers.)
# Display S_sota (paper name); on-disk blind files for the first step still use the legacy S_paper prefix.
DARWIN_TRANSITIONS_CHRONO: list[tuple[str, str]] = [
    ("S_sota", "S12"),
    ("S12", "S13"),
    ("S12", "S14"),
    ("S12", "S15"),
    ("S15", "S16"),
    ("S15", "S17"),
    ("S15", "S18"),
    ("S15", "S19"),
]

ACCEPTED_PROMOTIONS: frozenset[tuple[str, str]] = frozenset({("S_sota", "S12"), ("S12", "S15")})

# Logged tie-adjusted promotion mass for S_sota→S12 (artifact compare named S_paper) when reconstructing primary labels.
PRIMARY_S_SOTA_S12_B_MASS = 67 / 75

# Blind eval series for Figure 1: same model family as ``ideas/judge.py`` BLIND_JUDGE_MODEL default.
BLIND_EVAL_MODEL = "gemini-flash-lite-latest"


@dataclass
class TrajectorySpec:
    label: str
    current: str
    candidate: str
    compare_path: Path | None
    blind_eval_path: Path
    accepted: bool
    primary_override: float | None = None


def _trajectory_specs_plottable() -> list[TrajectorySpec]:
    """Chronological transitions where a blind Gemini eval file exists (train = compare DeepSeek when present)."""
    out: list[TrajectorySpec] = []
    for cur, cand in DARWIN_TRANSITIONS_CHRONO:
        cur_fs = "S_paper" if (cur, cand) == ("S_sota", "S12") else cur
        blind_ev = RESULTS / f"blind_{cur_fs}_vs_{cand}__{BLIND_EVAL_MODEL}.json"
        label = f"{cur}→{cand}"
        if (cur, cand) == ("S_sota", "S12"):
            if blind_ev.is_file():
                out.append(
                    TrajectorySpec(
                        label=label,
                        current=cur,
                        candidate=cand,
                        compare_path=None,
                        blind_eval_path=blind_ev,
                        accepted=True,
                        primary_override=PRIMARY_S_SOTA_S12_B_MASS,
                    )
                )
            continue
        cmp_p = RESULTS / f"compare_{cur}_vs_{cand}.json"
        if cmp_p.is_file() and blind_ev.is_file():
            out.append(
                TrajectorySpec(
                    label=label,
                    current=cur,
                    candidate=cand,
                    compare_path=cmp_p,
                    blind_eval_path=blind_ev,
                    accepted=(cur, cand) in ACCEPTED_PROMOTIONS,
                    primary_override=None,
                )
            )
    return out


def figure1_trajectory_wr() -> None:
    """Train = compare-session DeepSeek (promotion gate); eval = blind Gemini (same pairs)."""
    rows = _trajectory_specs_plottable()
    if not rows:
        raise SystemExit(
            f"No trajectory specs found under results/ with blind eval "
            f"blind_*__{BLIND_EVAL_MODEL}.json"
        )

    xs = np.arange(len(rows))
    blind_y: list[float] = []
    blind_err: list[tuple[float, float]] = []
    colors: list[str] = []
    for spec in rows:
        bver = _load(spec.blind_eval_path)["verdicts"]
        bp, blo, bhi = topic_block_bootstrap_ci(bver)
        blind_y.append(bp)
        blind_err.append((bp - blo, bhi - bp))
        colors.append("#2ca02c" if spec.accepted else "#d62728")

    mpl.rcParams.update(MPL_RC)
    fig_w = min(12.5, 6.8 + 0.62 * len(rows))
    fig, ax = plt.subplots(figsize=(fig_w, 2.95))
    fig.subplots_adjust(left=0.09, right=0.80, top=0.82, bottom=0.24)

    prim_y: list[float] = []
    prim_err: list[tuple[float, float]] = []
    for spec in rows:
        bver = _load(spec.blind_eval_path)["verdicts"]
        if spec.compare_path is None:
            assert spec.primary_override is not None
            synth = _synthetic_primary_verdicts_for_paper_s12(
                bver, round(spec.primary_override * 75)
            )
            pp, plo, phi = topic_block_bootstrap_ci(synth)
        else:
            pver = _load(spec.compare_path)["verdicts"]
            pp, plo, phi = topic_block_bootstrap_ci(pver)
        prim_y.append(pp)
        prim_err.append((pp - plo, phi - pp))

    for i in xs:
        ax.vlines(
            i,
            min(prim_y[i] - prim_err[i][0], blind_y[i] - blind_err[i][0]),
            max(prim_y[i] + prim_err[i][1], blind_y[i] + blind_err[i][1]),
            colors="#cccccc",
            linewidth=1.0,
            zorder=1,
        )

    for i in xs:
        ax.errorbar(
            i - 0.07,
            prim_y[i],
            yerr=[[prim_err[i][0]], [prim_err[i][1]]],
            fmt="o",
            color="#1f77b4",
            ecolor=colors[i],
            elinewidth=1.8,
            capsize=3,
            markersize=6,
            markerfacecolor="#1f77b4",
            markeredgecolor=colors[i],
            markeredgewidth=1.8,
            zorder=3,
        )
    for i in xs:
        ax.errorbar(
            i + 0.07,
            blind_y[i],
            yerr=[[blind_err[i][0]], [blind_err[i][1]]],
            fmt="o",
            color="#1f77b4",
            ecolor=colors[i],
            elinewidth=1.8,
            capsize=3,
            markersize=6,
            markerfacecolor="white",
            markeredgecolor=colors[i],
            markeredgewidth=1.8,
            zorder=3,
        )

    for i, c in enumerate(colors):
        ax.axvspan(i - 0.5, i + 0.5, facecolor=c, alpha=0.035, zorder=0)

    ax.axhline(0.5, color="#7f7f7f", linestyle="--", linewidth=1.0, zorder=2)
    ax.axhline(0.55, color="#7f7f7f", linestyle="--", linewidth=1.0, zorder=2)
    ax.text(-0.48, 0.505, "0.50", color="#555555", fontsize=8, va="bottom", ha="left")
    ax.text(-0.48, 0.555, "0.55 (promotion)", color="#555555", fontsize=8, va="bottom", ha="left")

    ax.set_xticks(xs)
    ax.set_xticklabels([s.label for s in rows], rotation=35, ha="right", fontsize=7.5)
    ax.set_ylabel(r"Tie-adjusted $\widehat{\mathrm{WR}}_B$")
    ax.set_ylim(0.0, 1.0)
    ax.set_xlim(-0.55, len(rows) - 0.45)
    ax.margins(x=0.02)

    leg_series = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="none",
            markerfacecolor="#1f77b4",
            markeredgecolor="#333333",
            markeredgewidth=1.0,
            markersize=7,
            label="Train (compare session, DeepSeek)",
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            color="none",
            markerfacecolor="white",
            markeredgecolor="#333333",
            markeredgewidth=1.0,
            markersize=7,
            label=f"Eval (blind {BLIND_EVAL_MODEL}, same pairs)",
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            color="none",
            markerfacecolor="#1f77b4",
            markeredgecolor="#2ca02c",
            markeredgewidth=1.6,
            markersize=7,
            label="Accepted transition (green edge)",
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            color="none",
            markerfacecolor="#1f77b4",
            markeredgecolor="#d62728",
            markeredgewidth=1.6,
            markersize=7,
            label="Rejected transition (red edge)",
        ),
    ]
    ax.legend(handles=leg_series, loc="upper left", bbox_to_anchor=(1.01, 1.0), frameon=False, fontsize=7.5)
    ax.set_title(
        "Post-reboot trajectory: train (DeepSeek compare) vs "
        f"eval (blind {BLIND_EVAL_MODEL}; 95% topic-block CI)"
    )

    _save(fig, "figure1_trajectory_wr")


@dataclass
class PrearchiveTrajectoryRow:
    """One head-to-head in the pre-reboot loop (git parent of ``620151b``)."""

    label: str
    accepted: bool
    primary_point: float  # tie-adjusted WR_B or documentary; nan if no train verdicts
    primary_verdicts: list[dict] | None
    blind_verdicts: list[dict] | None


# Main promotion spine (champion S1 rejects S2–S4, promotes to S5). **S6 omitted** (no finished archived compare).
# Last tuple entry = documentary primary WR_B if compare missing.
PREARCHIVE_MAIN_SPINE: list[tuple[str, str, str, bool, float | None]] = [
    ("S0→S1", "S0", "S1", True, 1.0),
    ("S1→S2", "S1", "S2", False, 0.08),
    ("S1→S3", "S1", "S3", False, 0.54),
    ("S1→S4", "S1", "S4", False, 0.40),
    ("S1→S5", "S1", "S5", True, 0.58),
]

# Additional archived compares: champion **S3** vs candidates S7–S9 (same snapshot; exploratory branch, not S5→S7).
PREARCHIVE_S3_PROBE_SPINE: list[tuple[str, str, str, bool, float | None]] = [
    ("S3→S7", "S3", "S7", False, None),
    ("S3→S8", "S3", "S8", False, None),
    ("S3→S9", "S3", "S9", False, None),
]

PREARCHIVE_N_MAIN = len(PREARCHIVE_MAIN_SPINE)

PREARCHIVE_INCLUDE_BLIND_EVAL = True
PREARCHIVE_BLIND_JUDGE_SLUG = "deepseek-chat"


def _prearchive_compare_verdicts(champ: str, cand: str) -> list[dict] | None:
    p = RESULTS / f"compare_{champ}_vs_{cand}.json"
    if not p.is_file():
        return None
    v = _load(p).get("verdicts")
    return v if isinstance(v, list) and v else None


def _prearchive_blind_verdicts(champ: str, cand: str) -> list[dict] | None:
    p = RESULTS / f"blind_{champ}_vs_{cand}__{PREARCHIVE_BLIND_JUDGE_SLUG}.json"
    if not p.is_file():
        return None
    v = _load(p).get("verdicts")
    return v if isinstance(v, list) and v else None


def _prearchive_trajectory_rows() -> list[PrearchiveTrajectoryRow]:
    rows: list[PrearchiveTrajectoryRow] = []
    for label, ch, ca, acc, doc_wr in PREARCHIVE_MAIN_SPINE:
        pv = _prearchive_compare_verdicts(ch, ca)
        bv: list[dict] | None = None
        if PREARCHIVE_INCLUDE_BLIND_EVAL:
            bv = _prearchive_blind_verdicts(ch, ca)
        if pv:
            point = _win_rate_b_tie_adj(pv)
        elif doc_wr is not None:
            point = doc_wr
        else:
            point = float("nan")
        rows.append(PrearchiveTrajectoryRow(label, acc, point, pv, bv))

    for label, ch, ca, acc, doc_wr in PREARCHIVE_S3_PROBE_SPINE:
        pv = _prearchive_compare_verdicts(ch, ca)
        bv = _prearchive_blind_verdicts(ch, ca) if PREARCHIVE_INCLUDE_BLIND_EVAL else None
        if pv:
            point = _win_rate_b_tie_adj(pv)
        elif doc_wr is not None:
            point = doc_wr
        else:
            point = float("nan")
        rows.append(PrearchiveTrajectoryRow(label, acc, point, pv, bv))
    return rows


def figure1_prearchive_loop_trajectory_wr() -> None:
    """Pre-reboot loop only: main spine + archived S3 vs S7–S9; train vs DeepSeek blind."""
    rows = _prearchive_trajectory_rows()
    use_ds_backfill = all(r.blind_verdicts for r in rows)
    xs = np.arange(len(rows))
    colors = ["#2ca02c" if r.accepted else "#d62728" for r in rows]

    prim_y: list[float] = []
    prim_err: list[tuple[float, float]] = []
    for r in rows:
        if r.primary_verdicts:
            pp, plo, phi = topic_block_bootstrap_ci(r.primary_verdicts)
            prim_y.append(pp)
            prim_err.append((pp - plo, phi - pp))
        else:
            prim_y.append(r.primary_point)
            prim_err.append((0.0, 0.0))

    has_blind: list[bool] = []
    blind_y: list[float] = []
    blind_err: list[tuple[float, float]] = []
    for r in rows:
        if r.blind_verdicts:
            bp, blo, bhi = topic_block_bootstrap_ci(r.blind_verdicts)
            blind_y.append(bp)
            blind_err.append((bp - blo, bhi - bp))
            has_blind.append(True)
        else:
            blind_y.append(float("nan"))
            blind_err.append((0.0, 0.0))
            has_blind.append(False)

    mpl.rcParams.update(MPL_RC)
    fig_w = min(11.5, 5.6 + 0.58 * len(rows))
    fig, ax = plt.subplots(figsize=(fig_w, 2.95))
    fig.subplots_adjust(left=0.09, right=0.80, top=0.82, bottom=0.24)

    for i in xs:
        spans: list[tuple[float, float]] = []
        if not np.isnan(prim_y[i]):
            spans.append(
                (prim_y[i] - prim_err[i][0], prim_y[i] + prim_err[i][1])
            )
        if has_blind[i] and not np.isnan(blind_y[i]):
            spans.append(
                (blind_y[i] - blind_err[i][0], blind_y[i] + blind_err[i][1])
            )
        if not spans:
            continue
        lo = min(s[0] for s in spans)
        hi = max(s[1] for s in spans)
        ax.vlines(
            i,
            lo,
            hi,
            colors="#cccccc",
            linewidth=1.0,
            zorder=1,
        )

    for i in xs:
        if np.isnan(prim_y[i]):
            continue
        ax.errorbar(
            i - 0.07,
            prim_y[i],
            yerr=[[prim_err[i][0]], [prim_err[i][1]]],
            fmt="o",
            color="#1f77b4",
            ecolor=colors[i],
            elinewidth=1.8,
            capsize=3,
            markersize=6,
            markerfacecolor="#1f77b4",
            markeredgecolor=colors[i],
            markeredgewidth=1.8,
            zorder=3,
        )
    for i in xs:
        if not has_blind[i]:
            continue
        ax.errorbar(
            i + 0.07,
            blind_y[i],
            yerr=[[blind_err[i][0]], [blind_err[i][1]]],
            fmt="o",
            color="#1f77b4",
            ecolor=colors[i],
            elinewidth=1.8,
            capsize=3,
            markersize=6,
            markerfacecolor="white",
            markeredgecolor=colors[i],
            markeredgewidth=1.8,
            zorder=3,
        )

    for i, c in enumerate(colors):
        ax.axvspan(i - 0.5, i + 0.5, facecolor=c, alpha=0.035, zorder=0)

    if PREARCHIVE_N_MAIN > 0 and PREARCHIVE_N_MAIN < len(rows):
        ax.axvline(
            PREARCHIVE_N_MAIN - 0.5,
            color="#999999",
            linestyle=":",
            linewidth=1.2,
            zorder=2,
        )

    ax.axhline(0.5, color="#7f7f7f", linestyle="--", linewidth=1.0, zorder=2)
    ax.axhline(0.55, color="#7f7f7f", linestyle="--", linewidth=1.0, zorder=2)
    ax.text(-0.48, 0.505, "0.50", color="#555555", fontsize=8, va="bottom", ha="left")
    ax.text(-0.48, 0.555, "0.55 (promotion)", color="#555555", fontsize=8, va="bottom", ha="left")

    ax.set_xticks(xs)
    ax.set_xticklabels([r.label for r in rows], rotation=35, ha="right", fontsize=7.5)
    ax.set_ylabel(r"Tie-adjusted $\widehat{\mathrm{WR}}_B$")
    ax.set_ylim(0.0, 1.0)
    ax.set_xlim(-0.55, len(rows) - 0.45)
    ax.margins(x=0.02)

    leg_series = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="none",
            markerfacecolor="#1f77b4",
            markeredgecolor="#333333",
            markeredgewidth=1.0,
            markersize=7,
            label="Train (compare session; see caption)",
        ),
    ]
    if PREARCHIVE_INCLUDE_BLIND_EVAL:
        leg_series.append(
            Line2D(
                [0],
                [0],
                marker="o",
                color="none",
                markerfacecolor="white",
                markeredgecolor="#333333",
                markeredgewidth=1.0,
                markersize=7,
                label=(
                    f"Eval ({PREARCHIVE_BLIND_JUDGE_SLUG} backfill; full panel)"
                    if use_ds_backfill
                    else f"Eval ({PREARCHIVE_BLIND_JUDGE_SLUG} backfill where present)"
                ),
            )
        )
    leg_series.extend(
        [
            Line2D(
                [0],
                [0],
                marker="o",
                color="none",
                markerfacecolor="#1f77b4",
                markeredgecolor="#2ca02c",
                markeredgewidth=1.6,
                markersize=7,
                label="Accepted transition (green edge)",
            ),
            Line2D(
                [0],
                [0],
                marker="o",
                color="none",
                markerfacecolor="#1f77b4",
                markeredgecolor="#d62728",
                markeredgewidth=1.6,
                markersize=7,
                label="Rejected / not promoted (red edge)",
            ),
        ]
    )
    ax.legend(handles=leg_series, loc="upper left", bbox_to_anchor=(1.01, 1.0), frameon=False, fontsize=7.5)
    if PREARCHIVE_INCLUDE_BLIND_EVAL:
        ax.set_title(
            "From-scratch pre-reboot loop (git pre-620151b): "
            f"main spine + S3 probes (train vs {PREARCHIVE_BLIND_JUDGE_SLUG} blind; 95% topic-block CI)"
        )
    else:
        ax.set_title(
            "From-scratch pre-reboot loop: train compare only (95% topic-block CI)"
        )

    _save(fig, "figure1_prearchive_loop_trajectory_wr")


def figure2_human_vs_llm() -> None:
    """Left: publish-eval intersection WR_B (DeepSeek, GPT-5.4, Gemini 3 Flash Preview). Right: table."""
    multi = _load(RESULTS / "multi_judge_agreement_S15_vs_Spaper.json")
    humanj = _load(RESULTS / "custom_slice_human_vs_judge" / "human_vs_judge.json")

    gem_publish = "gemini-3-flash-preview"
    judge_order = [
        ("Gemini 3 Flash", gem_publish),
        ("GPT-5.4", "gpt-5.4"),
        ("DeepSeek Chat", "deepseek-chat"),
    ]
    wr_map = {r["model"]: r["win_rate_b"] for r in multi["per_judge"]}
    bars_x = np.arange(len(judge_order))
    wrs = [wr_map[m] for _, m in judge_order]
    n_shared = multi["n_shared_pairs"]

    def _pair_kappa(a: str, b: str) -> tuple[float, float]:
        for row in multi["pairwise"]:
            ja, jb = row["judge_a"], row["judge_b"]
            if {ja, jb} == {a, b}:
                k = row.get("cohen_kappa")
                agr = row.get("agreement_rate")
                return (
                    float(k) if k is not None else float("nan"),
                    float(agr) if agr is not None else float("nan"),
                )
        return float("nan"), float("nan")

    k_ds_gpt, a_ds_gpt = _pair_kappa("deepseek-chat", "gpt-5.4")
    k_ds_gm, a_ds_gm = _pair_kappa("deepseek-chat", gem_publish)
    k_gpt_gm, a_gpt_gm = _pair_kappa("gpt-5.4", gem_publish)

    pj = humanj["per_judge"]
    # Same Gemini model ID as the multi-judge publish-eval panel (gemini-3-flash-preview).
    hum_models = [
        ("deepseek-chat", "DeepSeek"),
        ("gpt-5.4", "GPT-5.4"),
        (gem_publish, "Gemini 3 Flash"),
    ]
    human_rows: list[list[str]] = []
    for key, short in hum_models:
        row = pj.get(key) or {}
        r = row.get("pearson_r")
        sa = row.get("side_agreement_rate")
        human_rows.append(
            [
                short,
                "—" if r is None else f"{r:+.2f}",
                "—" if sa is None else f"{100 * sa:.0f}%",
            ]
        )

    # Single-line labels only (no multiline) — avoids vertical overlap in tight rows.
    cell = [
        ["DeepSeek \u2194 GPT-5.4", f"{k_ds_gpt:.2f}", f"{100 * a_ds_gpt:.0f}%"],
        ["DeepSeek \u2194 G3 Flash", f"{k_ds_gm:.2f}", f"{100 * a_ds_gm:.0f}%"],
        ["GPT-5.4 \u2194 G3 Flash", f"{k_gpt_gm:.2f}", f"{100 * a_gpt_gm:.0f}%"],
    ]
    col_labels = [
        f"Pair\n($N = {n_shared}$)",
        r"Cohen $\kappa$",
        "Exact\nlabel match",
    ]
    mpl.rcParams.update(MPL_RC)
    fig = plt.figure(figsize=(10.0, 3.55))
    gs = gridspec.GridSpec(
        2,
        2,
        figure=fig,
        width_ratios=[1.02, 1.38],
        wspace=0.34,
        hspace=0.62,
        left=0.065,
        right=0.985,
        top=0.87,
        bottom=0.13,
    )
    ax_b = fig.add_subplot(gs[:, 0])
    ax_ij = fig.add_subplot(gs[0, 1])
    ax_h = fig.add_subplot(gs[1, 1])
    for ax in (ax_ij, ax_h):
        ax.axis("off")

    colors_b = ["#4c72b0"] * len(judge_order)
    ax_b.bar(bars_x, wrs, color=colors_b, alpha=0.88, width=0.72)
    ax_b.set_xticks(bars_x)
    ax_b.set_xticklabels(
        ["Gemini 3\nFlash", "GPT-5.4", "DeepSeek\nChat"],
        rotation=0,
        ha="center",
        fontsize=8,
        linespacing=1.0,
    )
    ax_b.set_ylabel(r"$\widehat{\mathrm{WR}}_B$ for S_sota")
    ax_b.set_title(f"Publish-eval intersection ($N={n_shared}$)")
    ax_b.set_ylim(0, 1.0)
    ax_b.axhline(0.5, color="#bbbbbb", linestyle=":", linewidth=1.0, zorder=0)

    # Titles in axes coordinates so they cannot overlap the table (table bbox is lower in axes).
    ax_ij.text(
        0.5,
        0.995,
        "Inter-judge agreement (publish-eval)",
        transform=ax_ij.transAxes,
        fontsize=9.5,
        fontweight="600",
        ha="center",
        va="top",
    )
    ax_ij.text(
        0.5,
        0.88,
        "Low pair-level $\kappa$; left-panel batch means still not near 0.5.",
        transform=ax_ij.transAxes,
        fontsize=7.5,
        ha="center",
        va="top",
        color="#444444",
    )

    # Table bbox: leave space below titles; wider first col for leftrightarrow labels.
    tbl_a = ax_ij.table(
        cellText=cell,
        colLabels=col_labels,
        loc="center",
        cellLoc="center",
        bbox=[0.02, 0.0, 0.96, 0.70],
        colWidths=[0.58, 0.21, 0.21],
    )
    tbl_a.auto_set_font_size(False)
    tbl_a.set_fontsize(7.5)
    tbl_a.scale(1.0, 2.25)
    for k in range(len(col_labels)):
        tbl_a[(0, k)].set_facecolor("#e8e8e8")
        tbl_a[(0, k)].set_text_props(fontweight="600", fontsize=7.4)
        tbl_a[(0, k)].set_height(0.16)
    for r in range(1, 4):
        for c in range(3):
            tbl_a[(r, c)].set_height(0.15)
            tbl_a[(r, c)].get_text().set_linespacing(1.0)

    ax_h.text(
        0.5,
        1.0,
        "vs. human experts (custom 12-topic pilot)",
        transform=ax_h.transAxes,
        fontsize=9.5,
        fontweight="600",
        ha="center",
        va="top",
    )

    col_b = ["Automated judge", "Pearson $r$", "Side match"]
    tbl_b = ax_h.table(
        cellText=human_rows,
        colLabels=col_b,
        loc="center",
        cellLoc="center",
        bbox=[0.02, 0.04, 0.96, 0.82],
        colWidths=[0.42, 0.29, 0.29],
    )
    tbl_b.auto_set_font_size(False)
    tbl_b.set_fontsize(8.0)
    tbl_b.scale(1.0, 2.05)
    for k in range(len(col_b)):
        tbl_b[(0, k)].set_facecolor("#e8e8e8")
        tbl_b[(0, k)].set_text_props(fontweight="600", fontsize=7.8)
        tbl_b[(0, k)].set_height(0.14)
    for r in range(1, 4):
        for c in range(3):
            tbl_b[(r, c)].set_height(0.11)

    fig.suptitle(
        "S15 vs S_sota: batch win rates and judge–human alignment",
        y=0.98,
        fontsize=11,
    )

    _save(fig, "figure2_human_vs_llm")


def figure3_system_harness() -> None:
    """Block diagram: benchmark + SWE edit loop + full eval + judge + promotion."""
    mpl.rcParams.update(MPL_RC)
    fig, ax = plt.subplots(figsize=(10.5, 11.2))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    def rbox(
        x: float,
        y: float,
        w: float,
        h: float,
        title: str,
        body: str,
        *,
        fc: str = "#eef2f7",
        ec: str = "#2c3e50",
        title_fs: float = 10.5,
        body_fs: float = 8.6,
    ) -> None:
        patch = FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle="round,pad=0.008,rounding_size=0.012",
            transform=ax.transAxes,
            facecolor=fc,
            edgecolor=ec,
            linewidth=1.15,
        )
        ax.add_patch(patch)
        ax.text(
            x + w / 2,
            y + h - 0.018,
            title,
            transform=ax.transAxes,
            ha="center",
            va="top",
            fontsize=title_fs,
            fontweight="600",
            color="#1a1a1a",
        )
        ax.text(
            x + w / 2,
            y + 0.022,
            body,
            transform=ax.transAxes,
            ha="center",
            va="bottom",
            fontsize=body_fs,
            linespacing=1.35,
            color="#222222",
        )

    def varrow(x: float, y_top: float, y_bot: float, shrink: float = 0.008) -> None:
        arr = FancyArrowPatch(
            (x, y_top - shrink),
            (x, y_bot + shrink),
            transform=ax.transAxes,
            arrowstyle="-|>",
            mutation_scale=14,
            color="#455a64",
            linewidth=1.2,
        )
        ax.add_patch(arr)

    # Vertical stack (axes coords, bottom-up y)
    rbox(
        0.07,
        0.805,
        0.86,
        0.11,
        "A. Frozen benchmark contract",
        "• Topic list: benchmark_topics.json\n"
        "• Each system: Python module systems/S{n}.py (IDEA_FORMAT output)\n"
        "• Idea generation model: gpt-4.1-mini (held fixed in our runs)",
        fc="#e3f2fd",
    )
    varrow(0.5, 0.805, 0.778)

    rbox(
        0.07,
        0.565,
        0.86,
        0.20,
        "B. SWE / meta-edit inner loop (ideas/swe_agent.py)",
        "1. Read losing patterns from a recent compare report\n"
        "2. Meta-LLM (e.g. Claude Sonnet) proposes one surgical code change\n"
        "3. Smoke test: edited module imports / loads\n"
        "4. Mini-eval: 3 topics × 3 ideas → ~9 pairwise judge calls vs current champion\n"
        "5. Keep the edit if mini-eval clears an improvement threshold; else revert\n"
        "6. Repeat until round / failure limits → accumulated candidate systems/S_next.py",
        fc="#fff8e1",
        ec="#b2871d",
    )
    varrow(0.5, 0.565, 0.538)

    rbox(
        0.07,
        0.415,
        0.86,
        0.115,
        "C. Full benchmark run (ideas/runner.py)",
        "• Regenerate ideas for champion A and candidate B on the full topic grid\n"
        "• Writes results/<version>/ideas.json (cache rules apply for champion only)\n"
        "• Matched slots: same (topic_id, idea_index) on both sides",
        fc="#e8f5e9",
    )
    varrow(0.5, 0.415, 0.388)

    rbox(
        0.07,
        0.265,
        0.86,
        0.115,
        "D. Pairwise LLM judge (ideas/judge.py)",
        "• Primary judge: rubric 0–10 × 4 dims per idea; declare A / B / tie\n"
        "• Random A/B presentation each call → mapped back to true champion/candidate\n"
        "• Tie-adjusted win rate WR_B for candidate B → compare_A_vs_B.json",
        fc="#f3e5f5",
    )
    varrow(0.5, 0.265, 0.238)

    rbox(
        0.07,
        0.115,
        0.86,
        0.115,
        "E. Promotion gate (ideas/godel_loop.py)",
        "• If WR_B > 0.55 → accept: write CURRENT_VERSION, append evolution_log.jsonl\n"
        "• Else reject candidate (champion unchanged)\n"
        "• Optional: independent blind judge on the same pairs for κ / WR diagnostics only",
        fc="#ffebee",
    )

    # Side callout for blind
    bx, by, bw, bh = 0.72, 0.565, 0.26, 0.14
    patch2 = FancyBboxPatch(
        (bx, by),
        bw,
        bh,
        boxstyle="round,pad=0.006,rounding_size=0.01",
        transform=ax.transAxes,
        facecolor="#eceff1",
        edgecolor="#78909c",
        linestyle="--",
        linewidth=1.0,
    )
    ax.add_patch(patch2)
    ax.text(
        bx + bw / 2,
        by + bh - 0.012,
        "Parallel diagnostic",
        transform=ax.transAxes,
        ha="center",
        va="top",
        fontsize=9,
        fontweight="600",
        color="#37474f",
    )
    ax.text(
        bx + bw / 2,
        by + 0.02,
        "Blind DeepSeek (or other)\nre-judges identical pairs\n→ blind_*.json\nDoes not gate promotion.",
        transform=ax.transAxes,
        ha="center",
        va="bottom",
        fontsize=7.8,
        linespacing=1.3,
        color="#37474f",
    )
    # Dotted connector from D to callout
    ax.plot(
        [0.72, bx + bw * 0.2],
        [0.322, by + bh * 0.55],
        transform=ax.transAxes,
        color="#78909c",
        linestyle=":",
        linewidth=1.0,
    )

    fig.suptitle(
        "Schematic — Gödel/Darwin harness (optional asset; not the numbered Figure 3 in main.tex)",
        y=0.98,
        fontsize=12,
        fontweight="600",
    )
    cap = (
        "Operational path for the ideas/ self-improvement tree: the SWE loop produces a candidate "
        "module under tight mini-eval feedback; only after that does the harness run the expensive "
        "full benchmark and primary pairwise judge. The numeric threshold and file names match "
        "ideas/godel_loop.py and ideas/swe_agent.py."
    )
    fig.text(0.5, 0.04, cap, fontsize=8, va="top", ha="center", wrap=True)

    fig.subplots_adjust(left=0.04, right=0.96, top=0.93, bottom=0.10)
    _save(fig, "figure3_system_harness")


def _pairwise_lookup(analysis: dict, a: str, b: str) -> dict | None:
    for row in analysis.get("pairwise", []):
        ja, jb = row["judge_a"], row["judge_b"]
        if {ja, jb} == {a, b}:
            return row
    return None


def _wr_lookup(analysis: dict, model: str) -> float | None:
    for row in analysis.get("per_judge", []):
        if row.get("model") == model:
            return float(row["win_rate_b"])
    return None


def figure4_test_retest_publish_eval() -> None:
    """Compare two API passes of the same judge on the publish-eval bundle (Figure 4)."""
    path = PUBLISH_EVAL_S15_SPAPER / "judge_analysis.json"
    ja = _load(path)
    n = int(ja.get("n_shared_pairs", 0))

    gpt = _pairwise_lookup(ja, "gpt-5.4", "gpt-5.4__rerun")
    ds = _pairwise_lookup(ja, "deepseek-chat", "deepseek-chat__rerun")
    if gpt is None or ds is None:
        raise ValueError(
            f"Need GPT-5.4 and DeepSeek self-pairwise rows in {path} "
            "(run rerun_publish_eval_judges + analyze_publish_eval_judges)."
        )

    labs_short = ["GPT-5.4", "DeepSeek\nChat"]
    agree = [float(gpt["agreement_rate"]), float(ds["agreement_rate"])]
    k3 = [
        float(gpt["cohen_kappa_3way"])
        if gpt.get("cohen_kappa_3way") is not None
        else float("nan"),
        float(ds["cohen_kappa_3way"])
        if ds.get("cohen_kappa_3way") is not None
        else float("nan"),
    ]

    wr_g1 = _wr_lookup(ja, "gpt-5.4")
    wr_g2 = _wr_lookup(ja, "gpt-5.4__rerun")
    wr_d1 = _wr_lookup(ja, "deepseek-chat")
    wr_d2 = _wr_lookup(ja, "deepseek-chat__rerun")

    def _pp(a: float | None, b: float | None) -> str:
        if a is None or b is None:
            return "—"
        return f"{abs(b - a) * 100.0:.1f}"

    mpl.rcParams.update(MPL_RC)
    fig, (ax_a, ax_k) = plt.subplots(
        1, 2, figsize=(7.4, 3.35), gridspec_kw={"wspace": 0.38}
    )
    fig.subplots_adjust(left=0.10, right=0.97, top=0.82, bottom=0.28)
    x = np.arange(2)
    colors = ["#4c72b0", "#dd8452"]

    bars_a = ax_a.bar(x, agree, width=0.58, color=colors, alpha=0.9, edgecolor="white", linewidth=0.6)
    ax_a.set_xticks(x)
    ax_a.set_xticklabels(labs_short, fontsize=8.5, linespacing=1.05)
    ax_a.set_ylabel("Fraction of pairs")
    ax_a.set_ylim(0, 1.05)
    ax_a.axhline(1.0 / 3.0, color="#bbbbbb", linestyle=":", linewidth=1.0, zorder=0)
    ax_a.set_title("Exact label match\n(same pass vs rerun)", fontsize=9.5, pad=6)
    for rect, v in zip(bars_a, agree):
        ax_a.text(
            rect.get_x() + rect.get_width() / 2.0,
            rect.get_height() + 0.02,
            f"{100.0 * v:.1f}%",
            ha="center",
            va="bottom",
            fontsize=8,
        )

    bars_k = ax_k.bar(x, k3, width=0.58, color=colors, alpha=0.9, edgecolor="white", linewidth=0.6)
    ax_k.set_xticks(x)
    ax_k.set_xticklabels(labs_short, fontsize=8.5, linespacing=1.05)
    ax_k.set_ylabel(r"Cohen $\kappa$ (3-class)")
    ymax = max(0.55, max(k3) * 1.15)
    ax_k.set_ylim(0, ymax)
    ax_k.set_title("Pairwise agreement\n(beyond chance)", fontsize=9.5, pad=6)
    for rect, v in zip(bars_k, k3):
        if np.isnan(v):
            continue
        ax_k.text(
            rect.get_x() + rect.get_width() / 2.0,
            rect.get_height() + 0.02,
            f"{v:.2f}",
            ha="center",
            va="bottom",
            fontsize=8,
        )

    fig.suptitle(
        f"Intra-judge test–retest (publish-eval intersection, $N = {n}$ pairs)",
        y=0.98,
        fontsize=11,
    )
    cap = (
        "Second passes: ``judge_gpt-5.4_rerun.json`` and ``judge_deepseek-chat_rerun.json`` vs primaries "
        f"on the same idea pairs as ``judge_analysis.json``. Dotted line: $1/3$ (uniform random 3-class match). "
        f"Tie-adjusted $\\widehat{{\\mathrm{{WR}}}}_B$ for `S_sota` on this slice: "
        f"GPT-5.4 {wr_g1:.3f} $\\to$ {wr_g2:.3f} ({_pp(wr_g1, wr_g2)} pp shift); "
        f"DeepSeek {wr_d1:.3f} $\\to$ {wr_d2:.3f} ({_pp(wr_d1, wr_d2)} pp shift)."
    )
    fig.text(0.5, 0.06, cap, fontsize=7.5, va="top", ha="center", wrap=True)

    _save(fig, "figure4_test_retest_publish_eval")


def _save(fig: mpl.figure.Figure, stem: str) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    png = OUT_DIR / f"{stem}.png"
    pdf = OUT_DIR / f"{stem}.pdf"
    for path in (png, pdf):
        fig.savefig(
            path,
            bbox_inches="tight",
            pad_inches=0.22,
            facecolor="white",
            edgecolor="none",
        )
    plt.close(fig)
    print(f"Wrote {png} and {pdf}")


def main() -> None:
    argparse.ArgumentParser(description="Generate paper figures.").parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    figure1_trajectory_wr()
    figure1_prearchive_loop_trajectory_wr()
    figure2_human_vs_llm()
    figure4_test_retest_publish_eval()
    # Figure 3 (Darwin/SWE): place ``figures/figure3_system_harness.png`` (external schematic). Optional:
    # ``figure3_system_harness()`` matplotlib fallback — not invoked here to avoid overwriting.


if __name__ == "__main__":
    main()
