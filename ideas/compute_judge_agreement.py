#!/usr/bin/env python3
"""Compute primary-vs-blind agreement metrics for compare transitions."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

from workshop_result_excludes import PRIMARY_COMPARE_EXCLUDE, skip_compare_path


IDEAS_DIR = Path(__file__).resolve().parent
RESULTS_DIR = IDEAS_DIR / "results"


@dataclass
class TransitionAgreement:
    current: str
    candidate: str
    primary_file: str
    blind_file: str
    overlap_n: int
    agreement_rate: float | None
    decisive_overlap_n: int
    flip_rate_decisive: float | None
    cohen_kappa: float | None
    label_counts_primary: dict[str, int]
    label_counts_blind: dict[str, int]


def _load_verdicts(path: Path) -> list[dict]:
    obj = json.loads(path.read_text())
    return obj.get("verdicts", [])


def _counts(labels: list[str]) -> dict[str, int]:
    out = {"A": 0, "B": 0, "tie": 0}
    for lab in labels:
        if lab in out:
            out[lab] += 1
    return out


def _cohen_kappa(labels_a: list[str], labels_b: list[str]) -> float | None:
    if not labels_a or len(labels_a) != len(labels_b):
        return None
    n = len(labels_a)
    if n == 0:
        return None

    cats = ("A", "B", "tie")
    p0 = sum(1 for a, b in zip(labels_a, labels_b) if a == b) / n
    pa = {c: sum(1 for x in labels_a if x == c) / n for c in cats}
    pb = {c: sum(1 for x in labels_b if x == c) / n for c in cats}
    pe = sum(pa[c] * pb[c] for c in cats)
    denom = 1.0 - pe
    if abs(denom) < 1e-12:
        return None
    return (p0 - pe) / denom


def _pair_key(v: dict) -> tuple[str, int]:
    return (str(v["topic_id"]), int(v["idea_index"]))


def _transition_metrics(primary_path: Path, blind_path: Path) -> TransitionAgreement:
    p = _load_verdicts(primary_path)
    b = _load_verdicts(blind_path)
    mp = {_pair_key(v): str(v.get("winner")) for v in p}
    mb = {_pair_key(v): str(v.get("winner")) for v in b}
    keys = sorted(set(mp) & set(mb))

    labels_p = [mp[k] for k in keys]
    labels_b = [mb[k] for k in keys]
    overlap_n = len(keys)
    agreement_rate = None
    if overlap_n:
        agreement_rate = sum(1 for x, y in zip(labels_p, labels_b) if x == y) / overlap_n

    decisive_pairs = [
        (x, y)
        for x, y in zip(labels_p, labels_b)
        if x in ("A", "B") and y in ("A", "B")
    ]
    decisive_n = len(decisive_pairs)
    flip_rate = None
    if decisive_n:
        flips = sum(1 for x, y in decisive_pairs if x != y)
        flip_rate = flips / decisive_n

    kappa = _cohen_kappa(labels_p, labels_b)

    pobj = json.loads(primary_path.read_text())
    return TransitionAgreement(
        current=str(pobj.get("current", "")),
        candidate=str(pobj.get("candidate", "")),
        primary_file=str(primary_path.relative_to(IDEAS_DIR)),
        blind_file=str(blind_path.relative_to(IDEAS_DIR)),
        overlap_n=overlap_n,
        agreement_rate=agreement_rate,
        decisive_overlap_n=decisive_n,
        flip_rate_decisive=flip_rate,
        cohen_kappa=kappa,
        label_counts_primary=_counts(labels_p),
        label_counts_blind=_counts(labels_b),
    )


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Primary-vs-blind agreement summary")
    ap.add_argument(
        "--blind-model",
        default="gemini-flash-lite-latest",
        help="Blind model suffix used in blind_<A>_vs_<B>__<model>.json",
    )
    ap.add_argument(
        "--output",
        type=Path,
        default=RESULTS_DIR / "judge_agreement.json",
        help="Output JSON file path",
    )
    return ap.parse_args()


def main() -> int:
    args = parse_args()
    safe_model = args.blind_model.replace("/", "_")

    primary_files = sorted(RESULTS_DIR.glob("compare_*.json"))
    summaries: list[TransitionAgreement] = []
    missing_blind: list[str] = []

    for ppath in primary_files:
        if skip_compare_path(ppath.name):
            continue
        pobj = json.loads(ppath.read_text())
        current = pobj.get("current")
        candidate = pobj.get("candidate")
        if not current or not candidate:
            continue
        bpath = RESULTS_DIR / f"blind_{current}_vs_{candidate}__{safe_model}.json"
        if not bpath.exists():
            missing_blind.append(str(bpath.relative_to(IDEAS_DIR)))
            continue
        summaries.append(_transition_metrics(ppath, bpath))

    out = {
        "meta": {
            "blind_model": args.blind_model,
            "primary_pattern": "results/compare_*.json",
            "blind_pattern": f"results/blind_*__{safe_model}.json",
            "excluded_compare_files": sorted(PRIMARY_COMPARE_EXCLUDE),
        },
        "transitions": [s.__dict__ for s in summaries],
        "missing_blind_files": missing_blind,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(out, indent=2) + "\n")
    print(
        f"Wrote {len(summaries)} transitions to {args.output}"
        + (f" (missing blind: {len(missing_blind)})" if missing_blind else "")
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
