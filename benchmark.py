"""
Accuracy benchmark for the duplicate-detection pipeline.

Two scenarios are scored with pairwise precision / recall / F1:

1. synthetic     - cohorts from SyntheticADRGenerator (clean, easy, self-consistent).
2. real+injected - the real FAERS sample with known duplicates injected into it.
                   Each original report is treated as its own case; a random
                   subset is copied 1-2 times with realistic perturbations
                   (age drift, later receipt day, dropped drugs / reactions,
                   blanked fields). This keeps the real data's skewed drug and
                   reaction frequencies, which is what makes de-duplication hard.

The second scenario is a lower bound on precision: any genuine duplicates that
already exist in the FAERS sample are counted as false positives.

Usage:
    python benchmark.py
    python benchmark.py --threshold 0.8 --csv my_reports.csv
"""
import argparse
import dataclasses
import random
import warnings
from typing import List, Optional

import pandas as pd

from src.clustering import compute_pairwise_metrics
from src.generator import SyntheticADRGenerator
from src.pipeline import ADRDuplicatePipeline
from src.schemas import ADRReport

MANUAL_WEIGHTS = {
    "drug_jaccard": 2.5, "event_jaccard": 2.5, "age_sim": 1.0,
    "temporal_sim": 1.0, "sex_match": 0.5, "seriousness_match": 0.5,
}


def _thin(items: set, rng: random.Random, p_drop: float) -> set:
    """Randomly drop members of a set, always keeping at least one."""
    items = sorted(items)
    kept = [x for x in items if rng.random() >= p_drop]
    return set(kept) if kept else {rng.choice(items)}


def inject_duplicates(
    reports: List[ADRReport], duplicate_rate: float = 0.20, seed: int = 7
) -> List[ADRReport]:
    """Label every report as its own case and add perturbed copies of a subset."""
    rng = random.Random(seed)
    out: List[ADRReport] = []
    for case_id, r in enumerate(reports, start=1):
        out.append(dataclasses.replace(r, true_cluster_id=case_id))
        if rng.random() >= duplicate_rate:
            continue
        for copy_no in range(1, rng.randint(1, 2) + 1):
            age = r.age
            if age is not None:
                age = None if rng.random() < 0.05 else max(0.0, age + rng.choice([-1, 0, 0, 0, 1]))
            flip = {"Serious": "Non-serious", "Non-serious": "Serious"}
            out.append(dataclasses.replace(
                r,
                report_id=f"{r.report_id}-dup{copy_no}",
                age=age,
                sex=None if rng.random() < 0.04 else r.sex,
                seriousness=flip.get(r.seriousness, r.seriousness) if rng.random() < 0.05 else r.seriousness,
                report_day=r.report_day + rng.randint(0, 60),
                drugs=_thin(r.drugs, rng, 0.15),
                events=_thin(r.events, rng, 0.15),
                true_cluster_id=case_id,
            ))
    rng.shuffle(out)
    return out


def score(reports: List[ADRReport], mode: str, threshold: Optional[float]) -> dict:
    pipeline = ADRDuplicatePipeline(
        mode=mode,
        probability_threshold=threshold,
        weights=MANUAL_WEIGHTS if mode == "manual" else None,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        clusters, _, _ = pipeline.run(reports)
    found = clusters.set_index("report_id")["cluster_id"]
    metrics = compute_pairwise_metrics(
        [found[r.report_id] for r in reports],
        [r.true_cluster_id for r in reports],
    )
    metrics["max_cluster"] = int(clusters["cluster_size"].max())
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--threshold", type=float, default=None,
                        help="match threshold (default: the pipeline's per-mode default)")
    parser.add_argument("--csv", default="real_faers_sample.csv",
                        help="real report file used for the injected-duplicate scenario")
    args = parser.parse_args()

    scenarios = []
    for n_base, rate in ((300, 0.2), (1000, 0.2), (2000, 0.3)):
        reports = SyntheticADRGenerator().generate(n_base=n_base, duplicate_rate=rate)
        scenarios.append((f"synthetic n={len(reports)}", reports))

    real = [ADRReport.from_series(row) for _, row in pd.read_csv(args.csv).iterrows()]
    for seed in (7, 8, 9):
        reports = inject_duplicates(real, seed=seed)
        scenarios.append((f"real+injected seed={seed} n={len(reports)}", reports))

    print(f"threshold = {args.threshold if args.threshold is not None else 'per-mode default'}")
    print(f"{'scenario':34s} {'mode':7s} {'precision':>9s} {'recall':>7s} {'F1':>6s} {'max cluster':>12s}")
    for name, reports in scenarios:
        for mode in ("auto", "manual"):
            m = score(reports, mode, args.threshold)
            print(f"{name:34s} {mode:7s} {m['precision']:9.3f} {m['recall']:7.3f} "
                  f"{m['f1']:6.3f} {m['max_cluster']:12d}")


if __name__ == "__main__":
    main()
