"""Post-hoc seed x shared-cluster resampling; not the frozen v5 analysis."""

from __future__ import annotations

import math
import random
from collections import defaultdict
from collections.abc import Hashable, Sequence
from statistics import fmean


def crossed_bootstrap(
    differences: Sequence[Sequence[float]],
    *,
    cluster_ids: Sequence[Hashable],
    resamples: int = 10_000,
    seed: int = 20260905,
) -> dict:
    """Draw seeds and shared whole clusters independently, preserving pairing.

    Rows must use the SAME ordered coordinates. Whole clusters retain all their
    observations, including temporal dependence within a cluster. This assumes
    independent clusters, not independent packet windows within one cluster.
    Unequal clusters use a coordinate-weighted ratio, not equal-flow weighting.
    """
    if type(resamples) is not int or resamples < 100:
        raise ValueError("at least 100 integer resamples are required")
    if type(seed) is not int:
        raise ValueError("bootstrap seed must be an integer")
    if not differences or not cluster_ids:
        raise ValueError("nonempty seed rows and aligned clusters required")
    n = len(cluster_ids)
    if any(len(row) != n for row in differences) or any(
        type(value) not in (int, float) or not math.isfinite(value)
        for row in differences
        for value in row
    ):
        raise ValueError("loss differences must be finite aligned numeric rows")
    groups: dict[Hashable, list[int]] = defaultdict(list)
    for index, cluster in enumerate(cluster_ids):
        groups[cluster].append(index)
    if len(groups) < 2:
        raise ValueError("one cluster cannot estimate between-cluster uncertainty")
    clusters = list(groups.values())
    # Compress within-cluster observations without discarding their shared draws.
    totals = [[sum(row[i] for i in group) for group in clusters] for row in differences]
    if any(not math.isfinite(value) for row in totals for value in row):
        raise ValueError("cluster sums overflowed; no valid interval can be reported")
    sizes = [len(group) for group in clusters]
    seed_rng = random.Random(seed)
    cluster_rng = random.Random(seed ^ 0x5A17)
    sampled = []
    for _ in range(resamples):
        rows = [seed_rng.randrange(len(differences)) for _ in differences]
        columns = [cluster_rng.randrange(len(clusters)) for _ in clusters]
        denominator = len(rows) * sum(sizes[column] for column in columns)
        value = (
            sum(totals[row][column] for row in rows for column in columns) / denominator
        )
        if not math.isfinite(value):
            raise ValueError(
                "bootstrap sum overflowed; no valid interval can be reported"
            )
        sampled.append(value)
    sampled.sort()
    return {
        "analysis_contract": "crossed_seed_shared_whole_cluster/v1",
        "publication_evidence": False,
        "study_role": "posthoc_sensitivity",
        "mean_baseline_minus_candidate": fmean(fmean(row) for row in differences),
        "bootstrap_95pct_low": sampled[int(0.025 * resamples)],
        "bootstrap_95pct_high": sampled[min(resamples - 1, int(0.975 * resamples))],
        "seed_count": len(differences),
        "cluster_count": len(clusters),
        "coordinates_per_seed": n,
        "resamples": resamples,
        "bootstrap_seed": seed,
        "estimand": "equal_seed_mean_of_coordinate_weighted_loss_difference",
        "assumptions": [
            "seed and cluster factors are crossed; coordinates are aligned",
            "clusters are independent; entire within-cluster series is retained",
            "not an exact-coverage guarantee, especially with few seeds or clusters",
        ],
    }
