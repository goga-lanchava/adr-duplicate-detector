from collections import defaultdict
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd
import scipy.cluster.hierarchy as sch
from scipy.spatial.distance import squareform


def cluster_candidate_pairs_sparse(
    n_reports: int,
    candidate_pairs: List[Tuple[int, int]],
    match_probs: np.ndarray,
    threshold: float = 0.70
) -> Dict[int, int]:
    """
    Scalable O(N + |E|) graph clustering with local subgraph average-linkage.

    Connected components are extracted from the sparse thresholded graph, then
    average-linkage is applied *within* each component only. This avoids the dense
    N x N distance matrix while still preventing the single-linkage "percolation
    chaining" failure mode, where one weak edge merges hundreds of unrelated
    reports into a single mega-cluster.
    """
    if n_reports <= 0:
        return {}
    if n_reports == 1:
        return {0: 1}

    # 1. Sparse adjacency over edges meeting the threshold.
    #    A plain dict is used deliberately: membership tests below must not
    #    insert keys as a side effect, which a defaultdict would do.
    adj: Dict[int, Dict[int, float]] = {}
    for (i, j), prob in zip(candidate_pairs, match_probs):
        p_val = float(prob)
        if p_val >= threshold:
            adj.setdefault(i, {})[j] = p_val
            adj.setdefault(j, {})[i] = p_val

    visited = set()
    cluster_labels: Dict[int, int] = {}
    current_cluster_id = 1

    for node in range(n_reports):
        if node in visited:
            continue

        # Singleton: no edge survived the threshold.
        if node not in adj:
            cluster_labels[node] = current_cluster_id
            current_cluster_id += 1
            visited.add(node)
            continue

        # Component extraction (iterative, stack-based).
        component = []
        stack = [node]
        visited.add(node)

        while stack:
            curr = stack.pop()
            component.append(curr)
            for neighbor in adj.get(curr, {}):
                if neighbor not in visited:
                    visited.add(neighbor)
                    stack.append(neighbor)

        comp_size = len(component)

        # A component of size <= 2 needs no further partitioning.
        if comp_size <= 2:
            for member in component:
                cluster_labels[member] = current_cluster_id
            current_cluster_id += 1
            continue

        # Average-linkage on the local (k x k) submatrix only.
        node_to_local = {m: idx for idx, m in enumerate(component)}
        local_dist = np.ones((comp_size, comp_size), dtype=np.float64)
        np.fill_diagonal(local_dist, 0.0)

        for m in component:
            loc_m = node_to_local[m]
            for nbr, prob in adj.get(m, {}).items():
                loc_nbr = node_to_local.get(nbr)
                if loc_nbr is not None:
                    d = max(0.0, 1.0 - prob)
                    local_dist[loc_m, loc_nbr] = d
                    local_dist[loc_nbr, loc_m] = d

        condensed = squareform(local_dist, checks=False)
        Z = sch.linkage(condensed, method="average")
        sub_labels = sch.fcluster(Z, t=1.0 - threshold, criterion="distance")

        sub_map = defaultdict(list)
        for loc_idx, sl in enumerate(sub_labels):
            sub_map[int(sl)].append(component[loc_idx])

        for members in sub_map.values():
            for m in members:
                cluster_labels[m] = current_cluster_id
            current_cluster_id += 1

    return cluster_labels


def _n_pairs(counts: np.ndarray) -> float:
    """Number of unordered within-group pairs, summed over groups: Sum C(n, 2)."""
    c = np.asarray(counts, dtype=np.float64)
    return float(np.sum(c * (c - 1.0) / 2.0))


def compute_pairwise_metrics(
    found_clusters: List[int], true_clusters: List[int]
) -> Dict[str, float]:
    """Pairwise Precision, Recall, and F1 against ground-truth clusters.

    Computed from group-size contingency counts in O(N + G) rather than by
    enumerating O(N^2) report pairs, which is exact and stays cheap for large
    corpora:

        TP           = Sum over (found, true) cells of C(n_ft, 2)
        found_pairs  = Sum over found clusters of C(n_f, 2)
        true_pairs   = Sum over true  clusters of C(n_t, 2)
    """
    n = len(found_clusters)
    if n == 0 or len(true_clusters) != n or any(c is None for c in true_clusters):
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}

    df = pd.DataFrame({"found": found_clusters, "true": true_clusters})

    tp = _n_pairs(df.groupby(["found", "true"]).size().to_numpy())
    found_pairs = _n_pairs(df.groupby("found").size().to_numpy())
    true_pairs = _n_pairs(df.groupby("true").size().to_numpy())

    fp = found_pairs - tp
    fn = true_pairs - tp

    precision = tp / max(tp + fp, 1.0)
    recall = tp / max(tp + fn, 1.0)
    f1 = 2 * precision * recall / max(precision + recall, 1e-9)

    return {"precision": precision, "recall": recall, "f1": f1}
