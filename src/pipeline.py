from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd

from src.blocking import InvertedIndexBlocker
from src.clustering import cluster_candidate_pairs_sparse
from src.features import PairwiseFeatureExtractor
from src.linkage_model import FellegiSunterEM
from src.schemas import ADRReport


class ADRDuplicatePipeline:
    """
    End-to-End Hybrid Record Linkage Pipeline with Auto-ML and Expert Override support.
    """
    def __init__(
        self,
        mode: str = "auto",
        probability_threshold: float = 0.70,
        weights: Optional[Dict[str, float]] = None,
        max_day_delta: int = 90,
        max_age_diff: float = 15.0,
        age_conflict_years: float = 15.0,
        max_block_size: int = 300,
        max_pairs: int = 5_000_000,
        audit_min_probability: float = 0.30,
    ):
        self.mode = mode
        self.probability_threshold = probability_threshold
        self.audit_min_probability = audit_min_probability
        self.blocker = InvertedIndexBlocker(
            max_day_delta=max_day_delta,
            max_block_size=max_block_size,
            max_pairs=max_pairs,
        )
        self.extractor = PairwiseFeatureExtractor(
            max_age_diff=max_age_diff,
            max_day_diff=float(max_day_delta),
            age_conflict_years=age_conflict_years,
        )
        self.model = FellegiSunterEM(
            feature_names=self.extractor.feature_names,
            mode=mode,
            weights=weights
        )

    def _all_singletons(self, reports: List[ADRReport]) -> pd.DataFrame:
        return pd.DataFrame([{
            "report_id": r.report_id,
            "cluster_id": idx + 1,
            "cluster_size": 1,
            "is_candidate_duplicate": False,
            "true_cluster_id": r.true_cluster_id
        } for idx, r in enumerate(reports)])

    def run(self, reports: List[ADRReport]) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
        n_reports = len(reports)
        if n_reports == 0:
            return pd.DataFrame(), pd.DataFrame(), {}

        # 1. Candidate generation via disjunctive compound-key blocking
        candidate_pairs = sorted(self.blocker.generate_candidate_pairs(reports))

        if not candidate_pairs:
            return (
                self._all_singletons(reports),
                pd.DataFrame(),
                {
                    "mode": self.mode,
                    "learned_weights": {},
                    "blocking": self.blocker.stats,
                }
            )

        # 2. Extract feature matrix
        X = self.extractor.extract_features(reports, candidate_pairs)

        # 3. Train linkage model and score pairs
        self.model.fit(X)
        match_probs = self.model.predict_proba(X)

        # 3b. Hard demographic veto: reports with incompatible recorded age or
        #     sex cannot be the same case, whatever the text similarity says.
        conflict_mask = self.extractor.hard_conflict_mask(reports, candidate_pairs)
        n_conflicts = int(conflict_mask.sum())
        if n_conflicts:
            match_probs = np.where(conflict_mask, 0.0, match_probs)

        # 4. Sparse graph partitioning with local average-linkage
        cluster_map = cluster_candidate_pairs_sparse(
            n_reports, candidate_pairs, match_probs, threshold=self.probability_threshold
        )

        # 5. Scored-pair audit trail (vectorized mask over the retained pairs)
        keep = np.flatnonzero(match_probs >= self.audit_min_probability)
        linked_pair_records = []
        for row in keep:
            idx_a, idx_b = candidate_pairs[row]
            record = {
                "report_id_a": reports[idx_a].report_id,
                "report_id_b": reports[idx_b].report_id,
                "match_probability": float(match_probs[row]),
            }
            record.update({
                name: float(X[row, k])
                for k, name in enumerate(self.extractor.feature_names)
            })
            linked_pair_records.append(record)

        # 6. Consolidate output
        cluster_sizes: Dict[int, int] = defaultdict(int)
        for c in cluster_map.values():
            cluster_sizes[c] += 1

        report_rows = []
        for i, r in enumerate(reports):
            cid = cluster_map[i]
            csize = cluster_sizes[cid]
            report_rows.append({
                "report_id": r.report_id,
                "cluster_id": cid,
                "cluster_size": csize,
                "is_candidate_duplicate": (csize > 1),
                "true_cluster_id": r.true_cluster_id
            })

        cluster_df = (
            pd.DataFrame(report_rows)
            .sort_values(by=["cluster_id", "report_id"])
            .reset_index(drop=True)
        )
        pairs_df = (
            pd.DataFrame(linked_pair_records)
            .sort_values(by="match_probability", ascending=False)
            .reset_index(drop=True)
            if linked_pair_records else pd.DataFrame()
        )

        diagnostics = {
            "mode": self.mode,
            "learned_weights": self.model.learned_weights,
            "m_probs": self.model.m.tolist() if self.model.m is not None else [],
            "u_probs": self.model.u.tolist() if self.model.u is not None else [],
            "feature_names": self.extractor.feature_names,
            "em_iterations": self.model.n_iter_,
            "em_converged": self.model.converged_,
            "em_degenerate": self.model.degenerate_,
            "em_n_patterns": self.model.n_patterns_,
            "demographic_conflicts_vetoed": n_conflicts,
            "blocking": self.blocker.stats,
        }

        return cluster_df, pairs_df, diagnostics
