from typing import List, Tuple
import numpy as np
from src.schemas import ADRReport


class PairwiseFeatureExtractor:
    """
    Vectorizes candidate pairs into similarity vectors:
    [Age_Sim, Sex_Match, Seriousness_Match, Temporal_Sim, Drug_Jaccard, Event_Jaccard]

    Missing values are encoded as 0.0 (no agreement evidence), preventing false-positive
    agreement inflation in downstream probabilistic models.

    Scalar attributes are gathered into contiguous NumPy arrays and indexed with the
    pair index vectors, so age/sex/seriousness/temporal similarities are computed as
    whole-array operations instead of per-pair Python arithmetic. Only the set-overlap
    features require iteration, and those run against pre-resolved set objects.
    """
    def __init__(
        self,
        max_age_diff: float = 15.0,
        max_day_diff: float = 90.0,
        age_conflict_years: float = 15.0,
    ):
        self.max_age_diff = max_age_diff
        self.max_day_diff = max_day_diff
        # Age gap (in years) beyond which two reports with both ages recorded are
        # treated as demographically incompatible and vetoed outright.
        self.age_conflict_years = age_conflict_years
        self.feature_names = [
            "age_sim", "sex_match", "seriousness_match",
            "temporal_sim", "drug_jaccard", "event_jaccard"
        ]

    @staticmethod
    def _jaccard(set_a: set, set_b: set) -> float:
        if not set_a or not set_b:
            return 0.0
        intersection = len(set_a & set_b)
        if intersection == 0:
            return 0.0
        union = len(set_a) + len(set_b) - intersection
        return float(intersection / union) if union > 0 else 0.0

    def _gather_arrays(self, reports: List[ADRReport]):
        """Gather per-report scalar attributes into contiguous NumPy arrays.

        Sex / seriousness are integer-coded with 0 reserved for "missing", so
        "both sides known and equal" (or "known and different") is a single
        vectorised comparison downstream.
        """
        n = len(reports)
        age = np.full(n, np.nan, dtype=np.float64)
        day = np.zeros(n, dtype=np.float64)
        sex_codes = np.zeros(n, dtype=np.int32)
        ser_codes = np.zeros(n, dtype=np.int32)

        sex_lookup, ser_lookup = {}, {}
        for i, r in enumerate(reports):
            if r.age is not None:
                age[i] = float(r.age)
            day[i] = float(r.report_day)

            # code 0 is reserved for missing, so real values start at 1
            if r.sex and r.sex.strip():
                key = r.sex.strip().lower()
                sex_codes[i] = sex_lookup.setdefault(key, len(sex_lookup) + 1)
            if r.seriousness and r.seriousness.strip():
                key = r.seriousness.strip().lower()
                ser_codes[i] = ser_lookup.setdefault(key, len(ser_lookup) + 1)

        return age, day, sex_codes, ser_codes

    def hard_conflict_mask(
        self, reports: List[ADRReport], candidate_pairs: List[Tuple[int, int]]
    ) -> np.ndarray:
        """Boolean mask over ``candidate_pairs`` flagging pairs that cannot be the
        same case regardless of textual similarity:

        * both ages recorded and more than ``age_conflict_years`` apart, or
        * both sexes recorded and different.

        Applied as a hard veto on the match probability so that a single shared
        drug plus a single shared reaction can no longer merge demographically
        incompatible reports into one "duplicate" cluster.
        """
        n_pairs = len(candidate_pairs)
        if n_pairs == 0:
            return np.zeros(0, dtype=bool)

        pair_arr = np.asarray(candidate_pairs, dtype=np.int64)
        ia, ib = pair_arr[:, 0], pair_arr[:, 1]
        age, _day, sex_codes, _ser = self._gather_arrays(reports)

        age_a, age_b = age[ia], age[ib]
        both_age = ~(np.isnan(age_a) | np.isnan(age_b))
        age_conflict = both_age & (np.abs(age_a - age_b) > self.age_conflict_years)

        sa, sb = sex_codes[ia], sex_codes[ib]
        sex_conflict = (sa != 0) & (sb != 0) & (sa != sb)

        return age_conflict | sex_conflict

    def extract_features(
        self, reports: List[ADRReport], candidate_pairs: List[Tuple[int, int]]
    ) -> np.ndarray:
        n_pairs = len(candidate_pairs)
        n_features = len(self.feature_names)
        X = np.zeros((n_pairs, n_features), dtype=np.float32)

        if n_pairs == 0:
            return X

        pair_arr = np.asarray(candidate_pairs, dtype=np.int64)
        ia, ib = pair_arr[:, 0], pair_arr[:, 1]

        # --- Gather per-report attributes into contiguous arrays ---
        age, day, sex_codes, ser_codes = self._gather_arrays(reports)

        # 1. Age proximity (0.0 when either side is missing)
        age_a, age_b = age[ia], age[ib]
        age_valid = ~(np.isnan(age_a) | np.isnan(age_b))
        age_diff = np.abs(np.where(age_valid, age_a - age_b, 0.0))
        X[:, 0] = np.where(
            age_valid,
            np.maximum(0.0, 1.0 - age_diff / self.max_age_diff),
            0.0
        )

        # 2. Sex exact match (0.0 when either side is missing)
        sa, sb = sex_codes[ia], sex_codes[ib]
        X[:, 1] = ((sa != 0) & (sb != 0) & (sa == sb)).astype(np.float32)

        # 3. Seriousness exact match (0.0 when either side is missing)
        ra, rb = ser_codes[ia], ser_codes[ib]
        X[:, 2] = ((ra != 0) & (rb != 0) & (ra == rb)).astype(np.float32)

        # 4. Temporal proximity
        day_diff = np.abs(day[ia] - day[ib])
        X[:, 3] = np.maximum(0.0, 1.0 - day_diff / self.max_day_diff)

        # 5/6. Set overlaps (irreducibly per-pair, but against resolved sets)
        drug_sets = [r.drugs for r in reports]
        event_sets = [r.events for r in reports]
        jac = self._jaccard

        drug_col = X[:, 4]
        event_col = X[:, 5]
        for row in range(n_pairs):
            a = ia[row]
            b = ib[row]
            drug_col[row] = jac(drug_sets[a], drug_sets[b])
            event_col[row] = jac(event_sets[a], event_sets[b])

        return X
