from typing import Dict, List, Optional
import numpy as np


class FellegiSunterEM:
    """
    Hybrid Record Linkage Model:
    - mode='auto': Unsupervised Fellegi-Sunter EM estimating m_k, u_k likelihood ratios.
    - mode='manual': User-defined weight multipliers over continuous similarity dimensions.

    Performance note: features are binarized into agreement patterns, so for K
    features there are at most 2^K distinct rows (64 for K=6). EM is therefore
    fitted over *unique patterns weighted by their multiplicity* rather than over
    every candidate pair. This is algebraically identical to the naive per-pair
    formulation -- the E-step responsibility depends only on the pattern, and the
    M-step sums factor through the counts -- but turns a per-iteration cost of
    O(n_pairs * K) into O(2^K * K), making the fit independent of corpus size.
    """
    def __init__(
        self,
        feature_names: Optional[List[str]] = None,
        mode: str = "auto",
        weights: Optional[Dict[str, float]] = None,
        max_iter: int = 500,
        tol: float = 1e-6,
        agreement_threshold: float = 0.65,
        agreement_thresholds: Optional[Dict[str, float]] = None
    ):
        self.feature_names = feature_names or [
            "age_sim", "sex_match", "seriousness_match",
            "temporal_sim", "drug_jaccard", "event_jaccard"
        ]
        self.mode = mode
        self.weights = weights or {f: 1.0 for f in self.feature_names}
        self.max_iter = max_iter
        self.tol = tol
        self.agreement_threshold = agreement_threshold
        # Per-field overrides of the similarity level that counts as
        # "agreement". Drug / event lists of one case routinely differ a little
        # between reports, and follow-up reports arrive weeks apart, so those
        # fields need a lower bar than the scalar default.
        self.agreement_thresholds = dict(self.DEFAULT_AGREEMENT_THRESHOLDS)
        if agreement_thresholds:
            self.agreement_thresholds.update(agreement_thresholds)
        self.m: Optional[np.ndarray] = None
        self.u: Optional[np.ndarray] = None
        self.p: float = 0.05
        self.learned_weights: Dict[str, float] = {}
        self.n_iter_: int = 0
        self.converged_: bool = False
        self.degenerate_: bool = False
        self.n_patterns_: int = 0

    # Fallback priors used when EM is unidentifiable. These are conventional
    # Fellegi-Sunter starting values: a matching pair agrees on a field ~90% of
    # the time, a random pair ~10%.
    DEFAULT_AGREEMENT_THRESHOLDS = {
        "age_sim": 0.50,
        "drug_jaccard": 0.50,
        "event_jaccard": 0.50,
        "temporal_sim": 0.30,
    }

    MAX_M = 0.98
    # Below this many candidate pairs there is nothing to learn field
    # statistics from, so conventional priors are used instead of EM.
    MIN_PAIRS_FOR_EM = 30
    FALLBACK_M = 0.90
    FALLBACK_U = 0.10
    FALLBACK_P = 0.05

    def _binarize(self, X: np.ndarray) -> np.ndarray:
        thresholds = np.array([
            self.agreement_thresholds.get(name, self.agreement_threshold)
            for name in self.feature_names
        ], dtype=np.float64)
        gamma = (X >= thresholds).astype(np.float64)
        # Unknown stays unknown: neither agreement (1) nor disagreement (0).
        gamma[np.isnan(X)] = np.nan
        return gamma

    @staticmethod
    def _responsibilities(
        gamma: np.ndarray, m: np.ndarray, u: np.ndarray, p: float
    ) -> np.ndarray:
        """Posterior P(match | agreement pattern), computed in log space.

        A NaN entry (field missing on either report) contributes to neither
        likelihood, i.e. it is marginalised out.
        """
        log_m = np.log(m)
        log_1m = np.log1p(-m)
        log_u = np.log(u)
        log_1u = np.log1p(-u)

        agree = (gamma == 1.0).astype(np.float64)
        disagree = (gamma == 0.0).astype(np.float64)

        log_p_m = agree @ log_m + disagree @ log_1m
        log_p_u = agree @ log_u + disagree @ log_1u

        a = np.log(p) + log_p_m
        b = np.log1p(-p) + log_p_u
        max_log = np.maximum(a, b)

        num = np.exp(a - max_log)
        denom = num + np.exp(b - max_log)
        return num / (denom + 1e-12)

    def fit(self, X: np.ndarray) -> "FellegiSunterEM":
        if len(X) == 0:
            return self

        gamma_full = self._binarize(X)
        n_samples, n_features = gamma_full.shape

        # Collapse to unique agreement patterns with multiplicities.
        # (NaN is recoded to -1 for the uniqueness pass, then restored.)
        coded, counts = np.unique(
            np.where(np.isnan(gamma_full), -1.0, gamma_full), axis=0, return_counts=True
        )
        patterns = np.where(coded < 0, np.nan, coded)
        counts = counts.astype(np.float64)
        total = counts.sum()

        agree = (patterns == 1.0).astype(np.float64)
        observed = (~np.isnan(patterns)).astype(np.float64)

        # u (chance agreement among non-matches) is estimated directly rather
        # than left free in EM. Duplicates are a small minority of candidate
        # pairs, so the overall agreement rate of each field is a close,
        # stable estimate of its non-match agreement rate. Leaving u free lets
        # the two mixture components swap roles or latch onto some other
        # structure in the data (e.g. "same sex" vs "different sex"), which
        # made the learned weights swing wildly between similar inputs.
        self.u = np.clip(
            (agree * counts[:, None]).sum(axis=0)
            / ((observed * counts[:, None]).sum(axis=0) + 1e-9),
            1e-4, 1.0 - 1e-4,
        )
        self.m = np.maximum(np.full(n_features, 0.90), self.u)
        self.p = 0.05

        for iteration in range(self.max_iter):
            resp = self._responsibilities(patterns, self.m, self.u, self.p)

            w_match = resp * counts
            sum_match = w_match.sum()

            new_p = float(np.clip(sum_match / total, 1e-4, 0.50))
            new_m = (agree * w_match[:, None]).sum(axis=0) / (
                (observed * w_match[:, None]).sum(axis=0) + 1e-9
            )
            # A true match never agrees on a field LESS often than chance (the
            # floor keeps every field's evidence non-negative), and no field is
            # infallible (the cap keeps one disagreement from being an absolute
            # veto when everything else lines up).
            new_m = np.clip(np.maximum(new_m, self.u), 1e-4, self.MAX_M)

            delta = np.max(np.abs(self.m - new_m)) + abs(self.p - new_p)
            self.m, self.p = new_m, new_p
            self.n_iter_ = iteration + 1

            if delta < self.tol:
                self.converged_ = True
                break

        # --- Identifiability guard ---
        # Unsupervised Fellegi-Sunter can only separate the match/non-match
        # mixture if the agreement patterns actually vary. With a single distinct
        # pattern (or a candidate set with no disagreement to learn from), EM
        # drives m and u to the same point: the likelihood ratio becomes 1 and
        # every pair scores exactly p, regardless of how similar it is. Left
        # unchecked this silently reports "no duplicates" on small or homogeneous
        # inputs, so fall back to conventional priors instead.
        self.n_patterns_ = int(patterns.shape[0])
        ratios = self.m / (self.u + 1e-12)
        unidentifiable = (
            n_samples < self.MIN_PAIRS_FOR_EM
            or self.n_patterns_ < 2
            or np.allclose(self.m, self.u, atol=1e-3)
            or np.all(np.abs(np.log2(np.maximum(ratios, 1e-12))) < 1e-2)
        )

        if unidentifiable:
            self.degenerate_ = True
            self.m = np.full(n_features, self.FALLBACK_M)
            self.u = np.full(n_features, self.FALLBACK_U)
            self.p = self.FALLBACK_P
        else:
            self.degenerate_ = False

        for i, name in enumerate(self.feature_names):
            ratio = self.m[i] / (self.u[i] + 1e-9)
            self.learned_weights[name] = float(np.log2(max(ratio, 1e-3)))

        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if len(X) == 0:
            return np.zeros(0, dtype=np.float64)

        if self.mode == "auto":
            if self.m is None or self.u is None:
                raise RuntimeError("Model must be fitted before predict_proba in 'auto' mode.")
            gamma = self._binarize(X)
            probs = self._responsibilities(gamma, self.m, self.u, self.p)
        else:
            w_vec = np.array(
                [self.weights.get(name, 1.0) for name in self.feature_names],
                dtype=np.float64
            )
            # Weighted mean over the fields that are actually known for the pair.
            present = ~np.isnan(X)
            num = np.where(present, X, 0.0) @ w_vec
            den = present.astype(np.float64) @ w_vec
            probs = np.divide(num, den, out=np.zeros(len(X), dtype=np.float64), where=den > 0)

        probs = np.asarray(probs, dtype=np.float64)

        # Hard clinical vetoes: a duplicate must share at least one adverse event
        # AND at least one drug. Disjoint/missing sets score exactly 0.0 in the
        # feature extractor, so an exact zero means "no overlap".
        for veto_name in ("event_jaccard", "drug_jaccard"):
            if veto_name in self.feature_names:
                idx = self.feature_names.index(veto_name)
                probs = np.where(X[:, idx] == 0.0, 0.0, probs)

        # Require at least two independent similarity signals to agree before a
        # pair may be linked. Stops "one shared drug + one shared reaction,
        # everything else in conflict" from being reported as a match.
        signal_specs = (
            ("drug_jaccard", 1e-9),
            ("event_jaccard", 1e-9),
            ("age_sim", 0.5),
            ("temporal_sim", 0.5),
            ("sex_match", 0.5),
            ("seriousness_match", 0.5),
        )
        n_signals = np.zeros(len(X), dtype=np.int64)
        for name, thr in signal_specs:
            if name in self.feature_names:
                n_signals += (X[:, self.feature_names.index(name)] > thr).astype(np.int64)
        probs = np.where(n_signals >= 2, probs, 0.0)

        return probs
