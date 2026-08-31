import warnings
from collections import defaultdict
from typing import Dict, List, Set, Tuple

from src.schemas import ADRReport


class InvertedIndexBlocker:
    """
    Candidate generator using a disjunction of conjunctive blocking keys.

    Blocking on a single drug token is too permissive for pharmacovigilance data:
    the drug vocabulary is small relative to the corpus, so posting lists grow
    linearly with N and pair enumeration degenerates to O(N^2).

    This blocker instead unions several *conjunctive* key families, each far more
    selective than a bare drug token:

        K1 = (drug, event, time_bucket)   primary  - duplicates agree on both
        K2 = (drug, sex,   time_bucket)   fallback - survives event mis-normalization
        K3 = (event, sex,  time_bucket)   fallback - survives drug mis-normalization

    Recall is preserved by the disjunction (a pair need only collide under ONE
    family); precision comes from each family individually being narrow.

    Time bucketing uses width = max_day_delta and compares each bucket against
    itself and its immediate successor, which provably covers every pair within
    max_day_delta days while keeping per-block density bounded by bucket
    occupancy rather than by corpus size.

    Note on asymptotics: for any *bounded* key space, pair count is inherently
    Theta(N^2 / |keyspace|) -- selectivity buys a large constant factor, not a
    change of complexity class. Real FAERS data has a drug vocabulary in the
    thousands and a multi-year date range, so |keyspace| grows with the corpus
    and blocking behaves near-linearly. `max_block_size` is the safety valve for
    pathological blocks (hyper-frequent tokens such as aspirin).
    """

    def __init__(
        self,
        max_day_delta: int = 90,
        max_block_size: int = 300,
        max_pairs: int = 5_000_000,
    ):
        self.max_day_delta = max_day_delta
        self.max_block_size = max_block_size
        self.max_pairs = max_pairs
        self.stats: Dict[str, int] = {}

    def _bucket(self, day: int) -> int:
        width = max(1, self.max_day_delta)
        # Floor division handles negative report_day values correctly.
        return day // width

    def _build_index(
        self, reports: List[ADRReport]
    ) -> Dict[Tuple, Dict[int, List[int]]]:
        """Maps conjunctive key prefix -> time bucket -> list of report indices."""
        index: Dict[Tuple, Dict[int, List[int]]] = defaultdict(lambda: defaultdict(list))

        for idx, report in enumerate(reports):
            bucket = self._bucket(report.report_day)

            drugs = {d.strip().lower() for d in report.drugs if d and d.strip()}
            events = {e.strip().lower() for e in report.events if e and e.strip()}
            sex = (report.sex or "").strip().lower() or "_na_"

            # K1: drug x event
            for drug in drugs:
                for event in events:
                    index[("de", drug, event)][bucket].append(idx)

            # K2: drug x sex
            for drug in drugs:
                index[("ds", drug, sex)][bucket].append(idx)

            # K3: event x sex
            for event in events:
                index[("es", event, sex)][bucket].append(idx)

        return index

    def generate_candidate_pairs(
        self, reports: List[ADRReport]
    ) -> Set[Tuple[int, int]]:
        index = self._build_index(reports)

        candidate_pairs: Set[Tuple[int, int]] = set()
        skipped_blocks = 0
        truncated = False

        for _, bucket_map in index.items():
            if truncated:
                break

            for bucket, members in bucket_map.items():
                # Compare this bucket against itself and the next one, which
                # together cover the full +/- max_day_delta window.
                neighbours = bucket_map.get(bucket + 1, [])
                block = members + neighbours

                if len(block) < 2:
                    continue

                # Safety valve: hyper-frequent tokens carry no discriminative
                # signal and would dominate the candidate set.
                if len(block) > self.max_block_size:
                    skipped_blocks += 1
                    continue

                # Sorted sliding window with early termination on the day delta.
                block.sort(key=lambda i: reports[i].report_day)
                n_block = len(block)

                for i in range(n_block):
                    idx_a = block[i]
                    day_a = reports[idx_a].report_day

                    for j in range(i + 1, n_block):
                        idx_b = block[j]
                        if reports[idx_b].report_day - day_a > self.max_day_delta:
                            break
                        if idx_a != idx_b:
                            candidate_pairs.add(
                                (min(idx_a, idx_b), max(idx_a, idx_b))
                            )

                if len(candidate_pairs) > self.max_pairs:
                    truncated = True
                    break

        self.stats = {
            "n_reports": len(reports),
            "n_blocks": len(index),
            "n_candidate_pairs": len(candidate_pairs),
            "skipped_oversized_blocks": skipped_blocks,
            "truncated": int(truncated),
        }

        # Truncation silently destroys recall -- the remaining blocks are never
        # examined, so genuine duplicates are dropped without trace. Surface it.
        if truncated:
            warnings.warn(
                f"Blocking hit the max_pairs cap ({self.max_pairs:,}) and stopped "
                f"early. Results are INCOMPLETE and recall is unreliable. The input "
                f"is too dense for the current blocking keys: reduce max_day_delta, "
                f"lower max_block_size, or add a more selective blocking attribute.",
                RuntimeWarning,
                stacklevel=2,
            )

        if skipped_blocks and len(index):
            skip_ratio = skipped_blocks / len(index)
            if skip_ratio > 0.05:
                warnings.warn(
                    f"{skipped_blocks:,} blocks ({skip_ratio:.1%}) exceeded "
                    f"max_block_size={self.max_block_size} and were skipped. Pairs "
                    f"colliding only in those blocks were not evaluated.",
                    RuntimeWarning,
                    stacklevel=2,
                )

        return candidate_pairs
