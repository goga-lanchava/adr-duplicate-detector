from dataclasses import dataclass, field
from typing import List, Optional, Set
import pandas as pd
from src.normalizer import TextNormalizer


@dataclass(frozen=True)
class ADRReport:
    """Standardized Adverse Drug Reaction (ADR) Report entity with normalized clinical terms."""
    report_id: str
    age: Optional[float]
    sex: Optional[str]
    seriousness: Optional[str]
    report_day: int
    drugs: Set[str] = field(default_factory=set)
    events: Set[str] = field(default_factory=set)
    true_cluster_id: Optional[int] = None

    @classmethod
    def from_series(cls, row: pd.Series) -> "ADRReport":
        drugs_raw = row.get("Drugs", row.get("drugs", ""))
        events_raw = row.get("Events", row.get("events", ""))

        if isinstance(drugs_raw, str):
            drug_tokens = [d.strip() for d in drugs_raw.split(";") if d.strip()]
        elif isinstance(drugs_raw, (list, set)):
            drug_tokens = [str(d).strip() for d in drugs_raw if str(d).strip()]
        else:
            drug_tokens = []

        if isinstance(events_raw, str):
            event_tokens = [e.strip() for e in events_raw.split(";") if e.strip()]
        elif isinstance(events_raw, (list, set)):
            event_tokens = [str(e).strip() for e in events_raw if str(e).strip()]
        else:
            event_tokens = []

        # Standardize strings through clinical normalizer
        drugs = TextNormalizer.normalize_drug_set(drug_tokens)
        events = TextNormalizer.normalize_event_set(event_tokens)

        age_val = row.get("Age", row.get("age", None))
        try:
            age = float(age_val) if pd.notna(age_val) else None
        except (ValueError, TypeError):
            age = None

        sex_val = row.get("Sex", row.get("sex", ""))
        sex = str(sex_val).strip().capitalize() if pd.notna(sex_val) and str(sex_val).strip() else None

        ser_val = row.get("Seriousness", row.get("seriousness", ""))
        seriousness = str(ser_val).strip().capitalize() if pd.notna(ser_val) and str(ser_val).strip() else None

        raw_day = row.get("ReportDay", row.get("report_day", 0))
        try:
            report_day = int(float(raw_day)) if pd.notna(raw_day) else 0
        except (ValueError, TypeError):
            report_day = 0

        true_cluster = row.get("TrueClusterID", row.get("true_cluster_id", None))
        try:
            true_cluster_id = int(true_cluster) if pd.notna(true_cluster) else None
        except (ValueError, TypeError):
            true_cluster_id = None

        return cls(
            report_id=str(row.get("ReportID", row.get("report_id", ""))),
            age=age,
            sex=sex,
            seriousness=seriousness,
            report_day=report_day,
            drugs=drugs,
            events=events,
            true_cluster_id=true_cluster_id,
        )


@dataclass
class ScoredPair:
    report_id_a: str
    report_id_b: str
    match_probability: float
    feature_vector: list