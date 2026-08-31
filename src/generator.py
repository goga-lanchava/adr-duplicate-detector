import random
from typing import List
import numpy as np
import pandas as pd
from src.normalizer import TextNormalizer
from src.schemas import ADRReport


class SyntheticADRGenerator:
    """Generates synthetic ADR reports with noisy text variations and duplicates."""

    CANONICAL_DRUGS = [
        "atorvastatin", "lisinopril", "metformin", "amlodipine",
        "omeprazole", "levothyroxine", "losartan", "gabapentin",
        "hydrochlorothiazide", "sertraline", "simvastatin", "montelukast"
    ]

    # Noisy aliases injected into uncleaned duplicate reports
    DRUG_NOISE_MAP = {
        "atorvastatin": ["Lipitor 20mg", "Atorvastatin Calcium 40 mg tab", "lipitor tab", "atorvastatn"],
        "lisinopril": ["Zestril 10mg", "Prinivil 20mg oral", "Lisinopril 5 mg", "lisinopril tab"],
        "metformin": ["Glucophage 500mg", "Metformin HCl 850mg", "Glucophage XR", "metfomin"],
        "amlodipine": ["Norvasc 5mg", "Amlodipine Besylate 10mg", "norvasc 10 mg tab"],
        "omeprazole": ["Prilosec 20mg DR", "Omeprazole 40mg cap", "Prilosec OTC"],
        "sertraline": ["Zoloft 50mg", "Sertraline HCl 100mg", "Zoloft oral"],
    }

    CANONICAL_EVENTS = [
        "nausea", "headache", "dizziness", "rash", "fatigue",
        "dyspnea", "pruritus", "diarrhea", "arthralgia",
        "insomnia", "myalgia", "acute kidney injury", "hepatotoxicity"
    ]

    EVENT_NOISE_MAP = {
        "nausea": ["throwing up", "severe emesis", "nauseous feeling", "feeling sick"],
        "headache": ["migraine", "head ache", "severe cephalalgia"],
        "dizziness": ["lightheadedness", "vertigo", "feeling dizzy"],
        "rash": ["skin rash", "hives (urticaria)", "itchy skin rash"],
        "fatigue": ["extreme tiredness", "exhaustion", "lethargy"],
        "dyspnea": ["shortness of breath", "SOB", "dyspnoea"],
        "diarrhea": ["loose stools", "diarrhoea", "watery stool"],
        "pruritus": ["severe itching", "itchiness", "itchy skin"],
        "acute kidney injury": ["acute renal failure", "kidney damage", "AKI"],
        "hepatotoxicity": ["liver damage", "elevated LFTs", "hepatic injury"],
    }

    def __init__(self, seed: int = 42):
        self.seed = seed
        random.seed(seed)
        np.random.seed(seed)

    def _noisy_drug(self, drug: str) -> str:
        if drug in self.DRUG_NOISE_MAP and random.random() < 0.65:
            return random.choice(self.DRUG_NOISE_MAP[drug])
        return drug

    def _noisy_event(self, event: str) -> str:
        if event in self.EVENT_NOISE_MAP and random.random() < 0.65:
            return random.choice(self.EVENT_NOISE_MAP[event])
        return event

    def generate(
        self,
        n_base: int = 400,
        duplicate_rate: float = 0.20,
        max_duplicates: int = 2
    ) -> List[ADRReport]:
        reports: List[ADRReport] = []
        base_id_start = 100000
        dup_id_start = 200000

        for i in range(n_base):
            cluster_id = i + 1
            age = float(np.random.randint(18, 85)) if np.random.rand() > 0.04 else None
            sex = random.choices(["Male", "Female", None], weights=[0.48, 0.48, 0.04])[0]
            seriousness = random.choices(["Serious", "Non-serious"], weights=[0.35, 0.65])[0]
            report_day = int(np.random.randint(1, 700))

            n_drugs = random.randint(1, 3)
            base_drug_keys = random.sample(self.CANONICAL_DRUGS, n_drugs)

            n_events = random.randint(1, 3)
            base_event_keys = random.sample(self.CANONICAL_EVENTS, n_events)

            # Clean canonical base case
            base_report = ADRReport(
                report_id=str(base_id_start + cluster_id),
                age=age,
                sex=sex,
                seriousness=seriousness,
                report_day=report_day,
                drugs=set(base_drug_keys),
                events=set(base_event_keys),
                true_cluster_id=cluster_id,
            )
            reports.append(base_report)

            # Injected duplicates with noisy clinical text
            if np.random.rand() < duplicate_rate:
                n_copies = random.randint(1, max_duplicates)
                for _ in range(n_copies):
                    dup_id_start += 1

                    d_age = None if (age is None or np.random.rand() < 0.05) else max(1.0, min(100.0, age + np.random.randint(-2, 3)))
                    d_sex = sex if np.random.rand() > 0.04 else None
                    d_ser = ("Non-serious" if seriousness == "Serious" else "Serious") if np.random.rand() < 0.10 else seriousness
                    d_day = report_day + int(np.random.randint(0, 61))

                    # Apply noise to duplicate's drug and event list
                    noisy_drugs = {self._noisy_drug(d) for d in base_drug_keys}
                    noisy_events = {self._noisy_event(e) for e in base_event_keys}

                    # Pass through normalizer to mimic production ingestion
                    dup_report = ADRReport(
                        report_id=str(dup_id_start),
                        age=d_age,
                        sex=d_sex,
                        seriousness=d_ser,
                        report_day=d_day,
                        drugs=TextNormalizer.normalize_drug_set(noisy_drugs),
                        events=TextNormalizer.normalize_event_set(noisy_events),
                        true_cluster_id=cluster_id,
                    )
                    reports.append(dup_report)

        random.shuffle(reports)
        return reports

    @staticmethod
    def to_dataframe(reports: List[ADRReport]) -> pd.DataFrame:
        data = []
        for r in reports:
            data.append({
                "ReportID": r.report_id,
                "Age": r.age if r.age is not None else np.nan,
                "Sex": r.sex if r.sex else "",
                "Seriousness": r.seriousness if r.seriousness else "",
                "ReportDay": r.report_day,
                "Drugs": ";".join(sorted(r.drugs)),
                "Events": ";".join(sorted(r.events)),
                "TrueClusterID": r.true_cluster_id,
            })
        return pd.DataFrame(data)