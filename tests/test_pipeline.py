import pytest
import numpy as np
import pandas as pd

from src.normalizer import TextNormalizer
from src.schemas import ADRReport
from src.blocking import InvertedIndexBlocker
from src.linkage_model import FellegiSunterEM
from src.pipeline import ADRDuplicatePipeline


def test_text_normalizer():
    assert TextNormalizer.normalize_drug("Lipitor 20 mg oral tablet") == "atorvastatin"
    assert TextNormalizer.normalize_event("throwing up") == "nausea"


def test_fuzzy_normalizer_rejects_cross_ingredient_jumps():
    # Real spelling errors are still corrected...
    assert TextNormalizer.normalize_drug("atorvastatn") == "atorvastatin"
    assert TextNormalizer.normalize_drug("metfomin") == "metformin"
    # ...but a different ingredient must never be rewritten to another.
    assert TextNormalizer.normalize_drug("valsartan") == "valsartan"
    # Brand + salt/route/hydrate noise collapses to the ingredient.
    assert TextNormalizer.normalize_drug("Diovan") == "valsartan"
    assert TextNormalizer.normalize_drug("FORMOTEROL FUMARATE DIHYDRATE") == "formoterol"
    assert TextNormalizer.normalize_drug("Clotrimazole Topical") == "clotrimazole"


def test_event_normalizer_preserves_qualifier_terms():
    # "acute"/"chronic" is part of the term here, not a strippable modifier.
    assert (
        TextNormalizer.normalize_event("Chronic obstructive pulmonary disease")
        == "chronic obstructive pulmonary disease"
    )
    assert (
        TextNormalizer.normalize_event("acute myocardial infarction")
        == "acute myocardial infarction"
    )
    # Stripping is still applied when it resolves to a known term.
    assert TextNormalizer.normalize_event("severe migraine") == "headache"


def test_pipeline_vetoes_demographically_incompatible_pairs():
    # Same single drug + single reaction + same day, but ages 30 years apart:
    # must not be merged into one duplicate cluster.
    reports = [
        ADRReport("A", 40.0, "Female", "Serious", 100, {"atorvastatin"}, {"nausea"}, 1),
        ADRReport("B", 41.0, "Female", "Serious", 101, {"atorvastatin"}, {"nausea"}, 1),
        ADRReport("C", 72.0, "Female", "Serious", 101, {"atorvastatin"}, {"nausea"}, 2),
    ]
    c, _, diag = ADRDuplicatePipeline(mode="auto", probability_threshold=0.60).run(reports)
    cid = c.set_index("report_id")["cluster_id"]
    assert cid["A"] == cid["B"]          # genuine near-duplicate still linked
    assert cid["C"] != cid["A"]          # 32-year age gap vetoed
    assert diag["demographic_conflicts_vetoed"] >= 1


def test_hybrid_fellegi_sunter_modes():
    features = ["age_sim", "sex_match", "seriousness_match", "temporal_sim", "drug_jaccard", "event_jaccard"]
    X = np.array([
        [0.9, 1.0, 1.0, 0.9, 1.0, 1.0],  # Match
        [0.1, 0.0, 0.0, 0.2, 0.5, 0.0]   # Zero event overlap
    ])

    # 1. Auto Mode
    model_auto = FellegiSunterEM(feature_names=features, mode="auto")
    model_auto.fit(X)
    probs_auto = model_auto.predict_proba(X)
    assert probs_auto[0] > 0.50
    assert probs_auto[1] == 0.0  # Zero event guardrail

    # 2. Manual Mode
    weights = {"drug_jaccard": 3.0, "event_jaccard": 3.0, "age_sim": 0.5, "temporal_sim": 1.0, "sex_match": 0.5, "seriousness_match": 0.5}
    model_man = FellegiSunterEM(feature_names=features, mode="manual", weights=weights)
    model_man.fit(X)
    probs_man = model_man.predict_proba(X)
    assert probs_man[0] > 0.80
    assert probs_man[1] == 0.0


def test_end_to_end_pipeline_auto_and_manual():
    sample_reports = [
        ADRReport("101", 45.0, "Male", "Serious", 100, {"atorvastatin", "metformin"}, {"nausea", "headache"}, 1),
        ADRReport("102", 46.0, "Male", "Serious", 105, {"atorvastatin"}, {"nausea"}, 1),
        ADRReport("103", 72.0, "Female", "Non-serious", 500, {"lisinopril"}, {"cough"}, 2),
    ]

    # Test Auto Mode
    p_auto = ADRDuplicatePipeline(mode="auto", probability_threshold=0.60)
    c_auto, _, diag_auto = p_auto.run(sample_reports)
    assert len(c_auto) == 3
    assert diag_auto["mode"] == "auto"

    # Test Manual Mode
    p_man = ADRDuplicatePipeline(mode="manual", probability_threshold=0.60)
    c_man, _, diag_man = p_man.run(sample_reports)
    assert len(c_man) == 3
    assert diag_man["mode"] == "manual"