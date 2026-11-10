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

def test_drug_normalizer_keeps_inorganic_salts_distinct():
    # A leading cation is the drug itself, not a strippable counter-ion.
    assert TextNormalizer.normalize_drug("POTASSIUM CHLORIDE") == "potassium chloride"
    assert TextNormalizer.normalize_drug("SODIUM CHLORIDE") == "sodium chloride"
    assert TextNormalizer.normalize_drug("Calcium") == "calcium"
    assert TextNormalizer.normalize_drug("FERROUS SULFATE") == "ferrous sulfate"
    # A trailing counter-ion is still noise.
    assert TextNormalizer.normalize_drug("LOSARTAN POTASSIUM") == "losartan"
    assert TextNormalizer.normalize_drug("OMEPRAZOLE MAGNESIUM") == "omeprazole"
    # Parenthetical restatements collapse to one name.
    assert TextNormalizer.normalize_drug("ASPIRIN (ACETYLSALICYLIC ACID)") == "aspirin"
    assert TextNormalizer.normalize_drug("LIPITOR (ATORVASTATIN CALCIUM)") == "atorvastatin"


def test_event_normalizer_matches_whole_words_only():
    # "aki" is inside "pollakiuria", "emesis" inside "haematemesis".
    assert TextNormalizer.normalize_event("Pollakiuria") == "pollakiuria"
    assert TextNormalizer.normalize_event("Haematemesis") == "haematemesis"
    assert TextNormalizer.normalize_event("AKI") == "acute kidney injury"
    # Route / salt words are meaningful in reaction terms and must survive.
    assert TextNormalizer.normalize_event("Injection site pain") == "injection site pain"
    assert TextNormalizer.normalize_event("Blood calcium decreased") == "blood calcium decreased"


def _report(rid, age, drugs, events, case, sex="Female", day=100):
    return ADRReport(rid, age, sex, "Serious", day, set(drugs), set(events), case)


def test_rare_shared_terms_outweigh_common_ones():
    from src.features import PairwiseFeatureExtractor

    # "statin" is on every report; "rare_a" / "rare_b" on two each.
    reports = [
        _report("1", 50.0, {"statin", "rare_a"}, {"nausea"}, 1),
        _report("2", 50.0, {"statin", "rare_a"}, {"nausea"}, 1),
        _report("3", 50.0, {"statin", "rare_b"}, {"nausea"}, 2),
        _report("4", 50.0, {"statin", "rare_b"}, {"nausea"}, 2),
    ] + [_report(str(10 + i), 50.0, {"statin"}, {"nausea"}, 10 + i) for i in range(20)]
    ex = PairwiseFeatureExtractor()
    X = ex.extract_features(reports, [(0, 1), (0, 2)])
    drug = ex.feature_names.index("drug_jaccard")
    assert X[0, drug] == pytest.approx(1.0)      # same rare drug
    assert X[1, drug] < 0.2                      # only the ubiquitous drug shared


def test_missing_fields_are_neutral_not_disagreement():
    from src.features import PairwiseFeatureExtractor

    reports = [
        _report("1", 50.0, {"a"}, {"x"}, 1),
        _report("2", None, {"a"}, {"x"}, 1, sex=None),
    ]
    ex = PairwiseFeatureExtractor()
    X = ex.extract_features(reports, [(0, 1)])
    assert np.isnan(X[0, ex.feature_names.index("age_sim")])
    assert np.isnan(X[0, ex.feature_names.index("sex_match")])

    # With fixed parameters, an unknown field must score strictly between
    # agreement and disagreement on that field.
    model = FellegiSunterEM(feature_names=ex.feature_names, mode="auto")
    model.m, model.u, model.p = np.full(6, 0.9), np.full(6, 0.1), 0.05
    base = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
    agree, unknown, disagree = base.copy(), base.copy(), base.copy()
    unknown[0], disagree[0] = np.nan, 0.0
    p_agree, p_unknown, p_disagree = model.predict_proba(np.vstack([agree, unknown, disagree]))
    assert p_agree > p_unknown > p_disagree


def test_ambiguity_pruning_breaks_up_lookalike_crowds():
    # 30 distinct patients, all the same age and sex, same single drug and
    # reaction, same week: indistinguishable on the available fields. They
    # must not be reported as one 30-report "duplicate" cluster.
    crowd = [_report(f"c{i}", 55.0, {"atorvastatin"}, {"type 2 diabetes mellitus"}, i, day=100 + i % 5)
             for i in range(30)]
    # Background so the crowd's drug/event are not the only terms in the corpus.
    background = [_report(f"b{i}", 30.0 + i, {f"drug{i}"}, {f"event{i}"}, 100 + i, sex="Male", day=400)
                  for i in range(30)]
    clusters, _, diag = ADRDuplicatePipeline(mode="manual").run(crowd + background)
    assert clusters["cluster_size"].max() <= 3
    assert diag["ambiguous_matches_pruned"] > 0
    assert clusters["ambiguous_matches_pruned"].max() > 0


def test_pairwise_metrics():
    from src.clustering import compute_pairwise_metrics

    # found: {a,b} {c,d}; true: {a,b,c} {d}  -> TP=1 (a,b), FP=1 (c,d), FN=2
    m = compute_pairwise_metrics([1, 1, 2, 2], [1, 1, 1, 2])
    assert m["precision"] == pytest.approx(0.5)
    assert m["recall"] == pytest.approx(1 / 3)


def test_from_series_tolerates_blank_cells():
    row = pd.Series({"ReportID": "7", "Age": None, "Sex": None, "Seriousness": None,
                     "ReportDay": float("nan"), "Drugs": "Lipitor 20mg; aspirin", "Events": "throwing up"})
    r = ADRReport.from_series(row)
    assert r.age is None and r.sex is None and r.report_day == 0
    assert r.drugs == {"atorvastatin", "aspirin"} and r.events == {"nausea"}


def test_accuracy_floor_on_synthetic_cohort():
    from src.clustering import compute_pairwise_metrics
    from src.generator import SyntheticADRGenerator

    reports = SyntheticADRGenerator().generate(n_base=300, duplicate_rate=0.2)
    for mode in ("auto", "manual"):
        clusters, _, _ = ADRDuplicatePipeline(mode=mode).run(reports)
        found = clusters.set_index("report_id")["cluster_id"]
        m = compute_pairwise_metrics(
            [found[r.report_id] for r in reports], [r.true_cluster_id for r in reports]
        )
        assert m["f1"] > 0.93, (mode, m)


def test_accuracy_floor_on_real_data_with_injected_duplicates():
    # Guards the property the synthetic cohort cannot: precision on data with
    # realistic, heavily skewed drug / reaction frequencies.
    from pathlib import Path
    import benchmark

    csv = Path(__file__).resolve().parent.parent / "real_faers_sample.csv"
    real = [ADRReport.from_series(row) for _, row in pd.read_csv(csv).iterrows()]
    reports = benchmark.inject_duplicates(real, seed=7)
    for mode in ("auto", "manual"):
        m = benchmark.score(reports, mode, None)
        assert m["precision"] > 0.6 and m["recall"] > 0.6, (mode, m)
        assert m["max_cluster"] <= 5, (mode, m)


def test_app_runs_end_to_end():
    pytest.importorskip("streamlit")
    from pathlib import Path
    from streamlit.testing.v1 import AppTest

    app = Path(__file__).resolve().parent.parent / "app.py"
    at = AppTest.from_file(str(app), default_timeout=120).run()
    assert not at.exception

    next(b for b in at.button if "Generate" in b.label).click().run()
    next(b for b in at.button if "Run Entity Resolution" in b.label).click().run()
    assert not at.exception
    labels = [m.label for m in at.metric]
    assert "Candidate Clusters" in labels and "Ambiguous Matches Pruned" in labels

    # Same flow in manual mode.
    at.radio[1].set_value(at.radio[1].options[1]).run()
    next(b for b in at.button if "Run Entity Resolution" in b.label).click().run()
    assert not at.exception
