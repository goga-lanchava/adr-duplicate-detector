# ADR Duplicate Resolution Engine

Hybrid probabilistic record linkage for detecting duplicate **Adverse Drug Reaction (ADR)**
reports, with an interactive human-in-the-loop review UI built on Streamlit.

## Pipeline

```
blocking            disjunctive compound-key inverted index (drug×event, drug×sex, event×sex, time-bucketed)
  -> features       6 pairwise similarity dims: age, sex, seriousness, temporal, drug Jaccard, event Jaccard
  -> linkage_model  Fellegi-Sunter EM (unsupervised) OR manual expert weights
  -> vetoes         no shared drug / no shared event / demographic conflict / <2 agreeing signals -> not a duplicate
  -> clustering     connected components + local average-linkage (no dense N×N matrix)
  -> review UI      side-by-side cluster inspection, keep/remove decisions, de-duplicated + audit-trail CSV export
```

## Run

```bash
pip install -r requirements.txt
streamlit run app.py
```

Or with Docker:

```bash
docker build -t adr-dedup .
docker run -p 8501:8501 adr-dedup
```

**New here?** [WORKFLOW.md](WORKFLOW.md) is a step-by-step walkthrough of using the
app — loading data, choosing a scoring mode, setting the threshold, reviewing
clusters, and exporting results.

## Development

```bash
pip install -r requirements-dev.txt
pytest -q
```


## Data

- **Synthetic Generator** — builds labelled cohorts with noisy text variants and injected duplicates.
- **Upload CSV** — columns: `ReportID, Age, Sex, Seriousness, ReportDay, Drugs, Events`
  (`Drugs` / `Events` are `;`-separated). An optional `TrueClusterID` enables pairwise
  precision/recall/F1 against ground truth.
- `fetch_real_faers_data.py` pulls a real sample from the openFDA drug-event API;
  `real_faers_sample.csv` is a pre-fetched atorvastatin slice.

## Notes / limitations

- Reported F1 on synthetic data (~0.95) reflects data drawn from the generator's own
  assumptions; real spontaneous reports with only these columns can be genuinely
  unresolvable (e.g. many same-age, same-sex reports of the same drug/event submitted
  together). The review UI exists for exactly those cases.
- Blocking truncation and EM non-convergence are surfaced in the app's diagnostics panel.
