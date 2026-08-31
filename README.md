# ADR Duplicate Resolution Engine

## What this is

When a drug causes a side effect, someone files an **Adverse Drug Reaction (ADR)
report** — to a regulator, a manufacturer, or a database like the FDA's FAERS.
The same real-world case often gets reported more than once (by the patient, their
doctor, and the drug company), each time with slightly different wording, a
rounded age, or a different date. Those near-copies inflate safety statistics and
have to be found and merged.

This tool takes a pile of ADR reports and **groups the ones that look like the
same underlying case**, then hands you a side-by-side review screen to confirm or
reject each group before anything is merged. It runs as a local web app.

It's useful for pharmacovigilance analysts, data scientists working with safety
data, or anyone who needs to de-duplicate event reports and keep an audit trail of
why each record was kept or dropped.

## Quickstart

```bash
pip install -r requirements.txt
streamlit run app.py
```

Your browser opens at `http://localhost:8501`. Choose **Synthetic Generator** in
the sidebar, click **Generate Synthetic Cohort**, then **Run Entity Resolution** —
you'll have results to look at in a few seconds. No data or API key needed.

Docker alternative:

```bash
docker build -t adr-dedup .
docker run -p 8501:8501 adr-dedup
```

**Full walkthrough:** [WORKFLOW.md](WORKFLOW.md) explains every screen and control
step by step — loading data, picking a scoring mode, setting the threshold,
reviewing clusters, and exporting.

## How it works

Each report is compared only against plausibly-related ones, scored on how similar
they are, and then similar reports are grouped:

| Stage | Plain description |
|---|---|
| **Normalization** | Clean up the text: brand names → generic (`Lipitor` → `atorvastatin`), strip doses/salts, map colloquial symptoms to standard terms, fix typos. |
| **Blocking** | Avoid comparing every report to every other one. Only pairs that share, say, a drug *and* a reaction *and* a rough time window are considered. |
| **Features** | For each candidate pair, measure 6 similarities: age, sex, seriousness, report date, drug overlap, reaction overlap. |
| **Linkage model** | Turn those 6 numbers into one match probability. *Auto* mode learns the field weights from the data (Fellegi–Sunter EM); *Manual* mode lets you set them with sliders. |
| **Vetoes** | Hard rules that override the score: no shared drug, no shared reaction, incompatible age/sex, or fewer than two agreeing signals → **not** a duplicate. |
| **Clustering** | Build groups from the high-scoring pairs, with a safeguard so one weak link can't chain hundreds of unrelated reports together. |
| **Review UI** | You inspect each group, untick the reports you consider redundant, and export. |

Nothing is deleted automatically — the tool only proposes groupings.

## Data

- **Synthetic Generator** — invents labelled test data with noisy duplicates mixed
  in. Because it knows the true groupings, the app can show real
  precision / recall / F1 scores.
- **Upload CSV** — your own reports. Columns:
  `ReportID, Age, Sex, Seriousness, ReportDay, Drugs, Events`
  (`Drugs` and `Events` are `;`-separated). An optional `TrueClusterID` column
  turns on accuracy scoring.
- `real_faers_sample.csv` — 500 real reports already fetched from the FDA's
  openFDA API, ready to upload.
- `fetch_real_faers_data.py` — script to pull a fresh sample yourself.

## Limitations

- The ~0.95 F1 on synthetic data looks great partly because that data is generated
  from the same assumptions the model uses. Real reports are harder.
- With only these six fields, some cases are genuinely undecidable — e.g. several
  55-year-old women all reporting the same drug and reaction in the same week.
  The tool clusters them and flags it; the human review step is where the call
  gets made.
- If the input is too dense, the blocking stage caps itself and the app shows a
  warning that recall is unreliable for that run.

## Development

```bash
pip install -r requirements-dev.txt
pytest -q
```
