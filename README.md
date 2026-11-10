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
| **Features** | For each candidate pair, measure 6 similarities: age, sex, seriousness, report date, drug overlap, reaction overlap. Overlap is rarity-weighted: sharing an unusual drug counts for much more than sharing one half the reports mention. A field that is blank on either report counts as "unknown", not as a mismatch. |
| **Linkage model** | Turn those 6 numbers into one match score. *Auto* mode learns how much each field matters from the data (Fellegi–Sunter EM) and outputs a probability; *Manual* mode lets you set the weights with sliders. |
| **Vetoes** | Hard rules that override the score: no shared drug, no shared reaction, recorded ages more than 5 years apart, different recorded sex, or fewer than two agreeing signals → **not** a duplicate. |
| **Ambiguity pruning** | A report keeps only its two strongest matches. If it resembles dozens of others equally (many patients on the same common drug with the same common reaction), those can't all be duplicates and the data can't say which are — so they are not merged, and the report is flagged. |
| **Clustering** | Build groups from the surviving pairs, with a safeguard so one weak link can't chain unrelated reports together. |
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

## Accuracy

`python benchmark.py` scores the pipeline (pairwise precision / recall / F1) on two
kinds of data:

| Scenario | Auto mode F1 | Manual mode F1 |
|---|---|---|
| Synthetic cohorts (300 – 2,000 base cases) | 0.87 – 0.98 | 0.97 – 0.98 |
| Real FAERS sample with injected duplicates | 0.70 – 0.72 | 0.69 – 0.73 |

The second scenario is the more honest one: it keeps the real data's heavily skewed
drug and reaction frequencies and adds known duplicates (perturbed copies) to score
against. Its precision is a lower bound, because genuine duplicates already present
in the FAERS sample are counted as errors.

## Limitations

- Synthetic scores flatter the model: that data is generated from the same
  assumptions the model uses. Judge it by the real-data scenario.
- With only six fields, some cases are genuinely undecidable — e.g. several
  55-year-old women all reporting the same drug and reaction in the same week.
  The tool declines to merge such look-alike crowds rather than guess, which costs
  some recall; affected reports are flagged in the review screen.
- Auto mode needs data to learn from. Below 30 candidate pairs it falls back to
  conventional prior weights, and on very dense inputs it becomes conservative
  (higher precision, lower recall).
- The symptom dictionary is deliberately coarse (e.g. "vomiting" and "nausea" map
  to one term) to help matching; it is not a MedDRA coding tool.
- If the input is too dense, the blocking stage caps itself and the app shows a
  warning that recall is unreliable for that run.

## Development

```bash
pip install -r requirements-dev.txt
pytest -q             # unit tests, accuracy floors, and a headless run of the app
python benchmark.py   # accuracy table shown above
```
