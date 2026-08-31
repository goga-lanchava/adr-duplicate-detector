# Trying the tool — a walkthrough

This is a step-by-step guide to using the ADR Duplicate Resolution Engine for the
first time. It takes about 5 minutes with the built-in synthetic data.

---

## 1. Install and launch

```bash
git clone https://github.com/goga-lanchava/adr-duplicate-detector.git
cd adr-duplicate-detector

python -m venv .venv
.venv\Scripts\activate            # PowerShell / cmd
# source .venv/bin/activate       # macOS / Linux / Git Bash

pip install -r requirements.txt
streamlit run app.py
```

Your browser opens at `http://localhost:8501`. Everything below happens in that
page. The controls are in the **left sidebar**; results appear in the main area.

Prefer Docker? `docker build -t adr-dedup . && docker run -p 8501:8501 adr-dedup`,
then open `http://localhost:8501`.

---

## 2. Get some data in — Sidebar step 1 ("Data Ingestion")

Pick one of the two radio options:

### Option A — Synthetic Generator (easiest first run)

1. Leave **Data Source** on *Synthetic Generator*.
2. **Base Cases** — how many unique reports to invent (start with the default 300).
3. **Duplicate Rate** — fraction of those that also get 1–2 noisy near-copies
   injected (default 0.20).
4. Click **Generate Synthetic Cohort**. You'll see
   *"Generated N records."*

The generator also records the *true* cluster of every report, so after you run
the pipeline you get real precision / recall / F1 numbers to judge it by.

### Option B — Upload CSV (your own data)

Switch **Data Source** to *Upload CSV* and drop in a file with these columns:

| Column | Meaning | Example |
|---|---|---|
| `ReportID` | unique id | `100234` |
| `Age` | years, may be blank | `54` |
| `Sex` | `Male` / `Female` / blank | `Female` |
| `Seriousness` | `Serious` / `Non-serious` / blank | `Serious` |
| `ReportDay` | integer day index (any epoch) | `1764` |
| `Drugs` | `;`-separated | `atorvastatin;aspirin` |
| `Events` | `;`-separated reactions | `nausea;headache` |
| `TrueClusterID` | *optional* — ground-truth group, enables scoring | `12` |

Drug and event text is normalized automatically (brand→generic, salts/forms
stripped, colloquial symptoms→MedDRA-style terms, typo correction). Duplicate
`ReportID`s are suffixed `#2`, `#3`… with a warning so nothing is silently lost.

`real_faers_sample.csv` in the repo is a ready-made example (500 real openFDA
reports) — upload that if you want to try real data immediately.

---

## 3. Choose how pairs are scored — Sidebar step 2 ("Linkage Engine & Mode")

| Mode | Use when |
|---|---|
| **🧠 Auto-ML (Fellegi-Sunter EM)** | Default. Learns how much each field matters directly from the data. No knobs. |
| **🎛️ Manual Expert Override** | You want to dictate the weighting. Six sliders appear (drug / event / age / date / sex / seriousness importance). |

Start with **Auto-ML**.

---

## 4. Set the cut-off — Sidebar step 3 ("Linkage Threshold")

One slider, 0.50–0.95 (default **0.70**). Pairs scoring at or above this are
treated as the same case.

- **Raise it** (e.g. 0.85) → fewer, higher-confidence merges (more precision).
- **Lower it** (e.g. 0.60) → more aggressive merging (more recall).

You can re-run with different values freely.

---

## 5. Run it

Click **🚀 Run Entity Resolution**. After a moment the main panel fills in.

---

## 6. Read the dashboard

**Top row of metrics**

- **Total Reports** — records loaded.
- **Candidate Clusters** — groups of size > 1 (i.e. suspected duplicate sets).
- **Candidate Duplicates** — total reports sitting in those groups.
- **Pairwise F1** — accuracy vs ground truth (`N/A` if your data has no
  `TrueClusterID`). A green **Ground-Truth Performance** box adds precision /
  recall when available.

**Watch for warnings** (shown as coloured banners):

- *Blocking hit its candidate-pair cap* → input too dense; recall is unreliable,
  tighten the data or parameters.
- *EM did not converge* / *EM was unidentifiable* → learned weights may be shaky
  (common on very small or very uniform inputs).

**📊 Pipeline Diagnostics** (expander) — candidate-pair count, how many pairs were
vetoed for incompatible age/sex, EM iteration count, and the learned per-field
weights table (Auto mode).

**🔗 Scored Linked Pairs** (expander) — every scored pair above the audit
threshold with its individual feature similarities. This is the "why did these
two match" view.

---

## 7. Review clusters — the human-in-the-loop step

- **Left:** *"Select a Cluster to Inspect"* dropdown lists every candidate
  duplicate group.
- **Right:** *Side-by-Side Review* shows each report in that cluster with its
  age, sex, seriousness, day, drugs, and events laid out for comparison.
- Each record has a **Keep Record** checkbox, ticked by default. **Untick** the
  ones you judge to be redundant duplicates.

Work through the clusters that matter to you. Your decisions are remembered as
you switch between clusters.

---

## 8. Export — "Audit Trail & Data Export"

- **📥 Export De-duplicated Dataset** — CSV of only the records you left ticked
  (the label shows the surviving count).
- **📋 Export Regulatory Audit Trail** — CSV of *every* input report with its
  assigned `cluster_id`, `cluster_size`, and your `KEEP` / `REMOVE_DUPLICATE`
  decision — the paper trail for why each record was kept or dropped.

---

## 9. Iterate

Adjust the threshold or switch to Manual mode, click **Run Entity Resolution**
again, and compare. On synthetic data, use the F1 metric to tune; on real data,
rely on the side-by-side review and the Scored Linked Pairs view.

---

## Good to know

- With only these six fields, genuinely ambiguous cases exist — e.g. several
  same-age, same-sex reports of the same drug and reaction filed the same week.
  The tool surfaces them as a cluster; the review step is where you make the call.
- Nothing is deleted automatically. The tool only ever *proposes*; the exported
  de-duplicated file reflects your checkboxes.
