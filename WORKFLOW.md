# Trying the tool — a walkthrough

This is a step-by-step guide to using the ADR Duplicate Detection tool for the
first time. It takes about 5 minutes with the built-in practice data.

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
page. The controls are in the **left panel**; results appear in the main area.

Prefer Docker? `docker build -t adr-dedup . && docker run -p 8501:8501 adr-dedup`,
then open `http://localhost:8501`.

---

## 2. Load data — Step 1 in the left panel

Choose one of three sources:

| Source | What it is | Button |
|---|---|---|
| **Generated practice data** | Invented reports with known duplicates mixed in. Because the right answer is known, the app can show how accurate it was. Two sliders set how many cases to invent and what share get duplicates. | **Generate practice data** |
| **FDA sample (500 real reports)** | Real reports from the FDA adverse event database, included with the tool. | **Load FDA sample** |
| **Upload a CSV file** | Your own reports. | drop the file in |

For your own file, use these columns (a **Download a template** button gives you a
starting file):

| Column | Meaning | Example |
|---|---|---|
| `ReportID` | unique id | `100234` |
| `Age` | years, may be blank | `54` |
| `Sex` | `Male` / `Female` / blank | `Female` |
| `Seriousness` | `Serious` / `Non-serious` / blank | `Serious` |
| `ReportDay` | day the report was received, as a whole number | `1764` |
| `Drugs` | separated by `;` | `atorvastatin;aspirin` |
| `Events` | reactions, separated by `;` | `nausea;headache` |
| `TrueClusterID` | *optional* — the known correct group, enables accuracy scoring | `12` |

Drug and reaction wording is tidied automatically (brand names become generic
names, doses are dropped, everyday symptom words become standard terms, common
typos are fixed). A green note confirms how many reports were loaded, and the
main area shows them in a table.

---

## 3. Choose a matching method — Step 2

| Method | Use when |
|---|---|
| **Automatic (recommended)** | The tool learns from your data how much each field should count. Nothing to set. |
| **Manual weights** | You want to decide yourself. Six sliders appear, one per field (drugs, reactions, age, report date, sex, seriousness). |

Start with **Automatic**.

---

## 4. Set the strictness — Step 3

One slider, **Minimum match score**. Two reports are grouped when their score
reaches it. Each method starts at a sensible value (0.50 for automatic, 0.70 for
manual).

- **Higher** → fewer, more certain groups.
- **Lower** → more groups, more to review.

Hover the **?** next to any control for a short explanation.

---

## 5. Run it

Click **Find duplicates**. After a moment the results appear.

---

## 6. Read the summary

Four numbers at the top:

- **Reports analysed** — how many reports were loaded.
- **Duplicate groups found** — sets of reports that look like the same case.
- **Reports in groups** — how many reports sit in those sets.
- **Marked for removal** — how many you have chosen to remove so far.

With practice data (or a file that has `TrueClusterID`) a line underneath shows
the accuracy. A coloured banner appears only if something about the run needs
your attention, with a plain explanation.

---

## 7. Review the groups — "Review groups" tab

- Move between groups with **Previous** / **Next** or the drop-down
  ("Group 3 of 64 · 2 reports").
- The table shows the group's reports side by side. **Untick Keep** for any
  report you consider a redundant copy.
- Shortcuts: **Keep earliest only** keeps the first report received and marks
  the rest; **Keep all** treats them as separate cases.
- **Why these reports were grouped** shows the match score for each pair and how
  similar they are field by field.
- A yellow note means some reports in the group also resembled others outside it;
  check those groups more carefully.
- **Apply a decision to every group** lets you keep the earliest report in all
  groups at once, or reset all decisions.

Your choices are remembered as you move between groups and tabs.

---

## 8. The other tabs

- **All reports** — every report with its group and current Keep / Remove decision.
- **Matched pairs** — every notable pair of reports with its score, strongest first.
- **Run details** — how many pairs were compared, how many were ruled out by age
  or sex, how many were too ambiguous to match, accuracy (when known), and which
  fields the automatic method found most informative.

---

## 9. Export — "Export" tab

- **Download cleaned data** — only the reports you chose to keep.
- **Download decision log** — every report with its group and your keep / remove
  decision: the record of why each report was kept or dropped.

---

## 10. Try again with different settings

Change the method or the minimum match score and click **Find duplicates** again.
Note that a new run resets your keep / remove decisions.

---

## Good to know

- With only these six fields, some cases cannot be told apart — e.g. several
  same-age, same-sex reports of the same drug and reaction filed the same week.
  The tool does not merge such look-alike crowds; it keeps each report's best
  one or two matches and flags the group so you can make the call.
- Nothing is deleted automatically. The tool only suggests; the cleaned file
  reflects your choices.
