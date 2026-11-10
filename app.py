import dataclasses
from pathlib import Path

import pandas as pd
import streamlit as st

from src.clustering import compute_pairwise_metrics
from src.generator import SyntheticADRGenerator
from src.pipeline import ADRDuplicatePipeline
from src.schemas import ADRReport

APP_DIR = Path(__file__).resolve().parent
BUNDLED_SAMPLE = APP_DIR / "real_faers_sample.csv"

SOURCE_SYNTHETIC = "Generated practice data"
SOURCE_BUNDLED = "FDA sample (500 real reports)"
SOURCE_UPLOAD = "Upload a CSV file"

METHOD_AUTO = "Automatic (recommended)"
METHOD_MANUAL = "Manual weights"

# Plain-language names for the six comparison fields.
FIELD_LABELS = {
    "drug_jaccard": "Drugs",
    "event_jaccard": "Reactions",
    "age_sim": "Age",
    "temporal_sim": "Report date",
    "sex_match": "Sex",
    "seriousness_match": "Seriousness",
}

CSV_TEMPLATE = (
    "ReportID,Age,Sex,Seriousness,ReportDay,Drugs,Events\n"
    "1001,54,Female,Serious,120,atorvastatin;aspirin,nausea;headache\n"
    "1002,54,Female,Serious,131,Lipitor 20mg,nausea\n"
)


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------
def _ensure_unique_ids(reports):
    """Suffix repeated ReportIDs (#2, #3, ...) so each record stays addressable.

    Downstream state (rep_dict, keep_decisions, the audit trail) is keyed by
    report_id; without this a file with duplicate IDs silently loses records.
    Returns (deduped_reports, n_duplicates_suffixed).
    """
    seen: dict = {}
    out = []
    n_dupes = 0
    for r in reports:
        rid = r.report_id
        if rid in seen:
            seen[rid] += 1
            n_dupes += 1
            r = dataclasses.replace(r, report_id=f"{rid}#{seen[rid]}")
        else:
            seen[rid] = 1
        out.append(r)
    return out, n_dupes


def _reports_to_frame(reports) -> pd.DataFrame:
    return pd.DataFrame([{
        "report_id": r.report_id,
        "age": r.age,
        "sex": r.sex or "",
        "seriousness": r.seriousness or "",
        "report_day": r.report_day,
        "drugs": "; ".join(sorted(r.drugs)),
        "events": "; ".join(sorted(r.events)),
    } for r in reports])


def _set_reports(reports, label: str) -> None:
    """Install a new dataset and clear everything derived from the old one."""
    st.session_state.raw_reports = reports
    st.session_state.data_label = label
    st.session_state.keep_decisions = {r.report_id: True for r in reports}
    st.session_state.cluster_df = pd.DataFrame()
    st.session_state.pairs_df = pd.DataFrame()
    st.session_state.metrics = {}
    st.session_state.diagnostics = {}
    st.session_state.load_notice = ""
    st.session_state.pop("group_select", None)
    st.session_state.editor_version += 1


def _load_csv(source, label: str) -> None:
    df = pd.read_csv(source)
    reports = [ADRReport.from_series(row) for _, row in df.iterrows()]
    reports, n_dupe_ids = _ensure_unique_ids(reports)
    _set_reports(reports, label)
    if n_dupe_ids:
        st.session_state.load_notice = (
            f"{n_dupe_ids} repeated report ID(s) in the file were given a suffix "
            "(#2, #3, ...) so no records are lost."
        )


def _keep_earliest_only(members: pd.DataFrame) -> None:
    """Within one group, keep the earliest-received report and mark the rest."""
    ordered = members.sort_values(["report_day", "report_id"])
    for position, rid in enumerate(ordered["report_id"]):
        st.session_state.keep_decisions[str(rid)] = position == 0


def _step_group(group_ids, step: int) -> None:
    current = st.session_state.get("group_select", group_ids[0])
    position = group_ids.index(current) if current in group_ids else 0
    st.session_state.group_select = group_ids[(position + step) % len(group_ids)]


COLUMN_CONFIG = {
    "report_id": st.column_config.TextColumn("Report ID"),
    "cluster_id": st.column_config.NumberColumn("Group", format="%d"),
    "cluster_size": st.column_config.NumberColumn("Group size", format="%d"),
    "age": st.column_config.NumberColumn("Age", format="%d"),
    "sex": st.column_config.TextColumn("Sex"),
    "seriousness": st.column_config.TextColumn("Seriousness"),
    "report_day": st.column_config.NumberColumn(
        "Report day", format="%d", help="Day the report was received (day index)."
    ),
    "drugs": st.column_config.TextColumn("Drugs", width="large"),
    "events": st.column_config.TextColumn("Reactions", width="large"),
}


# --------------------------------------------------------------------------
# Page setup
# --------------------------------------------------------------------------
st.set_page_config(
    page_title="ADR Duplicate Detection",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
      .block-container { padding-top: 2.4rem; padding-bottom: 3rem; }
      h1 { font-weight: 600; letter-spacing: -0.01em; margin-bottom: 0.1rem; }
      h2, h3 { font-weight: 600; }
      [data-testid="stMetric"] {
          background: #F7F9FB; border: 1px solid #E1E6EB;
          border-radius: 6px; padding: 14px 16px;
      }
      [data-testid="stMetricLabel"] p {
          font-size: 0.78rem; color: #5B6B7A;
          text-transform: uppercase; letter-spacing: 0.05em;
      }
      [data-testid="stSidebar"] h2 {
          font-size: 0.82rem; color: #5B6B7A; text-transform: uppercase;
          letter-spacing: 0.06em; margin-top: 0.6rem;
      }
    </style>
    """,
    unsafe_allow_html=True,
)

defaults = {
    "raw_reports": [],
    "data_label": "",
    "cluster_df": pd.DataFrame(),
    "pairs_df": pd.DataFrame(),
    "metrics": {},
    "diagnostics": {},
    "keep_decisions": {},
    "last_upload": None,
    "load_notice": "",
    "editor_version": 0,
}
for key, value in defaults.items():
    if key not in st.session_state:
        st.session_state[key] = value

st.title("ADR Duplicate Detection")
st.caption(
    "Finds adverse drug reaction reports that describe the same case, "
    "and lets you review each suggested group before anything is removed."
)

# --------------------------------------------------------------------------
# Sidebar: data, method, threshold
# --------------------------------------------------------------------------
with st.sidebar:
    st.header("Step 1 · Data")
    sources = [SOURCE_SYNTHETIC] + ([SOURCE_BUNDLED] if BUNDLED_SAMPLE.exists() else []) + [SOURCE_UPLOAD]
    data_source = st.radio("Data source", sources, label_visibility="collapsed")

    if data_source == SOURCE_SYNTHETIC:
        st.caption(
            "Invented reports with known duplicates mixed in. Because the correct "
            "answer is known, accuracy can be measured."
        )
        n_base = st.slider(
            "Number of cases", min_value=50, max_value=2000, value=300, step=50,
            help="How many distinct cases to invent.",
        )
        dup_rate = st.slider(
            "Share with duplicates", min_value=0.05, max_value=0.50, value=0.20, step=0.05,
            format="%.2f", help="Fraction of cases that also get one or two noisy copies.",
        )
        if st.button("Generate practice data", use_container_width=True):
            reports = SyntheticADRGenerator().generate(n_base=n_base, duplicate_rate=dup_rate)
            _set_reports(reports, "generated practice data")

    elif data_source == SOURCE_BUNDLED:
        st.caption("A sample of real reports from the FDA adverse event database (openFDA).")
        if st.button("Load FDA sample", use_container_width=True):
            _load_csv(BUNDLED_SAMPLE, "FDA sample")

    else:
        uploaded_file = st.file_uploader("CSV file", type=["csv"], label_visibility="collapsed")
        st.caption(
            "Required columns: ReportID, Age, Sex, Seriousness, ReportDay, Drugs, Events. "
            "Separate multiple drugs or reactions with a semicolon."
        )
        st.download_button(
            "Download a template", CSV_TEMPLATE.encode("utf-8"),
            file_name="adr_reports_template.csv", mime="text/csv",
        )
        if uploaded_file is not None:
            signature = (uploaded_file.name, uploaded_file.size)
            if st.session_state.last_upload != signature:
                try:
                    _load_csv(uploaded_file, uploaded_file.name)
                    st.session_state.last_upload = signature
                except Exception as exc:  # malformed file: explain, don't crash
                    st.error(f"This file could not be read as a report table: {exc}")

    if st.session_state.raw_reports:
        st.success(
            f"{len(st.session_state.raw_reports):,} reports loaded "
            f"({st.session_state.data_label})."
        )
    if st.session_state.load_notice:
        st.warning(st.session_state.load_notice)

    st.header("Step 2 · Matching method")
    method = st.radio(
        "Matching method", [METHOD_AUTO, METHOD_MANUAL], label_visibility="collapsed",
        help=(
            "Automatic learns from your data how much each field should count. "
            "Manual lets you set the importance of each field yourself."
        ),
    )
    is_manual = method == METHOD_MANUAL

    weights_dict = {}
    if is_manual:
        with st.expander("Field importance", expanded=True):
            weights_dict = {
                "drug_jaccard": st.slider("Drugs", 0.0, 5.0, 2.5, 0.5),
                "event_jaccard": st.slider("Reactions", 0.0, 5.0, 2.5, 0.5),
                "age_sim": st.slider("Age", 0.0, 5.0, 1.0, 0.5),
                "temporal_sim": st.slider("Report date", 0.0, 5.0, 1.0, 0.5),
                "sex_match": st.slider("Sex", 0.0, 3.0, 0.5, 0.25),
                "seriousness_match": st.slider("Seriousness", 0.0, 3.0, 0.5, 0.25),
            }
    else:
        st.caption("Field importance is learned from the loaded data.")

    st.header("Step 3 · Strictness")
    # Each method has its own sensible default (and its own remembered slider
    # position), because the two methods score on different scales.
    mode_key = "manual" if is_manual else "auto"
    prob_threshold = st.slider(
        "Minimum match score", 0.30, 0.95,
        ADRDuplicatePipeline.DEFAULT_THRESHOLDS[mode_key], 0.05,
        key=f"threshold_{mode_key}",
        help=(
            "Two reports are grouped when their match score reaches this value. "
            "Higher = fewer, more certain groups. Lower = more groups, more to review."
            + ("" if is_manual else " In automatic mode the score is a probability, "
               "so 0.50 means 'more likely the same case than not'.")
        ),
    )

    st.divider()
    run_pipeline = st.button(
        "Find duplicates", type="primary", use_container_width=True,
        disabled=not st.session_state.raw_reports,
    )
    if not st.session_state.raw_reports:
        st.caption("Load data in step 1 to continue.")

# --------------------------------------------------------------------------
# Run
# --------------------------------------------------------------------------
if run_pipeline and st.session_state.raw_reports:
    with st.spinner("Comparing reports..."):
        pipeline = ADRDuplicatePipeline(
            mode="manual" if is_manual else "auto",
            probability_threshold=prob_threshold,
            weights=weights_dict if is_manual else None,
        )
        c_df, p_df, diag = pipeline.run(st.session_state.raw_reports)

        details = _reports_to_frame(st.session_state.raw_reports)
        c_df = c_df.merge(details, on="report_id", how="left")

        st.session_state.cluster_df = c_df
        st.session_state.pairs_df = p_df
        st.session_state.diagnostics = diag
        st.session_state.keep_decisions = {r.report_id: True for r in st.session_state.raw_reports}
        st.session_state.pop("group_select", None)
        st.session_state.editor_version += 1

        true_ids = c_df["true_cluster_id"].tolist()
        if all(pd.notna(t) for t in true_ids):
            st.session_state.metrics = compute_pairwise_metrics(c_df["cluster_id"].tolist(), true_ids)
        else:
            st.session_state.metrics = {}

# --------------------------------------------------------------------------
# Main area
# --------------------------------------------------------------------------
if not st.session_state.raw_reports:
    st.subheader("Getting started")
    s1, s2, s3 = st.columns(3)
    with s1.container(border=True):
        st.markdown("**1. Load data**")
        st.write(
            "Use the practice data to try the tool, load the FDA sample, "
            "or upload your own CSV file."
        )
    with s2.container(border=True):
        st.markdown("**2. Find duplicates**")
        st.write(
            "The tool compares reports on drugs, reactions, age, sex, "
            "seriousness and date, and groups likely duplicates."
        )
    with s3.container(border=True):
        st.markdown("**3. Review and export**")
        st.write(
            "Check each suggested group, choose which reports to keep, "
            "then download the cleaned data and a decision log."
        )
    st.info("Start in the panel on the left: choose a data source under Step 1.")

elif st.session_state.cluster_df.empty:
    st.subheader("Loaded reports")
    st.write(
        f"{len(st.session_state.raw_reports):,} reports are ready. "
        "Choose **Find duplicates** in the left panel to analyse them."
    )
    st.dataframe(
        _reports_to_frame(st.session_state.raw_reports),
        column_config=COLUMN_CONFIG, hide_index=True, use_container_width=True, height=460,
    )

else:
    clusters = st.session_state.cluster_df
    pairs = st.session_state.pairs_df
    diag = st.session_state.diagnostics
    blocking = diag.get("blocking", {})
    keep = st.session_state.keep_decisions

    grouped = clusters[clusters["is_candidate_duplicate"]]
    group_ids = sorted(grouped["cluster_id"].unique().tolist())
    group_sizes = grouped.groupby("cluster_id")["report_id"].count().to_dict()
    n_removed = sum(1 for rid in clusters["report_id"] if not keep.get(str(rid), True))

    # ---- Summary ----
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Reports analysed", f"{len(clusters):,}")
    m2.metric("Duplicate groups found", f"{len(group_ids):,}")
    m3.metric("Reports in groups", f"{len(grouped):,}")
    m4.metric("Marked for removal", f"{n_removed:,}")

    if st.session_state.metrics:
        acc = st.session_state.metrics
        st.caption(
            "Accuracy against the known answers in this dataset — "
            f"precision {acc['precision']:.2f} · recall {acc['recall']:.2f} · F1 {acc['f1']:.2f}"
        )

    # ---- Run-quality warnings (shown only when something needs attention) ----
    if blocking.get("truncated"):
        st.error(
            f"The comparison stopped early after {blocking.get('n_candidate_pairs', 0):,} "
            "report pairs because the data is very dense. Some duplicates were not "
            "examined, so results for this run are incomplete."
        )
    skipped = blocking.get("skipped_oversized_blocks", 0)
    if skipped:
        st.warning(
            f"{skipped:,} very large sets of look-alike reports were skipped. "
            "Duplicates that could only be found within them were not examined."
        )
    if diag.get("em_degenerate"):
        st.info(
            "There was too little data to learn field importance automatically, "
            "so standard default values were used."
        )
    if diag.get("mode") == "auto" and diag.get("em_converged") is False:
        st.warning(
            "Automatic learning did not fully settle on this data; scores may be "
            "less reliable than usual. Consider the manual method."
        )

    tab_review, tab_reports, tab_pairs, tab_details, tab_export = st.tabs(
        ["Review groups", "All reports", "Matched pairs", "Run details", "Export"]
    )

    # ---- Review ----
    with tab_review:
        if not group_ids:
            st.info(
                "No duplicate groups were found at this strictness. "
                "Lower the minimum match score in Step 3 to see weaker matches."
            )
        else:
            if st.session_state.get("group_select") not in group_ids:
                st.session_state.group_select = group_ids[0]

            nav_prev, nav_select, nav_next = st.columns([1, 4, 1])
            nav_prev.button(
                "Previous", use_container_width=True,
                on_click=_step_group, args=(group_ids, -1),
            )
            selected = nav_select.selectbox(
                "Group", group_ids, key="group_select", label_visibility="collapsed",
                format_func=lambda cid: (
                    f"Group {group_ids.index(cid) + 1} of {len(group_ids)}"
                    f"  ·  {group_sizes[cid]} reports"
                ),
            )
            nav_next.button(
                "Next", use_container_width=True,
                on_click=_step_group, args=(group_ids, 1),
            )

            members = clusters[clusters["cluster_id"] == selected].sort_values(
                ["report_day", "report_id"]
            )
            member_ids = [str(r) for r in members["report_id"]]

            st.markdown(
                "These reports appear to describe the same case. "
                "Untick **Keep** for any report you consider a redundant copy."
            )

            view = members[["report_id", "age", "sex", "seriousness", "report_day", "drugs", "events"]].copy()
            view.insert(0, "keep", [bool(keep.get(rid, True)) for rid in member_ids])
            edited = st.data_editor(
                view,
                column_config={**COLUMN_CONFIG, "keep": st.column_config.CheckboxColumn("Keep", width="small")},
                disabled=[c for c in view.columns if c != "keep"],
                hide_index=True, use_container_width=True,
                key=f"editor_{st.session_state.editor_version}_{selected}",
            )
            changed = False
            for rid, value in zip(edited["report_id"].astype(str), edited["keep"]):
                if bool(keep.get(rid, True)) != bool(value):
                    keep[rid] = bool(value)
                    changed = True
            if changed:
                st.rerun()

            act1, act2, _ = st.columns([2, 2, 5])
            if act1.button("Keep earliest only", use_container_width=True,
                           help="Keep the first report received and mark the others for removal."):
                _keep_earliest_only(members)
                st.session_state.editor_version += 1
                st.rerun()
            if act2.button("Keep all", use_container_width=True,
                           help="Treat these as separate cases."):
                for rid in member_ids:
                    keep[rid] = True
                st.session_state.editor_version += 1
                st.rerun()

            n_crowded = int((members["ambiguous_matches_pruned"] > 0).sum())
            if n_crowded:
                st.warning(
                    f"{n_crowded} report(s) in this group also closely resembled other "
                    "reports outside it. Only the strongest matches were kept, so "
                    "check this group with extra care."
                )

            if not pairs.empty:
                in_group = pairs[
                    pairs["report_id_a"].astype(str).isin(member_ids)
                    & pairs["report_id_b"].astype(str).isin(member_ids)
                ]
                if not in_group.empty:
                    with st.expander("Why these reports were grouped"):
                        st.caption(
                            "Similarity of each pair, field by field "
                            "(1.00 = identical, blank = not recorded on one of the reports)."
                        )
                        st.dataframe(
                            in_group.rename(columns={
                                "report_id_a": "Report A", "report_id_b": "Report B",
                                "match_probability": "Match score", **FIELD_LABELS,
                            }),
                            column_config={
                                "Match score": st.column_config.ProgressColumn(
                                    "Match score", min_value=0.0, max_value=1.0, format="%.2f"
                                ),
                                **{label: st.column_config.NumberColumn(label, format="%.2f")
                                   for label in FIELD_LABELS.values()},
                            },
                            hide_index=True, use_container_width=True,
                        )

            with st.expander("Apply a decision to every group"):
                st.caption("These actions replace the decisions made so far.")
                bulk1, bulk2, _ = st.columns([2, 2, 5])
                if bulk1.button("Keep earliest only, in all groups", use_container_width=True):
                    for cid in group_ids:
                        _keep_earliest_only(clusters[clusters["cluster_id"] == cid])
                    st.session_state.editor_version += 1
                    st.rerun()
                if bulk2.button("Reset all decisions", use_container_width=True):
                    for rid in list(keep):
                        keep[rid] = True
                    st.session_state.editor_version += 1
                    st.rerun()

    # ---- All reports ----
    with tab_reports:
        only_grouped = st.checkbox("Show only reports that belong to a duplicate group", value=False)
        table = grouped if only_grouped else clusters
        table = table[["report_id", "cluster_id", "cluster_size", "age", "sex",
                       "seriousness", "report_day", "drugs", "events"]].copy()
        table.insert(1, "decision", [
            "Keep" if keep.get(str(rid), True) else "Remove" for rid in table["report_id"]
        ])
        st.dataframe(
            table,
            column_config={**COLUMN_CONFIG, "decision": st.column_config.TextColumn("Decision")},
            hide_index=True, use_container_width=True, height=520,
        )

    # ---- Matched pairs ----
    with tab_pairs:
        if pairs.empty:
            st.info("No report pairs reached a reportable match score.")
        else:
            st.caption(
                "Every pair of reports with a notable match score, strongest first. "
                "Field columns show how similar the two reports are on that field."
            )
            st.dataframe(
                pairs.rename(columns={
                    "report_id_a": "Report A", "report_id_b": "Report B",
                    "match_probability": "Match score", **FIELD_LABELS,
                }),
                column_config={
                    "Match score": st.column_config.ProgressColumn(
                        "Match score", min_value=0.0, max_value=1.0, format="%.2f"
                    ),
                    **{label: st.column_config.NumberColumn(label, format="%.2f")
                       for label in FIELD_LABELS.values()},
                },
                hide_index=True, use_container_width=True, height=520,
            )

    # ---- Run details ----
    with tab_details:
        d1, d2, d3 = st.columns(3)
        d1.metric(
            "Pairs compared", f"{blocking.get('n_candidate_pairs', 0):,}",
            help="Report pairs similar enough to be worth scoring.",
        )
        d2.metric(
            "Ruled out by age or sex", f"{diag.get('demographic_conflicts_vetoed', 0):,}",
            help="Pairs rejected because the recorded ages or sexes are incompatible.",
        )
        d3.metric(
            "Too ambiguous to match", f"{diag.get('ambiguous_matches_pruned', 0):,}",
            help=(
                "Matches dropped because a report resembled too many others to tell "
                "which, if any, is its duplicate."
            ),
        )

        method_used = "Manual weights" if diag.get("mode") == "manual" else "Automatic"
        st.write(
            f"**Method:** {method_used}  \n"
            f"**Minimum match score:** {diag.get('probability_threshold', prob_threshold):.2f}"
        )

        if st.session_state.metrics:
            acc = st.session_state.metrics
            st.markdown("**Accuracy against known answers**")
            a1, a2, a3 = st.columns(3)
            a1.metric("Precision", f"{acc['precision']:.2f}",
                      help="Of the pairs grouped together, the share that truly are duplicates.")
            a2.metric("Recall", f"{acc['recall']:.2f}",
                      help="Of the true duplicate pairs, the share that were found.")
            a3.metric("F1 score", f"{acc['f1']:.2f}",
                      help="Single summary of precision and recall.")

        if diag.get("mode") == "auto" and diag.get("learned_weights"):
            st.markdown("**Learned field importance**")
            st.caption(
                "How strongly agreement on each field points to a duplicate. "
                "Higher means more informative."
            )
            weights_table = pd.DataFrame([
                {"Field": FIELD_LABELS.get(name, name), "Importance": round(weight, 2)}
                for name, weight in diag["learned_weights"].items()
            ]).sort_values("Importance", ascending=False)
            top = max(float(weights_table["Importance"].max()), 1.0)
            st.dataframe(
                weights_table,
                column_config={"Importance": st.column_config.ProgressColumn(
                    "Importance", min_value=0.0, max_value=top, format="%.2f"
                )},
                hide_index=True, use_container_width=True,
            )

    # ---- Export ----
    with tab_export:
        kept_ids = {rid for rid, kept in keep.items() if kept}
        clean_df = clusters[clusters["report_id"].astype(str).isin(kept_ids)].copy()

        st.write(
            f"**{len(clean_df):,}** of {len(clusters):,} reports will be kept; "
            f"**{n_removed:,}** {'is' if n_removed == 1 else 'are'} marked for removal."
        )
        if n_removed == 0 and group_ids:
            st.info(
                "No reports are marked for removal yet. Review the groups first, "
                "or use the bulk action at the bottom of the Review tab."
            )

        e1, e2 = st.columns(2)
        with e1.container(border=True):
            st.markdown("**Cleaned dataset**")
            st.caption("Only the reports you chose to keep.")
            st.download_button(
                label=f"Download cleaned data ({len(clean_df):,} reports)",
                data=clean_df.to_csv(index=False).encode("utf-8"),
                file_name="deduplicated_adr_reports.csv", mime="text/csv",
                type="primary", use_container_width=True,
            )
        with e2.container(border=True):
            st.markdown("**Decision log**")
            st.caption("Every report with its group and your keep / remove decision.")
            audit_log = clusters[["report_id", "cluster_id", "cluster_size"]].copy()
            audit_log["reviewer_decision"] = audit_log["report_id"].map(
                lambda rid: "KEEP" if keep.get(str(rid), True) else "REMOVE_DUPLICATE"
            )
            st.download_button(
                label="Download decision log",
                data=audit_log.to_csv(index=False).encode("utf-8"),
                file_name="adr_duplicate_audit_log.csv", mime="text/csv",
                use_container_width=True,
            )
