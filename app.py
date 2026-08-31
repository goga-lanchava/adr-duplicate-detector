import dataclasses

import pandas as pd
import streamlit as st
from src.clustering import compute_pairwise_metrics
from src.generator import SyntheticADRGenerator
from src.pipeline import ADRDuplicatePipeline
from src.schemas import ADRReport


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

st.set_page_config(
    page_title="ADR Duplicate Entity Resolution Tool",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("🛡️ Adverse Drug Reaction Duplicate Resolution Engine")
st.markdown(
    "Hybrid **Probabilistic Record Linkage (Fellegi-Sunter EM)** with an interactive **Human-in-the-Loop Review** interface."
)

# ----------------- Session State Initialization -----------------
if "raw_reports" not in st.session_state:
    st.session_state.raw_reports = []
if "cluster_df" not in st.session_state:
    st.session_state.cluster_df = pd.DataFrame()
if "pairs_df" not in st.session_state:
    st.session_state.pairs_df = pd.DataFrame()
if "metrics" not in st.session_state:
    st.session_state.metrics = {}
if "diagnostics" not in st.session_state:
    st.session_state.diagnostics = {}
if "keep_decisions" not in st.session_state:
    st.session_state.keep_decisions = {}
if "last_loaded_file" not in st.session_state:
    st.session_state.last_loaded_file = None

# ----------------- Sidebar: Ingestion & Hybrid Mode -----------------
with st.sidebar:
    st.header("1. Data Ingestion")
    data_source = st.radio("Select Data Source:", ["Synthetic Generator", "Upload CSV"])

    if data_source == "Synthetic Generator":
        n_base = st.slider("Base Cases:", min_value=50, max_value=2000, value=300, step=50)
        dup_rate = st.slider("Duplicate Rate:", min_value=0.05, max_value=0.50, value=0.20, step=0.05)
        if st.button("Generate Synthetic Cohort", use_container_width=True):
            gen = SyntheticADRGenerator()
            st.session_state.raw_reports = gen.generate(n_base=n_base, duplicate_rate=dup_rate)
            st.session_state.keep_decisions = {r.report_id: True for r in st.session_state.raw_reports}
            st.session_state.cluster_df = pd.DataFrame()
            st.session_state.metrics = {}
            st.session_state.diagnostics = {}
            st.session_state.last_loaded_file = None
            st.success(f"Generated {len(st.session_state.raw_reports)} records.")
    else:
        uploaded_file = st.file_uploader("Upload ADR CSV file:", type=["csv"])
        if uploaded_file is not None:
            if st.session_state.last_loaded_file != uploaded_file.name:
                df_upload = pd.read_csv(uploaded_file)
                reports = [ADRReport.from_series(row) for _, row in df_upload.iterrows()]
                reports, n_dupe_ids = _ensure_unique_ids(reports)
                if n_dupe_ids:
                    st.warning(
                        f"{n_dupe_ids} duplicate ReportID(s) in the file were suffixed "
                        f"(#2, #3, ...) so no records are lost."
                    )
                st.session_state.raw_reports = reports
                st.session_state.keep_decisions = {r.report_id: True for r in st.session_state.raw_reports}
                st.session_state.cluster_df = pd.DataFrame()
                st.session_state.metrics = {}
                st.session_state.diagnostics = {}
                st.session_state.last_loaded_file = uploaded_file.name
                st.success(f"Loaded {len(st.session_state.raw_reports)} records.")
            else:
                st.caption(f"Active file: `{uploaded_file.name}` ({len(st.session_state.raw_reports)} records)")

    st.header("2. Linkage Engine & Mode")
    mode_selection = st.radio(
        "Select Scoring Mode:",
        ["🧠 Auto-ML (Fellegi-Sunter EM)", "🎛️ Manual Expert Override"]
    )
    is_manual = "Manual" in mode_selection

    weights_dict = {}
    if is_manual:
        st.caption("Adjust clinical importance multipliers manually:")
        drug_w = st.slider("💊 Drug Overlap Weight:", 0.0, 5.0, 2.5, 0.5)
        event_w = st.slider("🩺 Event Overlap Weight:", 0.0, 5.0, 2.5, 0.5)
        age_w = st.slider("👤 Age Proximity Weight:", 0.0, 5.0, 1.0, 0.5)
        date_w = st.slider("📅 Date Proximity Weight:", 0.0, 5.0, 1.0, 0.5)
        sex_w = st.slider("⚧ Sex Match Weight:", 0.0, 3.0, 0.5, 0.25)
        ser_w = st.slider("⚠️ Seriousness Match Weight:", 0.0, 3.0, 0.5, 0.25)

        weights_dict = {
            "drug_jaccard": drug_w,
            "event_jaccard": event_w,
            "age_sim": age_w,
            "temporal_sim": date_w,
            "sex_match": sex_w,
            "seriousness_match": ser_w,
        }
    else:
        st.info("Unsupervised EM will learn optimal field weights directly from population distributions.")

    st.header("3. Linkage Threshold")
    if is_manual:
        threshold_label = "Match Score Threshold (weighted similarity):"
        st.caption(
            "Manual mode scores are a normalized weighted average of similarities "
            "in [0, 1] - a ranking score, not a calibrated probability."
        )
    else:
        threshold_label = "Posterior Match Probability Threshold:"
    prob_threshold = st.slider(threshold_label, 0.50, 0.95, 0.70, 0.05)

    run_pipeline = st.button("🚀 Run Entity Resolution", type="primary", use_container_width=True)

# ----------------- Pipeline Execution -----------------
if run_pipeline and st.session_state.raw_reports:
    with st.spinner("Executing Record Linkage Pipeline..."):
        pipeline = ADRDuplicatePipeline(
            mode="manual" if is_manual else "auto",
            probability_threshold=prob_threshold,
            weights=weights_dict if is_manual else None
        )
        c_df, p_df, diag = pipeline.run(st.session_state.raw_reports)

        rep_dict = {r.report_id: r for r in st.session_state.raw_reports}
        c_df["age"] = c_df["report_id"].map(lambda rid: rep_dict[rid].age)
        c_df["sex"] = c_df["report_id"].map(lambda rid: rep_dict[rid].sex or "")
        c_df["seriousness"] = c_df["report_id"].map(lambda rid: rep_dict[rid].seriousness or "")
        c_df["report_day"] = c_df["report_id"].map(lambda rid: rep_dict[rid].report_day)
        c_df["drugs"] = c_df["report_id"].map(lambda rid: "; ".join(sorted(rep_dict[rid].drugs)))
        c_df["events"] = c_df["report_id"].map(lambda rid: "; ".join(sorted(rep_dict[rid].events)))

        st.session_state.cluster_df = c_df
        st.session_state.pairs_df = p_df
        st.session_state.diagnostics = diag
        st.session_state.keep_decisions = {r.report_id: True for r in st.session_state.raw_reports}

        true_ids = [rep_dict[rid].true_cluster_id for rid in c_df["report_id"]]
        if any(t is not None for t in true_ids):
            st.session_state.metrics = compute_pairwise_metrics(
                c_df["cluster_id"].tolist(), true_ids
            )
        else:
            st.session_state.metrics = {}

# ----------------- Dashboard & Diagnostics -----------------
if not st.session_state.cluster_df.empty:
    clusters = st.session_state.cluster_df
    candidate_clusters = clusters[clusters["is_candidate_duplicate"]]
    n_candidate_clusters = candidate_clusters["cluster_id"].nunique()

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Reports", len(clusters))
    col2.metric("Candidate Clusters", n_candidate_clusters)
    col3.metric("Candidate Duplicates", len(candidate_clusters))
    col4.metric("Pairwise F1", f"{st.session_state.metrics.get('f1', 0.0):.3f}" if st.session_state.metrics else "N/A")

    if st.session_state.metrics and st.session_state.metrics.get("precision", 0) > 0:
        st.info(
            f"**Ground-Truth Performance:** Precision = `{st.session_state.metrics['precision']:.3f}` | "
            f"Recall = `{st.session_state.metrics['recall']:.3f}` | "
            f"Pairwise F1 = `{st.session_state.metrics['f1']:.3f}`"
        )

    # ----------------- Pipeline Diagnostics -----------------
    diag = st.session_state.diagnostics
    blocking = diag.get("blocking", {})

    # Blocking truncation silently destroys recall; surface it prominently
    # rather than leaving it in a server-side RuntimeWarning the user never sees.
    if blocking.get("truncated"):
        st.error(
            f"⚠️ Blocking hit its candidate-pair cap "
            f"({blocking.get('n_candidate_pairs', 0):,} pairs) and stopped early. "
            "Genuine duplicates were not evaluated — recall is unreliable for this run."
        )
    skipped = blocking.get("skipped_oversized_blocks", 0)
    if skipped:
        st.warning(
            f"{skipped:,} oversized blocks were skipped; pairs colliding only in "
            "those blocks were not scored."
        )
    if diag.get("em_degenerate"):
        st.warning(
            "Unsupervised EM was unidentifiable on this input (too few distinct "
            "agreement patterns) and fell back to conventional Fellegi-Sunter priors."
        )
    if diag.get("mode") == "auto" and diag.get("em_converged") is False:
        st.warning(
            f"EM did not converge within {diag.get('em_iterations', '?')} iterations; "
            "learned weights may be unstable."
        )

    if diag:
        with st.expander("📊 Pipeline Diagnostics", expanded=False):
            d1, d2, d3 = st.columns(3)
            d1.metric("Candidate Pairs", f"{blocking.get('n_candidate_pairs', 0):,}")
            d2.metric("Demographic Vetoes", f"{diag.get('demographic_conflicts_vetoed', 0):,}")
            d3.metric("EM Iterations", diag.get("em_iterations", "—"))

            if diag.get("mode") == "auto" and diag.get("learned_weights"):
                st.caption("Learned Fellegi-Sunter EM field weights (log-likelihood ratios)")
                w_data = [
                    {"Clinical Dimension": name, "Weight log2(m/u)": round(weight, 3)}
                    for name, weight in diag["learned_weights"].items()
                ]
                st.dataframe(pd.DataFrame(w_data), use_container_width=True)

    # Scored-pair audit trail (computed by the pipeline; shown here for review).
    if not st.session_state.pairs_df.empty:
        with st.expander("🔗 Scored Linked Pairs (audit trail)", expanded=False):
            st.dataframe(st.session_state.pairs_df, use_container_width=True)

    st.markdown("---")

    left_col, right_col = st.columns([1, 2])

    with left_col:
        st.subheader("Candidate Clusters (Size > 1)")
        cluster_summary = (
            candidate_clusters.groupby("cluster_id")
            .agg(size=("report_id", "count"), reports=("report_id", lambda s: ", ".join(s)))
            .reset_index()
        )

        if not cluster_summary.empty:
            selected_cluster_id = st.selectbox(
                "Select a Cluster to Inspect:",
                options=cluster_summary["cluster_id"].tolist(),
                format_func=lambda cid: f"Cluster #{cid} ({cluster_summary.loc[cluster_summary['cluster_id'] == cid, 'size'].values[0]} reports)",
            )
        else:
            st.warning("No candidate duplicate clusters found at this threshold.")
            selected_cluster_id = None

    with right_col:
        if selected_cluster_id is not None:
            st.subheader(f"Side-by-Side Review: Cluster #{selected_cluster_id}")
            st.caption("Uncheck 'Keep Record' to mark duplicates for removal.")

            cluster_members = clusters[clusters["cluster_id"] == selected_cluster_id].copy()

            for _, row in cluster_members.iterrows():
                r_id = str(row["report_id"])
                c1, c2 = st.columns([1, 4])
                with c1:
                    is_kept = st.checkbox(
                        "Keep Record",
                        value=st.session_state.keep_decisions.get(r_id, True),
                        key=f"chk_{r_id}",
                    )
                    st.session_state.keep_decisions[r_id] = is_kept
                with c2:
                    st.markdown(
                        f"**Report ID:** `{r_id}` | **Age:** {row['age']} | **Sex:** {row['sex']} | "
                        f"**Seriousness:** {row['seriousness']} | **Day:** {row['report_day']}\n\n"
                        f"*Drugs:* `{row['drugs']}`\n\n"
                        f"*Events:* `{row['events']}`"
                    )
                    st.divider()

    st.markdown("---")

    # ----------------- Export Section -----------------
    st.subheader("3. Audit Trail & Data Export")
    exp_col1, exp_col2 = st.columns(2)

    with exp_col1:
        kept_ids = {r_id for r_id, kept in st.session_state.keep_decisions.items() if kept}
        clean_df = clusters[clusters["report_id"].astype(str).isin(kept_ids)].copy()

        st.download_button(
            label=f"📥 Export De-duplicated Dataset ({len(clean_df)} records)",
            data=clean_df.to_csv(index=False).encode("utf-8"),
            file_name="deduplicated_adr_reports.csv",
            mime="text/csv",
            use_container_width=True,
        )

    with exp_col2:
        audit_log = clusters[["report_id", "cluster_id", "cluster_size"]].copy()
        audit_log["reviewer_decision"] = audit_log["report_id"].map(
            lambda rid: "KEEP" if st.session_state.keep_decisions.get(str(rid), True) else "REMOVE_DUPLICATE"
        )

        st.download_button(
            label="📋 Export Regulatory Audit Trail (CSV)",
            data=audit_log.to_csv(index=False).encode("utf-8"),
            file_name="adr_duplicate_audit_log.csv",
            mime="text/csv",
            use_container_width=True,
        )