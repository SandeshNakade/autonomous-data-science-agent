import streamlit as st
import pandas as pd
import json
from pathlib import Path
import os
from datetime import datetime

# Import AutoDS modules
from autods.orchestrator import AutonomousDataScienceOrchestrator
from autods.schemas import RunConfig, RunStatus, TaskType
from autods.registry import LocalModelRegistry
from autods.profiling import profile_dataset
from autods.monitoring import build_drift_baseline, compare_to_baseline, retraining_request, DriftBaseline
from autods.tracking import LocalExperimentTracker

# Configure page
st.set_page_config(page_title="AutoDS Agent", layout="wide", page_icon="🤖")

# Initialize directories
ARTIFACTS_DIR = Path("artifacts")
MODELS_DIR = Path("models")
DATASETS_DIR = Path("datasets")
RUNS_DIR = ARTIFACTS_DIR / "runs"

for d in [ARTIFACTS_DIR, MODELS_DIR, DATASETS_DIR, RUNS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# Helper function to save uploaded file
def save_uploaded_file(uploaded_file, dest_dir: Path) -> Path:
    dest_path = dest_dir / uploaded_file.name
    with open(dest_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    return dest_path

st.title("🤖 Autonomous Data Science Agent Dashboard")
st.markdown("A comprehensive control center for automated data science experimentation, tracking, and monitoring.")

# Create tabs
tab_new_exp, tab_history, tab_registry, tab_monitoring, tab_explorer = st.tabs([
    "New Experiment", 
    "Run History", 
    "Model Registry", 
    "Monitoring & Drift", 
    "Dataset Explorer"
])

# ==========================================
# TAB 1: New Experiment
# ==========================================
with tab_new_exp:
    st.header("Start a New ML Experiment")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        uploaded_csv = st.file_uploader("Upload Dataset (CSV)", type="csv", key="new_exp_csv")
        objective = st.text_area("ML Objective", placeholder="E.g., Predict whether a customer will churn based on their profile.", height=100)
    
    with col2:
        st.subheader("Run Configuration")
        target_col = st.text_input("Target Column (Optional)", placeholder="Let the agent infer if blank")
        acc_threshold = st.number_input("Acceptance Threshold", value=0.85, step=0.01, min_value=0.0, max_value=1.0)
        max_exp = st.slider("Max Experiments", min_value=1, max_value=5, value=3)
    
    if st.button("Run Experiment", type="primary"):
        if not uploaded_csv or not objective:
            st.error("Please provide both a dataset and an objective.")
        else:
            with st.spinner("Agent is analyzing the dataset and running experiments..."):
                try:
                    # Save dataset
                    dataset_path = save_uploaded_file(uploaded_csv, DATASETS_DIR)
                    
                    # Configure run
                    config = RunConfig(
                        max_experiments=max_exp,
                        target_column=target_col if target_col else None,
                        acceptance_threshold=acc_threshold
                    )
                    
                    # Run orchestrator
                    orchestrator = AutonomousDataScienceOrchestrator()
                    summary = orchestrator.run(dataset_path=dataset_path, objective=objective, config=config)
                    
                    st.session_state['latest_run'] = summary
                    st.success(f"Run completed with status: {summary.status.name}")
                except Exception as e:
                    st.error(f"Error during run: {e}")

    # Display results if available
    if 'latest_run' in st.session_state:
        summary = st.session_state['latest_run']
        
        st.divider()
        st.subheader(f"Run Details: {summary.run_id}")
        
        # Status Badge
        status_colors = {
            RunStatus.NEEDS_REVIEW: "green",
            RunStatus.STOPPED: "orange",
            RunStatus.FAILED: "red",
            RunStatus.ACCEPTED: "blue"
        }
        color = status_colors.get(summary.status, "gray")
        st.markdown(f"**Status:** :{color}[**{summary.status.name}**] - {summary.stop_reason}")
        
        col_plan, col_results = st.columns([1, 1.5])
        
        with col_plan:
            if summary.plan:
                st.markdown("### Agent's ML Plan")
                st.markdown(f"**Task Type:** {summary.plan.task_type.name}")
                st.markdown(f"**Target:** `{summary.plan.target_column}`")
                st.markdown(f"**Metric:** {summary.plan.metric} (Threshold: {summary.plan.acceptance_threshold})")
                st.markdown(f"**Candidates:** {', '.join(summary.plan.candidate_models)}")
                
                with st.expander("Agent Reasoning"):
                    for r in summary.plan.reasoning:
                        st.markdown(f"- {r}")
        
        with col_results:
            if summary.experiments:
                st.markdown("### Experiment Results")
                
                # Format experiments for table
                exp_data = []
                for exp in summary.experiments:
                    exp_data.append({
                        "Model": exp.model_name,
                        "Metric": exp.metric,
                        "Score": exp.score,
                        "Time (s)": exp.elapsed_seconds,
                        "Strategy": exp.strategy,
                        "Status": exp.status
                    })
                df_exp = pd.DataFrame(exp_data)
                st.dataframe(df_exp, width="stretch")
                
                # Chart
                if not df_exp['Score'].isnull().all():
                    chart_data = df_exp.dropna(subset=['Score']).set_index("Model")[["Score"]]
                    st.bar_chart(chart_data)
        
        st.divider()
        col_trace, col_promo = st.columns([1, 1])
        
        with col_trace:
            with st.expander("View Decision Trace"):
                st.json(summary.decision_trace)
            
            if summary.final_evaluation:
                st.markdown("### Final Holdout Evaluation")
                st.metric(label=f"Holdout {summary.final_evaluation.metric}", value=f"{summary.final_evaluation.score:.4f}")
        
        with col_promo:
            if summary.promotion_proposal_id:
                st.info(f"Model proposed for promotion: **{summary.promotion_proposal_id}**")
                approver = st.text_input("Approver Name", key="approver_new")
                if st.button("Approve Proposal"):
                    if approver:
                        registry = LocalModelRegistry(MODELS_DIR)
                        try:
                            registry.approve(summary.promotion_proposal_id, approver)
                            st.success("Model successfully approved for production!")
                        except Exception as e:
                            st.error(f"Approval failed: {e}")
                    else:
                        st.warning("Please provide an approver name.")

# ==========================================
# TAB 2: Run History
# ==========================================
with tab_history:
    st.header("Experiment History")
    
    if RUNS_DIR.exists():
        run_files = sorted(RUNS_DIR.glob("*/run.json"), key=lambda p: p.stat().st_mtime, reverse=True)
        if not run_files:
            st.info("No runs found. Start a new experiment from the first tab.")
        else:
            runs_data = []
            runs_dict = {}
            for f in run_files:
                try:
                    data = json.loads(f.read_text(encoding="utf-8"))
                    run_id = data.get("run_id", f.parent.name)
                    runs_dict[run_id] = data
                    
                    objective = data.get("objective", "N/A")
                    started = data.get("started_at", "")
                    
                    runs_data.append({
                        "Run ID": run_id,
                        "Started": started[:19] if started else "Unknown",
                        "Status": data.get("status", "unknown"),
                        "Objective": objective[:60] + "…" if len(objective) > 60 else objective,
                    })
                except Exception:
                    pass
            
            if runs_data:
                df_runs = pd.DataFrame(runs_data)
                st.dataframe(df_runs, width="stretch")
                
                selected_run = st.selectbox("Select a run to view details", df_runs["Run ID"])
                if selected_run and selected_run in runs_dict:
                    run_data = runs_dict[selected_run]
                    
                    # Show structured summary
                    col_s1, col_s2, col_s3 = st.columns(3)
                    col_s1.metric("Status", run_data.get("status", "unknown"))
                    col_s2.metric("Experiments", len(run_data.get("experiments", [])))
                    plan = run_data.get("plan")
                    col_s3.metric("Task Type", plan.get("task_type", "N/A") if plan else "N/A")
                    
                    # Experiments table
                    experiments = run_data.get("experiments", [])
                    if experiments:
                        exp_rows = [{
                            "Model": e.get("model_name"),
                            "Score": e.get("score"),
                            "Metric": e.get("metric"),
                            "Time (s)": e.get("elapsed_seconds"),
                            "Status": e.get("status"),
                        } for e in experiments]
                        st.dataframe(pd.DataFrame(exp_rows), width="stretch")
                    
                    with st.expander("Full Run JSON"):
                        st.json(run_data)
    else:
        st.info("Artifacts directory not found.")


# ==========================================
# TAB 3: Model Registry
# ==========================================
with tab_registry:
    st.header("Model Registry")
    
    registry = LocalModelRegistry(MODELS_DIR)
    
    st.subheader("Production Model")
    try:
        prod_meta = registry.production_metadata()
        if prod_meta:
            meta_info = prod_meta.get("metadata", {})
            col1, col2, col3 = st.columns(3)
            col1.metric("Model Name", meta_info.get("model_name", "Unknown"))
            col2.metric("Run ID", str(meta_info.get("run_id", "Unknown"))[:12])
            col3.metric("Approved By", prod_meta.get("approved_by", "Unknown"))
            with st.expander("Production Model Details"):
                st.json(prod_meta)
        else:
            st.info("No model currently in production.")
    except Exception as e:
        st.info("No production model found.")
        
    st.divider()
    st.subheader("Pending Proposals")
    
    pending_dir = registry.pending_root
    if pending_dir.exists():
        proposals = []
        for proposal_dir in sorted(pending_dir.iterdir(), reverse=True):
            meta_file = proposal_dir / "proposal.json"
            if meta_file.is_file():
                try:
                    meta = json.loads(meta_file.read_text(encoding="utf-8"))
                    if meta.get("status") == "pending_human_approval":
                        proposals.append(meta)
                except Exception:
                    pass
        
        if proposals:
            for p in proposals:
                proposal_id = p.get("proposal_id", "unknown")
                model_name = p.get("metadata", {}).get("model_name", "unknown")
                with st.expander(f"Proposal: {proposal_id[:12]}… — {model_name}"):
                    # Show key metrics
                    metadata = p.get("metadata", {})
                    col_p1, col_p2, col_p3 = st.columns(3)
                    col_p1.metric("Model", metadata.get("model_name", "N/A"))
                    col_p2.metric("Validation Score", f"{metadata.get('validation_score', 'N/A')}")
                    col_p3.metric("Holdout Score", f"{metadata.get('holdout_score', 'N/A')}")
                    
                    st.json(p)
                    approver_reg = st.text_input("Approver", key=f"app_{proposal_id}")
                    if st.button("Approve", key=f"btn_{proposal_id}"):
                        if approver_reg:
                            try:
                                registry.approve(proposal_id, approver_reg)
                                st.success("Approved! Refresh to see in production.")
                            except Exception as e:
                                st.error(f"Error: {e}")
                        else:
                            st.warning("Enter approver name.")
        else:
            st.info("No pending proposals.")
    else:
        st.info("No pending proposals.")


# ==========================================
# TAB 4: Monitoring & Drift
# ==========================================
with tab_monitoring:
    st.header("Production Data Monitoring")
    
    drift_csv = st.file_uploader("Upload Current Data for Drift Analysis", type="csv", key="drift_csv")
    threshold_input = st.slider("Drift Threshold", min_value=0.01, max_value=0.50, value=0.10, step=0.01)
    
    if st.button("Analyze Drift"):
        if not drift_csv:
            st.warning("Upload a dataset first.")
        else:
            with st.spinner("Analyzing drift..."):
                try:
                    registry = LocalModelRegistry(MODELS_DIR)
                    baseline_raw = registry.production_baseline()
                    
                    if not baseline_raw:
                        st.error("No production baseline found. Cannot compute drift.")
                    else:
                        baseline = DriftBaseline(**baseline_raw)
                        current_path = save_uploaded_file(drift_csv, DATASETS_DIR)
                        report = compare_to_baseline(baseline, current_path, threshold=threshold_input)
                        
                        if report.drift_detected:
                            st.error(f"🚨 Data Drift Detected in {len(report.drifted_columns)} column(s): {', '.join(report.drifted_columns)}")
                        else:
                            st.success("✅ No significant data drift detected.")
                            
                        # Show summary metrics
                        col_m1, col_m2 = st.columns(2)
                        col_m1.metric("Drifted Columns", len(report.drifted_columns))
                        col_m2.metric("Missing Columns", len(report.missing_columns))
                        
                        # Retraining request
                        if report.drift_detected:
                            try:
                                prod_meta = registry.production_metadata()
                                run_id = prod_meta.get("metadata", {}).get("run_id", "unknown")
                            except Exception:
                                run_id = "unknown"
                            retrain_req = retraining_request(report, run_id)
                            st.warning("Retraining Request Generated (Awaiting Operator Action):")
                            st.json(retrain_req)
                            
                        with st.expander("Detailed Drift Report JSON"):
                            st.json(report.to_dict())
                except Exception as e:
                    st.error(f"Error analyzing drift: {e}")

# ==========================================
# TAB 5: Dataset Explorer
# ==========================================
with tab_explorer:
    st.header("Dataset Explorer")
    
    explore_csv = st.file_uploader("Upload Dataset to Profile", type="csv", key="explore_csv")
    
    if explore_csv:
        exp_path = save_uploaded_file(explore_csv, DATASETS_DIR)
        
        with st.spinner("Profiling dataset..."):
            try:
                profile = profile_dataset(exp_path)
                df_exp = pd.read_csv(exp_path)
                
                col1, col2, col3 = st.columns(3)
                col1.metric("Rows", profile.rows)
                col2.metric("Columns", len(profile.columns))
                col3.metric("Likely Targets", len(profile.likely_targets))
                
                st.subheader("Data Preview")
                st.dataframe(df_exp.head(10), width="stretch")
                
                st.subheader("Column Profiles")
                col_data = []
                for c in profile.columns:
                    col_data.append({
                        "Name": c.name,
                        "Type": c.inferred_type,
                        "DType": c.pandas_dtype,
                        "Missing (%)": f"{c.missing_fraction:.2%}",
                        "Unique Count": c.unique_count,
                        "Unique (%)": f"{c.unique_fraction:.2%}",
                        "Identifier?": c.likely_identifier
                    })
                st.dataframe(pd.DataFrame(col_data), width="stretch")
                
                if profile.likely_targets:
                    st.info(f"Potential Target Columns: {', '.join(profile.likely_targets)}")
                    
            except Exception as e:
                st.error(f"Error profiling dataset: {e}")

