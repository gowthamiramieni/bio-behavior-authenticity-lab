"""
BioAuth Lab - Interactive Evaluation Dashboard
Plotly-powered biometric verification metrics, ROC curves, EER analysis,
model comparisons, and live threshold simulation.
"""
import sys
from pathlib import Path
import json
import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.db import get_session, User, Sample, ModelRecord, EvaluationResult
from core.models import load_dataset, compare_models, load_user_pipeline, load_user_fusion
from core.metrics import evaluate_predictions, compute_far_frr_at_threshold, compute_eer

st.set_page_config(
    page_title="Evaluation - BioAuth Lab",
    page_icon="📊",
    layout="wide"
)

# Dark theme palette for Plotly charts
PLOTLY_TEMPLATE = "plotly_dark"
COLOR_GENUINE = "#00e676"    # Green
COLOR_IMPOSTOR = "#ff5252"   # Red
COLOR_PRIMARY = "#00b0ff"    # Cyan / Blue
COLOR_ACCENT = "#ffd700"     # Gold
BG_COLOR = "#0e1117"


# -------------------------------------------------------------
# Cached Data Fetchers & Model Evaluators
# -------------------------------------------------------------
@st.cache_data(ttl=600)
def fetch_users():
    """Retrieve list of users for selection."""
    session = get_session()
    users = session.query(User).order_by(User.username.asc()).all()
    user_list = [{"id": u.id, "username": u.username} for u in users]
    session.close()
    return user_list


@st.cache_data(ttl=600)
def get_modality_eval_data(user_id: int, modality: str):
    """
    Loads dataset, trains/evaluates cross-validation scores, and produces
    prediction scores for evaluation.
    """
    X_df, y, feat_names = load_dataset(user_id=user_id, modality=modality)
    if len(X_df) < 6:
        return None, None, None

    comp = compare_models(X_df.values, y, n_splits=5)
    best_m = comp["best_model"]
    best_res = comp["results"][best_m]

    y_true = np.array(best_res["y_true"])
    y_scores = np.array(best_res["y_scores"])

    return y_true, y_scores, comp


@st.cache_data(ttl=600)
def get_multimodal_eval_data(user_id: int):
    """
    Computes genuine vs impostor scores for Multimodal Fusion.
    """
    modalities = ["keystroke", "mouse", "touch"]
    scores_dict = {}
    y_true = None

    for m in modalities:
        X_df, y, _ = load_dataset(user_id=user_id, modality=m)
        if len(X_df) < 6:
            return None, None, None
        if y_true is None:
            y_true = y
        pipe = load_user_pipeline(user_id, m)
        if pipe is not None:
            scores_dict[m] = pipe.predict_proba(X_df.values)[:, 1]
        else:
            scores_dict[m] = np.zeros(len(y))

    fus = load_user_fusion(user_id)
    weights = fus.weights if fus else [0.34, 0.33, 0.33]

    fused_scores = (
        weights[0] * scores_dict["keystroke"] +
        weights[1] * scores_dict["mouse"] +
        weights[2] * scores_dict["touch"]
    )
    return y_true, fused_scores, weights


@st.cache_data(ttl=600)
def fetch_before_after_results(user_id: int, modality: str):
    """Fetches stored before vs after improvement metrics from SQLite."""
    session = get_session()
    res_before = session.query(EvaluationResult).filter_by(
        user_id=user_id, modality=modality, stage="before_improvement"
    ).first()
    res_after = session.query(EvaluationResult).filter_by(
        user_id=user_id, modality=modality, stage="after_improvement"
    ).first()

    data = {
        "before": res_before.get_metrics() if res_before else None,
        "after": res_after.get_metrics() if res_after else None,
    }
    session.close()
    return data


# -------------------------------------------------------------
# Main Page Render
# -------------------------------------------------------------
def main():
    st.title("📊 Biometric Evaluation & Model Benchmarks")
    st.markdown("Comprehensive evaluation of behavioural biometric models, error trade-offs, and multimodal score fusion.")

    users = fetch_users()
    if not users:
        st.warning("No user data found in the database. Please run `python scripts/seed.py` first.")
        return

    # User & Modality Selectors
    top_col1, top_col2, top_col3 = st.columns([2, 2, 2])
    with top_col1:
        default_idx = 0
        current_uid = st.session_state.get("user_id")
        user_names = [u["username"] for u in users]
        if current_uid:
            matched = [i for i, u in enumerate(users) if u["id"] == current_uid]
            if matched:
                default_idx = matched[0]
        selected_uname = st.selectbox("Select User Profile", user_names, index=default_idx)
        selected_user = next(u for u in users if u["username"] == selected_uname)
        user_id = selected_user["id"]

    with top_col2:
        modality_options = ["keystroke", "mouse", "touch", "Multimodal Fusion"]
        selected_modality = st.selectbox(
            "Evaluation Modality",
            modality_options,
            format_func=lambda x: "🧬 Multimodal Fusion" if x == "Multimodal Fusion" else f"{x.capitalize()} Dynamics"
        )

    with top_col3:
        st.write("")
        st.write("")
        if st.button("🔄 Refresh Cached Metrics", use_container_width=True):
            st.cache_data.clear()
            st.rerun()

    # Load Data
    is_fusion = (selected_modality == "Multimodal Fusion")
    if is_fusion:
        y_true, y_scores, fusion_weights = get_multimodal_eval_data(user_id)
        comp_results = None
    else:
        y_true, y_scores, comp_results = get_modality_eval_data(user_id, selected_modality)
        fusion_weights = None

    if y_true is None or len(y_true) == 0:
        st.error(f"Not enough biometric samples available for user **{selected_uname}** on **{selected_modality}**.")
        return

    # Baseline EER and full metric curve calculation
    eval_data = evaluate_predictions(y_true, y_scores)
    eer_val = eval_data["eer"]
    default_thresh = eval_data["eer_threshold"]

    st.divider()

    # -------------------------------------------------------------
    # 7. Interactive Live Threshold Slider & Dynamic Metrics
    # -------------------------------------------------------------
    st.subheader("🎚️ Live Decision Threshold Simulator")
    st.caption("Adjust the decision boundary threshold to observe real-time trade-offs between Security (FAR) and Usability (FRR).")

    thresh_col, kpi_col = st.columns([2, 3])
    with thresh_col:
        threshold = st.slider(
            "Decision Threshold (τ)",
            min_value=0.0,
            max_value=1.0,
            value=float(default_thresh),
            step=0.01,
            help="Scores above or equal to this threshold are verified as GENUINE; scores below are classified as IMPOSTORS."
        )
        st.info(f"📍 **Equal Error Rate (EER) Point**: `τ = {default_thresh:.2f}` with `EER = {eer_val*100:.2f}%`")

    # Live recalculation at chosen threshold
    current_metrics = evaluate_predictions(y_true, y_scores, threshold=threshold)

    with kpi_col:
        k1, k2, k3, k4 = st.columns(4)
        k1.metric("FAR (False Accept)", f"{current_metrics['far']*100:.1f}%", help="Impostors wrongly verified as genuine")
        k2.metric("FRR (False Reject)", f"{current_metrics['frr']*100:.1f}%", help="Genuine user wrongly blocked")
        k3.metric("ROC AUC", f"{current_metrics['auc']:.4f}")
        k4.metric("Accuracy", f"{current_metrics['accuracy']*100:.1f}%")

        k5, k6, k7, k8 = st.columns(4)
        k5.metric("Precision", f"{current_metrics['precision']:.3f}")
        k6.metric("Recall (TPR)", f"{current_metrics['recall']:.3f}")
        k7.metric("F1-Score", f"{current_metrics['f1']:.3f}")
        k8.metric("Threshold τ", f"{threshold:.2f}")

    st.divider()

    # -------------------------------------------------------------
    # Charts Row 1: ROC Curve & FAR vs FRR Curve
    # -------------------------------------------------------------
    c1, c2 = st.columns(2)

    with c1:
        st.subheader("1. ROC Curve (Receiver Operating Characteristic)")
        # Plot ROC
        fpr = eval_data["roc_fpr"]
        tpr = eval_data["roc_tpr"]
        auc = eval_data["auc"]

        fig_roc = go.Figure()
        # ROC Line
        fig_roc.add_trace(go.Scatter(
            x=fpr, y=tpr,
            mode="lines",
            name=f"ROC (AUC = {auc:.4f})",
            line=dict(color=COLOR_PRIMARY, width=3)
        ))
        # Random guess diagonal
        fig_roc.add_trace(go.Scatter(
            x=[0, 1], y=[0, 1],
            mode="lines",
            name="Random Guess",
            line=dict(color="gray", dash="dash")
        ))
        # Current Operating Point
        fig_roc.add_trace(go.Scatter(
            x=[current_metrics["far"]], y=[current_metrics["recall"]],
            mode="markers",
            name=f"Current Operating Point (τ={threshold:.2f})",
            marker=dict(color=COLOR_ACCENT, size=12, symbol="star")
        ))

        fig_roc.update_layout(
            template=PLOTLY_TEMPLATE,
            xaxis_title="False Acceptance Rate (FAR / FPR)",
            yaxis_title="True Acceptance Rate (TAR / TPR = 1 - FRR)",
            legend=dict(x=0.4, y=0.1),
            margin=dict(l=40, r=20, t=30, b=40),
            height=380,
        )
        st.plotly_chart(fig_roc, use_container_width=True)

    with c2:
        st.subheader("2. Error Trade-off: FAR vs. FRR & EER Point")
        # FAR and FRR curves over thresholds
        thresh_series = np.linspace(0.0, 1.0, 101)
        far_series = []
        frr_series = []
        for t in thresh_series:
            fa, fr = compute_far_frr_at_threshold(y_true, y_scores, t)
            far_series.append(fa)
            frr_series.append(fr)

        fig_eer = go.Figure()
        fig_eer.add_trace(go.Scatter(
            x=thresh_series, y=far_series,
            mode="lines",
            name="FAR (False Acceptance Rate)",
            line=dict(color=COLOR_IMPOSTOR, width=2.5)
        ))
        fig_eer.add_trace(go.Scatter(
            x=thresh_series, y=frr_series,
            mode="lines",
            name="FRR (False Rejection Rate)",
            line=dict(color=COLOR_GENUINE, width=2.5)
        ))
        # EER intersection point
        fig_eer.add_trace(go.Scatter(
            x=[default_thresh], y=[eer_val],
            mode="markers+text",
            name=f"EER Point ({eer_val*100:.1f}%)",
            text=[f"EER={eer_val*100:.1f}%"],
            textposition="top center",
            marker=dict(color=COLOR_ACCENT, size=14, symbol="diamond")
        ))
        # Current threshold vertical marker
        fig_eer.add_vline(x=threshold, line_width=1.5, line_dash="dash", line_color="#ffffff")

        fig_eer.update_layout(
            template=PLOTLY_TEMPLATE,
            xaxis_title="Decision Threshold (τ)",
            yaxis_title="Error Rate (0.0 to 1.0)",
            legend=dict(x=0.05, y=0.95),
            margin=dict(l=40, r=20, t=30, b=40),
            height=380,
        )
        st.plotly_chart(fig_eer, use_container_width=True)

    st.divider()

    # -------------------------------------------------------------
    # Charts Row 2: Confusion Matrix & Score Distribution Histogram
    # -------------------------------------------------------------
    c3, c4 = st.columns(2)

    with c3:
        st.subheader("3. Confusion Matrix Heatmap (at τ = {:.2f})".format(threshold))
        cm = current_metrics["confusion_matrix"]
        tn, fp, fn, tp = cm[0][0], cm[0][1], cm[1][0], cm[1][1]

        z_data = [[tn, fp], [fn, tp]]
        annot_text = [
            [f"True Impostor Blocked (TN)<br><b>{tn}</b>", f"False Acceptance (FP)<br><b>{fp}</b>"],
            [f"False Rejection (FN)<br><b>{fn}</b>", f"True Genuine Verified (TP)<br><b>{tp}</b>"]
        ]

        fig_cm = ff_create_heatmap(z_data, annot_text)
        st.plotly_chart(fig_cm, use_container_width=True)

    with c4:
        st.subheader("4. Genuine vs. Impostor Score Distribution")
        genuine_scores = y_scores[y_true == 1]
        impostor_scores = y_scores[y_true == 0]

        fig_hist = go.Figure()
        fig_hist.add_trace(go.Histogram(
            x=impostor_scores,
            name="Impostor Samples",
            marker_color=COLOR_IMPOSTOR,
            opacity=0.7,
            xbins=dict(start=0.0, end=1.0, size=0.04)
        ))
        fig_hist.add_trace(go.Histogram(
            x=genuine_scores,
            name="Genuine Samples",
            marker_color=COLOR_GENUINE,
            opacity=0.7,
            xbins=dict(start=0.0, end=1.0, size=0.04)
        ))
        # Threshold divider
        fig_hist.add_vline(
            x=threshold,
            line_width=3,
            line_dash="dot",
            line_color=COLOR_ACCENT,
            annotation_text=f"τ = {threshold:.2f}",
            annotation_position="top right"
        )

        fig_hist.update_layout(
            barmode="overlay",
            template=PLOTLY_TEMPLATE,
            xaxis_title="Biometric Genuine Probability Score",
            yaxis_title="Sample Frequency Count",
            legend=dict(x=0.05, y=0.95),
            margin=dict(l=40, r=20, t=30, b=40),
            height=360,
        )
        st.plotly_chart(fig_hist, use_container_width=True)

    st.divider()

    # -------------------------------------------------------------
    # Charts Row 3: Model Comparison & Before vs After Improvement
    # -------------------------------------------------------------
    c5, c6 = st.columns(2)

    with c5:
        st.subheader("5. Model Benchmark & CV Comparison")
        if not is_fusion and comp_results:
            results_dict = comp_results["results"]
            models_list = list(results_dict.keys())
            eers = [results_dict[m]["mean_eer"] * 100 for m in models_list]
            aucs = [results_dict[m]["summary_metrics"]["auc"] * 100 for m in models_list]

            fig_comp = go.Figure(data=[
                go.Bar(name="EER % (Lower is better)", x=models_list, y=eers, marker_color="#ff7043"),
                go.Bar(name="AUC % (Higher is better)", x=models_list, y=aucs, marker_color="#42a5f5"),
            ])
            fig_comp.update_layout(
                barmode="group",
                template=PLOTLY_TEMPLATE,
                yaxis_title="Percentage (%)",
                xaxis_tickangle=-20,
                legend=dict(x=0.05, y=1.15, orientation="h"),
                margin=dict(l=40, r=20, t=30, b=40),
                height=380,
            )
            st.plotly_chart(fig_comp, use_container_width=True)

            # Tabular Comparison
            comp_table = []
            for m in models_list:
                s = results_dict[m]["summary_metrics"]
                comp_table.append({
                    "Model": m,
                    "Mean EER": f"{results_dict[m]['mean_eer']*100:.2f}%",
                    "AUC": f"{s['auc']:.4f}",
                    "Accuracy": f"{s['accuracy']*100:.1f}%",
                    "F1 Score": f"{s['f1']:.3f}",
                })
            st.dataframe(pd.DataFrame(comp_table), use_container_width=True, hide_index=True)
        else:
            # Multimodal Fusion Weights breakdown
            st.markdown(f"**Score-Level Fusion Configuration for {selected_uname}**")
            w_k, w_m, w_t = fusion_weights or [0.34, 0.33, 0.33]
            fig_pie = go.Figure(data=[go.Pie(
                labels=["Keystroke Weight", "Mouse Weight", "Touch Weight"],
                values=[w_k, w_m, w_t],
                hole=0.45,
                marker=dict(colors=["#ab47bc", "#29b6f6", "#26a69a"])
            )])
            fig_pie.update_layout(
                template=PLOTLY_TEMPLATE,
                margin=dict(l=20, r=20, t=30, b=30),
                height=320,
            )
            st.plotly_chart(fig_pie, use_container_width=True)

    with c6:
        st.subheader("6. Improvement Pipeline (Before vs. After)")
        target_mod = "keystroke" if is_fusion else selected_modality
        ba_data = fetch_before_after_results(user_id, target_mod)

        if ba_data.get("before") and ba_data.get("after"):
            b_metrics = ba_data["before"]
            a_metrics = ba_data["after"]

            metric_names = ["EER %", "FAR %", "FRR %", "Accuracy %"]
            before_vals = [b_metrics["eer"] * 100, b_metrics["far"] * 100, b_metrics["frr"] * 100, b_metrics["accuracy"] * 100]
            after_vals = [a_metrics["eer"] * 100, a_metrics["far"] * 100, a_metrics["frr"] * 100, a_metrics["accuracy"] * 100]

            fig_ba = go.Figure(data=[
                go.Bar(name="Before Pipeline (Baseline)", x=metric_names, y=before_vals, marker_color="#78909c"),
                go.Bar(name="After Pipeline (Tuned)", x=metric_names, y=after_vals, marker_color="#00e676"),
            ])
            fig_ba.update_layout(
                barmode="group",
                template=PLOTLY_TEMPLATE,
                yaxis_title="Percentage (%)",
                legend=dict(x=0.05, y=1.15, orientation="h"),
                margin=dict(l=40, r=20, t=30, b=40),
                height=380,
            )
            st.plotly_chart(fig_ba, use_container_width=True)

            delta_eer = (b_metrics["eer"] - a_metrics["eer"]) * 100
            delta_auc = (a_metrics["auc"] - b_metrics["auc"])
            st.caption(f"Pipeline: Outlier Removal $\\rightarrow$ StandardScaler $\\rightarrow$ SelectKBest (Mutual Info) $\\rightarrow$ GridSearchCV with class weighting.")
            c_d1, c_d2 = st.columns(2)
            c_d1.metric("EER Reduction", f"-{delta_eer:.2f}%", delta=f"{delta_eer:+.2f}%")
            c_d2.metric("AUC Improvement", f"{delta_auc:+.4f}", delta=f"{delta_auc:+.4f}")
        else:
            st.info("No before vs. after comparison records stored yet for this user. Run `python scripts/train_and_evaluate.py` to generate.")


def ff_create_heatmap(z, text):
    """Generates an annotated heatmap for confusion matrix."""
    fig = go.Figure(data=go.Heatmap(
        z=z,
        x=["Predicted Impostor", "Predicted Genuine"],
        y=["Actual Impostor", "Actual Genuine"],
        text=text,
        texttemplate="%{text}",
        textfont={"size": 14, "color": "white"},
        colorscale="Blues",
        showscale=False
    ))
    fig.update_layout(
        template=PLOTLY_TEMPLATE,
        margin=dict(l=40, r=20, t=30, b=40),
        height=360,
    )
    return fig


if __name__ == "__main__":
    main()
