"""
BioAuth Lab - Real-Time Authentication Page
Captures live behavioral samples, evaluates genuine scores (0 to 1),
computes decision verdicts, and breaks down modality contributions.
"""
import sys
from pathlib import Path
import json
import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.db import get_session, User, ModelRecord
from core.features import extract_features_for_modality
from core.models import load_user_pipeline, load_user_fusion, load_dataset
from components import capture_keystrokes, capture_mouse, capture_touch

st.set_page_config(
    page_title="Authenticate - BioAuth Lab",
    page_icon="🛡️",
    layout="wide"
)

# Plotly theme styling
PLOTLY_TEMPLATE = "plotly_dark"


def get_all_users():
    session = get_session()
    users = session.query(User).order_by(User.username.asc()).all()
    user_map = {u.username: u.id for u in users}
    session.close()
    return user_map


def main():
    st.title("🛡️ Real-Time Biometric Authentication")
    st.markdown("""
    Authenticate by performing natural behavioral interactions.
    The engine computes continuous anomaly scores and renders a security verdict with multimodal attribution.
    """)

    user_map = get_all_users()
    if not user_map:
        st.warning("No enrolled users found. Please seed or enroll a user first.")
        return

    # Configuration bar
    top_c1, top_c2, top_c3 = st.columns([2, 2, 2])
    with top_c1:
        current_uname = st.session_state.get("username")
        default_idx = 0
        user_names = list(user_map.keys())
        if current_uname and current_uname in user_names:
            default_idx = user_names.index(current_uname)

        claimed_user = st.selectbox("Claimed Identity (Profile to verify against)", user_names, index=default_idx)
        user_id = user_map[claimed_user]

    with top_c2:
        auth_modality = st.selectbox(
            "Authentication Mode",
            ["Multimodal Combined", "Keystroke Dynamics", "Mouse Dynamics", "Touch Gestures"]
        )

    with top_c3:
        threshold = st.slider(
            "Decision Threshold (τ)",
            min_value=0.0,
            max_value=1.0,
            value=0.50,
            step=0.02,
            help="Scores above or equal to τ grant access."
        )

    st.divider()

    # Session state for holding live verification scores
    if "live_auth_result" not in st.session_state:
        st.session_state["live_auth_result"] = None

    # Load pipelines
    kp = load_user_pipeline(user_id, "keystroke")
    mp = load_user_pipeline(user_id, "mouse")
    tp = load_user_pipeline(user_id, "touch")
    fus = load_user_fusion(user_id)

    if not kp and not mp and not tp:
        st.warning(f"No trained biometric models found for **{claimed_user}**. Please enroll and train models in **1_Enroll** first.")
        return

    # Split into capture and results view
    left_col, right_col = st.columns([1.1, 0.9])

    with left_col:
        st.subheader("Interactive Biometric Verification Input")

        if auth_modality == "Keystroke Dynamics":
            st.caption("Type the verification passphrase naturally.")
            ks_data = capture_keystrokes(
                target_phrase="bioauth secure access protocol",
                key=f"auth_ks_{user_id}"
            )
            if ks_data:
                process_single_modality(user_id, "keystroke", ks_data, kp, threshold)

        elif auth_modality == "Mouse Dynamics":
            st.caption("Click the three moving target circles.")
            ms_data = capture_mouse(key=f"auth_ms_{user_id}")
            if ms_data:
                process_single_modality(user_id, "mouse", ms_data, mp, threshold)

        elif auth_modality == "Touch Gestures":
            st.caption("Swipe diagonally upward across the touch pad.")
            tc_data = capture_touch(key=f"auth_tc_{user_id}")
            if tc_data:
                process_single_modality(user_id, "touch", tc_data, tp, threshold)

        else:  # Multimodal Combined
            st.caption("Complete any or all of the behavioral biometric tasks:")
            m_tabs = st.tabs(["⌨️ Keystroke", "🖱️ Mouse", "👆 Touch"])

            # Store multimodal sub-scores in session state
            if "multi_scores" not in st.session_state:
                st.session_state["multi_scores"] = {}

            with m_tabs[0]:
                k_res = capture_keystrokes(
                    target_phrase="bioauth secure access protocol",
                    key=f"multi_ks_{user_id}"
                )
                if k_res and kp:
                    feats = extract_features_for_modality("keystroke", k_res)
                    s_k = float(kp.predict_proba(pd.DataFrame([feats]).values)[0, 1])
                    st.session_state["multi_scores"]["keystroke"] = s_k
                    st.toast(f"Keystroke score evaluated: {s_k:.3f}")

            with m_tabs[1]:
                m_res = capture_mouse(key=f"multi_ms_{user_id}")
                if m_res and mp:
                    feats = extract_features_for_modality("mouse", m_res)
                    s_m = float(mp.predict_proba(pd.DataFrame([feats]).values)[0, 1])
                    st.session_state["multi_scores"]["mouse"] = s_m
                    st.toast(f"Mouse score evaluated: {s_m:.3f}")

            with m_tabs[2]:
                t_res = capture_touch(key=f"multi_tc_{user_id}")
                if t_res and tp:
                    feats = extract_features_for_modality("touch", t_res)
                    s_t = float(tp.predict_proba(pd.DataFrame([feats]).values)[0, 1])
                    st.session_state["multi_scores"]["touch"] = s_t
                    st.toast(f"Touch score evaluated: {s_t:.3f}")

            # Check if any subscores available
            multi = st.session_state["multi_scores"]
            if multi:
                s_k = multi.get("keystroke", 0.5)
                s_m = multi.get("mouse", 0.5)
                s_t = multi.get("touch", 0.5)

                if fus:
                    f_score, contribs = fus.fuse(s_k, s_m, s_t)
                else:
                    f_score = float((s_k + s_m + s_t) / 3.0)
                    contribs = {"keystroke": 33.3, "mouse": 33.3, "touch": 33.3}

                # Find which modality contributed most
                top_modality = max(contribs.keys(), key=lambda k: contribs[k])

                st.session_state["live_auth_result"] = {
                    "score": f_score,
                    "verdict": "VERIFIED" if f_score >= threshold else "REJECTED",
                    "sub_scores": {"keystroke": s_k, "mouse": s_m, "touch": s_t},
                    "contributions": contribs,
                    "top_contributor": top_modality,
                    "threshold": threshold,
                    "user": claimed_user,
                }

    with right_col:
        st.subheader("Verification Decision & Analysis")
        result = st.session_state.get("live_auth_result")

        if not result:
            st.info("Perform an interaction on the left to capture live behavioral biometrics.")
            return

        score = result["score"]
        verdict = result["verdict"]
        top_mod = result["top_contributor"]
        contribs = result["contributions"]

        # 1. Prominent Verdict Card
        if verdict == "VERIFIED":
            st.success(f"""
            ### 🟢 ACCESS GRANTED: GENUINE IDENTITY
            **Claimed Identity:** `{result['user']}`  
            **Genuine Confidence:** `{score:.4f}` (Above threshold `{threshold:.2f}`)
            """)
        else:
            st.error(f"""
            ### 🔴 ACCESS DENIED: IMPOSTOR ANOMALY DETECTED
            **Claimed Identity:** `{result['user']}`  
            **Genuine Confidence:** `{score:.4f}` (Below threshold `{threshold:.2f}`)
            """)

        # 2. Gauge Chart for Score
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=score,
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': f"Biometric Genuine Probability (τ = {threshold:.2f})", 'font': {'size': 14}},
            gauge={
                'axis': {'range': [0, 1], 'tickwidth': 1, 'tickcolor': "white"},
                'bar': {'color': "#00e676" if verdict == "VERIFIED" else "#ff5252"},
                'bgcolor': "#1e293b",
                'threshold': {
                    'line': {'color': "#ffd700", 'width': 4},
                    'thickness': 0.85,
                    'value': threshold
                }
            }
        ))
        fig_gauge.update_layout(
            template=PLOTLY_TEMPLATE,
            height=240,
            margin=dict(l=20, r=20, t=40, b=20)
        )
        st.plotly_chart(fig_gauge, use_container_width=True)

        # 3. Top Contributing Modality Callout
        st.info(f"🏆 **Highest Contributing Modality**: **{top_mod.upper()}** ({contribs.get(top_mod, 0.0):.1f}% contribution)")

        # 4. Modality Contributions Breakdown Bar Chart
        if contribs:
            c_df = pd.DataFrame([
                {"Modality": k.capitalize(), "Contribution %": v}
                for k, v in contribs.items()
            ])
            fig_bar = go.Figure(go.Bar(
                x=c_df["Contribution %"],
                y=c_df["Modality"],
                orientation="h",
                marker_color=["#ab47bc", "#29b6f6", "#26a69a"],
                text=c_df["Contribution %"].apply(lambda x: f"{x:.1f}%"),
                textposition="auto"
            ))
            fig_bar.update_layout(
                template=PLOTLY_TEMPLATE,
                title="Modality Contribution Share",
                xaxis_title="Contribution Percentage (%)",
                height=220,
                margin=dict(l=20, r=20, t=30, b=20)
            )
            st.plotly_chart(fig_bar, use_container_width=True)

        if st.button("🔄 Reset / Clear Attempt", use_container_width=True):
            st.session_state["live_auth_result"] = None
            if "multi_scores" in st.session_state:
                st.session_state["multi_scores"] = {}
            st.rerun()


def process_single_modality(user_id: int, modality: str, raw_payload: dict, pipeline, threshold: float):
    """Processes a single modality verification attempt."""
    if not pipeline:
        st.warning(f"No trained pipeline found for {modality}.")
        return

    feats = extract_features_for_modality(modality, raw_payload)
    df_feat = pd.DataFrame([feats])
    score = float(pipeline.predict_proba(df_feat.values)[0, 1])

    st.session_state["live_auth_result"] = {
        "score": score,
        "verdict": "VERIFIED" if score >= threshold else "REJECTED",
        "sub_scores": {modality: score},
        "contributions": {modality: 100.0},
        "top_contributor": modality,
        "threshold": threshold,
        "user": st.session_state.get("username", f"user_{user_id}"),
    }
    st.rerun()


if __name__ == "__main__":
    main()
