"""
BioAuth Lab - Settings, Model Management & Privacy Controls
Allows users to retrain biometric pipelines, tune decision thresholds,
and manage or delete personal biometric data.
"""
import sys
from pathlib import Path
import json
import streamlit as st

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.db import get_session, User, Sample, ModelRecord, EvaluationResult, DEFAULT_DB_PATH
from core.models import (
    train_and_save_user_models,
    run_improvement_benchmark,
    load_user_fusion,
    MODELS_DIR
)
from core.metrics import compute_eer

st.set_page_config(
    page_title="Settings - BioAuth Lab",
    page_icon="⚙️",
    layout="wide"
)


def get_all_users():
    session = get_session()
    users = session.query(User).order_by(User.username.asc()).all()
    user_map = {u.username: u.id for u in users}
    session.close()
    return user_map


def main():
    st.title("⚙️ System Settings & Biometric Management")
    st.markdown("Configure operational decision thresholds, retrain machine learning models, and control biometric data privacy.")

    user_map = get_all_users()
    if not user_map:
        st.warning("No registered users found in the system.")
        return

    # User Profile Switcher
    top_col1, top_col2 = st.columns([2, 4])
    with top_col1:
        current_uname = st.session_state.get("username")
        default_idx = 0
        user_names = list(user_map.keys())
        if current_uname and current_uname in user_names:
            default_idx = user_names.index(current_uname)

        selected_user = st.selectbox("Active Profile", user_names, index=default_idx)
        user_id = user_map[selected_user]

    st.divider()

    # Three functional sections
    sec_col1, sec_col2 = st.columns(2)

    # -------------------------------------------------------------
    # SECTION 1: MODEL RETRAINING STUDIO
    # -------------------------------------------------------------
    with sec_col1:
        st.subheader("🤖 Model Retraining & Tuning")
        st.markdown(f"Retrain classification pipelines and multimodal fusion weights for **{selected_user}**.")

        session = get_session()
        sample_counts = {
            "keystroke": session.query(Sample).filter_by(user_id=user_id, modality="keystroke").count(),
            "mouse": session.query(Sample).filter_by(user_id=user_id, modality="mouse").count(),
            "touch": session.query(Sample).filter_by(user_id=user_id, modality="touch").count(),
        }
        active_models = session.query(ModelRecord).filter_by(user_id=user_id).all()
        session.close()

        st.write(f"**Available Training Samples**:")
        sc1, sc2, sc3 = st.columns(3)
        sc1.metric("Keystroke", sample_counts["keystroke"])
        sc2.metric("Mouse", sample_counts["mouse"])
        sc3.metric("Touch", sample_counts["touch"])

        tune_hyperparams = st.checkbox("Run Extensive GridSearch Hyperparameter Tuning", value=True)

        if st.button("🚀 Retrain All Modality Pipelines & Fusion", type="primary", use_container_width=True):
            if sum(sample_counts.values()) < 6:
                st.error("Insufficient samples to train models. Please record samples in **1_Enroll** first.")
            else:
                with st.spinner("Retraining pipelines, calibrating anomaly wrappers, and tuning fusion weights..."):
                    try:
                        summary = train_and_save_user_models(user_id)
                        # Also run before vs after benchmark
                        for mod in ["keystroke", "mouse", "touch"]:
                            run_improvement_benchmark(user_id, mod)

                        st.success(f"Successfully retrained models for **{selected_user}**!")
                        fusion_res = summary.get("fusion", {})
                        if fusion_res:
                            st.write(f"**Updated Fusion Weights**: {fusion_res.get('weights')}")
                            f_metrics = fusion_res.get("metrics", {})
                            st.write(f"**Validation EER**: `{f_metrics.get('eer', 0.0)*100:.2f}%` | **AUC**: `{f_metrics.get('auc', 1.0):.4f}`")

                        st.cache_data.clear()
                    except Exception as e:
                        st.error(f"Retraining failed: {str(e)}")

        st.write("")
        st.write("**Active Deployed Model Records:**")
        if active_models:
            model_info = []
            for m in active_models:
                m_metrics = m.get_metrics()
                model_info.append({
                    "Modality": m.modality.capitalize(),
                    "Classifier Architecture": m.model_type,
                    "Saved Path": Path(m.filepath).name,
                })
            st.dataframe(model_info, use_container_width=True, hide_index=True)
        else:
            st.info("No trained models recorded for this user.")

    # -------------------------------------------------------------
    # SECTION 2: DECISION THRESHOLD CONFIGURATION
    # -------------------------------------------------------------
    with sec_col2:
        st.subheader("🎚️ Decision Threshold Management")
        st.markdown("Set the global authentication acceptance threshold $\\tau$. Scores $\\ge \\tau$ grant access.")

        if "user_threshold" not in st.session_state:
            st.session_state["user_threshold"] = 0.50

        current_thresh = st.session_state["user_threshold"]
        new_thresh = st.slider(
            "Global Acceptance Threshold (τ)",
            min_value=0.05,
            max_value=0.95,
            value=float(current_thresh),
            step=0.01,
            help="Higher values increase security (lower FAR) but increase false rejections (higher FRR)."
        )
        st.session_state["user_threshold"] = new_thresh

        st.info(f"""
        **Operating Policy**:
        - Current Threshold: **{new_thresh:.2f}**
        - Mode: **Balanced Biometric Authentication**
        """)

        # Auto-calibration to EER
        if st.button("⚖️ Auto-Calibrate to Equal Error Rate (EER)", use_container_width=True):
            fus = load_user_fusion(user_id)
            if fus:
                session = get_session()
                fusion_rec = session.query(ModelRecord).filter_by(user_id=user_id, modality="fusion").first()
                session.close()
                if fusion_rec:
                    metrics = fusion_rec.get_metrics()
                    eer_thresh = metrics.get("eer_threshold", 0.50)
                    st.session_state["user_threshold"] = eer_thresh
                    st.success(f"Calibrated threshold to optimal EER point: **{eer_thresh:.2f}**")
                    st.rerun()
            st.info("Calibrated to default standard threshold 0.50.")

    st.divider()

    # -------------------------------------------------------------
    # SECTION 3: PRIVACY & DATA PURGING CONTROLS
    # -------------------------------------------------------------
    st.subheader("🗑️ Privacy & Data Purge Controls")
    st.markdown("Under biometric privacy standards, users retain the right to audit, export, or permanently delete their behavioral signatures.")

    p1, p2, p3 = st.columns(3)

    with p1:
        st.markdown("##### Delete Specific Modality")
        purge_mod = st.selectbox("Select Modality", ["keystroke", "mouse", "touch"])
        if st.button(f"Purge {purge_mod.capitalize()} Samples", use_container_width=True):
            session = get_session()
            deleted = session.query(Sample).filter_by(user_id=user_id, modality=purge_mod).delete()
            session.commit()
            session.close()
            st.toast(f"Deleted {deleted} {purge_mod} samples.", icon="🗑️")
            st.cache_data.clear()
            st.rerun()

    with p2:
        st.markdown("##### Wipe Entire User Profile")
        confirm_wipe = st.checkbox(f"Confirm deletion of ALL data for {selected_user}", value=False)
        if st.button("Permanently Purge Profile", type="primary", disabled=not confirm_wipe, use_container_width=True):
            session = get_session()
            user_obj = session.query(User).filter_by(id=user_id).first()
            if user_obj:
                # Delete files from models_store
                for mod_file in MODELS_DIR.glob(f"model_user_{user_id}_*.joblib"):
                    try:
                        mod_file.unlink()
                    except Exception:
                        pass
                session.delete(user_obj)
                session.commit()
                st.success(f"User {selected_user} and all associated biometric records have been permanently deleted.")
                st.session_state["username"] = None
                st.session_state["user_id"] = None
                st.cache_data.clear()
                st.rerun()
            session.close()

    with p3:
        st.markdown("##### System Diagnostics")
        session = get_session()
        total_u = session.query(User).count()
        total_s = session.query(Sample).count()
        total_m = session.query(ModelRecord).count()
        total_r = session.query(EvaluationResult).count()
        session.close()

        db_size_kb = (DEFAULT_DB_PATH.stat().st_size / 1024) if DEFAULT_DB_PATH.exists() else 0.0

        st.write(f"- Database Path: `{DEFAULT_DB_PATH.name}` ({db_size_kb:.1f} KB)")
        st.write(f"- Total Users: `{total_u}`")
        st.write(f"- Total Stored Samples: `{total_s}`")
        st.write(f"- Active Model Files: `{total_m}`")
        st.write(f"- Benchmark Results: `{total_r}`")


if __name__ == "__main__":
    main()
