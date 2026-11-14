"""
BioAuth Lab - User Enrollment Page
Captures at least 10 biometric baseline samples per modality with progress indicator,
redo capability, and automated model training.
"""
import sys
from pathlib import Path
import json
import uuid
import streamlit as st

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.db import get_session, User, Sample, ModelRecord
from core.features import extract_features_for_modality
from core.models import train_and_save_user_models
from components import capture_keystrokes, capture_mouse, capture_touch

st.set_page_config(
    page_title="Enroll - BioAuth Lab",
    page_icon="📥",
    layout="wide"
)

TARGET_ENROLL_COUNT = 10


def get_all_users():
    session = get_session()
    users = session.query(User).order_by(User.username.asc()).all()
    user_map = {u.username: u.id for u in users}
    session.close()
    return user_map


def get_user_sample_count(user_id: int, modality: str) -> int:
    session = get_session()
    count = session.query(Sample).filter_by(
        user_id=user_id,
        modality=modality,
        label="genuine"
    ).count()
    session.close()
    return count


def delete_last_sample(user_id: int, modality: str) -> bool:
    session = get_session()
    last_sample = (
        session.query(Sample)
        .filter_by(user_id=user_id, modality=modality, label="genuine")
        .order_by(Sample.id.desc())
        .first()
    )
    if last_sample:
        session.delete(last_sample)
        session.commit()
        session.close()
        return True
    session.close()
    return False


def save_captured_sample(user_id: int, modality: str, raw_payload: dict):
    session = get_session()
    feats = extract_features_for_modality(modality, raw_payload)

    sample = Sample(
        user_id=user_id,
        modality=modality,
        raw_data=json.dumps(raw_payload),
        features=json.dumps(feats),
        label="genuine",
        session_id=str(uuid.uuid4())[:8],
    )
    session.add(sample)
    session.commit()
    session.close()
    return feats


def main():
    st.title("📥 Biometric Enrollment Studio")
    st.markdown("""
    Enrolling establishes your behavioral baseline profile.
    Capture at least **10 genuine samples** for each modality to train robust biometric models.
    """)

    # User Profile Selection
    user_map = get_all_users()
    if not user_map:
        st.warning("No users registered in system. Please run `python scripts/seed.py` or register in Home.")
        return

    top_c1, top_c2 = st.columns([2, 4])
    with top_c1:
        current_uname = st.session_state.get("username")
        default_idx = 0
        user_names = list(user_map.keys())
        if current_uname and current_uname in user_names:
            default_idx = user_names.index(current_uname)

        selected_user = st.selectbox("Active Enrollment Profile", user_names, index=default_idx)
        user_id = user_map[selected_user]

    st.divider()

    # Modality Tabs
    tab_keys, tab_mouse, tab_touch = st.tabs([
        "⌨️ 1. Keystroke Dynamics",
        "🖱️ 2. Mouse Dynamics",
        "👆 3. Touch Dynamics"
    ])

    # -------------------------------------------------------------
    # TAB 1: KEYSTROKE ENROLLMENT
    # -------------------------------------------------------------
    with tab_keys:
        k_count = get_user_sample_count(user_id, "keystroke")
        render_modality_enrollment_ui(
            user_id=user_id,
            modality="keystroke",
            sample_count=k_count,
            title="Keystroke Dynamics Enrollment",
            instruction="Type the passphrase exactly as shown. Maintain your natural typing cadence.",
            capture_fn=lambda: capture_keystrokes(
                target_phrase="bioauth secure access protocol",
                key=f"enroll_ks_{user_id}_{k_count}"
            )
        )

    # -------------------------------------------------------------
    # TAB 2: MOUSE ENROLLMENT
    # -------------------------------------------------------------
    with tab_mouse:
        m_count = get_user_sample_count(user_id, "mouse")
        render_modality_enrollment_ui(
            user_id=user_id,
            modality="mouse",
            sample_count=m_count,
            title="Mouse Dynamics Enrollment",
            instruction="Move your cursor and click the three numbered targets in order (1 -> 2 -> 3).",
            capture_fn=lambda: capture_mouse(
                key=f"enroll_ms_{user_id}_{m_count}"
            )
        )

    # -------------------------------------------------------------
    # TAB 3: TOUCH ENROLLMENT
    # -------------------------------------------------------------
    with tab_touch:
        t_count = get_user_sample_count(user_id, "touch")
        render_modality_enrollment_ui(
            user_id=user_id,
            modality="touch",
            sample_count=t_count,
            title="Touch & Swipe Dynamics Enrollment",
            instruction="Swipe diagonally upward across the pad with a single fluid motion.",
            capture_fn=lambda: capture_touch(
                key=f"enroll_tc_{user_id}_{t_count}"
            )
        )


def render_modality_enrollment_ui(user_id: int, modality: str, sample_count: int, title: str, instruction: str, capture_fn):
    """Generic enrollment UI block for any biometric modality."""
    st.subheader(title)
    st.caption(instruction)

    # Progress Indicator
    prog_col1, prog_col2, prog_col3 = st.columns([3, 1.5, 1.5])
    with prog_col1:
        clamped_progress = min(sample_count / float(TARGET_ENROLL_COUNT), 1.0)
        st.progress(clamped_progress)
        if sample_count < TARGET_ENROLL_COUNT:
            st.markdown(f"**Progress**: Sample **{sample_count} of {TARGET_ENROLL_COUNT}** ({TARGET_ENROLL_COUNT - sample_count} more needed)")
        else:
            st.success(f"✓ **Enrollment Complete**: {sample_count} samples stored (Meets minimum of {TARGET_ENROLL_COUNT})")

    with prog_col2:
        if st.button(f"↩️ Redo Last ({modality})", key=f"redo_{modality}", help="Delete the most recently recorded sample"):
            if delete_last_sample(user_id, modality):
                st.toast(f"Removed latest {modality} sample.")
                st.rerun()
            else:
                st.warning("No samples to undo.")

    with prog_col3:
        if sample_count >= TARGET_ENROLL_COUNT:
            if st.button(f"⚡ Train Models", key=f"train_{modality}", type="primary"):
                with st.spinner("Training and tuning biometric models..."):
                    res = train_and_save_user_models(user_id)
                    st.success(f"Models successfully trained for {modality}!")
                    st.cache_data.clear()
                    st.rerun()

    st.write("")

    # State tracking to avoid double-processing the same browser event
    last_processed_key = f"last_processed_{modality}_{user_id}"
    if last_processed_key not in st.session_state:
        st.session_state[last_processed_key] = None

    # Render interactive capture component
    captured_data = capture_fn()

    if captured_data is not None:
        t_stamp = captured_data.get("timestamp")
        if t_stamp != st.session_state[last_processed_key]:
            st.session_state[last_processed_key] = t_stamp
            with st.spinner("Extracting features and saving biometric sample..."):
                extracted_feats = save_captured_sample(user_id, modality, captured_data)
                st.toast(f"Sample recorded! ({sample_count + 1} of {TARGET_ENROLL_COUNT})", icon="✅")
                st.success(f"Recorded new genuine sample for **{modality}**! Extracted {len(extracted_feats)} features.")
                with st.expander("Inspect Extracted Feature Vector", expanded=False):
                    st.json(extracted_feats)
                st.rerun()


if __name__ == "__main__":
    main()
