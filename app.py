"""
BioAuth Lab - Behavioural Biometrics Authentication Platform
Main Entry Point & Dashboard Overview
"""
import sys
from pathlib import Path
import streamlit as st

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.db import get_session, User, Sample, ModelRecord
from core.auth import authenticate_user, create_user, list_users

# Page configuration
st.set_page_config(
    page_title="BioAuth Lab - Behavioural Biometrics",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize Session State
if "authenticated" not in st.session_state:
    st.session_state["authenticated"] = False
if "username" not in st.session_state:
    st.session_state["username"] = None
if "user_id" not in st.session_state:
    st.session_state["user_id"] = None


def render_sidebar_auth():
    """Sidebar user authentication or profile switch."""
    st.sidebar.title("🧬 BioAuth Lab")
    st.sidebar.caption("Continuous Behavioral Biometrics")

    session = get_session()
    all_users = list_users(session)

    if not st.session_state["authenticated"]:
        st.sidebar.subheader("Account Login")
        auth_mode = st.sidebar.radio("Mode", ["Select Demo User", "Sign In", "Sign Up"])

        if auth_mode == "Select Demo User":
            if all_users:
                user_options = {u.username: u.id for u in all_users}
                selected_uname = st.sidebar.selectbox("Choose Enrolled User", list(user_options.keys()))
                if st.sidebar.button("Load Profile", use_container_width=True, type="primary"):
                    st.session_state["authenticated"] = True
                    st.session_state["username"] = selected_uname
                    st.session_state["user_id"] = user_options[selected_uname]
                    st.rerun()
            else:
                st.sidebar.warning("No users found. Run `python scripts/seed.py` first.")

        elif auth_mode == "Sign In":
            uname = st.sidebar.text_input("Username")
            pword = st.sidebar.text_input("Password", type="password")
            if st.sidebar.button("Sign In", use_container_width=True, type="primary"):
                user = authenticate_user(session, uname, pword)
                if user:
                    st.session_state["authenticated"] = True
                    st.session_state["username"] = user.username
                    st.session_state["user_id"] = user.id
                    st.sidebar.success(f"Welcome back, {user.username}!")
                    st.rerun()
                else:
                    st.sidebar.error("Invalid credentials.")

        elif auth_mode == "Sign Up":
            new_uname = st.sidebar.text_input("New Username")
            new_pword = st.sidebar.text_input("New Password", type="password")
            if st.sidebar.button("Register", use_container_width=True):
                try:
                    new_user = create_user(session, new_uname, new_pword)
                    st.session_state["authenticated"] = True
                    st.session_state["username"] = new_user.username
                    st.session_state["user_id"] = new_user.id
                    st.sidebar.success(f"Registered {new_user.username}!")
                    st.rerun()
                except Exception as e:
                    st.sidebar.error(str(e))
    else:
        st.sidebar.success(f"👤 Logged in as **{st.session_state['username']}**")
        st.sidebar.write(f"User ID: `{st.session_state['user_id']}`")
        if st.sidebar.button("Switch / Logout", use_container_width=True):
            st.session_state["authenticated"] = False
            st.session_state["username"] = None
            st.session_state["user_id"] = None
            st.rerun()

    session.close()


def main():
    render_sidebar_auth()

    st.title("🧬 BioAuth Lab: Behavioural Biometrics Engine")
    st.markdown("""
    **BioAuth Lab** verifies identity not by *what you know* (passwords) or *what you have* (tokens),
    but by **how you interact** with your hardware: the micro-rhythms of your keyboard keystrokes,
    pointer trajectory velocities and curvatures, and touch-screen swipe physics.
    """)

    # High-level metrics
    session = get_session()
    user_count = session.query(User).count()
    sample_count = session.query(Sample).count()
    model_count = session.query(ModelRecord).count()
    session.close()

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Enrolled Users", f"{user_count}")
    with col2:
        st.metric("Total Biometric Samples", f"{sample_count}")
    with col3:
        st.metric("Active Model Records", f"{model_count}")
    with col4:
        active_user = st.session_state["username"] or "user_01 (default)"
        st.metric("Selected Context", active_user)

    st.divider()

    st.subheader("Platform Workflow & Modalities")
    m1, m2, m3 = st.columns(3)
    with m1:
        st.info("⌨️ **Keystroke Dynamics**\n\n- Dwell times (press-to-release duration)\n- Flight times (up-down, down-down digraph latencies)\n- Typing rhythm & speed (CPS/WPM)")
    with m2:
        st.info("🖱️ **Mouse Dynamics**\n\n- Instantaneous speed, acceleration, jerk\n- Straightness & path efficiency ratios\n- Click durations & angular curvature deviations")
    with m3:
        st.info("👆 **Touch & Swipe Dynamics**\n\n- Swipe velocity profiles\n- Parabolic pressure arcs\n- Arc straightness, length & stroke duration")

    st.divider()
    st.subheader("Navigation")
    st.markdown("""
    Use the sidebar to explore:
    - 📥 **[1_Enroll](pages/1_Enroll.py)**: Collect 10 baseline samples per modality with progress indicator and redo options.
    - 🛡️ **[2_Authenticate](pages/2_Authenticate.py)**: Real-time verification with genuine probability score and modality contributions.
    - 📊 **[3_Evaluation](pages/3_Evaluation.py)**: Interactive Plotly analysis of ROC curves, EER thresholds, FAR vs. FRR, and before/after improvements.
    - ⚙️ **[4_Settings](pages/4_Settings.py)**: Retrain models, configure acceptance thresholds, and manage your data.
    """)


if __name__ == "__main__":
    main()
