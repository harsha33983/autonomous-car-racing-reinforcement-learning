"""
Autonomous Car Racing AI - Interactive Streamlit Dashboard
==========================================================
Main entry point for autonomous racing web application.
"""

import os
import glob
import streamlit as st

from dashboard.overview import render_overview
from dashboard.analytics import render_analytics
from dashboard.comparison import render_comparison
from dashboard.simulation import render_simulation
from dashboard.evaluation_view import render_evaluation_view
from dashboard.model_info import render_model_info


# Page Configuration
st.set_page_config(
    page_title="Autonomous Car Racing AI",
    page_icon="🏎️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom Styling
st.markdown(
    """
    <style>
    .main {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }
    .metric-card {
        background-color: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-radius: 8px;
        padding: 16px;
        text-align: center;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 4px;
        padding: 8px 16px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


def main():
    # Sidebar
    st.sidebar.title("🏎️ Car Racing AI")
    st.sidebar.caption("Autonomous Reinforcement Learning")

    # Quick Status
    dqn_models = glob.glob("models/dqn/**/*.zip", recursive=True) + glob.glob("models/dqn/*.zip")
    ppo_models = glob.glob("models/ppo/**/*.zip", recursive=True) + glob.glob("models/ppo/*.zip")

    st.sidebar.markdown("### Model Status")
    if dqn_models:
        st.sidebar.success(f"DQN Models: {len(dqn_models)} found")
    else:
        st.sidebar.warning("DQN Models: None")

    if ppo_models:
        st.sidebar.success(f"PPO Models: {len(ppo_models)} found")
    else:
        st.sidebar.warning("PPO Models: None")

    st.sidebar.markdown("---")
    st.sidebar.markdown("### Navigation")

    nav_options = [
        "A. Project Overview",
        "B. Training Analytics",
        "C. Algorithm Comparison",
        "D. Live Racing Simulation",
        "E. Model Evaluation",
        "F. Model Architecture & Registry",
    ]
    selected_page = st.sidebar.radio("Select View:", nav_options, index=0)

    st.sidebar.markdown("---")
    st.sidebar.markdown(
        """
        **System Specs**:
        - **Gymnasium**: `1.4.0`
        - **PyTorch**: `2.13.0`
        - **Environment**: Box2D CarRacing-v3
        - **Algorithms**: Continuous PPO & Discrete DQN
        """
    )

    # Route Page
    if "Overview" in selected_page:
        render_overview()
    elif "Analytics" in selected_page:
        render_analytics()
    elif "Comparison" in selected_page:
        render_comparison()
    elif "Simulation" in selected_page:
        render_simulation()
    elif "Evaluation" in selected_page:
        render_evaluation_view()
    elif "Architecture" in selected_page:
        render_model_info()


if __name__ == "__main__":
    main()
