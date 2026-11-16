"""
Dashboard: Model Evaluation View
================================
Displays detailed 100-episode validation results, reward distributions,
and risk/success analysis.
"""

import os
import json
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


def render_evaluation_view():
    st.header("100-Episode Evaluation & Performance Analysis")

    eval_dir = "results/evaluation"
    dqn_summary_path = os.path.join(eval_dir, "dqn_eval_summary.json")
    ppo_summary_path = os.path.join(eval_dir, "ppo_eval_summary.json")
    dqn_csv_path = os.path.join(eval_dir, "dqn_eval_100eps.csv")
    ppo_csv_path = os.path.join(eval_dir, "ppo_eval_100eps.csv")

    eval_data = {}
    if os.path.exists(dqn_summary_path):
        with open(dqn_summary_path, "r") as f:
            eval_data["DQN"] = json.load(f)
    if os.path.exists(ppo_summary_path):
        with open(ppo_summary_path, "r") as f:
            eval_data["PPO"] = json.load(f)

    if not eval_data:
        st.warning("No evaluation results found. Please execute the evaluation pipeline first.")
        return

    algo_select = st.radio("Select Model to Inspect:", list(eval_data.keys()), horizontal=True)
    summary = eval_data[algo_select]

    # Metrics row
    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Mean Reward", f"{summary.get('mean_reward', 0):.2f}", delta=f"± {summary.get('std_reward', 0):.2f}")
    m2.metric("Median (IQR)", f"{summary.get('median_reward', 0):.2f}", delta=f"IQR: {summary.get('iqr_reward', 0):.2f}")
    m3.metric("Success Rate", f"{summary.get('success_rate_percent', 0):.1f}%")
    m4.metric("Crash Rate", f"{summary.get('crash_rate_percent', 0):.1f}%")
    m5.metric("Improvement vs Random", f"+{summary.get('improvement_over_random_percent', 0):.1f}%")

    st.markdown("---")

    col_left, col_right = st.columns(2)

    with col_left:
        st.subheader("Statistical Confidence Bounds")
        st.markdown(
            f"""
            - **Sample Size ($N$)**: {summary.get('n_episodes', 100)} Independent Test Episodes
            - **95% Confidence Interval for Mean**: `[{summary.get('ci_95_low', 0):.2f}, {summary.get('ci_95_high', 0):.2f}]`
            - **Max Episode Score**: `{summary.get('max_reward', 0):.2f}`
            - **Min Episode Score**: `{summary.get('min_reward', 0):.2f}`
            - **Average Steps per Episode**: `{summary.get('mean_episode_length', 0):.1f}` steps
            """
        )

    with col_right:
        csv_file = dqn_csv_path if algo_select == "DQN" else ppo_csv_path
        if os.path.exists(csv_file):
            df = pd.read_csv(csv_file)
            st.subheader("Reward Distribution Histogram")
            fig_hist = px.histogram(
                df,
                x="reward",
                nbins=25,
                color_discrete_sequence=["#2563EB" if algo_select == "DQN" else "#059669"],
                labels={"reward": "Episode Reward"},
            )
            fig_hist.add_vline(x=200, line_dash="dash", line_color="#EF4444", annotation_text="Solved Benchmark")
            fig_hist.update_layout(template="plotly_white", height=280, showlegend=False)
            st.plotly_chart(fig_hist, use_container_width=True)

    # Episode Table
    if os.path.exists(csv_file):
        st.markdown("---")
        st.subheader(f"Detailed 100-Episode Telemetry Log ({algo_select})")
        df = pd.read_csv(csv_file)
        st.dataframe(df, use_container_width=True)
