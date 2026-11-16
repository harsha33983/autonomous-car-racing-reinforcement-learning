"""
Dashboard: Autonomous Racing Evaluation View
============================================
Displays detailed validation results, reward distributions, track progress %,
and empirical stability analysis across evaluation episodes.
"""

import os
import json
import glob
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


def load_eval_summary(eval_dir: str, prefix: str):
    json_candidates = [
        os.path.join(eval_dir, f"{prefix}_carracing_eval_summary.json"),
        os.path.join(eval_dir, f"{prefix}_eval_summary.json"),
    ]
    csv_candidates = glob.glob(os.path.join(eval_dir, f"{prefix}_carracing_eval_*.csv")) + glob.glob(os.path.join(eval_dir, f"{prefix}_eval_*.csv"))

    json_path = next((p for p in json_candidates if os.path.exists(p)), None)
    csv_path = next((p for p in csv_candidates if os.path.exists(p)), None)

    summary = None
    if json_path:
        with open(json_path, "r") as f:
            summary = json.load(f)

    df = pd.read_csv(csv_path) if csv_path else None
    return summary, df


def render_evaluation_view():
    st.header("📊 Autonomous Racing Evaluation & Empirical Stability")

    eval_dir = "results/evaluation"
    dqn_summary, dqn_df = load_eval_summary(eval_dir, "dqn")
    ppo_summary, ppo_df = load_eval_summary(eval_dir, "ppo")

    eval_data = {}
    if dqn_summary:
        eval_data["DQN (Discrete)"] = (dqn_summary, dqn_df)
    if ppo_summary:
        eval_data["PPO (Continuous)"] = (ppo_summary, ppo_df)

    if not eval_data:
        st.warning("Evaluation results not found yet. Execute the evaluation pipeline to generate metrics.")
        return

    algo_select = st.radio("Select Model to Inspect:", list(eval_data.keys()), horizontal=True)
    summary, df = eval_data[algo_select]

    # Metrics row
    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Mean Reward", f"{summary.get('mean_reward', 0):.2f}", delta=f"± {summary.get('std_reward', 0):.2f}")
    m2.metric("Median (IQR)", f"{summary.get('median_reward', 0):.2f}", delta=f"IQR: {summary.get('iqr_reward', 0):.2f}")
    m3.metric("Mean Track Progress", f"{summary.get('mean_track_progress_percent', 0):.1f}%")
    m4.metric("Max Track Progress", f"{summary.get('max_track_progress_percent', 0):.1f}%")
    m5.metric("Improvement vs Random", f"+{summary.get('improvement_over_random_percent', 0):.1f}%")

    st.markdown("---")

    col_left, col_right = st.columns(2)

    with col_left:
        st.subheader("Statistical Confidence Bounds")
        st.markdown(
            f"""
            - **Sample Size ($N$)**: {summary.get('n_episodes', 10)} Independent Evaluation Tracks
            - **95% Confidence Interval**: `[{summary.get('ci_95_low', 0):.2f}, {summary.get('ci_95_high', 0):.2f}]`
            - **Max Episode Score**: `{summary.get('max_reward', 0):.2f}`
            - **Min Episode Score**: `{summary.get('min_reward', 0):.2f}`
            - **Mean Track Steps**: `{summary.get('mean_episode_length', 0):.1f}` steps
            - **Lap Completion Rate**: `{summary.get('track_completion_rate_percent', 0):.1f}%`
            """
        )

    with col_right:
        if df is not None and not df.empty:
            st.subheader("Evaluation Return Histogram")
            color_hex = "#2563EB" if "DQN" in algo_select else "#059669"
            fig_hist = px.histogram(
                df,
                x="reward",
                nbins=15,
                color_discrete_sequence=[color_hex],
                labels={"reward": "Episode Reward"},
            )
            fig_hist.add_vline(x=0, line_dash="dot", line_color="#6B7280", annotation_text="Baseline")
            fig_hist.update_layout(template="plotly_white", height=280, showlegend=False)
            st.plotly_chart(fig_hist, use_container_width=True)

    # Detailed Table
    if df is not None and not df.empty:
        st.markdown("---")
        st.subheader(f"Detailed Telemetry Log ({algo_select})")
        st.dataframe(df, use_container_width=True)
