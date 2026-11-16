"""
Dashboard: Training Analytics - Autonomous Car Racing AI
========================================================
Interactive Plotly visualizations and metric analytics for DQN and PPO racing agents.
"""

import os
import json
import glob
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def find_racing_training_data(log_dir: str, algo_name: str):
    csv_candidates = [
        os.path.join(log_dir, f"{algo_name}_racing_metrics.csv"),
        os.path.join(log_dir, f"{algo_name}_carracing_training.csv"),
        os.path.join(log_dir, f"{algo_name}_training_metrics.csv"),
    ]
    json_candidates = [
        os.path.join(log_dir, f"{algo_name}_racing_metrics_summary.json"),
        os.path.join(log_dir, f"{algo_name}_training_metrics_summary.json"),
    ]

    csv_path = next((p for p in csv_candidates if os.path.exists(p)), None)
    json_path = next((p for p in json_candidates if os.path.exists(p)), None)

    df = pd.read_csv(csv_path) if csv_path else None
    summary = None
    if json_path:
        with open(json_path, "r") as f:
            summary = json.load(f)

    return df, summary


def render_analytics():
    st.header("📈 Autonomous Racing Training Analytics")

    dqn_log_dir = "results/training_logs/dqn"
    ppo_log_dir = "results/training_logs/ppo"

    dqn_df, dqn_sum = find_racing_training_data(dqn_log_dir, "dqn")
    ppo_df, ppo_sum = find_racing_training_data(ppo_log_dir, "ppo")

    if dqn_df is None and ppo_df is None:
        st.warning("Training logs not found yet. Train agents using the training pipeline to view live learning dynamics.")
        return

    # Algorithm selector
    available_algos = []
    if dqn_df is not None:
        available_algos.append("DQN (Discrete)")
    if ppo_df is not None:
        available_algos.append("PPO (Continuous)")
    if dqn_df is not None and ppo_df is not None:
        available_algos.append("Compare Both (PPO vs DQN)")

    selected_view = st.radio("Select Analytics View:", available_algos, horizontal=True)

    smooth_window = st.slider("Reward Smoothing Window (Episodes):", min_value=1, max_value=50, value=5, step=1)

    # Summary Cards
    if selected_view in ("DQN (Discrete)", "Compare Both (PPO vs DQN)") and dqn_sum:
        st.subheader("Discrete DQN Racing Summary")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total Timesteps", f"{dqn_sum.get('total_timesteps', 0):,}")
        c2.metric("Total Episodes", f"{dqn_sum.get('total_episodes', 0):,}")
        c3.metric("Peak Reward", f"{dqn_sum.get('max_reward', 0):.1f}")
        c4.metric("Mean Reward", f"{dqn_sum.get('mean_reward', 0):.1f}")

    if selected_view in ("PPO (Continuous)", "Compare Both (PPO vs DQN)") and ppo_sum:
        st.subheader("Continuous PPO Racing Summary")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total Timesteps", f"{ppo_sum.get('total_timesteps', 0):,}")
        c2.metric("Total Episodes", f"{ppo_sum.get('total_episodes', 0):,}")
        c3.metric("Peak Reward", f"{ppo_sum.get('max_reward', 0):.1f}")
        c4.metric("Mean Reward", f"{ppo_sum.get('mean_reward', 0):.1f}")

    st.markdown("---")

    # Interactive Learning Curve
    st.subheader("Interactive Learning Curve: Cumulative Reward vs Training Progress")
    fig = go.Figure()

    if "DQN" in selected_view and dqn_df is not None:
        rolling_dqn = dqn_df["reward"].rolling(smooth_window, min_periods=1).mean()
        fig.add_trace(go.Scatter(
            x=dqn_df["episode"],
            y=dqn_df["reward"],
            mode="lines",
            name="DQN Raw Reward",
            opacity=0.25,
            line=dict(color="#3B82F6", width=1),
        ))
        fig.add_trace(go.Scatter(
            x=dqn_df["episode"],
            y=rolling_dqn,
            mode="lines",
            name=f"DQN Rolling Mean ({smooth_window} ep)",
            line=dict(color="#1D4ED8", width=3),
        ))

    if "PPO" in selected_view and ppo_df is not None:
        rolling_ppo = ppo_df["reward"].rolling(smooth_window, min_periods=1).mean()
        fig.add_trace(go.Scatter(
            x=ppo_df["episode"],
            y=ppo_df["reward"],
            mode="lines",
            name="PPO Raw Reward",
            opacity=0.25,
            line=dict(color="#10B981", width=1),
        ))
        fig.add_trace(go.Scatter(
            x=ppo_df["episode"],
            y=rolling_ppo,
            mode="lines",
            name=f"PPO Rolling Mean ({smooth_window} ep)",
            line=dict(color="#047857", width=3),
        ))

    fig.add_hline(
        y=0,
        line_dash="dot",
        line_color="#6B7280",
        annotation_text="Zero Reward Baseline",
        annotation_position="bottom right",
    )
    fig.add_hline(
        y=700,
        line_dash="dash",
        line_color="#EF4444",
        annotation_text="High Competence Target (+700 pts)",
        annotation_position="top right",
    )

    fig.update_layout(
        xaxis_title="Training Episodes",
        yaxis_title="Cumulative Episodic Reward",
        template="plotly_white",
        height=480,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        hovermode="x unified",
    )
    st.plotly_chart(fig, use_container_width=True)

    # Track Survival Duration
    st.subheader("Autonomous Track Survival Duration (Steps per Episode)")
    fig_len = go.Figure()

    if dqn_df is not None and "DQN" in selected_view:
        fig_len.add_trace(go.Scatter(
            x=dqn_df["episode"],
            y=dqn_df["length"].rolling(smooth_window, min_periods=1).mean(),
            mode="lines",
            name="DQN Track Steps",
            line=dict(color="#2563EB", width=2.5),
        ))
    if ppo_df is not None and "PPO" in selected_view:
        fig_len.add_trace(go.Scatter(
            x=ppo_df["episode"],
            y=ppo_df["length"].rolling(smooth_window, min_periods=1).mean(),
            mode="lines",
            name="PPO Track Steps",
            line=dict(color="#059669", width=2.5),
        ))

    fig_len.update_layout(
        xaxis_title="Training Episodes",
        yaxis_title="Steps on Track",
        template="plotly_white",
        height=380,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    st.plotly_chart(fig_len, use_container_width=True)
