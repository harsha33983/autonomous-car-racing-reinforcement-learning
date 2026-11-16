"""
Dashboard: Training Analytics
=============================
Interactive Plotly visualizations and metric analytics for DQN and PPO training runs.
"""

import os
import json
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def load_training_data(log_dir: str, prefix: str):
    csv_path = os.path.join(log_dir, f"{prefix}_training_metrics.csv")
    json_path = os.path.join(log_dir, f"{prefix}_training_metrics_summary.json")

    df = pd.read_csv(csv_path) if os.path.exists(csv_path) else None
    summary = None
    if os.path.exists(json_path):
        with open(json_path, "r") as f:
            summary = json.load(f)

    return df, summary


def render_analytics():
    st.header("Training Analytics & Learning Dynamics")

    dqn_log_dir = "results/training_logs/dqn/optimized"
    ppo_log_dir = "results/training_logs/ppo/optimized"

    dqn_df, dqn_sum = load_training_data(dqn_log_dir, "dqn")
    ppo_df, ppo_sum = load_training_data(ppo_log_dir, "ppo")

    if dqn_df is None and ppo_df is None:
        st.warning("Training logs not found. Please train models first using the training pipeline.")
        return

    # Algorithm selector
    available_algos = []
    if dqn_df is not None:
        available_algos.append("DQN")
    if ppo_df is not None:
        available_algos.append("PPO")
    if dqn_df is not None and ppo_df is not None:
        available_algos.append("Compare Both (DQN vs PPO)")

    selected_view = st.radio("Select Analytics View:", available_algos, horizontal=True)

    # Rolling window slider
    smooth_window = st.slider("Reward Smoothing Window (Episodes):", min_value=5, max_value=100, value=25, step=5)

    # 1. Metric Cards
    if selected_view in ("DQN", "Compare Both (DQN vs PPO)") and dqn_sum:
        st.subheader("Deep Q-Network (DQN) Summary")
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Total Timesteps", f"{dqn_sum.get('total_timesteps', 0):,}")
        c2.metric("Total Episodes", f"{dqn_sum.get('total_episodes', 0):,}")
        c3.metric("Peak Reward", f"{dqn_sum.get('max_reward', 0):.1f}")
        c4.metric("Final Rolling Mean", f"{dqn_sum.get('final_reward_mean_50', 0):.1f}")
        c5.metric("Training Time", f"{dqn_sum.get('total_training_time_seconds', 0) / 60:.1f} min")

    if selected_view in ("PPO", "Compare Both (DQN vs PPO)") and ppo_sum:
        st.subheader("Proximal Policy Optimization (PPO) Summary")
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Total Timesteps", f"{ppo_sum.get('total_timesteps', 0):,}")
        c2.metric("Total Episodes", f"{ppo_sum.get('total_episodes', 0):,}")
        c3.metric("Peak Reward", f"{ppo_sum.get('max_reward', 0):.1f}")
        c4.metric("Final Rolling Mean", f"{ppo_sum.get('final_reward_mean_50', 0):.1f}")
        c5.metric("Training Time", f"{ppo_sum.get('total_training_time_seconds', 0) / 60:.1f} min")

    st.markdown("---")

    # 2. Interactive Learning Curve
    st.subheader("Interactive Learning Curve: Cumulative Reward vs Progress")
    fig = go.Figure()

    if selected_view in ("DQN", "Compare Both (DQN vs PPO)") and dqn_df is not None:
        rolling_dqn = dqn_df["reward"].rolling(smooth_window, min_periods=1).mean()
        fig.add_trace(go.Scatter(
            x=dqn_df["episode"],
            y=dqn_df["reward"],
            mode="lines",
            name="DQN (Raw)",
            opacity=0.2,
            line=dict(color="#3B82F6", width=1),
        ))
        fig.add_trace(go.Scatter(
            x=dqn_df["episode"],
            y=rolling_dqn,
            mode="lines",
            name=f"DQN Rolling Mean ({smooth_window} ep)",
            line=dict(color="#1D4ED8", width=3),
        ))

    if selected_view in ("PPO", "Compare Both (DQN vs PPO)") and ppo_df is not None:
        rolling_ppo = ppo_df["reward"].rolling(smooth_window, min_periods=1).mean()
        fig.add_trace(go.Scatter(
            x=ppo_df["episode"],
            y=ppo_df["reward"],
            mode="lines",
            name="PPO (Raw)",
            opacity=0.2,
            line=dict(color="#10B981", width=1),
        ))
        fig.add_trace(go.Scatter(
            x=ppo_df["episode"],
            y=rolling_ppo,
            mode="lines",
            name=f"PPO Rolling Mean ({smooth_window} ep)",
            line=dict(color="#047857", width=3),
        ))

    # Add environment solved boundary
    fig.add_hline(
        y=200,
        line_dash="dash",
        line_color="#EF4444",
        annotation_text="Solved Benchmark (+200 pts)",
        annotation_position="bottom right",
    )

    fig.update_layout(
        xaxis_title="Training Episodes",
        yaxis_title="Cumulative Episodic Reward",
        template="plotly_white",
        height=500,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        hovermode="x unified",
    )
    st.plotly_chart(fig, use_container_width=True)

    # 3. Episode Length & Survival Duration
    st.subheader("Episode Duration Dynamics (Survival & Control Horizon)")
    fig_len = go.Figure()

    if dqn_df is not None and selected_view in ("DQN", "Compare Both (DQN vs PPO)"):
        fig_len.add_trace(go.Scatter(
            x=dqn_df["episode"],
            y=dqn_df["length"].rolling(smooth_window, min_periods=1).mean(),
            mode="lines",
            name=f"DQN Steps/Ep ({smooth_window} rolling)",
            line=dict(color="#2563EB", width=2.5),
        ))
    if ppo_df is not None and selected_view in ("PPO", "Compare Both (DQN vs PPO)"):
        fig_len.add_trace(go.Scatter(
            x=ppo_df["episode"],
            y=ppo_df["length"].rolling(smooth_window, min_periods=1).mean(),
            mode="lines",
            name=f"PPO Steps/Ep ({smooth_window} rolling)",
            line=dict(color="#059669", width=2.5),
        ))

    fig_len.update_layout(
        xaxis_title="Training Episodes",
        yaxis_title="Average Episode Length (Steps)",
        template="plotly_white",
        height=400,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    st.plotly_chart(fig_len, use_container_width=True)
