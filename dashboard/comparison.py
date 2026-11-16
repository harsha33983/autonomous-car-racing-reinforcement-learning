"""
Dashboard: Algorithm Comparison
===============================
Comparative benchmark analysis between DQN, PPO, and Random baseline,
including statistical significance testing and empirical distributions.
"""

import os
import json
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px


def render_comparison():
    st.header("Algorithm Benchmark & Statistical Comparison")

    comp_csv = "results/evaluation/algorithm_comparison.csv"
    comp_json = "results/evaluation/algorithm_comparison.json"
    dist_csv = "results/evaluation/combined_eval_episodes.csv"

    if not os.path.exists(comp_csv) or not os.path.exists(comp_json):
        st.warning("Comparative evaluation data not found. Run evaluation pipeline to generate results.")
        return

    with open(comp_json, "r") as f:
        comp_data = json.load(f)

    comp_df = pd.read_csv(comp_csv)
    dist_df = pd.read_csv(dist_csv) if os.path.exists(dist_csv) else None

    # Comparative Metrics Table
    st.subheader("1. Standardized 100-Episode Evaluation Metrics")
    st.dataframe(
        comp_df.style.highlight_max(subset=["Mean Reward", "Median Reward", "Success Rate (%)"], color="#DCFCE7")
                     .highlight_min(subset=["Crash Rate (%)"], color="#DCFCE7"),
        use_container_width=True,
    )

    # Bar Charts
    col1, col2 = st.columns(2)
    colors = ["#94A3B8", "#2563EB", "#059669"]

    with col1:
        fig_bar = go.Figure(data=[
            go.Bar(
                x=comp_df["Algorithm"],
                y=comp_df["Mean Reward"],
                error_y=dict(type="data", array=comp_df["Std Reward"]),
                marker_color=colors,
                text=comp_df["Mean Reward"].round(1),
                textposition="auto",
            )
        ])
        fig_bar.add_hline(y=200, line_dash="dash", line_color="#EF4444", annotation_text="Solved (+200)")
        fig_bar.update_layout(
            title="Mean Evaluation Return (+/- 1 Std Dev)",
            yaxis_title="Reward",
            template="plotly_white",
            height=380,
        )
        st.plotly_chart(fig_bar, use_container_width=True)

    with col2:
        fig_succ = go.Figure(data=[
            go.Bar(
                x=comp_df["Algorithm"],
                y=comp_df["Success Rate (%)"],
                marker_color=colors,
                text=[f"{v:.1f}%" for v in comp_df["Success Rate (%)"]],
                textposition="auto",
            )
        ])
        fig_succ.update_layout(
            title="Mission Landing Success Rate (%)",
            yaxis_title="Success Rate (%)",
            yaxis_range=[0, 105],
            template="plotly_white",
            height=380,
        )
        st.plotly_chart(fig_succ, use_container_width=True)

    st.markdown("---")

    # Empirical Distributions
    if dist_df is not None:
        st.subheader("2. Empirical Reward Distributions (100 Test Seeds)")
        fig_dist = px.box(
            dist_df,
            x="algorithm",
            y="reward",
            color="algorithm",
            color_discrete_map={"Random": "#94A3B8", "DQN": "#2563EB", "PPO": "#059669"},
            points="all",
            labels={"algorithm": "Algorithm", "reward": "Episodic Return"},
        )
        fig_dist.add_hline(y=200, line_dash="dash", line_color="#EF4444", annotation_text="Solved Threshold (+200)")
        fig_dist.update_layout(template="plotly_white", height=420)
        st.plotly_chart(fig_dist, use_container_width=True)

    # Statistical Hypothesis Testing Card
    st.markdown("---")
    st.subheader("3. Rigorous Statistical Hypothesis Testing")

    stat = comp_data.get("statistical_test", {})
    if stat:
        sc1, sc2, sc3, sc4 = st.columns(4)
        sc1.metric("Welch's t-statistic", f"{stat.get('welch_t_statistic', 0):.4f}")
        sc2.metric("p-value (Welch)", f"{stat.get('welch_p_value', 0):.4e}")
        sc3.metric("Cohen's d (Effect Size)", f"{stat.get('cohens_d', 0):.3f}")
        sc4.metric("Leading Model", stat.get("higher_mean_performer", "N/A"))

        sig_text = (
            "**Statistically Significant (p < 0.05)**: The performance difference between the two RL models is statistically significant."
            if stat.get("statistically_significant_005")
            else "**Statistical Parity (p >= 0.05)**: No statistically significant difference in mean returns between DQN and PPO under this sample size."
        )
        st.info(sig_text)
