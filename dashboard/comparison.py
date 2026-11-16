"""
Dashboard: Algorithm Benchmark & Comparison - Autonomous Car Racing AI
======================================================================
Comparative benchmark analysis between Continuous PPO, Discrete DQN, and Random Baseline,
including statistical hypothesis testing and track navigation metrics.
"""

import os
import json
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px


def render_comparison():
    st.header("🏁 Autonomous Racing Benchmark & Statistical Comparison")

    comp_csv = "results/evaluation/racing_algorithm_comparison.csv"
    comp_json = "results/evaluation/racing_algorithm_comparison.json"

    if not os.path.exists(comp_csv) or not os.path.exists(comp_json):
        st.warning("Comparative evaluation data not found yet. Run the evaluation benchmark to generate comparative results.")
        return

    with open(comp_json, "r") as f:
        comp_data = json.load(f)

    df_records = pd.read_csv(comp_csv)

    # 1. Standardized Comparison Table
    st.subheader("1. Standardized Multi-Agent Evaluation Metrics")
    
    # Aggregate summary metrics
    summary_rows = []
    for algo_name in ["Random", "DQN", "PPO"]:
        sub = df_records[df_records["algorithm"] == algo_name]
        if not sub.empty:
            prog_col = "track_progress_percent" if "track_progress_percent" in sub.columns else None
            summary_rows.append({
                "Algorithm": algo_name,
                "Control Paradigm": "Continuous [S,G,B]" if algo_name in ("PPO", "Random") else "Discrete (7 Actions)",
                "Mean Reward": round(sub["reward"].mean(), 2),
                "Std Reward": round(sub["reward"].std(), 2),
                "Median Reward": round(sub["reward"].median(), 2),
                "Mean Track Progress (%)": round(sub[prog_col].mean(), 1) if prog_col else 0.0,
                "Max Track Progress (%)": round(sub[prog_col].max(), 1) if prog_col else 0.0,
                "Mean Episode Steps": round(sub["length"].mean(), 1),
            })

    if summary_rows:
        sum_df = pd.DataFrame(summary_rows)
        st.dataframe(
            sum_df.style.highlight_max(subset=["Mean Reward", "Median Reward", "Mean Track Progress (%)"], color="#DCFCE7"),
            use_container_width=True,
        )

    # 2. Side-by-Side Comparative Charts
    col1, col2 = st.columns(2)
    palette = {"Random": "#94A3B8", "DQN": "#2563EB", "PPO": "#059669"}

    with col1:
        fig_bar = go.Figure()
        for row in summary_rows:
            algo = row["Algorithm"]
            fig_bar.add_trace(go.Bar(
                x=[algo],
                y=[row["Mean Reward"]],
                error_y=dict(type="data", array=[row["Std Reward"]]),
                name=algo,
                marker_color=palette.get(algo, "#3B82F6"),
                text=[f"{row['Mean Reward']:.1f}"],
                textposition="auto",
            ))
        fig_bar.add_hline(y=0, line_dash="dot", line_color="#6B7280", annotation_text="Zero Reward Baseline")
        fig_bar.update_layout(
            title="Mean Evaluation Return (+/- 1 Std Dev)",
            yaxis_title="Reward",
            template="plotly_white",
            height=380,
            showlegend=False,
        )
        st.plotly_chart(fig_bar, use_container_width=True)

    with col2:
        fig_prog = go.Figure()
        for row in summary_rows:
            algo = row["Algorithm"]
            fig_prog.add_trace(go.Bar(
                x=[algo],
                y=[row["Mean Track Progress (%)"]],
                name=algo,
                marker_color=palette.get(algo, "#3B82F6"),
                text=[f"{row['Mean Track Progress (%)']:.1f}%"],
                textposition="auto",
            ))
        fig_prog.update_layout(
            title="Mean Track Navigation Progress (%)",
            yaxis_title="Track Progress (%)",
            yaxis_range=[0, 105],
            template="plotly_white",
            height=380,
            showlegend=False,
        )
        st.plotly_chart(fig_prog, use_container_width=True)

    st.markdown("---")

    # 3. Statistical Hypothesis Testing
    st.subheader("2. Statistical Significance Testing")
    stat_tests = comp_data.get("statistical_tests", {})
    ppo_vs_dqn = stat_tests.get("ppo_vs_dqn", {})

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Welch's t-test p-value", f"{ppo_vs_dqn.get('p_value_welch_t', 1.0):.4f}")
    c2.metric("Mann-Whitney U p-value", f"{ppo_vs_dqn.get('p_value_mann_whitney', 1.0):.4f}")
    c3.metric("Cohen's d Effect Size", f"{ppo_vs_dqn.get('cohens_d', 0.0):.2f}")
    c4.metric("Superior Policy", ppo_vs_dqn.get("winner", "PPO"))

    st.markdown(
        f"""
        **Statistical Interpretation**:
        - **Null Hypothesis ($H_0$)**: Continuous PPO and Discrete DQN perform identically on CarRacing-v3.
        - **Alternative Hypothesis ($H_1$)**: The continuous control paradigm achieves distinct, superior track performance.
        - **Statistically Significant**: {'Yes (p < 0.05)' if ppo_vs_dqn.get('is_significant_at_05', False) else 'Pending higher N or comparable scores'}.
        """
    )
