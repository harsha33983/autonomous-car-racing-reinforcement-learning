"""
Training Plots and Performance Visualization for Autonomous Car Racing
======================================================================
Generates publication-quality Matplotlib/Seaborn visualizations from
autonomous racing training logs and evaluation benchmarks.
"""

import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import glob
from typing import Optional
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Set styling
sns.set_theme(style="whitegrid", font="sans-serif")
plt.rcParams.update({
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 13,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "figure.titlesize": 14,
    "figure.dpi": 150,
})

COLORS = {
    "DQN": "#2563EB",       # Royal Blue
    "PPO": "#059669",       # Emerald Green
    "RANDOM": "#9CA3AF",    # Neutral Gray
    "TRACK": "#D97706",     # Amber
    "TARGET": "#DC2626",    # Crimson Red
}


def plot_racing_learning_curve(
    csv_path: str,
    output_path: str,
    title: str = "CarRacing-v3 Training Learning Curve",
    algo_name: str = "PPO",
) -> None:
    """
    Plot episodic rewards and rolling averages with confidence bands for car racing.
    """
    if not os.path.exists(csv_path):
        print(f"Warning: {csv_path} does not exist, skipping plot.")
        return

    df = pd.read_csv(csv_path)
    if "reward" not in df.columns:
        return

    fig, ax1 = plt.subplots(figsize=(10, 5))
    color = COLORS.get(algo_name.upper(), "#059669")

    # Raw rewards
    ax1.plot(df["episode"], df["reward"], color=color, alpha=0.25, label="Raw Episode Reward")

    # Rolling averages
    if "rolling_mean_50" in df.columns and not df["rolling_mean_50"].isna().all():
        ax1.plot(df["episode"], df["rolling_mean_50"], color=color, linewidth=2.2, label="Rolling Mean (50 ep)")
        if "rolling_std_50" in df.columns and not df["rolling_std_50"].isna().all():
            ax1.fill_between(
                df["episode"],
                df["rolling_mean_50"] - df["rolling_std_50"],
                df["rolling_mean_50"] + df["rolling_std_50"],
                color=color,
                alpha=0.15,
                label="+/- 1 Std Dev",
            )
    elif "rolling_mean_10" in df.columns:
        ax1.plot(df["episode"], df["rolling_mean_10"], color=color, linewidth=2.0, label="Rolling Mean (10 ep)")

    # Baseline & Benchmark thresholds
    ax1.axhline(0, color="#6B7280", linestyle=":", linewidth=1.2, label="Zero Reward Baseline")
    ax1.axhline(700, color=COLORS["TARGET"], linestyle="--", linewidth=1.5, label="Competent Racing (>700)")

    ax1.set_xlabel("Episode")
    ax1.set_ylabel("Cumulative Episodic Reward")
    ax1.set_title(title, fontweight="bold")
    ax1.legend(loc="lower right", framealpha=0.9)
    plt.tight_layout()

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"[+] Saved: {output_path}")


def plot_racing_comparison_curves(
    dqn_csv: str,
    ppo_csv: str,
    output_path: str,
) -> None:
    """
    Plot side-by-side training comparison of Discrete DQN vs Continuous PPO on CarRacing-v3.
    """
    if not os.path.exists(dqn_csv) or not os.path.exists(ppo_csv):
        print(f"Notice: Waiting for both training CSVs ({dqn_csv}, {ppo_csv})")
        return

    dqn_df = pd.read_csv(dqn_csv)
    ppo_df = pd.read_csv(ppo_csv)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 5.5))

    # Plot 1: Reward vs Timesteps
    dqn_col = "rolling_mean_10" if "rolling_mean_10" in dqn_df.columns else "reward"
    ppo_col = "rolling_mean_10" if "rolling_mean_10" in ppo_df.columns else "reward"

    ax1.plot(dqn_df["timestep"], dqn_df[dqn_col], label="DQN (Discrete 7-Actions)", color=COLORS["DQN"], linewidth=2.2)
    ax1.plot(ppo_df["timestep"], ppo_df[ppo_col], label="PPO (Continuous Control)", color=COLORS["PPO"], linewidth=2.2)
    ax1.axhline(0, color="#6B7280", linestyle=":", label="Zero Reward Line")
    ax1.set_xlabel("Environment Timesteps")
    ax1.set_ylabel("Mean Episodic Reward")
    ax1.set_title("Sample Efficiency: Reward vs Simulation Timesteps", fontweight="bold")
    ax1.legend(loc="lower right")

    # Plot 2: Episode Duration / Survival Progression
    dqn_len = dqn_df["length"].rolling(5, min_periods=1).mean() if "length" in dqn_df.columns else []
    ppo_len = ppo_df["length"].rolling(5, min_periods=1).mean() if "length" in ppo_df.columns else []

    if len(dqn_len) > 0:
        ax2.plot(dqn_df["episode"], dqn_len, label="DQN Track Steps", color=COLORS["DQN"], linewidth=2.0)
    if len(ppo_len) > 0:
        ax2.plot(ppo_df["episode"], ppo_len, label="PPO Track Steps", color=COLORS["PPO"], linewidth=2.0)

    ax2.set_xlabel("Training Episodes")
    ax2.set_ylabel("Mean Steps on Track")
    ax2.set_title("Autonomous Track Survival Duration Progression", fontweight="bold")
    ax2.legend(loc="upper left")

    plt.suptitle("Autonomous Car Racing Training Dynamics: Discrete DQN vs Continuous PPO (CarRacing-v3)", fontsize=14, y=1.02)
    plt.tight_layout()

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[+] Saved: {output_path}")


def plot_racing_evaluation_distributions(
    comparison_csv: str,
    output_path: str,
) -> None:
    """
    Plot reward distribution and track progress across Random, DQN, and PPO agents.
    """
    if not os.path.exists(comparison_csv):
        print(f"Warning: {comparison_csv} not found.")
        return

    df = pd.read_csv(comparison_csv)
    palette = {"Random": COLORS["RANDOM"], "DQN": COLORS["DQN"], "PPO": COLORS["PPO"]}

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Box / Stripplot for Rewards
    sns.boxplot(
        data=df,
        x="algorithm",
        y="reward",
        palette=palette,
        ax=ax1,
        boxprops=dict(alpha=0.8),
    )
    sns.stripplot(
        data=df,
        x="algorithm",
        y="reward",
        color="black",
        alpha=0.35,
        jitter=0.2,
        size=5,
        ax=ax1,
    )
    ax1.set_title("Benchmark Episodic Reward Distribution", fontweight="bold")
    ax1.set_xlabel("Autonomous Racing Agent")
    ax1.set_ylabel("Episodic Return")

    # Boxplot for Track Progress %
    prog_col = "track_progress_percent" if "track_progress_percent" in df.columns else None
    if prog_col:
        sns.boxplot(
            data=df,
            x="algorithm",
            y=prog_col,
            palette=palette,
            ax=ax2,
            boxprops=dict(alpha=0.8),
        )
        sns.stripplot(
            data=df,
            x="algorithm",
            y=prog_col,
            color="black",
            alpha=0.35,
            jitter=0.2,
            size=5,
            ax=ax2,
        )
        ax2.set_title("Track Progress Completion Rate (%)", fontweight="bold")
        ax2.set_xlabel("Autonomous Racing Agent")
        ax2.set_ylabel("Track Progress %")
        ax2.set_ylim(-5, 105)

    plt.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"[+] Saved: {output_path}")


def plot_racing_comparison_bar_chart(
    comparison_csv: str,
    output_path: str,
) -> None:
    """
    Bar chart comparing Mean Reward and Mean Track Progress across agents.
    """
    if not os.path.exists(comparison_csv):
        return

    df = pd.read_csv(comparison_csv)
    # Aggregate by algorithm
    summary = df.groupby("algorithm").agg(
        mean_reward=("reward", "mean"),
        std_reward=("reward", "std"),
        mean_progress=("track_progress_percent", "mean") if "track_progress_percent" in df.columns else ("reward", "count"),
    ).reset_index()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    palette = [COLORS.get(a.upper(), "#2563EB") for a in summary["algorithm"]]

    # Mean Reward
    bars1 = ax1.bar(summary["algorithm"], summary["mean_reward"], yerr=summary["std_reward"].fillna(0), capsize=5, color=palette, alpha=0.85)
    ax1.set_title("Mean Episodic Reward (± 1 Std Dev)", fontweight="bold")
    ax1.set_ylabel("Mean Reward")
    ax1.grid(axis="y", linestyle="--", alpha=0.7)
    for bar in bars1:
        yval = bar.get_height()
        va = "bottom" if yval >= 0 else "top"
        ax1.text(bar.get_x() + bar.get_width() / 2, yval, f"{yval:.1f}", ha="center", va=va, fontweight="bold", fontsize=10)

    # Mean Track Progress
    if "track_progress_percent" in df.columns:
        bars2 = ax2.bar(summary["algorithm"], summary["mean_progress"], color=palette, alpha=0.85)
        ax2.set_title("Mean Track Navigation Progress (%)", fontweight="bold")
        ax2.set_ylabel("Track Progress (%)")
        ax2.set_ylim(0, 105)
        ax2.grid(axis="y", linestyle="--", alpha=0.7)
        for bar in bars2:
            yval = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width() / 2, yval + 2, f"{yval:.1f}%", ha="center", va="bottom", fontweight="bold", fontsize=10)

    plt.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"[+] Saved: {output_path}")


def generate_all_plots(results_dir: str = "results") -> None:
    """
    Master function to regenerate all charts from available experiment logs.
    """
    plots_dir = os.path.join(results_dir, "plots")
    logs_dir = os.path.join(results_dir, "training_logs")
    eval_dir = os.path.join(results_dir, "evaluation")

    dqn_csv = os.path.join(logs_dir, "dqn", "dqn_racing_metrics.csv")
    if not os.path.exists(dqn_csv):
        dqn_csv = os.path.join(logs_dir, "dqn", "dqn_carracing_training.csv")

    ppo_csv = os.path.join(logs_dir, "ppo", "ppo_racing_metrics.csv")
    if not os.path.exists(ppo_csv):
        ppo_csv = os.path.join(logs_dir, "ppo", "ppo_carracing_training.csv")
    comp_csv = os.path.join(eval_dir, "racing_algorithm_comparison.csv")

    if os.path.exists(dqn_csv):
        plot_racing_learning_curve(dqn_csv, os.path.join(plots_dir, "dqn_carracing_learning_curve.png"), "Discrete DQN Racing Training Curve", "DQN")
    if os.path.exists(ppo_csv):
        plot_racing_learning_curve(ppo_csv, os.path.join(plots_dir, "ppo_carracing_learning_curve.png"), "Continuous PPO Racing Training Curve", "PPO")
    if os.path.exists(dqn_csv) and os.path.exists(ppo_csv):
        plot_racing_comparison_curves(dqn_csv, ppo_csv, os.path.join(plots_dir, "racing_dqn_vs_ppo_training.png"))
    if os.path.exists(comp_csv):
        plot_racing_evaluation_distributions(comp_csv, os.path.join(plots_dir, "racing_evaluation_distribution.png"))
        plot_racing_comparison_bar_chart(comp_csv, os.path.join(plots_dir, "racing_algorithm_comparison_bars.png"))


if __name__ == "__main__":
    generate_all_plots()
