"""
Training Plots and Performance Visualization
============================================
Generates publication-quality Matplotlib/Seaborn and Plotly visualizations
from training logs and evaluation data.
"""

import os
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
    "Random": "#9CA3AF",    # Neutral Gray
    "Baseline": "#D97706",  # Amber
    "Optimized": "#7C3AED", # Purple
}


def plot_learning_curves(
    csv_path: str,
    output_path: str,
    title: str = "Training Learning Curve",
    algo_name: str = "RL Agent",
) -> None:
    """
    Plot episodic rewards and rolling averages with confidence bands.
    """
    if not os.path.exists(csv_path):
        print(f"Warning: {csv_path} does not exist, skipping plot.")
        return

    df = pd.read_csv(csv_path)
    if "reward" not in df.columns:
        return

    fig, ax1 = plt.subplots(figsize=(10, 5))

    color = COLORS.get(algo_name.upper(), "#2563EB")
    ax1.plot(df["episode"], df["reward"], color=color, alpha=0.25, label="Raw Episode Reward")

    if "rolling_mean_50" in df.columns:
        ax1.plot(df["episode"], df["rolling_mean_50"], color=color, linewidth=2.2, label="Rolling Mean (50 ep)")
        if "rolling_std_50" in df.columns:
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

    # Environment solved threshold for LunarLander
    ax1.axhline(200, color="#DC2626", linestyle="--", linewidth=1.5, label="Solved Threshold (+200)")

    ax1.set_xlabel("Episode")
    ax1.set_ylabel("Cumulative Episodic Reward")
    ax1.set_title(title, fontweight="bold")
    ax1.legend(loc="lower right", framealpha=0.9)
    plt.tight_layout()

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"[+] Saved: {output_path}")


def plot_dqn_vs_ppo_training(
    dqn_csv: str,
    ppo_csv: str,
    output_path: str,
) -> None:
    """
    Plot side-by-side training comparison of DQN vs PPO on identical environment.
    """
    if not os.path.exists(dqn_csv) or not os.path.exists(ppo_csv):
        print("Warning: CSV files missing for DQN vs PPO comparison.")
        return

    dqn_df = pd.read_csv(dqn_csv)
    ppo_df = pd.read_csv(ppo_csv)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 5.5))

    # Plot 1: Reward vs Timesteps
    ax1.plot(dqn_df["timestep"], dqn_df["rolling_mean_50"], label="DQN (Rolling 50)", color=COLORS["DQN"], linewidth=2.2)
    ax1.plot(ppo_df["timestep"], ppo_df["rolling_mean_50"], label="PPO (Rolling 50)", color=COLORS["PPO"], linewidth=2.2)
    ax1.axhline(200, color="#DC2626", linestyle="--", label="Solved Threshold (+200)")
    ax1.set_xlabel("Environment Timesteps")
    ax1.set_ylabel("Rolling Mean Reward")
    ax1.set_title("Sample Efficiency: Rolling Reward vs Timesteps", fontweight="bold")
    ax1.legend(loc="lower right")

    # Plot 2: Episode Length vs Episodes
    ax2.plot(dqn_df["episode"], dqn_df["length"].rolling(20, min_periods=1).mean(), label="DQN Length", color=COLORS["DQN"], linewidth=2.0)
    ax2.plot(ppo_df["episode"], ppo_df["length"].rolling(20, min_periods=1).mean(), label="PPO Length", color=COLORS["PPO"], linewidth=2.0)
    ax2.set_xlabel("Training Episodes")
    ax2.set_ylabel("Mean Steps per Episode")
    ax2.set_title("Episode Duration (Survival Time) Progression", fontweight="bold")
    ax2.legend(loc="upper left")

    plt.suptitle("Reinforcement Learning Training Dynamics: DQN vs PPO (LunarLander-v3)", fontsize=15, y=1.02)
    plt.tight_layout()

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[+] Saved: {output_path}")


def plot_reward_distributions(
    combined_eval_csv: str,
    output_path: str,
) -> None:
    """
    Plot box plot and KDE distribution for 100 evaluation episodes.
    """
    if not os.path.exists(combined_eval_csv):
        print(f"Warning: {combined_eval_csv} not found.")
        return

    df = pd.read_csv(combined_eval_csv)
    palette = {"Random": COLORS["Random"], "DQN": COLORS["DQN"], "PPO": COLORS["PPO"]}

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Box / Violin Plot
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
        alpha=0.3,
        jitter=0.2,
        size=4,
        ax=ax1,
    )
    ax1.axhline(200, color="#DC2626", linestyle="--", label="Solved Threshold")
    ax1.set_title("100-Episode Evaluation Reward Distribution", fontweight="bold")
    ax1.set_xlabel("Agent / Algorithm")
    ax1.set_ylabel("Episodic Return")
    ax1.legend()

    # KDE Density Curve
    for algo in ["Random", "DQN", "PPO"]:
        sub = df[df["algorithm"] == algo]["reward"]
        sns.kdeplot(sub, ax=ax2, label=algo, color=palette.get(algo, "#000"), fill=True, alpha=0.2, linewidth=2)
    ax2.axvline(200, color="#DC2626", linestyle="--", label="Solved Threshold")
    ax2.set_title("Empirical Return Density (KDE)", fontweight="bold")
    ax2.set_xlabel("Episodic Return")
    ax2.set_ylabel("Probability Density")
    ax2.legend()

    plt.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"[+] Saved: {output_path}")


def plot_comparison_bar_chart(
    comparison_csv: str,
    output_path: str,
) -> None:
    """
    Bar chart comparing Mean Reward and Success Rate across models.
    """
    if not os.path.exists(comparison_csv):
        return

    df = pd.read_csv(comparison_csv)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    palette = ["#9CA3AF", "#2563EB", "#059669"]

    # Mean Reward
    bars1 = ax1.bar(df["Algorithm"], df["Mean Reward"], yerr=df["Std Reward"], capsize=5, color=palette, alpha=0.85)
    ax1.axhline(200, color="#DC2626", linestyle="--", label="Solved (>=200)")
    ax1.set_title("Mean Episodic Reward (± 1 Std Dev)", fontweight="bold")
    ax1.set_ylabel("Mean Reward")
    ax1.grid(axis="y", linestyle="--", alpha=0.7)
    for bar in bars1:
        yval = bar.get_height()
        va = "bottom" if yval >= 0 else "top"
        ax1.text(bar.get_x() + bar.get_width() / 2, yval, f"{yval:.1f}", ha="center", va=va, fontweight="bold", fontsize=10)

    # Success Rate
    bars2 = ax2.bar(df["Algorithm"], df["Success Rate (%)"], color=palette, alpha=0.85)
    ax2.set_title("Mission Landing Success Rate (%)", fontweight="bold")
    ax2.set_ylabel("Success Rate (%)")
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

    dqn_csv = os.path.join(logs_dir, "dqn", "optimized", "dqn_training_metrics.csv")
    ppo_csv = os.path.join(logs_dir, "ppo", "optimized", "ppo_training_metrics.csv")
    comp_csv = os.path.join(eval_dir, "algorithm_comparison.csv")
    combined_eval = os.path.join(eval_dir, "combined_eval_episodes.csv")

    if os.path.exists(dqn_csv):
        plot_learning_curves(dqn_csv, os.path.join(plots_dir, "dqn_learning_curve.png"), "DQN Training Learning Curve", "DQN")
    if os.path.exists(ppo_csv):
        plot_learning_curves(ppo_csv, os.path.join(plots_dir, "ppo_learning_curve.png"), "PPO Training Learning Curve", "PPO")
    if os.path.exists(dqn_csv) and os.path.exists(ppo_csv):
        plot_dqn_vs_ppo_training(dqn_csv, ppo_csv, os.path.join(plots_dir, "dqn_vs_ppo_training.png"))
    if os.path.exists(combined_eval):
        plot_reward_distributions(combined_eval, os.path.join(plots_dir, "evaluation_reward_distribution.png"))
    if os.path.exists(comp_csv):
        plot_comparison_bar_chart(comp_csv, os.path.join(plots_dir, "algorithm_comparison_bars.png"))


if __name__ == "__main__":
    generate_all_plots()
