"""
Evaluation Metrics
==================
Statistical computations for reinforcement learning policy evaluation.
"""

from typing import Any, Dict, List, Union
import numpy as np
from scipy import stats


def compute_rl_metrics(
    rewards: List[float],
    lengths: List[int],
    success_threshold: float = 200.0,
    random_baseline_mean: float = -180.0,
) -> Dict[str, Any]:
    """
    Calculate comprehensive statistical metrics from evaluation episodes.

    Args:
        rewards: List of cumulative episodic rewards
        lengths: List of episode lengths (steps)
        success_threshold: Threshold reward for task success
        random_baseline_mean: Benchmark random agent reward for relative improvement

    Returns:
        Dictionary of formatted statistics
    """
    rewards_arr = np.array(rewards, dtype=np.float64)
    lengths_arr = np.array(lengths, dtype=np.float64)
    n_episodes = len(rewards_arr)

    if n_episodes == 0:
        return {}

    mean_rew = float(np.mean(rewards_arr))
    std_rew = float(np.std(rewards_arr, ddof=1)) if n_episodes > 1 else 0.0
    median_rew = float(np.median(rewards_arr))
    min_rew = float(np.min(rewards_arr))
    max_rew = float(np.max(rewards_arr))
    iqr_rew = float(stats.iqr(rewards_arr)) if n_episodes > 1 else 0.0

    # 95% Confidence interval for mean reward
    if n_episodes > 1 and std_rew > 0:
        se = std_rew / np.sqrt(n_episodes)
        ci_95 = stats.t.interval(0.95, df=n_episodes - 1, loc=mean_rew, scale=se)
        ci_low, ci_high = float(ci_95[0]), float(ci_95[1])
    else:
        ci_low, ci_high = mean_rew, mean_rew

    # Success rate
    successes = (rewards_arr >= success_threshold)
    success_rate = float(np.mean(successes)) * 100.0

    # Crash / severe penalty rate (reward < -100)
    crashes = (rewards_arr < -100.0)
    crash_rate = float(np.mean(crashes)) * 100.0

    # Length statistics
    mean_len = float(np.mean(lengths_arr))
    median_len = float(np.median(lengths_arr))
    std_len = float(np.std(lengths_arr, ddof=1)) if n_episodes > 1 else 0.0

    # Percentage improvement over random baseline
    improvement = ((mean_rew - random_baseline_mean) / abs(random_baseline_mean)) * 100.0

    return {
        "n_episodes": n_episodes,
        "mean_reward": round(mean_rew, 2),
        "std_reward": round(std_rew, 2),
        "median_reward": round(median_rew, 2),
        "min_reward": round(min_rew, 2),
        "max_reward": round(max_rew, 2),
        "iqr_reward": round(iqr_rew, 2),
        "ci_95_low": round(ci_low, 2),
        "ci_95_high": round(ci_high, 2),
        "mean_episode_length": round(mean_len, 2),
        "median_episode_length": round(median_len, 2),
        "std_episode_length": round(std_len, 2),
        "success_rate_percent": round(success_rate, 2),
        "crash_rate_percent": round(crash_rate, 2),
        "improvement_over_random_percent": round(improvement, 2),
    }


def compare_two_distributions(
    rewards_a: List[float],
    rewards_b: List[float],
    label_a: str = "DQN",
    label_b: str = "PPO",
) -> Dict[str, Any]:
    """
    Perform statistical significance testing between two agent reward distributions.
    """
    arr_a = np.array(rewards_a)
    arr_b = np.array(rewards_b)

    # Welch's two-sample t-test (assumes unequal variances)
    t_stat, p_val_t = stats.ttest_ind(arr_a, arr_b, equal_var=False)

    # Mann-Whitney U test (non-parametric rank-sum test)
    u_stat, p_val_u = stats.mannwhitneyu(arr_a, arr_b, alternative="two-sided")

    # Cohen's d effect size
    pooled_std = np.sqrt(((len(arr_a) - 1) * np.var(arr_a, ddof=1) + (len(arr_b) - 1) * np.var(arr_b, ddof=1)) / (len(arr_a) + len(arr_b) - 2))
    cohen_d = (np.mean(arr_a) - np.mean(arr_b)) / (pooled_std + 1e-8)

    winner = label_a if np.mean(arr_a) > np.mean(arr_b) else label_b

    return {
        "label_a": label_a,
        "label_b": label_b,
        "mean_a": round(float(np.mean(arr_a)), 2),
        "mean_b": round(float(np.mean(arr_b)), 2),
        "welch_t_statistic": round(float(t_stat), 4),
        "welch_p_value": float(p_val_t),
        "statistically_significant_005": bool(p_val_t < 0.05),
        "mann_whitney_u": round(float(u_stat), 2),
        "mann_whitney_p_value": float(p_val_u),
        "cohens_d": round(float(cohen_d), 4),
        "higher_mean_performer": winner,
    }
