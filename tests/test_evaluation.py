"""
Unit Tests: Car Racing Evaluation Metrics & Comparative Testing
===============================================================
"""

import pytest
import numpy as np
from evaluation.metrics import compute_racing_metrics, compare_racing_distributions


def test_compute_racing_metrics_calculation():
    """Verify statistical formulas for racing rewards and track progress."""
    rewards = [150.0, 300.0, -20.0, 450.0, 100.0]
    lengths = [200, 350, 100, 500, 180]
    progress = [35.0, 60.0, 10.0, 95.0, 25.0]

    metrics = compute_racing_metrics(
        rewards=rewards,
        lengths=lengths,
        track_progresses=progress,
        completion_threshold=800.0,
        random_baseline_mean=-50.0,
    )

    assert metrics["n_episodes"] == 5
    assert metrics["mean_reward"] == pytest.approx(196.0, rel=1e-2)
    assert metrics["max_reward"] == 450.0
    assert metrics["min_reward"] == -20.0
    assert metrics["mean_track_progress_percent"] == pytest.approx(45.0, rel=1e-2)
    assert metrics["max_track_progress_percent"] == 95.0
    # One episode reached 95% progress
    assert metrics["track_completion_rate_percent"] == 20.0
    assert "ci_95_low" in metrics
    assert "ci_95_high" in metrics


def test_compare_racing_distributions():
    """Verify statistical hypothesis testing between two racing agents."""
    ppo_rewards = [400.0, 450.0, 500.0, 550.0, 600.0]
    dqn_rewards = [100.0, 150.0, 200.0, 250.0, 300.0]

    test_res = compare_racing_distributions(
        rewards_a=ppo_rewards,
        rewards_b=dqn_rewards,
        label_a="PPO",
        label_b="DQN",
    )

    assert test_res["winner"] == "PPO"
    assert test_res["is_significant_at_05"] is True
    assert test_res["cohens_d"] > 0
    assert "p_value_welch_t" in test_res
    assert "p_value_mann_whitney" in test_res
