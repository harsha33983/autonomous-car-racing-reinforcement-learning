"""
Unit Tests: Evaluation Metrics & Policy Evaluation
==================================================
"""

import pytest
import numpy as np
from evaluation.metrics import compute_rl_metrics, compare_two_distributions
from evaluation.evaluate import evaluate_policy
from environments.lunar_lander_env import make_lunar_lander_env


class DummyUniformAgent:
    """Mock agent for fast deterministic testing."""
    def predict(self, observation, deterministic=True):
        return 0, None


def test_compute_rl_metrics_calculation():
    """Verify statistical formulas for rewards and episode lengths."""
    rewards = [210.0, 250.0, 180.0, -120.0, 220.0]
    lengths = [300, 320, 290, 150, 310]

    metrics = compute_rl_metrics(rewards, lengths, success_threshold=200.0, random_baseline_mean=-180.0)

    assert metrics["n_episodes"] == 5
    assert metrics["mean_reward"] == pytest.approx(148.0, rel=1e-2)
    assert metrics["max_reward"] == 250.0
    assert metrics["min_reward"] == -120.0
    # Successes: 210, 250, 220 -> 3/5 = 60.0%
    assert metrics["success_rate_percent"] == 60.0
    # Crashes: -120 -> 1/5 = 20.0%
    assert metrics["crash_rate_percent"] == 20.0
    assert metrics["mean_episode_length"] == pytest.approx(274.0, rel=1e-2)


def test_compare_two_distributions():
    """Verify statistical hypothesis testing output."""
    dist_a = [210.0, 220.0, 230.0, 240.0, 250.0]
    dist_b = [100.0, 110.0, 120.0, 130.0, 140.0]

    test_res = compare_two_distributions(dist_a, dist_b, label_a="AgentA", label_b="AgentB")
    assert test_res["higher_mean_performer"] == "AgentA"
    assert test_res["statistically_significant_005"] is True
    assert test_res["cohens_d"] > 0


def test_evaluate_policy_loop():
    """Verify evaluation loop runs over 3 episodes without crash."""
    env = make_lunar_lander_env(seed=42)
    dummy_agent = DummyUniformAgent()

    results = evaluate_policy(
        agent=dummy_agent,
        env=env,
        n_episodes=3,
        deterministic=True,
        seed_start=100,
        verbose=False,
    )

    assert results["n_episodes"] == 3
    assert len(results["detailed_records"]) == 3
    assert "mean_reward" in results
    env.close()
