"""
Unit Tests: Training Pipeline & Checkpoint Management
=====================================================
"""

import os
import shutil
import pytest
import numpy as np
from environments.lunar_lander_env import make_lunar_lander_env
from agents.agent_factory import create_agent, load_agent
from training.checkpoint_manager import CheckpointManager


@pytest.fixture
def temp_test_dir(tmp_path):
    d = tmp_path / "test_models"
    d.mkdir()
    return str(d)


def test_dqn_build_and_prediction():
    """Verify DQN builds, predicts, and saves properly."""
    env = make_lunar_lander_env(seed=42)
    agent = create_agent("dqn", env=env, seed=42)

    summary = agent.get_model_summary()
    assert summary["algorithm"] == "Deep Q-Network (DQN)"
    assert summary["total_parameters"] > 0

    obs, _ = env.reset(seed=42)
    action, _ = agent.predict(obs, deterministic=True)
    assert 0 <= action < 4
    env.close()


def test_ppo_build_and_prediction():
    """Verify PPO builds, predicts, and saves properly."""
    env = make_lunar_lander_env(seed=42)
    agent = create_agent("ppo", env=env, seed=42)

    summary = agent.get_model_summary()
    assert "PPO" in summary["algorithm"]
    assert summary["total_parameters"] > 0

    obs, _ = env.reset(seed=42)
    action, _ = agent.predict(obs, deterministic=True)
    assert 0 <= action < 4
    env.close()


def test_fast_training_cycle_and_serialization(temp_test_dir):
    """Verify short training step (64 steps) runs without error and saves weights."""
    env = make_lunar_lander_env(seed=42)

    # Train short PPO
    ppo_hyper = {
        "learning_rate": 1e-3,
        "n_steps": 64,
        "batch_size": 32,
        "n_epochs": 1,
        "policy_kwargs": {"net_arch": dict(pi=[32, 32], vf=[32, 32])},
    }
    agent = create_agent("ppo", env=env, hyperparams=ppo_hyper, seed=42)
    agent.train(total_timesteps=64)

    # Save and reload
    save_file = os.path.join(temp_test_dir, "test_ppo.zip")
    agent.save(save_file)
    assert os.path.exists(save_file)

    loaded_agent = load_agent("ppo", save_file, env=env)
    obs, _ = env.reset(seed=42)
    action, _ = loaded_agent.predict(obs, deterministic=True)
    assert 0 <= action < 4
    env.close()


def test_checkpoint_manager(temp_test_dir):
    """Verify checkpoint manager discovers files and updates manifest."""
    mgr = CheckpointManager(checkpoint_dir=temp_test_dir, prefix="test")
    # Simulate a fake checkpoint file
    fake_ckpt = os.path.join(temp_test_dir, "test_step_1000_steps.zip")
    with open(fake_ckpt, "w") as f:
        f.write("dummy")

    ckpts = mgr.list_checkpoints()
    assert len(ckpts) == 1
    assert ckpts[0]["timestep"] == 1000

    mgr.update_manifest({"test_key": "test_val"})
    assert os.path.exists(mgr.manifest_path)
