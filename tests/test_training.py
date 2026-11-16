"""
Unit Tests: Training Callbacks and Checkpoint Management
========================================================
"""

import os
import pytest
from training.checkpoint_manager import RacingCheckpointManager
from training.callbacks import RacingMetricsCallback, BestRacingModelCallback


@pytest.fixture
def temp_checkpoint_dir(tmp_path):
    d = tmp_path / "checkpoints"
    d.mkdir()
    return str(d)


def test_racing_checkpoint_manager(temp_checkpoint_dir):
    """Verify RacingCheckpointManager lists files and updates manifest."""
    mgr = RacingCheckpointManager(checkpoint_dir=temp_checkpoint_dir, prefix="ppo")
    
    # Simulate a fake checkpoint
    fake_ckpt = os.path.join(temp_checkpoint_dir, "ppo_step_5000_steps.zip")
    with open(fake_ckpt, "w") as f:
        f.write("mock_model_data")

    ckpts = mgr.list_checkpoints()
    assert len(ckpts) == 1
    assert ckpts[0]["timestep"] == 5000
    assert ckpts[0]["filename"] == "ppo_step_5000_steps.zip"

    mgr.update_manifest({"algorithm": "PPO", "best_reward": 450.0})
    manifest = mgr.load_manifest()
    assert manifest["metadata"]["algorithm"] == "PPO"
    assert manifest["metadata"]["best_reward"] == 450.0


def test_racing_metrics_callback_initialization(temp_checkpoint_dir):
    """Verify RacingMetricsCallback creates log directory and structure."""
    cb = RacingMetricsCallback(
        log_dir=temp_checkpoint_dir,
        filename_prefix="test_metrics",
        save_freq_steps=100,
    )
    assert os.path.exists(temp_checkpoint_dir)
    assert cb.csv_path.endswith("test_metrics.csv")
    assert cb.json_path.endswith("test_metrics_summary.json")
