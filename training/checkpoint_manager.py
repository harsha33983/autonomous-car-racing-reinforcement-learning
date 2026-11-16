"""
Checkpoint Manager
==================
Handles saving, rotating, manifesting, and resuming model checkpoints.
"""

import os
import json
import glob
from typing import Any, Dict, List, Optional
from stable_baselines3.common.callbacks import CheckpointCallback


class CheckpointManager:
    """
    Manages model checkpoints and manifests for reproducible MLOps pipelines.
    """

    def __init__(self, checkpoint_dir: str, prefix: str = "model"):
        self.checkpoint_dir = checkpoint_dir
        self.prefix = prefix
        self.manifest_path = os.path.join(checkpoint_dir, f"{prefix}_manifest.json")
        os.makedirs(self.checkpoint_dir, exist_ok=True)

    def get_sb3_callback(self, save_freq: int = 25000, verbose: int = 1) -> CheckpointCallback:
        """Create a Stable-Baselines3 CheckpointCallback."""
        return CheckpointCallback(
            save_freq=save_freq,
            save_path=self.checkpoint_dir,
            name_prefix=f"{self.prefix}_step",
            save_replay_buffer=False,
            save_vecnormalize=True,
            verbose=verbose,
        )

    def list_checkpoints(self) -> List[Dict[str, Any]]:
        """List all checkpoint files and their metadata."""
        pattern = os.path.join(self.checkpoint_dir, f"{self.prefix}_step_*.zip")
        files = glob.glob(pattern)
        checkpoints = []

        for f in sorted(files, key=os.path.getmtime):
            filename = os.path.basename(f)
            # Extract step number
            try:
                step_str = filename.replace(f"{self.prefix}_step_", "").replace("_steps.zip", "").replace(".zip", "")
                step = int(step_str)
            except ValueError:
                step = 0

            checkpoints.append({
                "path": os.path.abspath(f),
                "filename": filename,
                "timestep": step,
                "modified_time": os.path.getmtime(f),
                "size_bytes": os.path.getsize(f),
            })

        return sorted(checkpoints, key=lambda x: x["timestep"])

    def get_latest_checkpoint(self) -> Optional[str]:
        """Return the path to the most recent checkpoint."""
        ckpts = self.list_checkpoints()
        if ckpts:
            return ckpts[-1]["path"]
        return None

    def update_manifest(self, additional_info: Optional[Dict[str, Any]] = None) -> None:
        """Update manifest JSON with all discovered checkpoints and optional metadata."""
        ckpts = self.list_checkpoints()
        manifest_data = {
            "prefix": self.prefix,
            "checkpoint_count": len(ckpts),
            "checkpoints": ckpts,
            "latest_checkpoint": ckpts[-1]["path"] if ckpts else None,
            "metadata": additional_info or {},
        }
        with open(self.manifest_path, "w") as f:
            json.dump(manifest_data, f, indent=2)
