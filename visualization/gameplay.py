"""
Autonomous Racing Gameplay Demonstration & Video/GIF Recording
=============================================================
Executes trained autonomous racing agent in rendered Gymnasium CarRacing-v3,
overlays real-time racing HUD telemetry (Steering, Gas, Brake, Track Progress, Score),
and exports high-resolution replay animations (GIF / MP4).
"""

import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import argparse
from typing import List, Optional, Tuple, Dict, Any, Union
import numpy as np
import imageio
from PIL import Image, ImageDraw, ImageFont

from environments.car_racing_env import make_car_racing_env
from environments.discrete_car_racing_env import make_discrete_car_racing_env, DISCRETE_ACTIONS
from agents.agent_factory import load_racing_agent


def overlay_racing_telemetry(
    frame: np.ndarray,
    algorithm: str,
    step: int,
    action: Union[int, np.ndarray, list],
    total_reward: float,
    telemetry: Optional[Dict[str, Any]] = None,
) -> np.ndarray:
    """
    Overlay autonomous racing HUD telemetry on (400, 600, 3) CarRacing frame.
    """
    img = Image.fromarray(frame)
    
    # Semi-transparent HUD overlay banner
    banner_height = 54
    overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
    overlay_draw = ImageDraw.Draw(overlay)
    overlay_draw.rectangle([0, 0, img.width, banner_height], fill=(15, 23, 42, 225))  # Dark Slate
    img = Image.alpha_composite(img.convert("RGBA"), overlay)
    draw = ImageDraw.Draw(img)

    # Format action values
    algo_lower = algorithm.lower()
    if isinstance(action, (int, np.integer)):
        act_idx = int(action)
        act_def = DISCRETE_ACTIONS.get(act_idx, [0.0, 0.0, 0.0])
        steer, gas, brake = act_def[0], act_def[1], act_def[2]
        act_str = f"DQN Act {act_idx} [Steer:{steer:+.1f} Gas:{gas:.1f} Brk:{brake:.1f}]"
    else:
        act_arr = np.array(action).flatten()
        steer = float(act_arr[0]) if len(act_arr) > 0 else 0.0
        gas = float(act_arr[1]) if len(act_arr) > 1 else 0.0
        brake = float(act_arr[2]) if len(act_arr) > 2 else 0.0
        act_str = f"Steer: {steer:+.2f} | Gas: {gas:.2f} | Brake: {brake:.2f}"

    algo_color = "#38BDF8" if "dqn" in algo_lower else "#34D399"
    rew_color = "#22C55E" if total_reward >= 0 else "#EF4444"

    # Row 1: Agent Name, Step, Reward Score
    draw.text((12, 6), f"AGENT: {algorithm.upper()}", fill=algo_color)
    draw.text((210, 6), f"STEP: {step}", fill="#E2E8F0")
    draw.text((320, 6), f"REWARD: {total_reward:+.1f}", fill=rew_color)

    # Row 2: Action details and track progress
    draw.text((12, 28), f"ACT: {act_str}", fill="#FCD34D")

    if telemetry:
        progress = float(telemetry.get("track_progress", 0.0))
        tiles = int(telemetry.get("tiles_visited", 0))
        off_track = int(telemetry.get("off_track_count", 0))
        prog_color = "#38BDF8" if progress > 50 else "#F59E0B"
        draw.text((370, 28), f"PROGRESS: {progress:.1f}% ({tiles} tiles)", fill=prog_color)
        if off_track > 0:
            draw.text((515, 6), "OFF-TRACK", fill="#EF4444")
        else:
            draw.text((515, 6), "ON-TRACK", fill="#10B981")

    return np.array(img.convert("RGB"))


def record_autonomous_racing(
    algorithm: str,
    model_path: str,
    output_gif_path: str,
    seed: int = 1042,
    fps: int = 25,
    max_steps: int = 600,
    grayscale: bool = True,
    frame_skip: int = 2,
    frame_subsample: int = 2,
) -> Tuple[float, int, List[np.ndarray]]:
    """
    Simulate an autonomous racing episode with render frames, overlay HUD, and save GIF.
    """
    os.makedirs(os.path.dirname(os.path.abspath(output_gif_path)), exist_ok=True)
    algo_lower = algorithm.lower()

    if algo_lower == "dqn":
        env = make_discrete_car_racing_env(
            seed=seed,
            grayscale=grayscale,
            frame_skip=frame_skip,
            render_mode="rgb_array",
        )
    else:
        env = make_car_racing_env(
            continuous=True,
            seed=seed,
            grayscale=grayscale,
            frame_skip=frame_skip,
            render_mode="rgb_array",
        )

    agent = load_racing_agent(algorithm, model_path, env=env)

    obs, info = env.reset(seed=seed)
    done = False
    truncated = False
    step = 0
    total_reward = 0.0
    frames = []

    print(f"[*] Simulating autonomous racing for {algorithm.upper()} (Seed={seed})...")

    # Initial frame
    initial_frame = env.render()
    if initial_frame is not None:
        annotated = overlay_racing_telemetry(
            initial_frame, algorithm, step, 0 if algo_lower == "dqn" else [0.0, 0.0, 0.0],
            total_reward, info.get("telemetry")
        )
        frames.append(annotated)

    step_counter = 0
    while not (done or truncated) and step < max_steps:
        action, _ = agent.predict(obs, deterministic=True)
        obs, reward, done, truncated, info = env.step(action)
        total_reward += float(reward)
        step += 1
        step_counter += 1

        if step_counter % frame_subsample == 0:
            frame = env.render()
            if frame is not None:
                annotated = overlay_racing_telemetry(
                    frame, algorithm, step, action, total_reward, info.get("telemetry")
                )
                frames.append(annotated)

    env.close()

    # Save animation
    if frames:
        print(f"[*] Compiling {len(frames)} frames into animation: {output_gif_path} ...")
        # Ensure sizes are consistent
        imageio.mimsave(output_gif_path, frames, fps=fps, loop=0)
        file_size_mb = os.path.getsize(output_gif_path) / (1024 * 1024)
        print(f"[+] Successfully exported {output_gif_path} ({file_size_mb:.2f} MB)")

    return total_reward, step, frames


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Record Autonomous Racing Gameplay")
    parser.add_argument("--algo", type=str, default="ppo", choices=["ppo", "dqn"])
    parser.add_argument("--model", type=str, required=True, help="Path to saved agent model (.zip)")
    parser.add_argument("--output", type=str, default="results/videos/autonomous_racing.gif")
    parser.add_argument("--seed", type=int, default=1042)
    parser.add_argument("--max_steps", type=int, default=500)
    args = parser.parse_args()

    record_autonomous_racing(
        algorithm=args.algo,
        model_path=args.model,
        output_gif_path=args.output,
        seed=args.seed,
        max_steps=args.max_steps,
    )
