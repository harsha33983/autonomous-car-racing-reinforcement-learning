"""
Autonomous Gameplay Demonstration & Video/GIF Recording
=======================================================
Executes trained agent in rendered Gymnasium environment, overlays real-time HUD telemetry,
and saves high-resolution replay animations (GIF / MP4).
"""

import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import argparse
from typing import List, Optional, Tuple, Dict, Any
import numpy as np
import imageio
from PIL import Image, ImageDraw, ImageFont

from environments.lunar_lander_env import make_lunar_lander_env, ACTION_MEANINGS
from agents.agent_factory import load_agent


def overlay_telemetry(
    frame: np.ndarray,
    algorithm: str,
    episode: int,
    step: int,
    action: int,
    step_reward: float,
    total_reward: float,
    telemetry: Optional[Dict[str, Any]] = None,
) -> np.ndarray:
    """
    Overlay professional HUD telemetry on RGB environment frame.
    """
    img = Image.fromarray(frame)
    draw = ImageDraw.Draw(img)

    # Semi-transparent top HUD banner
    banner_height = 56
    overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
    overlay_draw = ImageDraw.Draw(overlay)
    overlay_draw.rectangle([0, 0, img.width, banner_height], fill=(15, 23, 42, 220))  # Slate dark
    img = Image.alpha_composite(img.convert("RGBA"), overlay)
    draw = ImageDraw.Draw(img)

    # Status text
    algo_color = "#38BDF8" if "dqn" in algorithm.lower() else "#34D399"
    act_str = ACTION_MEANINGS.get(action, f"Action {action}")

    # Top line info
    draw.text((12, 6), f"AGENT: {algorithm.upper()}", fill=algo_color)
    draw.text((180, 6), f"EP: {episode}", fill="#E2E8F0")
    draw.text((260, 6), f"STEP: {step}", fill="#E2E8F0")
    rew_color = "#22C55E" if total_reward >= 0 else "#EF4444"
    draw.text((370, 6), f"SCORE: {total_reward:+.1f}", fill=rew_color)

    # Second line info
    draw.text((12, 28), f"ACTION: {act_str}", fill="#FCD34D")
    if telemetry:
        vx = telemetry.get("linear_velocity_x", 0.0)
        vy = telemetry.get("linear_velocity_y", 0.0)
        left_c = int(telemetry.get("left_leg_contact", 0))
        right_c = int(telemetry.get("right_leg_contact", 0))
        draw.text((330, 28), f"Vx:{vx:+.2f} Vy:{vy:+.2f} Legs:[{left_c},{right_c}]", fill="#94A3B8")

    return np.array(img.convert("RGB"))


def record_autonomous_gameplay(
    algorithm: str,
    model_path: str,
    output_gif_path: str,
    seed: int = 1042,
    fps: int = 30,
    max_steps: int = 1000,
) -> Tuple[float, int, List[np.ndarray]]:
    """
    Play an autonomous episode, overlay HUD telemetry, and write animated GIF.
    """
    os.makedirs(os.path.dirname(os.path.abspath(output_gif_path)), exist_ok=True)
    env = make_lunar_lander_env(seed=seed, render_mode="rgb_array")
    agent = load_agent(algorithm, model_path, env=env)

    obs, info = env.reset(seed=seed)
    done = False
    truncated = False
    step = 0
    total_reward = 0.0
    frames = []

    print(f"[*] Simulating autonomous episode for {algorithm.upper()} (Seed={seed})...")

    initial_frame = env.render()
    if initial_frame is not None:
        annotated = overlay_telemetry(
            initial_frame, algorithm, 1, step, 0, 0.0, total_reward, info.get("telemetry")
        )
        frames.append(annotated)

    while not (done or truncated) and step < max_steps:
        action, _ = agent.predict(obs, deterministic=True)
        obs, reward, done, truncated, info = env.step(action)
        step += 1
        total_reward += reward

        raw_frame = env.render()
        if raw_frame is not None:
            annotated = overlay_telemetry(
                raw_frame, algorithm, 1, step, action, reward, total_reward, info.get("telemetry")
            )
            frames.append(annotated)

    env.close()

    # Save GIF
    print(f"[*] Rendering {len(frames)} frames to GIF: {output_gif_path} (FPS={fps})")
    imageio.mimsave(output_gif_path, frames, fps=fps, loop=0)
    print(f"[+] Saved gameplay animation! Final Score: {total_reward:.2f} | Steps: {step}")

    return total_reward, step, frames


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Record Autonomous Gameplay Demonstration")
    parser.add_argument("--algo", type=str, required=True, choices=["dqn", "ppo"])
    parser.add_argument("--model", type=str, required=True, help="Model path (.zip)")
    parser.add_argument("--output", type=str, default="results/videos/gameplay.gif")
    parser.add_argument("--seed", type=int, default=1042)
    parser.add_argument("--fps", type=int, default=30)
    args = parser.parse_args()

    record_autonomous_gameplay(
        algorithm=args.algo,
        model_path=args.model,
        output_gif_path=args.output,
        seed=args.seed,
        fps=args.fps,
    )
