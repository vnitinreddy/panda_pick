"""
Computes evaluation metrics (Success Rate, Mean Reward) and visualizes the policy.
"""

import os
import sys
import time
import argparse
import numpy as np
import zipfile
import json
import torch
from typing import Tuple, Type
from stable_baselines3 import PPO, SAC, TD3, A2C, DDPG, DQN
from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.common.save_util import load_from_zip_file
from datetime import datetime

import envs


def parse_args() -> argparse.Namespace:
    """Parses command line arguments for the evaluation script."""
    parser = argparse.ArgumentParser(description="Evaluate trained SB3 policy on PandaEnv.")

    # Model Path
    parser.add_argument("--model-path", type=str, default="finetuned_panda_ppo.zip", 
        help="Path to trained model .zip file")

    # Number of episodes for evaluation
    parser.add_argument("--episodes", type=int, default=20, 
        help="Number of evaluation episodes")

    # Is rendering needed?
    parser.add_argument("--render", action="store_true", 
        help="Enable 3D window rendering for visual evaluation")

    parser.add_argument("--render-delay", type=float, default=0.02,
        help="Delay in seconds between steps during rendering (default: 0.02)")

    # Is recording of evaluation episodes needed?
    parser.add_argument("--record", action="store_true",
        help="Enable video recording of evaluation episodes using Gymnasium wrapper")

    # Path to save recorded videos
    parser.add_argument("--video-dir", type=str, default="eval_videos",
        help="Directory where recorded episode videos will be saved")

    # seed for environment
    parser.add_argument("--seed", type=int, default=42,
        help="Random seed for environment and action sampling reproducibility")

    # environment identifier
    parser.add_argument("--env-id", type=str, default="PandaGymnasium",
        help="Gym environment ID.")

    # Deterministic or Stochastic?
    parser.add_argument("--stochastic", action="store_true",
        help="Use stochastic actions (with policy noise) instead of deterministic optimal policy actions")

    # Where to save the results?
    parser.add_argument("--output-path", type=str, default="results/eval_results.json",
        help="Path to save evaluation summary as JSON")

    # Verbose output
    parser.add_argument("--verbose", action="store_true",
        help="Print additional debug and step-level details during evaluation")

    return parser.parse_args()


def print_args(args: argparse.Namespace) -> None:
    """Prints the parsed arguments in a clean, readable format."""
    print("=" * 50)
    print("      EVALUATION CONFIGURATION")
    print("=" * 50)
    for arg, value in vars(args).items():
        print(f"  • {arg:<15}: {value}")
    print("=" * 50 + "\n")


def create_env(args):
    import gymnasium

    render_mode = "human" if args.render else "rgb_array"
    env = gymnasium.make(args.env_id, render_mode=render_mode)

    if args.record:
        if args.render:
            print("Skipping recording as rendering is enabled")
            return env
        try:
            import moviepy  # noqa: F401
        except ImportError:
            raise RuntimeError(
                "Video recording requires 'moviepy' and 'imageio-ffmpeg'.\n"
                "Please run: pip install moviepy imageio-ffmpeg"
            )

        from gymnasium.wrappers import RecordVideo

        # Create unique timestamp subfolder path (do NOT call os.makedirs beforehand)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        video_folder = os.path.join(os.path.abspath(args.video_dir), timestamp)

        env = RecordVideo(
            env,
            video_folder=video_folder,
            episode_trigger=lambda ep_id: True,
            name_prefix="eval",
            disable_logger=True
        )

        print(f"Recording videos to: {video_folder}")

    return env


# Order algorithms: candidate in filename first, then common continuous/discrete algorithms
ALGORITHMS: list[Tuple[str, Type[BaseAlgorithm]]] = [
    ("PPO", PPO),
    ("SAC", SAC),
    ("TD3", TD3),
    ("A2C", A2C),
    ("DDPG", DDPG),
    ("DQN", DQN),
]


def load_sb3_model(
    model_path: str, env=None, device: str = "auto"
) -> BaseAlgorithm:
    """
    Attempts to load an SB3 model archive by checking filename heuristics
    first, and then falling back to trial-loading known algorithm classes.
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at: {model_path}")

    filename_upper = os.path.basename(model_path).upper()

    # Prioritize algorithms matching substrings in the file path
    candidates = sorted(
        ALGORITHMS,
        key=lambda item: 0 if item[0] in filename_upper else 1
    )

    errors = {}
    for algo_name, algo_class in candidates:
        try:
            model = algo_class.load(model_path, env=env, device=device)
            print(f"Successfully loaded '{model_path}' using {algo_name}.load()")
            return model 
        except Exception as e:
            errors[algo_name] = str(e)

    error_msg = "\n".join([f"  • {name}: {err}" for name, err in errors.items()])
    raise RuntimeError(
        f"Failed to load '{model_path}' with any supported SB3 algorithm:\n{error_msg}"
    )


def print_model_details(model: BaseAlgorithm, args) -> None:
    """Prints structural details, architecture, and spaces of the loaded model."""
    total_params = sum(p.numel() for p in model.policy.parameters() if p.requires_grad)

    print("=" * 50)
    print("      MODEL DETAILS & ARCHITECTURE")
    print("=" * 50)
    print(f"  • Algorithm:        {model.__class__.__name__}")
    print(f"  • Policy Class:     {model.policy.__class__.__name__}")
    print(f"  • Device:           {model.device}")
    print(f"  • Learning Rate:    {model.learning_rate}")
    print(f"  • Total Parameters: {total_params:,}")
    print(f"  • Obs Space Shape:  {model.observation_space.shape}")
    print(f"  • Act Space Shape:  {model.action_space.shape}")
    print("-" * 50)
    if args.verbose:
        print("  • Network Architecture:")
        print(model.policy)
        print("=" * 50 + "\n")


def print_n_save_results(
    model,
    args,
    time_taken,
    episode_rewards: list[float],
    episode_lengths: list[int],
    successes: list[float]
) -> None:
    """Computes and displays aggregated evaluation statistics."""

    results = {
        "n_episodes": args.episodes,
        "mean_reward": float(np.mean(episode_rewards)),
        "std_reward": float(np.std(episode_rewards)),
        "min_reward": float(np.min(episode_rewards)),
        "max_reward": float(np.max(episode_rewards)),
        "median_reward": float(np.median(episode_rewards)),
        "mean_length": float(np.mean(episode_lengths)),
        "std_length": float(np.std(episode_lengths)),
        "success_rate": float(np.mean(successes)),
        "total_successes": int(np.sum(successes)),
    }
    results["timestamp"] = datetime.now().isoformat()
    results["time_seconds"] = round(time_taken, 2)
    results["algorithm"] = model.__class__.__name__
    results["model_path"] = os.path.abspath(args.model_path)
    results["deterministic"] = not args.stochastic
    results["seed"] = args.seed

    # Print Evaluation Summary
    print("-" * 50)
    print(f"Evaluation Summary ({args.episodes} Episodes):")
    print(f"  • Success Rate:   {(results['success_rate'] * 100):.1f}%")
    print(f"  • Mean Reward:    {results['mean_reward']:.2f} +/- {results['std_reward']:.2f}")
    print(f"  • Median Reward:  {results['median_reward']:.2f}")
    print(f"  • Mean episode length:  {results['mean_length']:.1f}"
        f"  ± {results['std_length']:.1f}"
    )
    print(f"  • Min / Max reward   : {results['min_reward']:.2f}"
        f"  /  {results['max_reward']:.2f}"
    )
    print(f"  • Wall time: {time_taken:.1f}s "
          f"({time_taken / args.episodes:.2f}s per episode)")
    print("-" * 50)

    # Save if output_path is provided!
    if args.output_path:
        save_dir = os.path.dirname(args.output_path)
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)

        full_results = {
            **results,
            "per_episode": {
                "rewards": episode_rewards,
                "lengths": episode_lengths,
                "successes": successes,
            },
        }

        with open(args.output_path, "w") as f:
            json.dump(full_results, f, indent=2)

        print(f"  Results saved to: {os.path.abspath(args.output_path)}")


def evaluate_policy(args, env):
    """Evaluates a trained model over multiple episodes."""
    if not os.path.exists(args.model_path):
        raise FileNotFoundError(f"Model file not found at: {args.model_path}")

    # Load trained model
    print(f"Loading model from: {args.model_path}")
    model = load_sb3_model(args.model_path, env=env)
    print_model_details(model, args)

    episode_rewards = []
    episode_lengths = []
    successes = []

    deterministic = not args.stochastic

    # Seed global NumPy random state once
    if args.seed is not None:
        np.random.seed(args.seed)

    print(f"\nStarting Evaluation for {args.episodes} episodes...\n" + "-" * 50)

    start_time = time.time()

    for episode in range(1, args.episodes + 1):
        # Increment seed per episode for distinct, reproducible initial scenes
        episode_seed = args.seed + episode if args.seed is not None else None
        obs, info = env.reset(seed=episode_seed)
        
        done = False
        truncated = False
        episode_reward = 0.0
        steps = 0

        while not (done or truncated):
            action, _states = model.predict(obs, deterministic=deterministic)
            obs, reward, done, truncated, info = env.step(action)
            episode_reward += reward
            steps += 1

            if args.render:
                time.sleep(args.render_delay)

        is_success = info.get("is_success", False)
        distance = info.get("distance", 0.0)

        episode_rewards.append(episode_reward)
        episode_lengths.append(steps)
        successes.append(1.0 if is_success else 0.0)

        if args.verbose or not args.render:
            print(f"Episode {episode:02d} | "
                  f"Reward: {episode_reward:7.2f} | "
                  f"Steps: {steps:3d} | "
                  f"Final Dist: {distance:.3f}m | "
                  f"Success: {'YES' if is_success else 'NO'}")

    time_taken = time.time() - start_time

    print_n_save_results(
        model=model,
        args=args,
        time_taken=time_taken,
        episode_rewards=episode_rewards,
        episode_lengths=episode_lengths,
        successes=successes
    )

    # Cleanly close video recorder wrapper using Gymnasium get_wrapper_attr API
    if hasattr(env, "get_wrapper_attr"):
        try:
            recorder = env.get_wrapper_attr("video_recorder")
            if recorder is not None:
                recorder.close()
        except AttributeError:
            pass


def main():
    args = parse_args()
    print_args(args)

    env = create_env(args=args)

    try:
        evaluate_policy(args=args, env=env)
    finally:
        if hasattr(env, "close"):
            env.close()

        # Brief pause to allow OS file handles to release .mp4 files safely
        time.sleep(0.5)


if __name__ == "__main__":
    main()
