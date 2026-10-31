import argparse
import os

# Trigger Gymnasium registration & custom environment import
import envs # Import for PandaEnv
from envs.reach_panda import PandaEnv

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import (
    CallbackList,
    CheckpointCallback,
    EvalCallback,
)
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import SubprocVecEnv


def parse_args():
    parser = argparse.ArgumentParser(
        description="Fine-tune SB3 PPO on PandaEnv."
    )

    # Large n-envs gives us 1) faster training 2) safer, smoother ppo updates
    # and 3) better sample diversity. The cons include increased IPC, 
    # core thrashing if n-envs crosses number of cores, and increased rollout
    # batch size (n_steps x n_envs). If your batch size gets too huge, 
    # PPO takes fewer gradient steps per million environment interactions. 
    # The policy can become overly conservative or slow to adapt early in 
    # training unless you adjust parameters like lowering n_steps proportionally
    # and/or adjusting learning rate and/or batch size.
    parser.add_argument("--n-envs", type=int, default=4,
        help="Number of parallel environments to run")

    # The number 42 comes from Douglas Adams' famous sci-fi series, 
    # The Hitchhiker's Guide to the Galaxy, where a supercomputer spends 
    # 7.5 million years calculating the "Answer to the Ultimate Question 
    # of Life, the Universe, and Everything," only to reveal that the 
    # answer is simply 42.
    # TBD: Multi seed evaluation: It is best practice to train across 
    # 3 to 5 different seeds and average the performance curve to 
    # ensure your agent's success isn't just a lucky fluke! 
    parser.add_argument("--seed", type=int, default=42,
        help="Random seed for environment and training reproducibility")

    # Typical range from scratch is 1M+
    # For finetuning, 100K to 500K
    # Best to chose by looking at mean episode reward (rollout/ep_rew_mean) plot
    # against timesteps. Chose the value where the curve plateaus.
    # A common way is to use EvalCallback continuously save best model!
    parser.add_argument("--timesteps", type=int, default=500_000,
        help="Total timesteps for fine-tuning")


    # The total rollout buffer is sliced into mini batches of batch_size.
    # Small values (32-64): Introduce noise into gradient steps, which can help
    # escape local minima early in training
    # Large values (128 - 512): Provide smooth, stable gradients and leverage
    # GPU parallelism efficiently, but require higher learning rates to make rapid progress!
    parser.add_argument("--batch-size", type=int, default=64,
        help="Minibatch size")

    # The number of environment steps each parallel worker collects before 
    # pausing simulation to update the neural network.
    # Smaller values (128 - 512): Great for short, reactive tasks with instantaneous 
    # rewards. However, the agent struggles to credit actions taken far in the past.
    # Larger values (2048 - 4096): Gives agent long enough trajectories
    # to learn multi-step reaching motions.
    # Reducing to 1024 reduced the success rate to 50% with timesteps passed as 100,000
    # and everything else is at default value.
    parser.add_argument("--n-steps", type=int, default=2048,
        help="Rollout buffer length (per update)")

    # The number of times the network iterates over the entire rollout 
    # buffer during a single update cycle.
    parser.add_argument("--n-epochs", type=int, default=10,
        help="Number of SGD epochs per update")

    # This dictates the step size that PyTorch takes along the gradient direction
    # to update policy and value network weights during optimization.
    # How ot works: RL creates a moving target problem. 1) policy collects data based
    # on its current weights 2) It updates its weights using the collected data and
    # 3) the new weights change how the agent acts, which changes the distribution of the 
    # data it collects next.
    # Becuse the policy determines its own distribution, setting learning rate requires
    # a careful balance between learning speed and policy stability.
    # Too High (1e-2 to 1e-3): The optimizer takes massive leaps across loss landscape.
    # In traning, we might see policy collapse (arm suddenly forgetting everything learned
    # and making wild and jerky movements or thrashing), unstable loss curves (policy_gradient_loss,
    # approx_kl), and entropy loss drop (entropy_loss crashing) indicating policy prematurely
    # committing to sub-optimal actions (stasis or thrashing)
    # Too Low (1e-6 or lower): Takes long to train and convergence is slow!
    # Instead of keeping constant, we can use a linear decay schedule (explore aggressively
    # early in training and progressively shrink the step size neart the end to settle
    # precisely into optimal joint control) 
    parser.add_argument("--learning_rate", type=float, default=3e-4,
        help="Learning rate for fine-tuning")

    # The interval of environment timesteps between eval benchmark runs.
    # This should be a multiple of total rollout steps (n_steps * n_envs)
    # as otherwise there won't be updated weights by the time we run eval!
    # Set such that you get enough evaluations over total timesteps
    parser.add_argument("--eval-freq", type=int, default=10_000,
        help="Evaluation frequency per env")

    # Number of complete evaluation episodes the agent runs during each
    # benchmark phase to compute average reward (eval/mean_reward) and 
    # average episode length.
    # When we set the sampling to deterministic (no exploratory noise), the
    # agent plays through eval_episodes from start to finish, logs the
    # average performance to TensorBoard, and saves a checkpoint if the
    # score beats the previous record.
    # Running 1 or 2 episodes can result in accidental accuracy! Good to
    # run 5 to 10 episodes!
    parser.add_argument("--eval-episodes", type=int, default=10,
        help="Number of evaluation episodes per benchmark run")

    parser.add_argument("--checkpoint-freq", type=int, default=100_000,
        help="Checkpoint save frequency per env")

    parser.add_argument("--models-dir", type=str, default="./models/",
        help="Directory to store all model artifacts and checkpoints")
    
    parser.add_argument("--pretrained-path", type=str, default="./models/pretrained_panda_ppo.zip",
        help="Path to pre-trained model .zip file")

    parser.add_argument("--output-filename", type=str, default="finetuned_panda_ppo.zip",
        help="Filename for the final fine-tuned model inside models directory",
    )

    parser.add_argument("--log-dir", type=str, default="./tensorboard_logs/",
        help="TensorBoard log directory")

    parser.add_argument("--verbose", action="store_true",
        help="Verbose SB3 output")

    return parser.parse_args()

def print_args(args):
    args_dict = vars(args)
    
    print()
    print("=" * 62)
    print("  Panda Reach — Training Configuration")
    print("=" * 62)
    
    # Dynamically print all parsed arguments cleanly
    for key, val in args_dict.items():
        formatted_val = f"{val:,}" if isinstance(val, int) else str(val)
        print(f"  {key:<20} : {formatted_val:>35}")
        
    print("-" * 62)
    print(f"  TensorBoard          : tensorboard --logdir {args.log_dir}")
    print("=" * 62)
    print()

def main():
    args = parse_args()

    # Ensure output model directories exist
    models_dir = args.models_dir
    checkpoint_dir = os.path.join(models_dir, "checkpoints")
    best_model_dir = os.path.join(models_dir, "best_model")
    
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(best_model_dir, exist_ok=True)

    output_path = os.path.join(models_dir, args.output_filename)

    # ---------------------------------------------------------------------
    # 0. Setup Environments
    # ---------------------------------------------------------------------
    # Parallel training environments via Gym registry ID & SubprocVecEnv
    train_env = make_vec_env(
        "PandaGymnasium",
        n_envs=args.n_envs,
        seed=args.seed,
        vec_env_cls=SubprocVecEnv,
    )

    # Single, standalone evaluation environment monitored directly
    eval_env = Monitor(PandaEnv())
    eval_env.reset(seed=args.seed)  # Seeds the evaluation environment generator

    # ---------------------------------------------------------------------
    # 1. Setup SB3 Callbacks
    # ---------------------------------------------------------------------
    checkpoint_callback = CheckpointCallback(
        save_freq=max(1, args.checkpoint_freq),
        save_path=checkpoint_dir,
        name_prefix="panda_ppo_checkpoint",
        verbose=1,
       save_vecnormalize=True,
    )

    eval_callback = EvalCallback(
        eval_env=eval_env,
        best_model_save_path=best_model_dir,
        log_path=args.log_dir,
        eval_freq=max(1, args.eval_freq // args.n_envs),
        n_eval_episodes=args.eval_episodes,
        deterministic=True,
        verbose=1,
        render=False,
    )

    callback_list = CallbackList([checkpoint_callback, eval_callback])

    # ---------------------------------------------------------------------
    # 2. Load Pre-trained Model or Initialize New
    # ---------------------------------------------------------------------
    if os.path.exists(args.pretrained_path):
        print(f"Loading pre-trained model from '{args.pretrained_path}'...")
        model = PPO.load(
            args.pretrained_path,
            env=train_env,
            tensorboard_log=args.log_dir,
            custom_objects={"learning_rate": args.learning_rate},
        )
    else:
        print(f"No checkpoint found at '{args.pretrained_path}'. Initializing new model...")
        model = PPO(
            policy="MlpPolicy",
            env=train_env,
            learning_rate=args.learning_rate,
            n_steps=args.n_steps,
            batch_size=args.batch_size,
            n_epochs=args.n_epochs,
            verbose=1 if args.verbose else 0,
            seed=args.seed,
            tensorboard_log=args.log_dir,
        )

    # ---------------------------------------------------------------------
    # 3. Execute Fine-Tuning
    # ---------------------------------------------------------------------
    print_args(args)

    model.learn(
        total_timesteps=args.timesteps,
        callback=callback_list,
        reset_num_timesteps=False,
        tb_log_name="Panda_PPO_FineTuning",
        progress_bar=True,
    )

    # ---------------------------------------------------------------------
    # 4. Save Final Output to models directory
    # ---------------------------------------------------------------------
    model.save(output_path)
    print(f"Final fine-tuned model saved to '{output_path}'")
    print(f"Best model saved to '{os.path.join(best_model_dir, 'best_model.zip')}'")

    train_env.close()
    eval_env.close()


if __name__ == "__main__":
    main()
