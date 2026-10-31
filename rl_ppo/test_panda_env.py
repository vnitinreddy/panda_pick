
import gymnasium as gym
from envs.reach_panda import PandaEnv

# Create environment directly or via gym.make
env = PandaEnv(render_mode=None)

obs, info = env.reset(seed=42)
print("Initial Observation Keys:", obs.keys())
print("Achieved Goal Position:", obs["achieved_goal"])
print("Target Goal Position:", obs["desired_goal"])

for step in range(5):
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    print(f"Step {step+1} | Reward: {reward:.4f} | Dist: {info['distance_to_target']:.4f}")

env.close()
