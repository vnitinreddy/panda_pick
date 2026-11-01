# panda_pick_rl

This project uses StableBaselines Reinforcement Learning (RL) algorithms for fine tuning a model to predict actions
for Pick-and-Place tasks of robot arms in MuJoCo simulation.

Fine-tuned OpenVLA with reinforcement learning (Stable-Baselines3 Proximal Policy Optimization) and evaluated a Panda arm’s success rate on pick-and-place tasks.

MuJoCo is an open source advanced physics simulation frame work from Google Deepmind and 
you can learn about it in https://mujoco.org/

The StableBaselines3 provides reliable RL implementations in PyTorch and you can learn more at
https://stable-baselines3.readthedocs.io/en/master/

Gymnaisum: An API standard for reinforcement learning with a diverse collection of 
reference environments. You can learn more at https://gymnasium.farama.org/index.html


1. Setup the environment (install the required libraries and dependencies), run

conda env create -f env_panda.yml

Note the environment is setup for Macbook with Intel processor!

2. Activate the environment

conda activate arm_rl_x86

3. Test the ARM control

mjpython scripts/control_panda.py 

The scripts assume ARM is using position control. You can run the below script to check
whether it is configured for position control or not!

python scripts/find_panda_actuator_config_type.py

4. Set up the Gymnaisum environment for the agent to manipulate panda

envs/__init__.py: This script registers the PandaEnv with Gymanisum.

register(id="PandaGymnasium", entry_point="envs.panda_env:PandaEnv")

The id is an identifier for the Panda Environment.
Entry Point defines the required class PandaEnv in the required file envs/panda_env.py 

envs/reach_panda.py: Implements the Panda manipulation tasks with Gymanisum interface.

5. Use RL for fine tuning the models

python train_panda.py --timesteps 20000

6. Evalure the fine tuned model

mjpython evaluate_panda.py --model-path models/finetuned_panda_ppo.zip --render --episodes 100

