#Panda Pick

This project uses OpenVLA to control Panda arm for Pick up tasks.

The franka_emika_panda folder has the MuJoCo simulation of robotic ARM.

The franka_agent folder has the code for the client (or agent) and jupyter notebooks used for controlling the Panda using OpenVLA predictions.

The arm_agent folder has the code to use a fine tuned OpenVLA model and LIBERO environment (containing Panda and exposed through robosuite API) for the Pick and Place Tasks.

The rl_ppo contains the logic for using StableBaselines Reinforcement Learning (RL) algorithm for fine tuning the Panda.
