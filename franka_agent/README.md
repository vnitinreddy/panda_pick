#Panda Pick

An inference-to-control pipeline for a Franka Panda arm to execute pick-and-place tasks based on action predictions from the OpenVLA model hosted on Google Colab.

test_scripts/ : This folder contains various test scripts used to be able to understand and debug issues.
local_agent/ : This folder contains the logic where the agent script (client) that launches the robot arm simulation and controls runs on local machine and connects to the OpenVLA model running in Google Colab over the network for predictions. See details in the README.md file in this folder on how to use this split (simulation with screen rendering local and OpenVLA in headless Google Colab) method.

The below 2 files contain the logic to running both the Panda and OpenVLA in Google Colab to avoid the slowness due to network latency. Just copy the notebook (franka_agent/open_vla_on_colab.ipynb) to your colab and run it on an env with GPU (I tested it on A100 GPU).


