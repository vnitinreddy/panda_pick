#Panda Pick

An inference-to-control pipeline for a Franka Panda arm to execute pick-and-place tasks based on action predictions from the OpenVLA model hosted on Google Colab.

test_scripts/ : This folder contains various test scripts used to be able to understand and debug issues.
backup/ : This folder contains the backup scripts and jupyter notebooks used. The bridge_to_brain.py is a client that can run on local machine and connect with the OpenVLA running on Google colab (or similar environment) using the open_vla_only_on_colab.ipynb.

The franka_agent/open_vla_and_mujoco_on_colab4.ipynb notebook contains the logic to run both the Panda and OpenVLA in Google Colab. Just copy the notebook to your colab and run it on an env with GPU (I tested it on A100 GPU). It sets up the required packages and then runs the inferencing loop needed for the given pick and place task.





