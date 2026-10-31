#Panda Pick

This project uses OpenVLA to control Panda arm for Pick up tasks.

Built an inference-to-control pipeline for a Franka Panda arm to execute pick-and-place tasks based on action predictions from the OpenVLA model hosted on Google Colab

franka_agent/move_robot.py: Allows you to move the robot to a given point
franka_agent/camera_to_xml.py:
franka_agent/franka_agent.py:

For testing Panda with OpenVLA running in Google Colab and client running in your laptop:
1. Copy the notebook (franka_agent/open_vla_on_colab.ipynb) to your colab and run it on an env with GPU (I tested it on A100 GPU).
2. Run the bridge_to_brain.py in your laptop

For testing Panda with both OpenVLA and client running in Google Colab:
1. Copy the notebook (franka_agent/open_vla_on_colab.ipynb) to your colab and run it on an env with GPU (I tested it on A100 GPU).


For testing LIBERO:
1. Copy the notebook (arm_agent/vla_libero.ipynb) to your colab and run it on an env with GPU (I tested it on A100 GPU).
2. dds


