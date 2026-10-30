#Panda Pick

This project uses OpenVLA to control robotic arms (Franka Emika Panda and LIBERO) for Pick up tasks.

For testing LIBERO:
1. Copy the notebook (arm_agent/vla_libero.ipynb) to your colab and run it on an env with GPU (I tested it on A100 GPU).
2. dds

For testing Panda with OpenVLA running in Google Colab and client running in your laptop:
1. Copy the notebook (franka_agent/open_vla_on_colab.ipynb) to your colab and run it on an env with GPU (I tested it on A100 GPU).
2. Run the bridge_to_brain.py in your laptop

For testing Panda with both OpenVLA and client running in Google Colab:
1. Copy the notebook (franka_agent/open_vla_on_colab.ipynb) to your colab and run it on an env with GPU (I tested it on A100 GPU).


franka_agent/move_robot.py: Allows you to move the robot to a given point
franka_agent/camera_to_xml.py:
franka_agent/franka_agent.py:
