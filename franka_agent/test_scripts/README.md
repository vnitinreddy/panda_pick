#Panda Pick

An inference-to-control pipeline for a Franka Panda arm to execute pick-and-place tasks based on action predictions from the OpenVLA model hosted on Google Colab.

Below are few python scripts I have used to be able to understand how robot can be controlled and how to calculate a good camera location.

* franka_agent/move_robot.py: This is a basic visualization/demo script to animate a robot arm and verify that the MuJoCo model responds to joint control inputs.

* franka_agent/capture_test.py: This file is a small MuJoCo visualization script used to generate a sample camera observation from a robot task scene.

* franka_agent/reach_cube.py: This script continuously “pulls” the gripper toward the cube by applying force-like control commands proportional to the distance between them. It’s a very basic, direct control loop rather than a full motion-planning or inverse-kinematics controller.

* franka_agent/render_image.py: This file is meant to generate a static visual observation image from the MuJoCo environment, for feeding into a vision model or pipeline such as OpenVLA.

* franka_agent/test_mj_step.py: This is a basic animation/demo script to visually verify that the robot model loads correctly and responds to simple joint control inputs.




For testing Panda with OpenVLA running in Google Colab and client running in your laptop:
1. Copy the notebook (franka_agent/open_vla_on_colab.ipynb) to your colab and run it on an env with GPU (I tested it on A100 GPU).
2. Run the bridge_to_brain.py in your laptop

For testing Panda with both OpenVLA and client running in Google Colab:
1. Copy the notebook (franka_agent/open_vla_on_colab.ipynb) to your colab and run it on an env with GPU (I tested it on A100 GPU).


For testing LIBERO:
1. Copy the notebook (arm_agent/vla_libero.ipynb) to your colab and run it on an env with GPU (I tested it on A100 GPU).
2. dds


