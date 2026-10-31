#Panda Pick

An inference-to-control pipeline for a Franka Panda arm to execute pick-and-place tasks based on action predictions from the OpenVLA model hosted on Google Colab.

test_scripts/ : This folder contains various test scripts used to be able to understand and debug issues.
backup/ : This folder contains the backup scripts and jupyter notebooks used. The bridge_to_brain.py is a client that can run on local machine and connect with the OpenVLA running on Google colab (or similar environment) using the open_vla_only_on_colab.ipynb.

The franka_agent/open_vla_and_mujoco_on_colab4.ipynb notebook contains the logic to run both the Panda and OpenVLA in Google Colab. Just copy the notebook to your colab and run it on an env with GPU (I tested it on A100 GPU). It sets up the required packages and then runs the inferencing loop needed for the given pick and place task.

At a high level, it’s a Colab-based robotics experiment notebook that connects a MuJoCo Panda environment to an OpenVLA model for simulated grasping and manipulation. The notebook does the following:

* Sets up a persistent Python environment in Colab
* Installs the required ML and simulation dependencies (PyTorch, Transformers, MuJoCo, MediaPipe, etc.)
* Mounts Google Drive and prepares a working directory for the project
* Verifies the environment and package versions
* Loads or initializes a MuJoCo model for the Panda robot
* Builds a custom control loop that:
    * renders a camera image from the simulated scene
    * sends the image plus a task prompt to the OpenVLA model
    * gets an action prediction
    * converts that action into target movement / gripper commands
    * steps the MuJoCo simulation to move the arm toward the target
Repeats this in a loop to simulate a robot picking and placing an object







