Simply copy the jupyter notebook and run it. It sets up the environment, loads and runs the model on GPU, and downloads the agent script (arm_agent.py) and runs it as a subprocess. The agent script controls the simulated robot arm by taking a snapshot of robot arm environment, sending it to OpenVLA for action prediction, and use the action predicted to move the Panda robot arm inside the LIBERO environment.

robot camera → vision-language model → action command → LIBERO environment → repeat

The arm_agent.py 
----------------
This script is a robot-control runner for a simulated robot arm in the LIBERO benchmark, using a pretrained OpenVLA vision-language-action model.

High level:
- It loads a robot task environment from LIBERO.
- It takes a natural-language instruction like “pick up the red bowl”.
- It captures the robot’s camera view.
- It feeds the image + instruction to OpenVLA.
- The model predicts a low-level action vector for the robot arm.
- The action is executed in the simulation.
- It repeats until success or a max step limit.
- It records a video of the rollout.

What each part does:

1. Configuration
Defines the task suite, image size, steps, output folder, and GPU.

2. LIBERO environment setup

  a. create_environment(task_info, image_size):
  Finds the BDDL task file for the chosen LIBERO task.
  Builds an OffScreenRenderEnv environment.
  Seeds it for reproducibility.
  
  b. get_task(task_suite_name, task_number):
  Loads the benchmark definition for a task suite.
  Retrieves the selected task instance.

3. Image processing

  a. prepare_image():
  Takes the agent camera image from LIBERO.
  Flips it 180° because the camera view is rotated.
  Resizes it to the model input size.
  This is important because OpenVLA expects images in a specific format.
  
  b. resize_image():
  Encodes the image to JPEG, decodes it, resizes it with TensorFlow, and converts to uint8.
  This keeps the image compatible with the model.

4. Gripper conversion: process_gripper(action_vector):
  LIBERO uses a different gripper convention than OpenVLA output.
  OpenVLA predicts gripper values in [0, 1], while LIBERO expects [-1, +1].
  This function converts and clamps the gripper value to binary open/close.
  It also flips the sign because LIBERO uses the opposite convention.

5. No-op action
  Returns [0, 0, 0, 0, 0, 0, -1]
  This is used during warmup so the simulation can settle before real actions.

6. Video recording: save_video():
  Stores all frames from a rollout as an MP4 in outputs/videos. Uses imageio. Filenames include the episode number and a safe version of the instruction.

7. OpenVLA wrapper: OpenVLA.init():
  a. Loads the processor and model from Hugging Face.
  Example model names:
  openvla/openvla-7b-finetuned-libero-object
  openvla/openvla-7b-finetuned-libero-spatial
  openvla/openvla-7b-finetuned-libero-goal
  openvla/openvla-7b-finetuned-libero-10
  Uses flash attention and bfloat16 on the GPU.
  
  b. predict():
  Converts the image to PIL RGB.
  Builds a text prompt:
    “In: What action should the robot take to [instruction]?”
  Passes the prompt and image into the model processor.
  Calls self.model.predict_action(...)
  Returns a predicted action vector.

8. ArmAgent
  This is the main logic class.
  
  a. init():
  Chooses the correct OpenVLA checkpoint for the selected suite.
  Loads LIBERO task info and initial states.
  Creates the simulation environment.
  Picks the correct normalization statistics for action scaling.
  
  b. run(instruction, episode_number):
  Resets the environment.
  Loads the initial state for the episode.
  For a while:
    During warmup, it keeps sending no-op actions.
    Then it captures a camera image.
    It records the frame.
    It asks OpenVLA to predict the next action.
    It converts the gripper command.
    It executes the action in the simulation.
    If the task is finished, it breaks.
  At the end, if not successful, it prints failure.
  Saves the video of the full episode.

9. Command line usage
read_arguments():
  Parses:
  --prompt: instruction for the robot
  --task: LIBERO suite
  --task_id: task number
  --image_resize
  --output_video

main:
  Creates ArmAgent
  Runs the episode
  Prints whether it succeeded


vla_libero3.ipynb 
-----------------
This jupyter notebook 
