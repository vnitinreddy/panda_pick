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

Configuration
Defines the task suite, image size, steps, output folder, and GPU.
Example:
model_type = "openvla"
task_suite = "libero_object"
maximum_steps = 280
warmup_steps = 20
LIBERO environment setup
create_environment(task_info, image_size):

Finds the BDDL task file for the chosen LIBERO task.
Builds an OffScreenRenderEnv environment.
Seeds it for reproducibility.
get_task(task_suite_name, task_number):

Loads the benchmark definition for a task suite.
Retrieves the selected task instance.
Image processing
prepare_image():

Takes the agent camera image from LIBERO.
Flips it 180° because the camera view is rotated.
Resizes it to the model input size.
This is important because OpenVLA expects images in a specific format.
resize_image():

Encodes the image to JPEG, decodes it, resizes it with TensorFlow, and converts to uint8.
This keeps the image compatible with the model.
Gripper conversion
process_gripper(action_vector):
LIBERO uses a different gripper convention than OpenVLA output.
OpenVLA predicts gripper values in [0, 1], while LIBERO expects [-1, +1].
This function converts and clamps the gripper value to binary open/close.
It also flips the sign because LIBERO uses the opposite convention.
No-op action
no_op_action():
Returns [0, 0, 0, 0, 0, 0, -1]
This is used during warmup so the simulation can settle before real actions.
Video recording
save_video():
Stores all frames from a rollout as an MP4 in outputs/videos.
Uses imageio.
Filenames include the episode number and a safe version of the instruction.
OpenVLA wrapper
OpenVLA.init():

Loads the processor and model from Hugging Face.
Example model names:
openvla/openvla-7b-finetuned-libero-object
openvla/openvla-7b-finetuned-libero-spatial
openvla/openvla-7b-finetuned-libero-goal
openvla/openvla-7b-finetuned-libero-10
Uses flash attention and bfloat16 on the GPU.
predict():

Converts the image to PIL RGB.
Builds a text prompt:
“In: What action should the robot take to [instruction]?”
Passes the prompt and image into the model processor.
Calls self.model.predict_action(...)
Returns a predicted action vector.
ArmAgent
This is the main logic class.

init():

Chooses the correct OpenVLA checkpoint for the selected suite.
Loads LIBERO task info and initial states.
Creates the simulation environment.
Picks the correct normalization statistics for action scaling.
run(instruction, episode_number):

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
Command line usage
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
In plain English: This file is a simulation-based robot policy runner. It uses a language-conditioned vision model to decide what movement the robot should make to complete a task in LIBERO, and it records the whole process as a video.

It is basically:

robot camera → vision-language model → action command → LIBERO environment → repeat
One important detail: This is not controlling a real robot. It is controlling a simulated robot inside LIBERO. The code is built for benchmarking or evaluation, not for physical hardware.

If you want, I can also walk through:

the exact action format used by LIBERO,
how OpenVLA’s prompt is constructed,
or a line-by-line explanation of the control loop.

vla_libero3.ipynb 
-----------------
This jupyter notebook 
