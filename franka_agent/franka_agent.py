
#@title Inference
# ============================================================
# OpenVLA + Robosuite Panda Inference
#
# Pipeline:
#
#   Robosuite camera
#        ↓
#   image preprocessing
#        ↓
#   OpenVLA 7B (base model)
#        ↓
#   bridge_orig 7-D action
#        ↓
#   Robosuite OSC_POSE controller
#        ↓
#   Panda
#
# Action:
#   [dx, dy, dz, dRx, dRy, dRz, gripper]
#
# The robot controller / IK is handled by Robosuite.
# ============================================================


# ============================================================
# 1. IMPORTS
# ============================================================

import os
import time
import math
import imageio

import numpy as np
import torch

from PIL import Image
from transformers import AutoProcessor, AutoModelForVision2Seq

import robosuite as suite
from robosuite.controllers import load_controller_config


# ============================================================
# 2. CONFIGURATION
# ============================================================

class Config:

    # --------------------------------------------------------
    # OpenVLA
    # --------------------------------------------------------

    model_name = "openvla/openvla-7b"

    # IMPORTANT:
    # This is the action normalization key used by the
    # general/base OpenVLA model for BridgeData.
    normalization_key = "bridge_orig"

    gpu = "cuda:0"

    # --------------------------------------------------------
    # Task
    # --------------------------------------------------------

    instruction = "pick up the red cube"

    # --------------------------------------------------------
    # Camera
    # --------------------------------------------------------

    camera_name = "agentview"

    camera_height = 224
    camera_width = 224

    # --------------------------------------------------------
    # Simulation
    # --------------------------------------------------------

    control_freq = 20

    maximum_steps = 300

    # Let the simulation settle before asking OpenVLA
    # for the first action.
    warmup_steps = 20

    # --------------------------------------------------------
    # Output
    # --------------------------------------------------------

    output_folder = "/content/PickAgent/outputs/videos"

    episode_number = 0

    # --------------------------------------------------------
    # Action safety
    # --------------------------------------------------------

    # Number of repeated controller steps per VLA action.
    #
    # Start with 1.
    #
    # If movement is too slow, this can be increased later.
    action_repeat = 1

    # --------------------------------------------------------
    # Debug
    # --------------------------------------------------------

    print_actions = True
    print_robot_state = True


# ============================================================
# 3. RUNTIME ENVIRONMENT
# ============================================================

os.environ["TOKENIZERS_PARALLELISM"] = "false"

# We are running this script with:
#
# /content/openvla_env/bin/python
#
# so the following paths are useful when running the script
# directly from PickAgent.
#
# They are harmless if the paths already exist in sys.path.

import sys

PROJECT_PATHS = [
    "/content/PickAgent",
    "/content/openvla",
]

for path in PROJECT_PATHS:
    if path not in sys.path:
        sys.path.insert(0, path)


# ============================================================
# 4. OPENVLA
# ============================================================

class OpenVLA:
    """
    Small wrapper around the general OpenVLA model.
    """

    def __init__(
        self,
        model_name,
        gpu="cuda:0",
        normalization_key="bridge_orig",
    ):

        self.gpu = gpu
        self.normalization_key = normalization_key

        print()
        print("=" * 70)
        print("Loading OpenVLA")
        print("=" * 70)

        print("Model:", model_name)
        print("GPU:", gpu)
        print("Normalization:", normalization_key)

        # ----------------------------------------------------
        # Processor
        # ----------------------------------------------------

        print()
        print("Loading processor...")

        self.processor = AutoProcessor.from_pretrained(
            model_name,
            trust_remote_code=True,
        )

        # ----------------------------------------------------
        # Model
        # ----------------------------------------------------

        print("Loading model...")

        self.model = AutoModelForVision2Seq.from_pretrained(
            model_name,

            # Use Flash Attention when available.
            attn_implementation="flash_attention_2",

            torch_dtype=torch.bfloat16,

            low_cpu_mem_usage=True,

            trust_remote_code=True,
        ).to(gpu)

        self.model.eval()

        print()
        print("OpenVLA loaded successfully.")

        print(
            "Model dtype:",
            next(self.model.parameters()).dtype,
        )

        print(
            "Model device:",
            next(self.model.parameters()).device,
        )

        print("=" * 70)
        print()


    @torch.inference_mode()
    def predict(
        self,
        image,
        instruction,
    ):
        """
        Predict one OpenVLA action.

        Returns:
            np.ndarray with shape (7,)
        """

        # ----------------------------------------------------
        # Convert image to PIL
        # ----------------------------------------------------

        if isinstance(image, np.ndarray):

            image = Image.fromarray(
                image
            ).convert("RGB")

        elif not isinstance(image, Image.Image):

            raise TypeError(
                f"Unsupported image type: {type(image)}"
            )

        # ----------------------------------------------------
        # OpenVLA prompt
        # ----------------------------------------------------

        prompt = (
            "In: What action should the robot take to "
            f"{instruction.lower()}?\nOut:"
        )

        # ----------------------------------------------------
        # Processor
        # ----------------------------------------------------

        inputs = self.processor(
            prompt,
            image,
        )

        # Move tensors to GPU.
        #
        # We do NOT blindly convert every object to bfloat16.
        # Integer tensors such as input_ids must remain integer.
        #
        inputs = {
            key: value.to(self.gpu)
            if hasattr(value, "to")
            else value
            for key, value in inputs.items()
        }

        # ----------------------------------------------------
        # Predict
        # ----------------------------------------------------

        action = self.model.predict_action(
            **inputs,

            unnorm_key=self.normalization_key,

            do_sample=False,
        )

        # ----------------------------------------------------
        # Normalize output type
        # ----------------------------------------------------

        if isinstance(action, torch.Tensor):

            action = action.detach().float().cpu().numpy()

        else:

            action = np.asarray(action)

        action = np.asarray(
            action,
            dtype=np.float32,
        ).reshape(-1)

        if action.shape[0] != 7:

            raise RuntimeError(
                "OpenVLA returned an unexpected action shape: "
                f"{action.shape}. Expected 7 values."
            )

        return action


# ============================================================
# 5. ROBOSUITE ENVIRONMENT
# ============================================================

def create_environment():
    """
    Create a Robosuite Panda environment.

    OSC_POSE receives a normalized 7-D action:

        [dx, dy, dz, dRx, dRy, dRz, gripper]

    Robosuite performs the robot-level control internally.
    """

    print()
    print("=" * 70)
    print("Creating Robosuite environment")
    print("=" * 70)

    # --------------------------------------------------------
    # Controller
    # --------------------------------------------------------

    controller_config = load_controller_config(
        default_controller="OSC_POSE"
    )

    print()
    print("Controller:")
    print("  OSC_POSE")

    # --------------------------------------------------------
    # Environment
    # --------------------------------------------------------

    env = suite.make(
        "Lift",

        robots="Panda",

        controller_configs=controller_config,

        # ----------------------------------------------------
        # Rendering
        # ----------------------------------------------------

        has_renderer=False,

        has_offscreen_renderer=True,

        render_camera=Config.camera_name,

        # ----------------------------------------------------
        # Camera observations
        # ----------------------------------------------------

        use_camera_obs=True,

        camera_names=Config.camera_name,

        camera_heights=Config.camera_height,

        camera_widths=Config.camera_width,

        # ----------------------------------------------------
        # Simulation
        # ----------------------------------------------------

        control_freq=Config.control_freq,

        horizon=Config.maximum_steps,

        # Reward is useful for diagnostics.
        reward_shaping=True,

        # ----------------------------------------------------
        # Determinism
        # ----------------------------------------------------

        ignore_done=False,
    )

    print()
    print("Robosuite environment created successfully.")

    print("=" * 70)
    print()

    return env


# ============================================================
# 6. IMAGE PROCESSING
# ============================================================

def prepare_image(observation):
    """
    Convert Robosuite's camera observation into the image
    expected by OpenVLA.

    IMPORTANT:

    Robosuite camera observations are vertically flipped
    relative to the normal image convention.

    Therefore we flip Y only:

        image[::-1]

    We do NOT rotate 180 degrees like the LIBERO code.
    """

    if Config.camera_name not in observation:

        raise RuntimeError(
            f"Camera '{Config.camera_name}' not found in "
            f"observation keys: {list(observation.keys())}"
        )

    image = observation[
        Config.camera_name
    ]

    image = np.asarray(
        image,
        dtype=np.uint8,
    )

    # --------------------------------------------------------
    # Robosuite / MuJoCo camera convention
    # --------------------------------------------------------

    image = image[::-1]

    # --------------------------------------------------------
    # Ensure RGB
    # --------------------------------------------------------

    if image.ndim != 3:

        raise RuntimeError(
            f"Unexpected camera image shape: {image.shape}"
        )

    if image.shape[2] != 3:

        raise RuntimeError(
            f"Expected RGB image, got: {image.shape}"
        )

    return image


# ============================================================
# 7. OPENVLA → ROBOSUITE ACTION
# ============================================================

def convert_vla_action_to_robosuite(
    vla_action,
):
    """
    Convert OpenVLA's 7-D BridgeData action to the action
    expected by the Robosuite OSC_POSE controller.

    OpenVLA:

        [dx, dy, dz, dRx, dRy, dRz, gripper]

    Robosuite OSC_POSE:

        [dx, dy, dz, dRx, dRy, dRz, gripper]

    The important point is that Robosuite owns the
    robot-level IK / OSC computation.

    We therefore DO NOT calculate a Jacobian here.

    --------------------------------------------------------

    Gripper:

        OpenVLA:
            0 = open
            1 = closed

        Robosuite:
            -1 = open
            +1 = closed

    """

    action = np.asarray(
        vla_action,
        dtype=np.float32,
    ).copy()

    if action.shape != (7,):

        action = action.reshape(-1)

        if action.shape[0] != 7:

            raise RuntimeError(
                f"Expected 7-D action, got {action.shape}"
            )

    # --------------------------------------------------------
    # Copy the Cartesian pose action.
    # --------------------------------------------------------

    robosuite_action = action.copy()

    # --------------------------------------------------------
    # Gripper conversion
    # --------------------------------------------------------

    # OpenVLA:
    #
    #     0 -> open
    #     1 -> closed
    #
    # Robosuite:
    #
    #     -1 -> open
    #     +1 -> closed

    robosuite_action[6] = (
        2.0 * action[6] - 1.0
    )

    # --------------------------------------------------------
    # Clamp gripper.
    # --------------------------------------------------------

    robosuite_action[6] = np.clip(
        robosuite_action[6],
        -1.0,
        1.0,
    )

    return robosuite_action


# ============================================================
# 8. ROBOT STATE DIAGNOSTICS
# ============================================================

def print_robot_state(
    env,
    step_number,
):
    """
    Print the Panda end-effector state.

    This is diagnostic only.
    """

    if not Config.print_robot_state:
        return

    try:

        robot = env.robots[0]

        # End-effector position
        eef_pos = np.asarray(
            robot.sim.data.site_xpos[
                robot.eef_site_id
            ]
        )

        # End-effector orientation
        eef_rot = np.asarray(
            robot.sim.data.site_xmat[
                robot.eef_site_id
            ]
        ).reshape(3, 3)

        print(
            f"  EEF position: "
            f"[{eef_pos[0]:+.4f}, "
            f"{eef_pos[1]:+.4f}, "
            f"{eef_pos[2]:+.4f}]"
        )

        print(
            "  EEF rotation matrix:"
        )

        print(eef_rot)

    except Exception as exc:

        print(
            "  Could not read EEF state:",
            repr(exc),
        )


# ============================================================
# 9. VIDEO
# ============================================================

def save_video(
    frames,
    instruction,
    episode_number,
):
    """
    Save recorded RGB frames as MP4.
    """

    os.makedirs(
        Config.output_folder,
        exist_ok=True,
    )

    safe_instruction = (
        instruction
        .lower()
        .replace(" ", "_")
        .replace("\n", "_")
        .replace(".", "_")
    )[:60]

    video_file = os.path.join(
        Config.output_folder,
        (
            f"episode={episode_number}"
            f"--prompt={safe_instruction}.mp4"
        ),
    )

    print()
    print("Saving video:")
    print(video_file)

    writer = imageio.get_writer(
        video_file,
        fps=30,
    )

    try:

        for frame in frames:

            writer.append_data(frame)

    finally:

        writer.close()

    print("Video saved.")

    return video_file


# ============================================================
# 10. SINGLE EPISODE
# ============================================================

def run_episode(
    env,
    vla,
    instruction,
):
    """
    Run one complete OpenVLA → Robosuite episode.
    """

    print()
    print("=" * 70)
    print("STARTING EPISODE")
    print("=" * 70)

    print()
    print("Instruction:")
    print(instruction)

    # --------------------------------------------------------
    # Reset
    # --------------------------------------------------------

    observation = env.reset()

    frames = []

    successful = False

    # --------------------------------------------------------
    # Warm-up
    # --------------------------------------------------------

    print()
    print(
        f"Warming up for {Config.warmup_steps} steps..."
    )

    # Robosuite action dimension:
    #
    # [dx,dy,dz,dRx,dRy,dRz,gripper]

    no_op = np.zeros(
        7,
        dtype=np.float32,
    )

    # Gripper open during warmup.
    no_op[6] = -1.0

    for warmup_step in range(
        Config.warmup_steps
    ):

        observation, reward, done, info = env.step(
            no_op
        )

    print("Warm-up complete.")

    # --------------------------------------------------------
    # Main loop
    # --------------------------------------------------------

    for step_number in range(
        Config.maximum_steps
    ):

        print()
        print("-" * 70)
        print(
            f"STEP {step_number + 1} / "
            f"{Config.maximum_steps}"
        )
        print("-" * 70)

        # ----------------------------------------------------
        # Image
        # ----------------------------------------------------

        camera_frame = prepare_image(
            observation
        )

        frames.append(
            camera_frame.copy()
        )

        if step_number == 0:

            print()
            print(
                "Camera image shape:",
                camera_frame.shape,
            )

            print(
                "Camera image dtype:",
                camera_frame.dtype,
            )

        # ----------------------------------------------------
        # OpenVLA
        # ----------------------------------------------------

        t0 = time.time()

        vla_action = vla.predict(
            image=camera_frame,
            instruction=instruction,
        )

        inference_time = (
            time.time() - t0
        )

        # ----------------------------------------------------
        # Convert action
        # ----------------------------------------------------

        robosuite_action = (
            convert_vla_action_to_robosuite(
                vla_action
            )
        )

        # ----------------------------------------------------
        # Diagnostics
        # ----------------------------------------------------

        if Config.print_actions:

            print()
            print(
                "OpenVLA action:"
            )

            print(
                np.array2string(
                    vla_action,
                    precision=5,
                    suppress_small=False,
                )
            )

            print()
            print(
                "Robosuite action:"
            )

            print(
                np.array2string(
                    robosuite_action,
                    precision=5,
                    suppress_small=False,
                )
            )

            print()
            print(
                f"Inference time: "
                f"{inference_time:.3f} sec"
            )

        # ----------------------------------------------------
        # Execute action
        # ----------------------------------------------------

        for repeat_number in range(
            Config.action_repeat
        ):

            (
                observation,
                reward,
                done,
                info,
            ) = env.step(
                robosuite_action
            )

        # ----------------------------------------------------
        # State diagnostics
        # ----------------------------------------------------

        print_robot_state(
            env,
            step_number,
        )

        # ----------------------------------------------------
        # Reward
        # ----------------------------------------------------

        print(
            f"  Reward: {reward}"
        )

        # ----------------------------------------------------
        # Success
        # ----------------------------------------------------

        # Robosuite's Lift environment provides reward/success
        # information through its reward / info machinery.
        #
        # We check both where possible.

        is_success = False

        if isinstance(info, dict):

            is_success = bool(
                info.get(
                    "success",
                    False,
                )
            )

        if not is_success:

            try:

                is_success = bool(
                    env._check_success()
                )

            except Exception:

                pass

        if is_success:

            successful = True

            print()
            print("=" * 70)
            print(
                "SUCCESS: Robosuite reports task success."
            )
            print("=" * 70)

            break

        if done:

            print()
            print(
                "Environment reported done."
            )

            break

    # --------------------------------------------------------
    # Failure
    # --------------------------------------------------------

    if not successful:

        print()
        print("=" * 70)
        print(
            "FAILED / TIMEOUT: "
            "Task was not completed."
        )
        print("=" * 70)

    # --------------------------------------------------------
    # Save video
    # --------------------------------------------------------

    video_file = save_video(
        frames=frames,
        instruction=instruction,
        episode_number=Config.episode_number,
    )

    return successful, video_file


# ============================================================
# 11. MAIN
# ============================================================

def main():

    print()
    print("=" * 70)
    print("OPENVLA + ROBOSUITE PANDA")
    print("=" * 70)

    print()
    print("Python:")
    print(sys.executable)

    print()
    print("PyTorch:")
    print(torch.__version__)

    print()
    print("CUDA available:")
    print(torch.cuda.is_available())

    if torch.cuda.is_available():

        print(
            "GPU:",
            torch.cuda.get_device_name(0),
        )

    # --------------------------------------------------------
    # Create VLA
    # --------------------------------------------------------

    vla = OpenVLA(
        model_name=Config.model_name,
        gpu=Config.gpu,
        normalization_key=Config.normalization_key,
    )

    # --------------------------------------------------------
    # Create Robosuite
    # --------------------------------------------------------

    env = create_environment()

    try:

        # ----------------------------------------------------
        # Run
        # ----------------------------------------------------

        successful, video_file = run_episode(
            env=env,
            vla=vla,
            instruction=Config.instruction,
        )

        print()
        print("=" * 70)
        print("SIMULATION COMPLETE")
        print("=" * 70)

        print()
        print(
            "Result:",
            "SUCCESS" if successful else "FAILURE",
        )

        print()
        print(
            "Video:",
            video_file,
        )

    finally:

        # ----------------------------------------------------
        # Cleanup
        # ----------------------------------------------------

        print()
        print("Closing Robosuite environment...")

        env.close()

        print("Environment closed.")


# ============================================================
# 12. COMMAND-LINE ENTRY POINT
# ============================================================

def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Run OpenVLA + Robosuite Panda inference."
    )
    parser.add_argument(
        "--instruction",
        default=Config.instruction,
        help="Robot task instruction.",
    )
    parser.add_argument(
        "--model",
        default=Config.model_name,
        help="Hugging Face OpenVLA model name.",
    )
    parser.add_argument(
        "--output-folder",
        default=Config.output_folder,
        help="Folder in which to save the episode video.",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=Config.maximum_steps,
        help="Maximum number of Robosuite control steps.",
    )
    parser.add_argument(
        "--warmup-steps",
        type=int,
        default=Config.warmup_steps,
        help="Number of no-op warmup steps.",
    )
    parser.add_argument(
        "--action-repeat",
        type=int,
        default=Config.action_repeat,
        help="Number of Robosuite steps per OpenVLA action.",
    )
    parser.add_argument(
        "--episode",
        type=int,
        default=Config.episode_number,
        help="Episode number used in the output filename.",
    )

    args = parser.parse_args()

    # Apply command-line configuration.
    Config.instruction = args.instruction
    Config.model_name = args.model
    Config.output_folder = args.output_folder
    Config.maximum_steps = args.max_steps
    Config.warmup_steps = args.warmup_steps
    Config.action_repeat = args.action_repeat
    Config.episode_number = args.episode

    main_start = time.time()

    print()
    print("=" * 70)
    print("OPENVLA + ROBOSUITE PANDA")
    print("=" * 70)

    print()
    print("Python:")
    print(sys.executable)

    print()
    print("PyTorch:")
    print(torch.__version__)

    print()
    print("CUDA available:")
    print(torch.cuda.is_available())

    if torch.cuda.is_available():
        print("GPU:", torch.cuda.get_device_name(0))

    # ------------------------------------------------------------
    # Create VLA
    # ------------------------------------------------------------

    vla = OpenVLA(
        model_name=Config.model_name,
        gpu=Config.gpu,
        normalization_key=Config.normalization_key,
    )

    # ------------------------------------------------------------
    # Create Robosuite
    # ------------------------------------------------------------

    env = create_environment()

    try:
        successful, video_file = run_episode(
            env=env,
            vla=vla,
            instruction=Config.instruction,
        )

        print()
        print("=" * 70)
        print("SIMULATION COMPLETE")
        print("=" * 70)

        print()
        print("Result:", "SUCCESS" if successful else "FAILURE")
        print("Video:", video_file)
        print(f"Total runtime: {time.time() - main_start:.2f} sec")

    finally:
        print()
        print("Closing Robosuite environment...")
        env.close()
        print("Environment closed.")


if __name__ == "__main__":
    main()
