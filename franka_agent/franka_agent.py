#@title Inference

# ============================================================
# franka_agent.py
#
# OpenVLA + Robosuite Franka Panda
#
# Pipeline:
#
'''
observation
    ↓
prepare_image(observation)
    ↓
observation["agentview_image"]
    ↓
resize_image(image, 1024)
    ↓
1024 × 1024 uint8 image
    ↓
OpenVLA.predict(...)
    ↓
Image.fromarray(image).convert("RGB")
    ↓
processor(...)
'''
#
# IMPORTANT:
#
# The OpenVLA -> Robosuite action conversion intentionally
# follows the existing franka_agent.py implementation:
#
#   OpenVLA:
#       [dx, dy, dz, dRx, dRy, dRz, gripper]
#
#   First six:
#       copied unchanged
#
#   Gripper:
#       OpenVLA 0 = open
#       OpenVLA 1 = closed
#
#       Robosuite -1 = open
#       Robosuite +1 = closed
#
#       robosuite_gripper = 2 * openvla_gripper - 1
#
# NO additional position / rotation scaling is performed here.
# ============================================================


# ============================================================
# 1. ENVIRONMENT
# ============================================================

import os

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


# ============================================================
# 2. IMPORTS
# ============================================================

import sys
import time
import argparse
from pathlib import Path

import numpy as np
import torch

from PIL import Image

import imageio.v2 as imageio

import robosuite as suite
from robosuite.controllers import load_controller_config

from transformers import (
    AutoProcessor,
    AutoModelForVision2Seq,
)

import tensorflow as tf


# ============================================================
# 3. PATHS
# ============================================================

# Keep the same paths used by the working Colab environment.

PANDA_PICK_PATH = "/content/panda_pick"
OPENVLA_PATH = "/content/openvla"

for path in [
    PANDA_PICK_PATH,
    OPENVLA_PATH,
]:
    if path not in sys.path:
        sys.path.insert(0, path)


# ============================================================
# 4. CONFIGURATION
# ============================================================

class Config:

    # --------------------------------------------------------
    # OpenVLA
    # --------------------------------------------------------

    MODEL_NAME = "openvla/openvla-7b"

    NORMALIZATION_KEY = "bridge_orig"

    DEVICE = "cuda:0"

    # --------------------------------------------------------
    # Task
    # --------------------------------------------------------

    INSTRUCTION = "pick up the red cube"

    # This is the prompt used by the old working code.
    PROMPT_TEMPLATE = (
        "In: What action should the robot take to "
        "{instruction}?\nOut:"
    )

    # --------------------------------------------------------
    # Robosuite
    # --------------------------------------------------------

    ENV_NAME = "Lift"

    ROBOT = "Panda"

    CONTROLLER = "OSC_POSE"

    CONTROL_FREQ = 20

    MAX_STEPS = 300

    WARMUP_STEPS = 20

    ACTION_REPEAT = 1

    # --------------------------------------------------------
    # Camera
    # --------------------------------------------------------

    CAMERA_NAME = "agentview"

    CAMERA_WIDTH = 224

    CAMERA_HEIGHT = 224

    # --------------------------------------------------------
    # Output
    # --------------------------------------------------------

    OUTPUT_FOLDER = (
        "/content/panda_pick/"
        "franka_agent/outputs/videos"
    )

    EPISODE = 1

    # --------------------------------------------------------
    # Debug
    # --------------------------------------------------------

    PRINT_ACTIONS = True

    PRINT_STATE = True

    PRINT_CAMERA_INFO = True


# ============================================================
# 5. OPENVLA WRAPPER
# ============================================================

class OpenVLAWrapper:

    def __init__(
        self,
        model_name,
        device,
        normalization_key,
    ):

        self.model_name = model_name
        self.device = device
        self.normalization_key = normalization_key

        print()
        print("=" * 70)
        print("OPENVLA")
        print("=" * 70)

        print(
            "Model:",
            model_name,
        )

        print(
            "Device:",
            device,
        )

        print(
            "Normalization key:",
            normalization_key,
        )

        # ----------------------------------------------------
        # Processor
        # ----------------------------------------------------

        print()
        print("Loading processor...")

        self.processor = (
            AutoProcessor.from_pretrained(
                model_name,
                trust_remote_code=True,
            )
        )

        # ----------------------------------------------------
        # Model
        # ----------------------------------------------------

        print("Loading model...")

        self.model = (
            AutoModelForVision2Seq.from_pretrained(
                model_name,
                torch_dtype=torch.bfloat16,
                low_cpu_mem_usage=True,
                trust_remote_code=True,
                attn_implementation="flash_attention_2",
            )
            .to(device)
        )

        self.model.eval()

        print()
        print("OpenVLA loaded.")

        try:

            print(
                "Model device:",
                next(
                    self.model.parameters()
                ).device,
            )

            print(
                "Model dtype:",
                next(
                    self.model.parameters()
                ).dtype,
            )

        except Exception:
            pass

        print("=" * 70)


    # ========================================================
    # Predict
    # ========================================================

    @torch.inference_mode()
    def predict(self, image, instruction):
        image = Image.fromarray(image).convert("RGB")
    
        model_prompt = (
            "In: What action should the robot take to "
            f"{instruction.lower()}?\nOut:"
        )
    
        model_inputs = self.processor(
            model_prompt,
            image
        ).to(
            self.device,
            dtype=torch.bfloat16
        )
    
        action = self.model.predict_action(
            **model_inputs,
            unnorm_key=self.unnorm_key,
            do_sample=False
        )
    
        return action

# ============================================================
# 6. ROBOSUITE ENVIRONMENT
# ============================================================

def create_environment():

    print()
    print("=" * 70)
    print("ROBOSUITE")
    print("=" * 70)

    # --------------------------------------------------------
    # Controller configuration
    # --------------------------------------------------------

    controller_config = (
        load_controller_config(
            default_controller=Config.CONTROLLER
        )
    )

    print()
    print(
        "Controller:",
        Config.CONTROLLER,
    )

    # --------------------------------------------------------
    # Create environment
    # --------------------------------------------------------

    env = suite.make(
        Config.ENV_NAME,

        robots=Config.ROBOT,

        controller_configs=controller_config,

        # ----------------------------------------------------
        # Rendering
        # ----------------------------------------------------

        has_renderer=False,

        has_offscreen_renderer=True,

        render_camera=Config.CAMERA_NAME,

        # ----------------------------------------------------
        # Camera observations
        # ----------------------------------------------------

        use_camera_obs=True,

        camera_names=Config.CAMERA_NAME,

        camera_heights=Config.CAMERA_HEIGHT,

        camera_widths=Config.CAMERA_WIDTH,

        # ----------------------------------------------------
        # Simulation
        # ----------------------------------------------------

        control_freq=Config.CONTROL_FREQ,

        horizon=Config.MAX_STEPS,

        reward_shaping=True,

        ignore_done=False,
    )

    print()
    print("Robosuite environment created.")

    # --------------------------------------------------------
    # Action information
    # --------------------------------------------------------

    try:

        print()
        print("Robot action information:")
        print("-" * 70)

        env.robots[0].print_action_info()

    except Exception as exc:

        print(
            "Could not print action information:",
            repr(exc),
        )

    # --------------------------------------------------------
    # Camera information
    # --------------------------------------------------------

    if Config.PRINT_CAMERA_INFO:

        try:

            model = env.sim.model

            print()
            print("MuJoCo cameras:")
            print("-" * 70)

            for cam_id in range(
                model.ncam
            ):

                camera_name = (
                    model.camera_id2name(
                        cam_id
                    )
                )

                print(
                    f"camera {cam_id}: "
                    f"{camera_name}"
                )

                print(
                    "  position:",
                    model.cam_pos[
                        cam_id
                    ],
                )

                print(
                    "  quaternion:",
                    model.cam_quat[
                        cam_id
                    ],
                )

        except Exception as exc:

            print(
                "Could not inspect cameras:",
                repr(exc),
            )

    print("=" * 70)

    return env


# ============================================================
# 7. CAMERA IMAGE
# ============================================================
def resize_image(image, target_size):
    image = tf.image.encode_jpeg(image)
    image = tf.io.decode_image(
        image,
        expand_animations=False,
        dtype=tf.uint8,
    )
    image = tf.image.resize(
        image,
        (target_size, target_size),
        method="lanczos3",
        antialias=True,
    )
    image = tf.cast(
        tf.clip_by_value(tf.round(image), 0, 255),
        tf.uint8,
    )
    return image.numpy()


def prepare_image(observation):
    camera_key = f"{Config.CAMERA_NAME}_image"

    if camera_key not in observation:
        raise RuntimeError(
            f"Camera image '{camera_key}' not found in observation keys: "
            f"{list(observation.keys())}"
        )

    image = np.asarray(observation[camera_key], dtype=np.uint8)

    if image.ndim != 3 or image.shape[2] != 3:
        raise RuntimeError(
            f"Unexpected camera image shape: {image.shape}"
        )

    # Robosuite agentview does NOT need the LIBERO-specific 180° rotation.
    image = resize_image(image, 1024)

    return image

# ============================================================
# 8. OPENVLA → ROBOSUITE ACTION
# ============================================================

def convert_vla_action_to_robosuite(
    vla_action,
):
    """
    Convert OpenVLA's 7-D BridgeData action into the
    7-D Robosuite action.

    THIS IS THE SAME CONVERSION USED IN THE EXISTING
    franka_agent.py.

    OpenVLA:

        [dx, dy, dz, dRx, dRy, dRz, gripper]

    Robosuite:

        [dx, dy, dz, dRx, dRy, dRz, gripper]

    First six values:
        copied unchanged.

    Gripper:

        OpenVLA:
            0 = open
            1 = closed

        Robosuite:
            -1 = open
            +1 = closed

        Therefore:

            robosuite_gripper
                = 2 * vla_gripper - 1

    IMPORTANT:

    There is intentionally NO:

        position scaling
        rotation scaling
        coordinate-frame rotation
        Jacobian calculation
        IK calculation

    Robosuite's OSC_POSE controller handles the robot control.
    """

    # --------------------------------------------------------
    # Convert to numpy
    # --------------------------------------------------------

    action = np.asarray(
        vla_action,
        dtype=np.float32,
    ).copy()

    # --------------------------------------------------------
    # Validate
    # --------------------------------------------------------

    if action.shape != (7,):

        action = action.reshape(-1)

        if action.shape[0] != 7:

            raise RuntimeError(
                "Expected 7-D OpenVLA action, "
                f"got {action.shape}"
            )

    # --------------------------------------------------------
    # Copy all seven values
    # --------------------------------------------------------

    robosuite_action = (
        action.copy()
    )

    # --------------------------------------------------------
    # Gripper conversion
    # --------------------------------------------------------
    #
    # OpenVLA:
    #
    #     0 -> open
    #     1 -> closed
    #
    # Robosuite:
    #
    #    -1 -> open
    #    +1 -> closed
    #
    # Therefore:
    #
    #     0 -> -1
    #     1 -> +1
    #

    robosuite_action[6] = (
        2.0 * action[6] - 1.0
    )

    # --------------------------------------------------------
    # Clamp gripper
    # --------------------------------------------------------

    robosuite_action[6] = np.clip(
        robosuite_action[6],
        -1.0,
        1.0,
    )

    return robosuite_action


# ============================================================
# 9. ROBOT STATE
# ============================================================

def get_eef_state(
    env,
):

    robot = env.robots[0]

    eef_site_id = (
        robot.eef_site_id
    )

    eef_pos = np.asarray(
        robot.sim.data.site_xpos[
            eef_site_id
        ]
    ).copy()

    eef_rot = np.asarray(
        robot.sim.data.site_xmat[
            eef_site_id
        ]
    ).reshape(
        3,
        3,
    ).copy()

    return (
        eef_pos,
        eef_rot,
    )


# ============================================================
# 10. CUBE STATE
# ============================================================

def get_cube_position_from_observation(
    observation,
):

    if "cube_pos" in observation:

        return np.asarray(
            observation["cube_pos"],
            dtype=np.float64,
        ).copy()

    return None


def get_cube_position_from_sim(
    env,
):

    try:

        cube_body_id = (
            env.obj_body_id["cube"]
        )

        return np.asarray(
            env.sim.data.body_xpos[
                cube_body_id
            ]
        ).copy()

    except Exception:

        return None


# ============================================================
# 11. SAVE VIDEO
# ============================================================

def save_video(
    frames,
    output_folder,
    episode,
    instruction,
):

    output_folder = Path(
        output_folder
    )

    output_folder.mkdir(
        parents=True,
        exist_ok=True,
    )

    safe_instruction = (
        instruction
        .strip()
        .lower()
        .replace(" ", "_")
        .replace("/", "_")
        .replace("\\", "_")
        .replace(":", "_")
    )

    video_path = (
        output_folder
        / (
            f"episode={episode}"
            f"--prompt={safe_instruction}.mp4"
        )
    )

    print()
    print("=" * 70)
    print("SAVING VIDEO")
    print("=" * 70)

    print(
        "Path:",
        video_path,
    )

    if len(frames) == 0:

        print(
            "WARNING: No frames were recorded."
        )

        return None

    writer = imageio.get_writer(
        str(video_path),
        fps=30,
    )

    try:

        for frame in frames:

            writer.append_data(
                np.asarray(
                    frame,
                    dtype=np.uint8,
                )
            )

    finally:

        writer.close()

    print(
        "Video saved."
    )

    return str(video_path)


# ============================================================
# 12. RUN ONE EPISODE
# ============================================================

def run_episode(
    env,
    vla,
    instruction,
    episode,
):

    print()
    print("=" * 70)
    print("STARTING EPISODE")
    print("=" * 70)

    print()
    print(
        "Instruction:",
        instruction,
    )

    print(
        "Episode:",
        episode,
    )

    # --------------------------------------------------------
    # Reset
    # --------------------------------------------------------

    observation = env.reset()

    print()
    print(
        "Observation keys:"
    )

    print(
        list(
            observation.keys()
        )
    )

    # --------------------------------------------------------
    # Verify camera immediately
    # --------------------------------------------------------

    camera_key = (
        f"{Config.CAMERA_NAME}_image"
    )

    if camera_key not in observation:

        raise RuntimeError(
            f"Expected camera key "
            f"'{camera_key}' was not found. "
            f"Observation keys: "
            f"{list(observation.keys())}"
        )

    # --------------------------------------------------------
    # Video frames
    # --------------------------------------------------------

    frames = []

    # --------------------------------------------------------
    # Initial image
    # --------------------------------------------------------

    initial_image = prepare_image(
        observation
    )

    print()
    print(
        "Initial camera image:",
        initial_image.shape,
        initial_image.dtype,
    )

    # Save initial image for debugging.

    debug_image_path = (
        Path(Config.OUTPUT_FOLDER)
        / "first_vla_frame.png"
    )

    debug_image_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    Image.fromarray(
        initial_image
    ).save(
        debug_image_path
    )

    print(
        "Initial image saved:",
        debug_image_path,
    )

    # --------------------------------------------------------
    # Warmup
    # --------------------------------------------------------

    print()
    print(
        f"Warmup: "
        f"{Config.WARMUP_STEPS} steps"
    )

    # Robosuite gripper:
    #
    #     -1 = open
    #

    no_op = np.zeros(
        7,
        dtype=np.float32,
    )

    no_op[6] = -1.0

    for warmup_step in range(
        Config.WARMUP_STEPS
    ):

        (
            observation,
            reward,
            done,
            info,
        ) = env.step(
            no_op
        )

        if done:

            print(
                "Environment ended during warmup."
            )

            break

    print(
        "Warmup complete."
    )

    # --------------------------------------------------------
    # Main control loop
    # --------------------------------------------------------

    successful = False

    for step in range(
        Config.MAX_STEPS
    ):

        print()
        print(
            "=" * 70
        )

        print(
            f"STEP {step + 1} / "
            f"{Config.MAX_STEPS}"
        )

        print(
            "=" * 70
        )

        # ====================================================
        # 1. IMAGE
        # ====================================================

        image = prepare_image(
            observation
        )

        # Store image for video.

        frames.append(
            image.copy()
        )

        # ====================================================
        # 2. ROBOT STATE BEFORE ACTION
        # ====================================================

        try:

            eef_before, eef_rot_before = (
                get_eef_state(env)
            )

        except Exception:

            eef_before = None
            eef_rot_before = None

        cube_before = (
            get_cube_position_from_observation(
                observation
            )
        )

        if cube_before is None:

            cube_before = (
                get_cube_position_from_sim(
                    env
                )
            )

        # ====================================================
        # 3. OPENVLA
        # ====================================================

        prediction_start = (
            time.time()
        )

        vla_action = vla.predict(
            image=image,
            instruction=instruction,
        )

        prediction_time = (
            time.time()
            - prediction_start
        )

        # ====================================================
        # 4. CONVERT OPENVLA ACTION
        # ====================================================

        robot_action = (
            convert_vla_action_to_robosuite(
                vla_action
            )
        )

        # ====================================================
        # 5. PRINT ACTION
        # ====================================================

        if Config.PRINT_ACTIONS:

            print()
            print(
                "OpenVLA action:"
            )

            print(
                np.array2string(
                    vla_action,
                    precision=6,
                    suppress_small=False,
                )
            )

            print()
            print(
                "Robosuite action:"
            )

            print(
                np.array2string(
                    robot_action,
                    precision=6,
                    suppress_small=False,
                )
            )

            print()
            print(
                f"OpenVLA inference: "
                f"{prediction_time:.3f} sec"
            )

        # ====================================================
        # 6. EXECUTE ROBOT ACTION
        # ====================================================

        for repeat in range(
            Config.ACTION_REPEAT
        ):

            (
                observation,
                reward,
                done,
                info,
            ) = env.step(
                robot_action
            )

        # ====================================================
        # 7. ROBOT STATE AFTER ACTION
        # ====================================================

        try:

            eef_after, eef_rot_after = (
                get_eef_state(env)
            )

        except Exception:

            eef_after = None
            eef_rot_after = None

        if Config.PRINT_STATE:

            if eef_after is not None:

                print()
                print(
                    "EEF position:"
                )

                print(
                    np.array2string(
                        eef_after,
                        precision=6,
                        suppress_small=False,
                    )
                )

            if (
                eef_before is not None
                and eef_after is not None
            ):

                actual_motion = (
                    eef_after
                    - eef_before
                )

                print()
                print(
                    "Actual EEF motion:"
                )

                print(
                    np.array2string(
                        actual_motion,
                        precision=6,
                        suppress_small=False,
                    )
                )

        # ====================================================
        # 8. CUBE / EEF DISTANCE
        # ====================================================

        cube_after = (
            get_cube_position_from_observation(
                observation
            )
        )

        if cube_after is None:

            cube_after = (
                get_cube_position_from_sim(
                    env
                )
            )

        if (
            cube_after is not None
            and eef_after is not None
        ):

            eef_to_cube = (
                cube_after
                - eef_after
            )

            distance = (
                np.linalg.norm(
                    eef_to_cube
                )
            )

            print()
            print(
                "Cube position:"
            )

            print(
                np.array2string(
                    cube_after,
                    precision=6,
                )
            )

            print(
                "EEF → cube distance:",
                f"{distance:.5f} m",
            )

        # ====================================================
        # 9. REWARD
        # ====================================================

        print()
        print(
            "Reward:",
            reward,
        )

        # ====================================================
        # 10. SUCCESS
        # ====================================================

        success = False

        if isinstance(
            info,
            dict,
        ):

            success = bool(
                info.get(
                    "success",
                    False,
                )
            )

        # Robosuite environments commonly expose
        # _check_success().

        if not success:

            try:

                success = bool(
                    env._check_success()
                )

            except Exception:

                pass

        if success:

            successful = True

            print()
            print(
                "=" * 70
            )

            print(
                "SUCCESS"
            )

            print(
                "=" * 70
            )

            break

        # ====================================================
        # 11. DONE
        # ====================================================

        if done:

            print()
            print(
                "Environment reported done."
            )

            break

    # ========================================================
    # RESULT
    # ========================================================

    if not successful:

        print()
        print(
            "=" * 70
        )

        print(
            "EPISODE FINISHED WITHOUT SUCCESS"
        )

        print(
            "=" * 70
        )

    # ========================================================
    # VIDEO
    # ========================================================

    video_file = save_video(
        frames=frames,
        output_folder=Config.OUTPUT_FOLDER,
        episode=episode,
        instruction=instruction,
    )

    return (
        successful,
        video_file,
    )


# ============================================================
# 13. MAIN
# ============================================================

def main():

    print()
    print("=" * 70)
    print("OPENVLA + ROBOSUITE FRANKA PANDA")
    print("=" * 70)

    print()
    print(
        "Python:",
        sys.executable,
    )

    print(
        "PyTorch:",
        torch.__version__,
    )

    print(
        "CUDA available:",
        torch.cuda.is_available(),
    )

    if torch.cuda.is_available():

        print(
            "GPU:",
            torch.cuda.get_device_name(0),
        )

    print()
    print(
        "MUJOCO_GL:",
        os.environ.get(
            "MUJOCO_GL"
        ),
    )

    # --------------------------------------------------------
    # OpenVLA
    # --------------------------------------------------------

    vla = OpenVLAWrapper(
        model_name=Config.MODEL_NAME,
        device=Config.DEVICE,
        normalization_key=(
            Config.NORMALIZATION_KEY
        ),
    )

    # --------------------------------------------------------
    # Robosuite
    # --------------------------------------------------------

    env = create_environment()

    try:

        successful, video_file = (
            run_episode(
                env=env,
                vla=vla,
                instruction=Config.INSTRUCTION,
                episode=Config.EPISODE,
            )
        )

        print()
        print("=" * 70)
        print("COMPLETE")
        print("=" * 70)

        print()
        print(
            "Result:",
            "SUCCESS"
            if successful
            else "FAILURE",
        )

        print(
            "Video:",
            video_file,
        )

        return successful

    finally:

        print()
        print(
            "Closing Robosuite environment..."
        )

        env.close()

        print(
            "Environment closed."
        )


# ============================================================
# 14. COMMAND LINE
# ============================================================

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description=(
            "OpenVLA + Robosuite "
            "Franka Panda agent"
        )
    )

    parser.add_argument(
        "--instruction",
        default=Config.INSTRUCTION,
        help="Robot task instruction.",
    )

    parser.add_argument(
        "--model",
        default=Config.MODEL_NAME,
        help="OpenVLA model.",
    )

    parser.add_argument(
        "--output-folder",
        default=Config.OUTPUT_FOLDER,
        help="Video output directory.",
    )

    parser.add_argument(
        "--max-steps",
        type=int,
        default=Config.MAX_STEPS,
        help="Maximum number of control steps.",
    )

    parser.add_argument(
        "--warmup-steps",
        type=int,
        default=Config.WARMUP_STEPS,
        help="Number of warmup steps.",
    )

    parser.add_argument(
        "--action-repeat",
        type=int,
        default=Config.ACTION_REPEAT,
        help="Number of Robosuite steps per VLA action.",
    )

    parser.add_argument(
        "--episode",
        type=int,
        default=Config.EPISODE,
        help="Episode number.",
    )

    args = parser.parse_args()

    # --------------------------------------------------------
    # Apply CLI arguments
    # --------------------------------------------------------

    Config.INSTRUCTION = (
        args.instruction
    )

    Config.MODEL_NAME = (
        args.model
    )

    Config.OUTPUT_FOLDER = (
        args.output_folder
    )

    Config.MAX_STEPS = (
        args.max_steps
    )

    Config.WARMUP_STEPS = (
        args.warmup_steps
    )

    Config.ACTION_REPEAT = (
        args.action_repeat
    )

    Config.EPISODE = (
        args.episode
    )

    # --------------------------------------------------------
    # Run
    # --------------------------------------------------------

    main()
