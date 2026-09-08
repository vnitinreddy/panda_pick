# ============================================================
# Simplified OpenVLA + LIBERO Inference
# ============================================================

import argparse
import math
import os
import imageio

import numpy as np
import tensorflow as tf
import torch

from PIL import Image
from transformers import AutoModelForVision2Seq, AutoProcessor

from libero.libero.envs import OffScreenRenderEnv
from libero.libero import benchmark, get_libero_path


# ============================================================
# Configuration
# ============================================================

class Config:
    model_type = "openvla"
    task_suite = "libero_object"
    task_number = 0

    camera_size = 256
    model_image_size = 1024

    warmup_steps = 20
    maximum_steps = 280

    output_folder = "outputs/videos"
    gpu = "cuda:0"


# ============================================================
# LIBERO
# ============================================================

def create_environment(task_info, image_size):
    """Create a LIBERO simulation environment."""

    bddl_path = os.path.join(
        get_libero_path("bddl_files"),
        task_info.problem_folder,
        task_info.bddl_file
    )

    environment = OffScreenRenderEnv(
        bddl_file_name=bddl_path,
        camera_heights=image_size,
        camera_widths=image_size
    )

    environment.seed(0)

    return environment


def get_task(task_suite_name, task_number):
    """Load one task from a LIBERO task suite."""

    benchmark_list = benchmark.get_benchmark_dict()

    suite = benchmark_list[task_suite_name]()

    task_info = suite.get_task(task_number)

    return task_info, suite


# ============================================================
# IMAGE PROCESSING
# ============================================================

def resize_image(image, target_size):
    """Resize an image using the same JPEG/resizing pipeline."""

    image = tf.image.encode_jpeg(image)

    image = tf.io.decode_image(
        image,
        expand_animations=False,
        dtype=tf.uint8
    )

    image = tf.image.resize(
        image,
        (target_size, target_size),
        method="lanczos3",
        antialias=True
    )

    image = tf.cast(
        tf.clip_by_value(
            tf.round(image),
            0,
            255
        ),
        tf.uint8
    )

    return image.numpy()


def prepare_image(environment_observation, target_size):
    """Extract and preprocess the LIBERO camera image."""

    camera_image = environment_observation["agentview_image"]

    # LIBERO camera orientation must be rotated 180 degrees.
    camera_image = camera_image[::-1, ::-1]

    camera_image = resize_image(
        camera_image,
        target_size
    )

    return camera_image


# ============================================================
# GRIPPER
# ============================================================

def process_gripper(action_vector):
    """
    Convert the gripper action from [0, 1]
    to the environment's [-1, +1] representation.
    """

    action_vector[..., -1] = (
        2.0 * action_vector[..., -1] - 1.0
    )

    # Convert to either -1 or +1.
    action_vector[..., -1] = np.sign(
        action_vector[..., -1]
    )

    # LIBERO uses the opposite gripper convention.
    action_vector[..., -1] *= -1.0

    return action_vector


def no_op_action():
    """Action used while waiting for the simulation to settle."""

    return [0, 0, 0, 0, 0, 0, -1]


# ============================================================
# VIDEO
# ============================================================

def save_video(frames, episode_number, instruction, output_folder):
    """Save the recorded episode as an MP4."""

    os.makedirs(
        output_folder,
        exist_ok=True
    )

    safe_instruction = (
        instruction
        .lower()
        .replace(" ", "_")
        .replace("\n", "_")
        .replace(".", "_")
    )[:50]

    video_file = os.path.join(
        output_folder,
        f"episode={episode_number}--prompt={safe_instruction}.mp4"
    )

    writer = imageio.get_writer(
        video_file,
        fps=30
    )

    for frame in frames:
        writer.append_data(frame)

    writer.close()

    return video_file


# ============================================================
# OpenVLA
# ============================================================

class OpenVLA:
    """Small wrapper around the OpenVLA model."""

    def __init__(
        self,
        model_name,
        gpu="cuda:0"
    ):

        self.gpu = gpu

        print(f"Loading model: {model_name}")

        self.processor = AutoProcessor.from_pretrained(
            model_name,
            trust_remote_code=True
        )

        self.model = AutoModelForVision2Seq.from_pretrained(
            model_name,
            attn_implementation="flash_attention_2",
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            trust_remote_code=True
        ).to(gpu)

        print("OpenVLA loaded successfully.")

    def predict(
        self,
        image,
        instruction,
        normalization_key
    ):
        """Predict one robot action."""

        image = Image.fromarray(image).convert("RGB")

        model_prompt = (
            "In: What action should the robot take to "
            f"{instruction.lower()}?\nOut:"
        )

        model_inputs = self.processor(
            model_prompt,
            image
        ).to(
            self.gpu,
            dtype=torch.bfloat16
        )

        predicted_action = self.model.predict_action(
            **model_inputs,
            unnorm_key=normalization_key,
            do_sample=False
        )

        return predicted_action


# ============================================================
# ARM AGENT
# ============================================================

class ArmAgent:
    """Controls a LIBERO robot using OpenVLA."""

    def __init__(
        self,
        task_suite="libero_object",
        task_number=0,
        image_size=1024,
        output_folder="outputs/videos"
    ):

        self.task_suite_name = task_suite
        self.task_number = task_number
        self.image_size = image_size
        self.output_folder = output_folder

        # ----------------------------------------------------
        # Select the correct OpenVLA checkpoint
        # ----------------------------------------------------

        model_name = self.select_model(task_suite)

        self.vla = OpenVLA(
            model_name=model_name,
            gpu=Config.gpu
        )

        # ----------------------------------------------------
        # Load LIBERO
        # ----------------------------------------------------

        self.task_info, self.task_suite = get_task(
            task_suite,
            task_number
        )

        self.initial_states = (
            self.task_suite.get_task_init_states(
                task_number
            )
        )

        self.environment = create_environment(
            self.task_info,
            Config.camera_size
        )

        # ----------------------------------------------------
        # Action normalization statistics
        # ----------------------------------------------------

        self.normalization_key = task_suite

        if (
            hasattr(self.vla.model, "norm_stats")
            and self.normalization_key not in self.vla.model.norm_stats
        ):

            alternative_key = (
                f"{self.normalization_key}_no_noops"
            )

            if alternative_key in self.vla.model.norm_stats:
                self.normalization_key = alternative_key

        print(
            f"Using normalization key: "
            f"{self.normalization_key}"
        )

    def select_model(self, task_suite):
        """Select the OpenVLA checkpoint for the task suite."""

        models = {
            "libero_object":
                "openvla/openvla-7b-finetuned-libero-object",

            "libero_spatial":
                "openvla/openvla-7b-finetuned-libero-spatial",

            "libero_goal":
                "openvla/openvla-7b-finetuned-libero-goal",

            "libero_10":
                "openvla/openvla-7b-finetuned-libero-10",

            "general":
                "openvla/openvla-7b"
        }

        return models.get(
            task_suite,
            "openvla/openvla-7b"
        )

    def run(
        self,
        instruction,
        episode_number=0
    ):
        """Run one complete LIBERO episode."""

        print()
        print("=" * 60)
        print(f"Instruction: {instruction}")
        print("=" * 60)

        # ----------------------------------------------------
        # Reset simulation
        # ----------------------------------------------------

        self.environment.reset()

        environment_observation = (
            self.environment.set_init_state(
                self.initial_states[episode_number]
            )
        )

        frames = []
        successful = False

        # ----------------------------------------------------
        # Main control loop
        # ----------------------------------------------------

        for step_number in range(
            Config.maximum_steps + Config.warmup_steps
        ):

            # -----------------------------------------------
            # Let the simulation settle
            # -----------------------------------------------

            if step_number < Config.warmup_steps:

                environment_observation, _, finished, _ = (
                    self.environment.step(
                        no_op_action()
                    )
                )

                continue

            # -----------------------------------------------
            # Get camera image
            # -----------------------------------------------

            camera_frame = prepare_image(
                environment_observation,
                self.image_size
            )

            frames.append(camera_frame)

            # -----------------------------------------------
            # Ask OpenVLA for an action
            # -----------------------------------------------

            action_vector = self.vla.predict(
                image=camera_frame,
                instruction=instruction,
                normalization_key=self.normalization_key
            )

            print(
                f"Step {step_number}: "
                f"Action = {action_vector}"
            )

            # -----------------------------------------------
            # Process gripper
            # -----------------------------------------------

            action_vector = process_gripper(
                action_vector
            )

            # -----------------------------------------------
            # Execute action in LIBERO
            # -----------------------------------------------

            (
                environment_observation,
                reward,
                finished,
                info
            ) = self.environment.step(
                action_vector.tolist()
            )

            # -----------------------------------------------
            # Check success
            # -----------------------------------------------

            if finished:

                successful = True

                print()
                print(
                    f"SUCCESS: Episode finished "
                    f"at step {step_number}"
                )

                break

        # ----------------------------------------------------
        # Timeout
        # ----------------------------------------------------

        if not successful:

            print()
            print(
                "FAILED: Maximum number of steps reached."
            )

        # ----------------------------------------------------
        # Save video
        # ----------------------------------------------------

        video_file = save_video(
            frames=frames,
            episode_number=self.task_number,
            instruction=instruction,
            output_folder=self.output_folder
        )

        print()
        print(f"Video saved to: {video_file}")

        return successful, frames


# ============================================================
# COMMAND-LINE ARGUMENTS
# ============================================================

def read_arguments():

    argument_parser = argparse.ArgumentParser(
        description="OpenVLA LIBERO ArmAgent"
    )

    argument_parser.add_argument(
        "--prompt",
        type=str,
        required=True,
        help="Instruction for the robot"
    )

    argument_parser.add_argument(
        "--task",
        type=str,
        default="libero_object",
        choices=[
            "libero_object",
            "libero_spatial",
            "libero_goal",
            "libero_10",
            "general"
        ],
        help="LIBERO task suite"
    )

    argument_parser.add_argument(
        "--task_id",
        type=int,
        default=0,
        help="LIBERO task number"
    )

    argument_parser.add_argument(
        "--image_resize",
        type=int,
        default=1024,
        help="Image size sent to OpenVLA"
    )

    argument_parser.add_argument(
        "--output_video",
        type=str,
        default="outputs/videos",
        help="Directory for rollout videos"
    )

    return argument_parser.parse_args()


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    command_line = read_arguments()

    robot_agent = ArmAgent(
        task_suite=command_line.task,
        task_number=command_line.task_id,
        image_size=command_line.image_resize,
        output_folder=command_line.output_video
    )

    result, _ = robot_agent.run(
        instruction=command_line.prompt
    )

    print()
    print(
        "Simulation completed:",
        "SUCCESS" if result else "FAILURE"
    )
