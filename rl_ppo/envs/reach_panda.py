"""
Native MuJoCo Gymnasium Environment for Franka Emika Panda Reach Task.
Provides a flat 20D observation vector compatible with Stable-Baselines3 MlpPolicy.
"""

import os
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import mujoco
import mujoco.viewer

class PandaEnv(gym.Env):
    def __init__(
        self, 
        render_mode: str | None = None, 
        max_episode_steps: int = 100,
        action_scale: float = 0.05, 
        success_tolerance: float = 0.05, 
        num_substeps: int = 20
    ):
        super().__init__()

        # human mode or not
        self.render_mode = render_mode
        # termination condition
        self.max_episode_steps = max_episode_steps
        # scale multiplier for action
        self.action_scale = action_scale
        # What is the distance tolerance for success!
        self.success_tolerance = success_tolerance
        # We move the arm in many substeps for smooth movement!
        self.num_substeps = num_substeps

        # Model and Data Init
        scene_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "franka_emika_panda", "scene.xml"))
        self.model = mujoco.MjModel.from_xml_path(scene_path)
        self.data = mujoco.MjData(self.model)

        # Cache element indices for direct C-level access
        self._hand_id = self.model.body("hand").id
        self._home_key_id = self.model.key("home").id

        # Spaces Setup
        self.n_arm_joints = 7
        
        # Continuous action space for 7 joint position deltas [-1, 1]
        # Gymnasium needs this
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(self.n_arm_joints,), dtype=np.float32)

        # Flat 20D observation vector [joint_pos(7) + joint_vel(7) + ee_pos(3) + target_pos(3)]
        # Gymnasium needs this
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(20,), dtype=np.float64)

        # Workspace sampling volume boundaries for target spawning
        self._workspace_low = np.array([0.3, -0.3, 0.2], dtype=np.float64)
        self._workspace_high = np.array([0.7, 0.3, 0.6], dtype=np.float64)

        # Control limits cache
        self._ctrl_min = self.model.actuator_ctrlrange[:self.n_arm_joints, 0]
        self._ctrl_max = self.model.actuator_ctrlrange[:self.n_arm_joints, 1]

        # State Counters & Rendering Handles
        self._elapsed_steps = 0
        self._passive_viewer = None
        self._offscreen_renderer = None


    # Gymnasium API

    # Tells render mode and render_fps to Gymnasium
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 25}

    # Reset the Panda and state, set random target position, and return the
    # observation and info after that!
    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)

        # Restore system state to default keyframe
        mujoco.mj_resetDataKeyframe(self.model, self.data, self._home_key_id)

        # Set random target position using mocap body
        random_target_position = self.np_random.uniform(self._workspace_low, self._workspace_high)
        self.data.mocap_pos[0] = random_target_position

        # Compute forward kinematics to update initial Cartesian positions
        mujoco.mj_forward(self.model, self.data)

        # Reset the steps
        self._elapsed_steps = 0

        return self._build_observation(), self._build_info()

    # Take an action, evaluate the success, and return the observation, reward, info, 
    # and terminated or truncated!
    def step(self, action: np.ndarray):
        # Clip normalized action array
        norm_action = np.clip(action, -1.0, 1.0, dtype=np.float64)

        # Vectorized joint command calculation and saturation clipping
        target_ctrl = self.data.ctrl[:self.n_arm_joints] + (norm_action * self.action_scale)
        self.data.ctrl[:self.n_arm_joints] = np.clip(target_ctrl, self._ctrl_min, self._ctrl_max)

        # Keep gripper fully closed if present in XML model
        # The self.model.nu tells us number of actuators. If it is more than joints, we have a 
        # gripper!
        # In the standard Franka Emika Panda MuJoCo model, the finger actuator is configured 
        # as a position/tendon actuator with a control range of [0, 255], where 255.0 corresponds 
        # to maximum closing force/position (fully closed).
        if self.model.nu > self.n_arm_joints:
            self.data.ctrl[self.n_arm_joints] = 255.0

        # Step high-frequency simulation loop
        # Each mj_step advances physics only by 0.002s. 
        # We need many substeps to advance longer time! 20 steps gives us 0.04s advance
        # in a smooth way
        for _ in range(self.num_substeps):
            mujoco.mj_step(self.model, self.data)

        # count the steps
        self._elapsed_steps += 1

        # Build observation
        observation = self._build_observation()
        # Build info that includes distance
        info = self._build_info()

        # Distance-based reward calculation
        dist = info["distance"] # Get Eucledean distance to target
        # if the distance to target is less than success tolerance, we have a successful case!
        is_successful = dist < self.success_tolerance
        # We reward +10 for success and negative distance if we have not reached target yet!
        reward = -dist + (10.0 if is_successful else 0.0)

        # terminate if successful
        info["is_success"] = is_successful
        terminated = is_successful

        # terminate as we reached the time limit
        truncated = self._elapsed_steps >= self.max_episode_steps

        # render if requested
        if self.render_mode == "human":
            self.render()

        return observation, reward, terminated, truncated, info

    # render the graphics if human mode is requested!
    def render(self):
        if self.render_mode == "human":
            if self._passive_viewer is None:
                self._passive_viewer = mujoco.viewer.launch_passive(self.model, self.data)
            self._passive_viewer.sync()
        elif self.render_mode == "rgb_array":
            if self._offscreen_renderer is None:
                self._offscreen_renderer = mujoco.Renderer(self.model, height=480, width=640)
            self._offscreen_renderer.update_scene(self.data)
            return self._offscreen_renderer.render()
        return None

    # close the simulation
    def close(self):
        if self._passive_viewer is not None:
            self._passive_viewer.close()
            self._passive_viewer = None
        if self._offscreen_renderer is not None:
            del self._offscreen_renderer
            self._offscreen_renderer = None


    # Helpers

    # observation data (20D state vector)
    def _build_observation(self) -> np.ndarray:
        """Constructs 20D state vector using direct C-array views."""
        return np.hstack([
            self.data.qpos[:self.n_arm_joints],
            self.data.qvel[:self.n_arm_joints],
            self.data.xpos[self._hand_id],
            self.data.mocap_pos[0],
        ]).astype(np.float64)

    # 
    def _build_info(self) -> dict:
        """Computes Euclidean distance between end-effector and target."""
        dist = float(np.linalg.norm(self.data.xpos[self._hand_id] - self.data.mocap_pos[0]))
        return {"distance": dist}
