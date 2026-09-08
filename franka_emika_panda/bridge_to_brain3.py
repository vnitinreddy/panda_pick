import os
import sys
import io
import time
import numpy as np
import mujoco
import mujoco.viewer
import requests
import cv2
import base64
import PIL.Image
from scipy.spatial.transform import Rotation as R

# ==========================================
# CONFIGURATION & CONSTANTS
# ==========================================
SERVER_URL = "https://surface-shuffling-recede.ngrok-free.dev/predict"
SERVER_GET_URL = "https://surface-shuffling-recede.ngrok-free.dev/"

HEADERS = {
    "ngrok-skip-browser-warning": "true",
    "Content-Type": "application/json"
}

TASK_PROMPT = "Pick up the red cube."

EMA_ALPHA = 0.35
PHYSICS_STEPS_PER_ACTION = 10
POSITION_SCALE = 0.6
ROTATION_SCALE = 0.5
DAMPING = 0.05

# Joint & Actuator indices
ARM_JOINT_IDS = list(range(7))      # Joint IDs 0..6
GRIPPER_CTRL_ID = 7               # Actuator index for Franka finger control

# Franka Panda Gripper Bounds
GRIPPER_OPEN = 0.04    # Fully open
GRIPPER_CLOSE = 0.00   # Fully closed

class IK:
    def __init__(self, model, data):
        """Initialize Inverse Kinematic instance.

        Args:
        - model: Mujoco model.
        - data: Mujoco data.
        """
        self.model = model
        self.data = data
    
    def jacobian(self, actuator_angles) -> np.ndarray:
        """
        Get the Jacobian of actuator w.r.t gripper pose.

        Returns:
                np.ndarray: Jacobian matrix
        """
        self.J = np.zeros((6, 7))
        pe = self.data.xpos[9]
        for i in range(len(actuator_angles)):
            zi = self.data.xaxis[i]
            pi = self.data.xanchor[i]
            self.J[:3, i] = np.cross(zi, pe - pi)
            self.J[3:, i] = zi
        return self.J
    
    def damped_inverse(self, lam=1e-3) -> np.ndarray:
        """
        Get the damped inverse of a matrix. Helps with non-square matricies.

        Returns:
                np.ndarray: Damped inverse matrix.
        """
        JT = self.J.T
        JJt = self.J @ JT
        return JT @ np.linalg.inv(JJt + (lam**2) * np.eye(self.J.shape[0]))

    def calculate(self, delta_x) -> np.ndarray:
        """
        Calculate the delta in actuators angles. 

        Returns:
                np.ndarray: new actuator angles.
        """
        actuator_angles = self.data.qpos[0:7]
        J = self.jacobian(actuator_angles)
        Jhash = self.damped_inverse()
        dtheta = Jhash @ delta_x
        return actuator_angles + dtheta

# ==========================================
# CONTROLLER STATE
# ==========================================
class OpenVLAController:
    def __init__(self, model, data):
        self.model = model
        self.data = data
        
        # EMA buffer for spatial action (XYZ + RPY)
        self.smoothed_spatial_action = np.zeros(6)
        
        # Binary latching state for gripper (Starts OPEN)
        self.gripper_target = GRIPPER_OPEN
        
        # Camera ID lookup
        self.cam_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, "vla_camera")
        if self.cam_id == -1:
            self.cam_id = 0

        # End-effector / Reference frame setup (Site vs Body fallback)
        self.site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "attachment_site")
        self.use_body = False
        if self.site_id == -1:
            body_name = "hand" if mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "hand") != -1 else "link7"
            self.site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)
            self.use_body = True

        # Target joint position accumulator
        self.target_qpos = None

        print(f"🔍 Testing connection to Cloud Brain at {SERVER_GET_URL}...")
        try:
            test_resp = requests.get(SERVER_GET_URL, headers=HEADERS, timeout=5)
            test_resp.raise_for_status() 
            print(f"✅ Server Response: {test_resp.text.strip()}")
            print("✅ Connection test successful.")
        except Exception as e:
            print(f"❌ CONNECTION ERROR: Cannot reach server.\nDetails: {e}")
            sys.exit(1)

    def query_openvla_server(self, image_str):
        """Queries OpenVLA server for 7D action."""
        payload = {
            "image": image_str,
            "prompt": TASK_PROMPT
        }

        try:
            response = requests.post(SERVER_URL, json=payload, headers=HEADERS, timeout=10)
            if response.status_code == 200:
                return np.array(response.json()["action"], dtype=np.float64)
        except Exception as e:
            print(f"⚠️ Server Request Failed: {e}")
        
        return np.array([0, 0, 0, 0, 0, 0, 1.0])

    def process_raw_action(self, raw_action):
        """Processes 7D action: EMA on spatial [:6], latching on gripper [6]."""
        raw_spatial = raw_action[:6]
        self.smoothed_spatial_action = (
            (1.0 - EMA_ALPHA) * self.smoothed_spatial_action + 
            EMA_ALPHA * raw_spatial
        )

        raw_gripper = raw_action[6]
        if raw_gripper > 0.8:
            self.gripper_target = GRIPPER_OPEN
        elif raw_gripper < 0.2:
            self.gripper_target = GRIPPER_CLOSE

        return self.smoothed_spatial_action, self.gripper_target

    def scale_gripper(self, vla_gripper_val):
        """Maps OpenVLA continuous gripper output to MuJoCo range."""
        normalized_val = np.clip(vla_gripper_val, 0.0, 1.0)
        return 0.04 * (1.0 - normalized_val)

    def compute_target_controls(self, smoothed_spatial, raw_gripper_val):
        """Calculates 6D Weighted Differential IK and returns target joint positions and gripper command."""
        raw_xyz = np.clip(smoothed_spatial[:3], -1.0, 1.0) * POSITION_SCALE
        raw_rpy = np.clip(smoothed_spatial[3:6], -1.0, 1.0) * ROTATION_SCALE

        # Axis alignment correction
        raw_xyz[0] = -raw_xyz[0]

        # Deadband filter to prevent micro-jitter
        raw_xyz[np.abs(raw_xyz) < 0.002] = 0.0
        raw_rpy[np.abs(raw_rpy) < 0.01] = 0.0

        # Transform camera-relative translation to world coordinates
        R_cam = self.data.cam_xmat[self.cam_id].reshape(3, 3)
        delta_xyz_world = R_cam @ raw_xyz

        # Transform orientation delta to world frame (using correct body/site lookup)
        if self.use_body:
            ee_rot_mat = self.data.xmat[self.site_id].reshape(3, 3)
        else:
            ee_rot_mat = self.data.site_xmat[self.site_id].reshape(3, 3)

        rot_delta = R.from_euler('xyz', raw_rpy).as_matrix()
        rot_world = ee_rot_mat @ rot_delta @ ee_rot_mat.T
        delta_rpy_world = R.from_matrix(rot_world).as_rotvec()

        delta_pose = np.concatenate([delta_xyz_world, delta_rpy_world])

        # 6D Jacobians (Position + Rotation) with correct body/site method
        jacp = np.zeros((3, self.model.nv))
        jacr = np.zeros((3, self.model.nv))
        if self.use_body:
            mujoco.mj_jacBody(self.model, self.data, jacp, jacr, self.site_id)
        else:
            mujoco.mj_jacSite(self.model, self.data, jacp, jacr, self.site_id)

        J_full = np.vstack([jacp[:, :7], jacr[:, :7]])

        # Weighted DLS Pseudo-Inverse (penalize base joint for extension)
        W = np.diag([5.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
        W_inv = np.linalg.inv(W)
        inv_matrix = np.linalg.inv(J_full @ W_inv @ J_full.T + DAMPING * np.eye(6))
        J_pinv = W_inv @ J_full.T @ inv_matrix

        delta_q = J_pinv @ delta_pose
        delta_q = np.clip(delta_q, -0.05, 0.05)

        # Initialize target_qpos accumulator if first step
        if self.target_qpos is None:
            self.target_qpos = self.data.qpos[ARM_JOINT_IDS].copy()
            
        self.target_qpos += delta_q

        # Clip to control bounds
        target_q_clipped = np.zeros(7)
        for i, act_id in enumerate(ARM_JOINT_IDS):
            lo, hi = self.model.actuator_ctrlrange[act_id]
            target_q_clipped[i] = np.clip(self.target_qpos[i], lo, hi)

        gripper_ctrl = self.scale_gripper(raw_gripper_val)

        return target_q_clipped, gripper_ctrl, delta_xyz_world

# ==========================================
# MAIN EXECUTION LOOP WITH FULL DEBUG PRINTS
# ==========================================
def run_simulation(xml_path):
    model = mujoco.MjModel.from_xml_path(xml_path)
    data = mujoco.MjData(model)

    # --- CRITICAL FIX: Configure stiff actuator gains to hold position ---
    kp, kd = 3000.0, 2.0 * np.sqrt(3000.0)
    for i in range(min(7, model.nu)):
        params = model.actuator_biasprm[i].copy()
        params[0] = 0.0
        params[1] = -kp
        params[2] = -kd
        model.actuator_biasprm[i] = params

    controller = OpenVLAController(model, data)
    mujoco.mj_resetData(model, data)

    # Set bent "ready" pose to break singularity
    ready_qpos = [0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785]
    data.qpos[ARM_JOINT_IDS] = ready_qpos
    controller.target_qpos = np.array(ready_qpos)
    mujoco.mj_forward(model, data)

    renderer = mujoco.Renderer(model, height=224, width=224)
    cube_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "red_cube")

    print("🚀 Starting MuJoCo OpenVLA Control Loop...")

    with mujoco.viewer.launch_passive(model, data) as viewer:
        while viewer.is_running():
            step_start = time.time()

            # 1. Render frame from vla_camera perspective
            renderer.update_scene(data, camera=controller.cam_id)
            pixels = renderer.render()

            # 2. Compress image frame to JPEG base64
            img = PIL.Image.fromarray(pixels)
            buffered = io.BytesIO()
            img.save(buffered, format="JPEG")
            img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")

            # 3. Query OpenVLA server
            raw_action = controller.query_openvla_server(img_str)

            # 4. Process action
            smoothed_spatial, raw_gripper = controller.process_raw_action(raw_action)

            # 5. Compute target controls via controller
            target_q, target_gripper, delta_xyz_world = controller.compute_target_controls(smoothed_spatial, raw_gripper)

            # 6. Apply explicitly to local physics data
            data.ctrl[ARM_JOINT_IDS] = target_q
            data.ctrl[GRIPPER_CTRL_ID] = target_gripper

            # -------------------------------------------------------------
            # FULL DEBUG PRINTING LOGS
            # -------------------------------------------------------------
            if cube_id != -1:
                cube_pos = data.xpos[cube_id]
                print(f"📍 Cube World Pos -> X: {cube_pos[0]:.4f}, Y: {cube_pos[1]:.4f}, Z: {cube_pos[2]:.4f}")

            act_str = ", ".join([f"{x:+.6f}" for x in raw_action])
            print(f"📡 Action: [{act_str}]")
            print(f"Cam XYZ: {np.round(smoothed_spatial[:3], 8)} --> World XYZ: {np.round(delta_xyz_world, 8)}")
            
            print("data.qpos:")
            print(np.round(data.qpos, 8))
            print(ARM_JOINT_IDS)
            print(GRIPPER_CTRL_ID)
            
            print("data.ctrl:")
            print(np.round(data.ctrl, 8))
            print("gripper_action (Raw Prediction):")
            print(raw_action[6])
            print("latched_gripper_ctrl (Actual Actuator Target):")
            print(target_gripper)
            print("-" * 60)
            # -------------------------------------------------------------

            # Step physics forward
            for _ in range(PHYSICS_STEPS_PER_ACTION):
                mujoco.mj_step(model, data)

            viewer.sync()

            # Maintain real-time frame timing
            time_until_next_frame = model.opt.timestep * PHYSICS_STEPS_PER_ACTION - (time.time() - step_start)
            if time_until_next_frame > 0:
                time.sleep(time_until_next_frame)

    print("👋 Exiting simulation bridge.")


import gymnasium as gym
import logging
import numpy as np
import mujoco
import mujoco.viewer
from gymnasium import spaces

from src.inversekinematics.ik import IK
from src.sim.camera import Camera

class ArmEnv(gym.Env):

    metadata = {"render_modes": ["human"], "render_fps": 60}

    def __init__(self, args : dict):
        """Initialize Arm Environment for VLA.

        Args:
        - render: boolean for viewer rendering.
        """
        path = 'src/armx7/scene.xml'
        with open(path, 'r') as f:
            xml = f.read()

        self.action_space = spaces.Box(low=-np.inf, high=np.inf, shape=(7,))
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.cam_width, self.cam_height, 3))

        self.cam_width = 480
        self.cam_height = 480

        self.num_dof = 7
        self.sample_hz = 5
        self.timestep = 0.002
        self.iters = (1/self.sample_hz)/self.timestep

        self.model = mujoco.MjModel.from_xml_string(xml)
        self.data = mujoco.MjData(self.model)
        self.data.qpos[:self.num_dof] = [0, 0.427, 0, 0.357, 0, -0.0109, 1.57]
        self.data.ctrl[:self.num_dof] = [0, 0.427, 0, 0.357, 0, -0.0109, 1.57]
        self.paused = False
        self.logging = args['logging']
        self.render = args['render']
        args = {'cam_width': self.cam_width, 'cam_height': self.cam_height}
        self.camera = Camera(args, self.model, self.data, "cam")

        if self.logging:
            self.logger = logging.basicConfig(filename='data/log/app.log',
                        level=logging.DEBUG,
                        format='%(asctime)s - %(levelname)s - %(message)s')

        self.ik = IK(self.model, self.data)

        for i in range(self.model.ncam):
            print(f"Camera ID {i}: name = '{self.model.cam(i).name}'")

        if self.render:
            self.viewer =  mujoco.viewer.launch_passive(
                self.model, 
                self.data, 
                key_callback=self.key_callback, 
                show_left_ui=False, 
                show_right_ui=False,
            ) 
            self.viewer.sync()
    
    def get_image(self) -> np.ndarray:
        """
        Get image of the current scene from camera.

        Returns:
                np.ndarray: image of scene from camera.
        """
        return self.camera.image

    def _get_info(self):
        """Get current information."""
        touch_bottom = self.data.sensordata[self.bottom_sensorid]

        contacts = []
        for i in range(self.data.ncon):
            forces = np.zeros(6)
            contact = self.data.contact[i]
            geom1_id = contact.geom[0]
            geom2_id = contact.geom[1]
            mujoco.mj_contactForce(self.model, self.data, i, forces)
            contact = {
                "geom1": self.geom_names[geom1_id],
                "geom2": self.geom_names[geom2_id],
                "forces": forces,
            }
            contacts.append(contact)

        return {
            "touch_bottom": np.array([touch_bottom]),
            "contacts": contacts,
        }

    def step(self, action : np.ndarray):
        """
        Step the environment and put action into the simulator.

        Args:
        - action: nparray of change in pose of the gripper from VLA.
        """
        new_action = self.ik.calculate(action)
        self.data.ctrl[:7] = new_action
        for i in range(self.iters):
            mujoco.mj_step(self.model, self.data)
            if self.logging:
                logging.info(f"action: {new_action} \r\n")
                logging.info(f"results: {self._get_info()} \r\n")
            if self.render:
                self.viewer.sync()
    
    def close(self):
        """
        Close the renderer if option.
        """
        if self.render:
            self.viewer.close()

    def key_callback(self, keycode):
        """
        Callback for viewer pause.
        """
        key = chr(keycode)
        if key == ' ':
            self.paused = not self.paused


if __name__ == "__main__":
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    XML_FILE_PATH = os.path.join(SCRIPT_DIR, "vla_task.xml")
    run_simulation(XML_FILE_PATH)