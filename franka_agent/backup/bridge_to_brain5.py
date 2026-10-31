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
from PIL import Image
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

EMA_ALPHA = 1.0 # 0.35
PHYSICS_STEPS_PER_ACTION = 10
POSITION_SCALE = 0.5 # 0.6
ROTATION_SCALE = 0.02
DAMPING = 0.05

# Joint & Actuator indices
ARM_JOINT_IDS = list(range(7))      # Joint IDs 0..6
GRIPPER_CTRL_ID = 7               # Actuator index for Franka finger control

# Franka Panda Gripper Bounds
GRIPPER_OPEN = 0.04    # Fully open
GRIPPER_CLOSE = 0.00   # Fully closed

class IK:
    def __init__(self, model, data, site_id):
        self.model = model
        self.data = data
        self.site_id = site_id

    def calculate(self, target_pos, target_rot):

        # MuJoCo computes the geometric Jacobian
        jacp = np.zeros((3, self.model.nv))
        jacr = np.zeros((3, self.model.nv))

        mujoco.mj_jacSite(
            self.model,
            self.data,
            jacp,
            jacr,
            self.site_id
        )

        #print(jacp[:, :7])
        #print("rank:", np.linalg.matrix_rank(jacp[:, :7]))

        # Panda has 7 arm DoFs
        J = np.vstack(
            (
                jacp[:, :7],
                jacr[:, :7]
            )
        )

        # Damped least-squares inverse
        lam = 0.01

        J_inv = (
            J.T @
            np.linalg.inv(
                J @ J.T +
                (lam ** 2) * np.eye(6)
            )
        )

        # Current EE pose
        current_pos = self.data.site_xpos[self.site_id].copy()
        current_rot = self.data.site_xmat[self.site_id].reshape(3,3)

        # Position error
        pos_error = target_pos - current_pos

        # Rotation error
        rot_error_mat = target_rot @ current_rot.T

        rot_error = R.from_matrix(rot_error_mat).as_rotvec()

        # 6D Cartesian error
        delta_pose = np.concatenate(
            [
                pos_error,
                rot_error
            ]
        )

        # Limit Cartesian motion per iteration
        max_pos_step = 0.005  # 5 mm

        pos_norm = np.linalg.norm(delta_pose[:3])

        if pos_norm > max_pos_step:
            delta_pose[:3] *= max_pos_step / pos_norm

        # Differential IK:
        # Δq = J⁺ Δx
        dq = J_inv @ delta_pose
        print(dq)

        # Limit step size
        dq = np.clip(dq, -0.01, 0.01)
        #print("clipped dq:", dq)

        predicted = J @ dq
        #print("Cartesian predicted: ", predicted)
        #print("Desired delta_pose: ", delta_pose)
        #print("pos error: ", pos_error)

        # Current Panda joint positions
        q_current = self.data.qpos[:7].copy()

        joint_ranges = self.model.jnt_range[ARM_JOINT_IDS]

        return np.clip(
            q_current + dq,
            joint_ranges[:,0],
            joint_ranges[:,1]
        )

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
        print("cam_id is ", self.cam_id)
        if self.cam_id == -1:
            print("overriding the cam_id to 0 as vla_camera does not exist")
            self.cam_id = 0

        # End-effector / Reference frame setup (Site vs Body fallback)
        self.site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "pinch")
        print("site_id: ", self.site_id)
        if self.site_id == -1:
            print("pinch does not exist")
            exit(0)

        # Target end-effector pose accumulator
        self.target_pos = None
        self.target_rot = None

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

    def compute_delta_pose(self, smoothed_spatial):

        raw_xyz = np.clip(smoothed_spatial[:3], -1.0, 1.0) * POSITION_SCALE

        raw_rpy = np.clip(smoothed_spatial[3:6], -1.0, 1.0) * ROTATION_SCALE

        return np.concatenate(
            [
                raw_xyz,
                raw_rpy
            ]
        ), raw_xyz

    def compute_delta_pose1(self, smoothed_spatial):
        """Transforms policy prediction space into world-frame cartesian increments."""
        raw_xyz = np.clip(smoothed_spatial[:3], -1.0, 1.0) * POSITION_SCALE
        raw_rpy = np.clip(smoothed_spatial[3:6], -1.0, 1.0) * ROTATION_SCALE

        # Axis alignment correction
        raw_xyz[0] = -raw_xyz[0]

        # Deadband filter to prevent micro-jitter
        raw_xyz[np.abs(raw_xyz) < 0.002] = 0.0
        raw_rpy[np.abs(raw_rpy) < 0.01] = 0.0

        # Transform camera-relative translation to world coordinates
        R_cam = self.data.cam_xmat[self.cam_id].reshape(3, 3)
        #delta_xyz_world = R_cam @ raw_xyz
        delta_xyz_world = raw_xyz

        # Transform orientation delta to world frame
        ee_rot_mat = self.data.site_xmat[self.site_id].reshape(3, 3)

        rot_delta = R.from_euler('xyz', raw_rpy).as_matrix()
        #rot_world = ee_rot_mat @ rot_delta @ ee_rot_mat.T
        rot_world = ee_rot_mat @ rot_delta
        delta_rpy_world = R.from_matrix(rot_world).as_rotvec()

        return np.concatenate([delta_xyz_world, delta_rpy_world]), delta_xyz_world


# ==========================================
# MAIN EXECUTION LOOP WITH FULL DEBUG PRINTS
# ==========================================
def run_simulation(xml_path):
    model = mujoco.MjModel.from_xml_path(xml_path)
    data = mujoco.MjData(model)

    print("===== MuJoCo Model Info =====")
    print("Number of actuators (model.nu):", model.nu)
    print("Number of joints (model.njnt):", model.njnt)

    print("\n===== Actuators =====")
    for i in range(model.nu):
        print(i, mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i))

    print("\n===== Joints =====")
    for i in range(model.njnt):
        print(i, mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i))

    print("\n===== Sites =====")
    for i in range(model.nsite):
        print(i, mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_SITE, i))

    print("actuator details")
    for i in range(model.nu):
        print(
            i,
            "trntype=", model.actuator_trntype[i],
            "gaintype=", model.actuator_gaintype[i],
            "biastype=", model.actuator_biastype[i],
            "gear=", model.actuator_gear[i],
            "ctrlrange=", model.actuator_ctrlrange[i]
        )
    for i in range(7):
        print(
            i,
            "gain:",
            model.actuator_gainprm[i],
            "bias:",
            model.actuator_biasprm[i]
        )

    mujoco.mj_resetData(model, data)
    # Set bent "ready" pose to break singularity
    ready_qpos = [0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785]
    data.qpos[ARM_JOINT_IDS] = ready_qpos
    mujoco.mj_forward(model, data)
    print("Initial data.ctrl: ", data.ctrl)

    controller = OpenVLAController(model, data)

    # Initialize target EE pose
    ee_pos = data.site_xpos[controller.site_id].copy()
    ee_rot = data.site_xmat[controller.site_id].reshape(3,3).copy()
    controller.target_pos = ee_pos
    controller.target_rot = ee_rot
    print("Initial EE position:", controller.target_pos)
    target_gripper = GRIPPER_OPEN

    renderer = mujoco.Renderer(model, height=224, width=224)
    cube_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "cube")
    ik = IK(model, data, controller.site_id)

    print("🚀 Starting MuJoCo OpenVLA Control Loop...")

    with mujoco.viewer.launch_passive(model, data) as viewer:

        while viewer.is_running():
            # ======================================================
            # 1. RENDER CAMERA IMAGE
            # ======================================================
            renderer.update_scene(data, camera=controller.cam_id)
            pixels = renderer.render()
            img = PIL.Image.fromarray(pixels)
            buffered = io.BytesIO()
            img.save(buffered, format="JPEG")
            img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")

            # ======================================================
            # 2. GET NEW ACTION ONLY AFTER HOLD TIME
            # ======================================================
            raw_action = controller.query_openvla_server(img_str)
            print("NEW ACTION: ", raw_action)

            # Process VLA action
            smoothed_spatial, target_gripper = (controller.process_raw_action(raw_action))

            # Compute delta_pose
            _, delta_xyz_world = (controller.compute_delta_pose(smoothed_spatial))

            # ==========================================
            # UPDATE TARGET EE POSITION
            # ==========================================
            TARGET_STEP = 0.02   # 2 cm per action update
            distance = np.linalg.norm(delta_xyz_world)
            if distance > TARGET_STEP:
                delta_xyz_world = (delta_xyz_world / distance) * TARGET_STEP

            current_pos = data.site_xpos[controller.site_id].copy()
            controller.target_pos = current_pos + delta_xyz_world

            # ==========================================
            # UPDATE TARGET EE ROTATION
            # ==========================================
            raw_rpy = (np.clip( smoothed_spatial[3:6], -1.0, 1.0) * ROTATION_SCALE)

            if np.linalg.norm(raw_rpy) > 1e-6:
                rot_delta = R.from_euler('xyz', raw_rpy).as_matrix()

                current_rot = data.site_xmat[controller.site_id].reshape(3,3)

                controller.target_rot = current_rot @ rot_delta

            # ======================================================
            # 4. APPLY CONTROL
            # ======================================================
            data.ctrl[GRIPPER_CTRL_ID] = target_gripper

            # ======================================================
            # 5. DEBUG
            # ======================================================
            current_ee = data.site_xpos[controller.site_id].copy()
            error = controller.target_pos - current_ee

            print("CURRENT EE:", np.round(data.site_xpos[controller.site_id], 5))
            print("TARGET EE:", np.round(controller.target_pos,5))
            print("EE error:", np.round(np.linalg.norm(error),6), "meters")
            print("Joint CTRL:", np.round(data.ctrl[:7],4))

            print("NEW ACTION:", raw_action)

            print("smoothed:", smoothed_spatial)

            print("delta_xyz_world:", delta_xyz_world)

            print("target_pos:", controller.target_pos)
            print("target_rot: ", controller.target_rot)

            print("CURRENT EE:", data.site_xpos[controller.site_id])

            print("TARGET EE:", controller.target_pos)

            print("EE error:", np.linalg.norm(
                controller.target_pos -
                data.site_xpos[controller.site_id]
            ))

            print("qpos before ik.calculate: ", data.qpos[:7])
            
            # ======================================================
            # 6. PHYSICS STEPS
            # ======================================================
            for _ in range(PHYSICS_STEPS_PER_ACTION):
                target_q = ik.calculate(controller.target_pos, controller.target_rot)
                data.ctrl[ARM_JOINT_IDS] = target_q
                print("target_q: ", target_q)
                print("qpos: ", data.qpos[:7])
                print("ctrl: ", data.ctrl[:7])
                print("time: ", data.time)
                mujoco.mj_step(model, data)
                viewer.sync()
                time.sleep(0.1)

    print("👋 Exiting simulation bridge.")

if __name__ == "__main__":
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    XML_FILE_PATH = os.path.join(SCRIPT_DIR, "vla_task.xml")
    run_simulation(XML_FILE_PATH)