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
from datetime import datetime
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

EMA_ALPHA = 1.0
POSITION_SCALE = 0.7 # 0.6
ROTATION_SCALE = 0.02
DAMPING = 0.05

# Joint & Actuator indices
ARM_JOINT_IDS = list(range(7))      # Joint IDs 0..6
GRIPPER_CTRL_ID = 7               # Actuator index for Franka finger control

class IK:
    def __init__(self, site_id):
        self.site_id = site_id

    def get_position_error(self, data, target_pos, dbg_print=False):
        current_pos = data.site_xpos[self.site_id].copy()
        pos_error = target_pos - current_pos
        if dbg_print:
            print("CURRENT EE:", np.round(data.site_xpos[self.site_id], 5))
            print("TARGET EE:", np.round(target_pos, 5))
            print("EE error:", pos_error, "meters")
        return pos_error

    def get_rotation_error(self, data, target_rot, dbg_print=False):
        current_rot = data.site_xmat[self.site_id].reshape(3,3)
        rot_error_mat = target_rot @ current_rot.T
        rot_error = R.from_matrix(rot_error_mat).as_rotvec()
        if dbg_print:
            print("CURRENT EE ROT:", np.round(data.site_xmat[self.site_id], 5))
            print("TARGET EE ROT:", np.round(target_rot, 5))
            print("EE ROT error:", rot_error, "radians")
        return rot_error

    def calculate(self, model, data, target_pos, target_rot):

        # MuJoCo computes the geometric Jacobian
        jacp = np.zeros((3, model.nv))
        jacr = np.zeros((3, model.nv))

        mujoco.mj_jacSite(
            model,
            data,
            jacp,
            jacr,
            self.site_id
        )

        # Panda has 7 arm DoFs
        J = np.vstack((jacp[:, :7], jacr[:, :7]))

        # Damped least-squares inverse
        lam = 0.01
        J_inv = (J.T @ np.linalg.inv(J @ J.T + (lam ** 2) * np.eye(6)))

        # Position error
        pos_error = self.get_position_error(data, target_pos)

        # Rotation error
        rot_error = self.get_rotation_error(data, target_rot)

        # 6D Cartesian error
        delta_pose = np.concatenate(
            [
                pos_error,
                rot_error
            ]
        )

        # Limit Cartesian motion per iteration
        max_pos_step = 0.07  # 50 mm
        pos_norm = np.linalg.norm(delta_pose[:3])
        if pos_norm > max_pos_step:
            delta_pose[:3] *= max_pos_step / pos_norm

        # Differential IK:
        # Δq = J⁺ Δx
        dq = J_inv @ delta_pose
        #print(dq)

        # Limit step size
        #dq = np.clip(dq, -0.5, 0.05)

        # Current Panda joint positions
        q_current = data.qpos[:7].copy()

        joint_ranges = model.jnt_range[ARM_JOINT_IDS]

        return np.clip(
            q_current + dq,
            joint_ranges[:,0],
            joint_ranges[:,1]
        )

# ==========================================
# CONTROLLER STATE
# ==========================================
class OpenVLAController:
    def __init__(self):
        print(f"🔍 Testing connection to Cloud Brain at {SERVER_GET_URL}...")
        # EMA buffer for spatial action (XYZ + RPY)
        self.smoothed_spatial_action = np.zeros(6)
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

        return self.smoothed_spatial_action, raw_gripper

    def compute_delta_pose(self, smoothed_spatial, R_cam):
        raw_xyz = np.clip(smoothed_spatial[:3], -1.0, 1.0) * POSITION_SCALE
        raw_rpy = np.clip(smoothed_spatial[3:6], -1.0, 1.0) * ROTATION_SCALE
        delta_xyz_world = R_cam.T @ raw_xyz
        delta_rotvec_world = R_cam.T @ raw_rpy
        #world1 = R_cam @ raw_xyz
        #world2 = R_cam.T @ raw_xyz
        #print(world1)
        #print(world2)

        print("raw_xyz: ", raw_xyz)
        print("world_xyz: ", delta_xyz_world)
        print("raw_rpy: ", raw_rpy)
        print("world_rotvec: ", delta_rotvec_world)

        return delta_xyz_world, delta_rotvec_world 


import numpy as np

def freecam_to_xml(viewer, fovy=45):
    cam = viewer.cam

    lookat = np.array(cam.lookat, dtype=float)
    dist = cam.distance
    az = np.deg2rad(cam.azimuth)
    el = np.deg2rad(cam.elevation)

    #
    # Camera position
    #
    pos = lookat + dist * np.array([
        -np.cos(el) * np.sin(az),
        -np.cos(el) * np.cos(az),
        -np.sin(el),
    ])

    #
    # Camera forward direction
    #
    forward = lookat - pos
    forward /= np.linalg.norm(forward)

    #
    # World up
    #
    world_up = np.array([0.0, 0.0, 1.0])

    #
    # Camera right axis
    #
    right = np.cross(forward, world_up)
    right /= np.linalg.norm(right)

    #
    # Camera up axis
    #
    up = np.cross(right, forward)
    up /= np.linalg.norm(up)

    #
    # MuJoCo xyaxes
    #
    xyaxes = np.concatenate([right, up])

    print("\n===== MuJoCo XML =====")
    print(
        f'<camera name="vla_camera"\n'
        f'        pos="{pos[0]:.6f} {pos[1]:.6f} {pos[2]:.6f}"\n'
        f'        xyaxes="{xyaxes[0]:.6f} {xyaxes[1]:.6f} {xyaxes[2]:.6f} '
        f'{xyaxes[3]:.6f} {xyaxes[4]:.6f} {xyaxes[5]:.6f}"\n'
        f'        fovy="{fovy}"/>')

    print("\nPosition :", pos)
    print("Forward  :", forward)
    print("Right    :", right)
    print("Up       :", up)

    return pos, xyaxes

def scale_gripper(model, raw_gripper, actuator_id=7,
                  prev_target=None, alpha=None):
    """
    Convert OpenVLA gripper output (0..1) to the MuJoCo actuator range.

    Parameters
    ----------
    model : mujoco.MjModel
    raw_gripper : float
        OpenVLA gripper output (expected in [0,1])
    actuator_id : int
        Gripper actuator index (default: 7)
    prev_target : float or None
        Previous commanded value (for smoothing)
    alpha : float or None
        EMA smoothing factor.
        None -> no smoothing
        0.2  -> slow/smooth
        0.5  -> medium
        1.0  -> no smoothing

    Returns
    -------
    float
        Target actuator command in the model's ctrlrange.
    """

    raw = np.clip(raw_gripper, 0.0, 1.0)

    ctrl_min, ctrl_max = model.actuator_ctrlrange[actuator_id]

    target = ctrl_min + raw * (ctrl_max - ctrl_min)

    if alpha is not None and prev_target is not None:
        target = alpha * target + (1.0 - alpha) * prev_target

    return target

def open_gripper(model, data, viewer):
    print("Open")
    data.ctrl[7] = scale_gripper(model, 1.0)   # fully open
    print("gripper open: ", data.ctrl[7])

    for _ in range(200):
        mujoco.mj_step(model, data)
        viewer.sync()

    print(data.qpos[7:9])

def close_gripper(model, data, viewer):
    print("Close")
    data.ctrl[7] = scale_gripper(model, 0.0)   # fully close
    print("gripper close: ", data.ctrl[7])

    for _ in range(200):
        mujoco.mj_step(model, data)
        viewer.sync()

    print(data.qpos[7:9])

def test_gripper(model, data, viewer):
    open_gripper(model, data, viewer)

    time.sleep(2)

    close_gripper(model, data, viewer)

def run_simulation(xml_path):
    model = mujoco.MjModel.from_xml_path(xml_path)
    data = mujoco.MjData(model)
    mujoco.mj_resetData(model, data)

    for i in range(model.nu):
        print(
        f"Actuator {i}:",
        mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i),
        "trnid =", model.actuator_trnid[i],
        "ctrlrange =", model.actuator_ctrlrange[i]
        )

    # Set bent "ready" pose to break singularity
    ready_qpos = [0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785]
    data.qpos[ARM_JOINT_IDS] = ready_qpos
    mujoco.mj_forward(model, data)
    renderer = mujoco.Renderer(model, height=224, width=224)
    print("data.qpos for ready pose: ", data.qpos)
    print("data.ctrl in ready pose: ", data.ctrl)

    # End-effector / Reference frame setup (Site vs Body fallback)
    site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "pinch")
    print("site_id: ", site_id)
    if site_id == -1:
        print("pinch does not exist")
        exit(0)

    ik = IK(site_id)

    # Initialize target EE pose
    current_pos = data.site_xpos[site_id].copy()
    current_rot = data.site_xmat[site_id].reshape(3,3).copy()
    print("Initial EE position:", current_pos)
    print("Initial EE rotation:", current_rot)
    target_pos = current_pos
    target_rot = current_rot
    data.ctrl[GRIPPER_CTRL_ID] = scale_gripper(model, 1.0)   # fully open
    mujoco.mj_step(model, data)


    print("🚀 Starting MuJoCo OpenVLA Control Loop...")

    # Camera ID lookup
    cam_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, "vla_camera")
    print("cam_id is ", cam_id)
    if cam_id == -1:
        print("overriding the cam_id to 0 as vla_camera does not exist")
        cam_id = 0

    cube_id = model.body("cube").id
    cube_pos = data.xpos[cube_id] 
    print("Cube position:", cube_pos)
    print("Hand position:", data.xpos[model.body("hand").id])

    controller = OpenVLAController()

    with mujoco.viewer.launch_passive(model, data) as viewer:

        while viewer.is_running():

            '''
            test_gripper(model, data, viewer)
            cam = viewer.cam
            print("lookat:", cam.lookat)
            print("distance:", cam.distance)
            print("azimuth:", cam.azimuth)
            print("elevation:", cam.elevation)
            freecam_to_xml(viewer)
            '''

            # ======================================================
            # 1. RENDER CAMERA IMAGE
            # ======================================================
            renderer.update_scene(data, camera="vla_camera")
            pixels = renderer.render()
            img = PIL.Image.fromarray(pixels)
            buffered = io.BytesIO()
            img.save(buffered, format="JPEG")
            img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
            filename = f"vla_observation_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}.jpg"
            img.save(filename, format="JPEG")

            # ======================================================
            # 2. GET NEW ACTION ONLY AFTER HOLD TIME
            # ======================================================
            raw_action = controller.query_openvla_server(img_str)
            print("NEW ACTION: ", raw_action)

            # Process VLA action
            smoothed_spatial, _= (controller.process_raw_action(raw_action))
            #print("Smoothed Action: ", smoothed_spatial)
            #print("raw_gripper: ", raw_gripper)

            # Compute delta_pose
            R_cam = data.cam_xmat[cam_id].reshape(3, 3)
            delta_xyz_world, delta_rotvec_world = (controller.compute_delta_pose(smoothed_spatial, R_cam))
            #print("delta_xyz_world: ", delta_xyz_world)
            #print("delta_rotvec_world: ", delta_rotvec_world)

            # ==========================================
            # UPDATE TARGET EE POSITION
            # ==========================================
            TARGET_STEP = 0.02   # 2 cm per action update
            distance = np.linalg.norm(delta_xyz_world)
            if distance > TARGET_STEP:
                delta_xyz_world = (delta_xyz_world / distance) * TARGET_STEP

            current_pos = data.site_xpos[site_id].copy()
            target_pos = current_pos + delta_xyz_world
            #print("target_pos:", target_pos)

            '''
            # ==========================================
            # UPDATE TARGET EE ROTATION
            # ==========================================
            if np.linalg.norm(delta_rotvec_world) > 1e-6:
                rot_delta = R.from_rotvec(delta_rotvec_world).as_matrix()
                #target_rot = target_rot @ rot_delta
                current_rot = data.site_xmat[site_id].reshape(3,3).copy()
                target_rot = current_rot @ rot_delta
            #print("target_rot: ", target_rot)
            '''

            # ======================================================
            # 4. APPLY CONTROL
            # ======================================================
            '''
            target_gripper = scale_gripper(
                model,
                raw_gripper,
                actuator_id=GRIPPER_CTRL_ID,
                prev_target=target_gripper,
                alpha=1.0           
            )
            '''

            # ======================================================
            # 5. DEBUG
            # ======================================================
            _ = ik.get_position_error(data, target_pos, dbg_print=True)
            #_ = ik.get_rotation_error(data, target_rot, dbg_print=True)

            """
            # ======================================================
            # 6. PHYSICS STEPS
            # ======================================================
            for _ in range(250):
                target_q = ik.calculate(model, data, target_pos, target_rot)
                data.ctrl[ARM_JOINT_IDS] = target_q
                #print("target_q: ", target_q)
                #print("qpos: ", data.qpos[:7])
                #print("ctrl: ", data.ctrl[:7])
                mujoco.mj_step(model, data)
                viewer.sync()

                ee = data.site_xpos[site_id]
                err = np.linalg.norm(target_pos - ee)

                if err < 0.001:      # 1 mm
                    break

            for outer in range(3):
                # Solve IK once
                target_q = ik.calculate(model, data, target_pos, target_rot)
                # Command joints
                data.ctrl[ARM_JOINT_IDS] = target_q
                ee = data.site_xpos[site_id]
                err = np.linalg.norm(target_pos - ee)
                while err > 0.001:
                    mujoco.mj_step(model, data)

                    ee = data.site_xpos[site_id]
                    err = np.linalg.norm(target_pos - ee)
                    steps += 1
            """

            steps = 0
            for outer in range(30):

                target_q = ik.calculate(model, data, target_pos, target_rot)
                data.ctrl[:7] = target_q

                for _ in range(100):
                    mujoco.mj_step(model, data)
                    steps += 1

                ee = data.site_xpos[site_id]
                err = np.linalg.norm(target_pos - ee)

                if err < 0.005:
                    break
            viewer.sync()
            #print("Settled in steps: ", steps)

            #print("CURRENT EE After Step: ", np.round(data.site_xpos[site_id], 5))
            
            new_pos = data.site_xpos[site_id].copy()
            print("Actual EE motion:", new_pos - current_pos)
 
            error = target_pos - data.site_xpos[site_id]
            # Note target_pos is where we want to be with this new action
            print("Remaining error in reaching target pos:", np.round(error, 5))
            #print("Norm:", np.linalg.norm(error))   

            #print remaining rotation error
            #print("Remaining rotation error in reaching target pos:", np.round(error, 5))
            #_ = ik.get_rotation_error(data, target_rot, dbg_print=True)

            # data.ctrl[GRIPPER_CTRL_ID] = target_gripper
            current_pos = data.site_xpos[site_id].copy()
            distance = np.linalg.norm(cube_pos - current_pos)
            print("Distance from cube: ", distance)
            if distance < 0.02:
                print("Closong the gripper")
                close_gripper(model, data, viewer)

            '''
            grip_pos = data.site_xpos[site_id]
            grip_rot = data.site_xmat[site_id].reshape(3,3)
            grip_axis = -grip_rot[:, 2]    # negative local Z
            to_cube = cube_pos - grip_pos
            to_cube /= np.linalg.norm(to_cube)
            dot = np.clip(np.dot(grip_axis, to_cube), -1.0, 1.0)
            angle_deg = np.degrees(np.arccos(dot))
            print(f"Grasp alignment: {angle_deg:.1f} deg")
            '''

    print("👋 Exiting simulation bridge.")

if __name__ == "__main__":
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    XML_FILE_PATH = os.path.join(SCRIPT_DIR, "vla_task.xml")
    run_simulation(XML_FILE_PATH)