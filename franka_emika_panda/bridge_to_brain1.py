import os
import sys
import io
import time
import base64
import requests
import numpy as np
import PIL.Image
import mujoco
import mujoco.viewer

from scipy.spatial.transform import Rotation as R

# === CONFIG ===
SERVER_URL = "https://surface-shuffling-recede.ngrok-free.dev/predict"
SERVER_GET_URL = "https://surface-shuffling-recede.ngrok-free.dev/"
HEADERS = {"ngrok-skip-browser-warning": "true"}

# Define the explicit task instructions here!
TASK_PROMPT = "Pick up the red cube."

# Global MuJoCo variables
model = None
'''
In the standard Franka Emika Panda XML, data.ctrl has 9 elements:
0-6: Arm joints
7: Finger 1
8: Finger 2
'''
data = None
renderer = None

# --- INITIALIZE MUJOCO ---
try:
    model = mujoco.MjModel.from_xml_path("vla_task.xml")
    data = mujoco.MjData(model)
    renderer = mujoco.Renderer(model, height=224, width=224)
    print("✅ MuJoCo scene initialized successfully.")
except Exception as e:
    print(f"❌ Failed to load MuJoCo structures: {e}")
    sys.exit(1)

# --- TEST SERVER PIPELINE ---
print(f"🔍 Testing connection to Cloud Brain at {SERVER_GET_URL}...")
try:
    test_resp = requests.get(SERVER_GET_URL, headers=HEADERS, timeout=5)
    test_resp.raise_for_status() 
    print(f"✅ Server Response: {test_resp.text.strip()}")
    print("✅ Connection test successful (Server is reachable).")
except Exception as e:
    print(f"❌ CONNECTION ERROR: Cannot reach the server. \nDetails: {e}")
    sys.exit(1)

def scale_gripper(vla_gripper_val):
    """
    Converts OpenVLA gripper values into MuJoCo actuator commands.
    
    OpenVLA: 
      - ~0.0 to 0.1 = OPEN
      - ~0.9 to 1.0 = CLOSE
    
    Franka MuJoCo Gripper Actuators:
      - 0.04 = Fully Open (4 cm limits per finger)
      - 0.00 = Fully Closed
    """
    # OpenVLA close to 1 means closed (0.0 limit), close to 0 means open (0.04 limit)
    # We clip to make sure values stay inside the Franka's physical limits
    normalized_val = np.clip(vla_gripper_val, 0.0, 1.0)
    target_width = 0.04 * (1.0 - normalized_val)
    
    return target_width

def run_inverse_kinematics(model, data, target_pos, target_quat, site_name="pinch"):
    """
    Calculates the joint positions needed to reach target_pos (3D) and target_quat (4D).
    Uses MuJoCo's built-in Jacobian features to solve IK numerically.
    """
    # Get the ID of our end-effector site ('pinch')
    site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, site_name)
    
    # Initialize Jacobians (3 rows for translation, 3 for rotation)
    jacp = np.zeros((3, model.nv))
    jacr = np.zeros((3, model.nv))
    
    # Damping factor to prevent crazy joint speeds near singularities
    damping = 1e-4
    step_size = 0.2
    tolerance = 1e-3
    max_steps = 100

    # Save original joint positions to restore state if needed
    qpos_init = data.qpos.copy()

    for _ in range(max_steps):
        # Update forward kinematics
        mujoco.mj_forward(model, data)
        
        # Get current site position
        current_pos = data.site_xpos[site_id]
        pos_err = target_pos - current_pos
        
        # Get current site orientation as quaternion
        current_mat = data.site_xmat[site_id].reshape(3, 3)
        current_quat = np.zeros(4)
        mujoco.mju_mat2Quat(current_quat, current_mat.flatten())
        
        # Calculate rotation error
        neg_current_quat = np.zeros(4)
        mujoco.mju_negQuat(neg_current_quat, current_quat)
        error_quat = np.zeros(4)
        mujoco.mju_mulQuat(error_quat, target_quat, neg_current_quat)
        rot_err = error_quat[1:] * np.sign(error_quat[0])
        
        # Combine errors into a 6D vector [dx, dy, dz, droll, dpitch, dyaw]
        error = np.concatenate([pos_err, rot_err])
        
        if np.linalg.norm(error) < tolerance:
            break
            
        # Get translation and rotation Jacobians for 'pinch'
        mujoco.mj_jacSite(model, data, jacp, jacr, site_id)
        J = np.vstack([jacp, jacr])
        
        # Damped Least Squares: J_inverse = J^T * (J * J^T + lambda^2 * I)^-1
        J_inv = J.T @ np.linalg.inv(J @ J.T + damping * np.eye(6))
        
        # Apply step to the actual joint positions (qpos)
        delta_q = J_inv @ error
        data.qpos[:7] += step_size * delta_q[:7]
        
    # Get the resulting target joint angles
    target_joints = data.qpos[:7].copy()
    
    # Restore the original physics state so the simulation doesn't "teleport"
    data.qpos[:] = qpos_init
    mujoco.mj_forward(model, data)
    
    return target_joints

def main():
    global model, data, renderer

    print("🚀 Starting Passive MuJoCo Viewer...")
    viewer = mujoco.viewer.launch_passive(model, data)
    
    # Force initial forward kinematics to prevent a false 0.0000 baseline reading
    mujoco.mj_forward(model, data)
    
    cube_id = model.body("cube").id 
    initial_cube_height = data.xpos[cube_id][2] 
    print(f"📦 Cube initialized on table at Z: {initial_cube_height:.4f}")
    print(data.xpos)

    while viewer.is_running():
        step_start = time.time()
        
        # 1. Synchronously render frame from the viewpoint
        renderer.update_scene(data, camera="vla_camera")
        pixels = renderer.render()
        
        # 2. Synchronously compress and encode the image frame
        img = PIL.Image.fromarray(pixels)
        buffered = io.BytesIO()
        img.save(buffered, format="JPEG")
        img_str = base64.b64encode(buffered.getvalue()).decode()
        
        # 3. Send synchronous blocking request to the OpenVLA Cloud Brain
        #print("📤 Posting image and task instruction to Cloud...")
        payload = {
            "image": img_str,
            "prompt": TASK_PROMPT
        }


        '''
        MuJoCo is designed to work in the MKS (Meters-Kgs-Seconds) system.
        openVLA outputs a 7D action vector:
        * 0-2: dx, dy, dz (deltas)
        * 3-5: Rotation (dr, dp, dy): End-effector orientation change (often Euler angles or axis-angle)
        * 6: Gripper State: Scalar value representing open (0.0)  or closed (1).0

        // MuJoCo
        data.qpos (Joint Positions): Radians: The 7 ARM joints
        data.qpos (Linear/Gripper): Meters: For sliding joints (e.g., the 2 finger parallel gripper)
        data.site_xpos: Meters: Global (x, y, z) coordinates of sites like the 'pinch'
        data.qvel (Joint Velocities):  Rad/s or m/s: Time derivative of positions
        '''

        POSITION_SCALE = 0.05 # Max meters per step (5cm)
        ROTATION_SCALE = 0.1 # Max radians per step (~5.7 degrees)
        try:
            response = requests.post(SERVER_URL, json=payload, headers=HEADERS, timeout=10)
            
            if response.status_code == 200:
                latest_action = np.array(response.json()["action"])
                print(f"📡 Raw Action from Server: {np.array2string(latest_action, precision=4, suppress_small=True)}")
                print(f"📊 Min/Max of predicted action: {latest_action.min():.4f} / {latest_action.max():.4f}")
                
                '''
                Translation (dx, dy, dz) Normalized [-1, 1] -- So, multiply by 0.02 to 0.05 for meters per step.
                Rotation (dr, dp, dy) Normalized to [-1, 1] -- So, multiply by 0.05 to 0.10 for radians per step.
                Gripper: [0, 1]: Map to gripper's open/close range.
                '''
                
                # 1. Split the 7D VLA action into Cartesian EE delta and gripper command
                # First, translate end effector Translation coordinates (dx, dy, dz)
                ee_delta_translation = latest_action[0:3] * POSITION_SCALE

                # Next, scale the rotation (dr, dp, dy)
                # latest_action[3:6] contains [d_roll, d_pitch, d_yaw]
                ee_delta_rotation = latest_action[3:6] * ROTATION_SCALE 

                # Copy gripper action
                gripper_action = latest_action[6] # Usually 0 (open) or 1 (close)

                # 2. Get current end-effector (EE) pose from MuJoCo
                # (Usually tracked via a site placed at the center of the gripper fingers)
                current_ee_pos = data.site('pinch').xpos
                # 3. Get the current end-effector rotation matrix from MuJoCo
                current_rot_matrix = data.site('pinch').xmat.reshape(3, 3)

                # 4. Calculate target Cartesian pose
                target_ee_pos = current_ee_pos + ee_delta_translation

                # Extract the raw 3D rotation delta from the server action (radians)
                # Convert the delta angles into a rotation matrix
                # VLA rotation deltas are typically local extrinsic Euler angles (extrinsic 'xyz')
                # Convert the loval Euler deltas to a rotation matrix and apply to current
                # Integrate them! Multiply delta * current (intrinsic/local rotation step)
                # This computes the new target rotation matrix
                delta_rot_matrix = R.from_euler('xyz', ee_delta_rotation).as_matrix()
                # data.site('gripper_site').xmat is a flat 9-element array; reshape it to 3x3
                target_rot_matrix = delta_rot_matrix @ current_rot_matrix

                # 5. Convert to quaternion for the IK solver (MuJoCo expects [w, x, y, z])
                quat_scipy = R.from_matrix(target_rot_matrix).as_quat()
                # Convert SciPy quaternion [x, y, z, w] to MuJoCo quaternion [w, x, y, z]
                target_quat_mj = np.array([quat_scipy[3], quat_scipy[0], quat_scipy[1], quat_scipy[2]])

                # 6. Calculate joint angles via IK
                # (Passing model and data objects to the IK function)
                new_targets = run_inverse_kinematics(model, data, target_ee_pos, target_quat_mj)

                # Apply targets to arm joints first
                data.ctrl[:7] = new_targets # Arm joints

                # Handle Gripper (Franka expects 0.0 (closed) and 0.04 (open)
                # VLA open = 0.0 and closed: 1.0
                # OpenVLA Open (0.0) -> 0.04 (MuJoCo Open)
                # OpenVLA Closed (1.0) -> 0.0 (MuJoCo closed)
                target_width = scale_gripper(gripper_action) # Gripper actuator(s)
                data.ctrl[7] = target_width

                # Use a larger macro physics window so the arm can move the distance
                physics_steps = 150
            else:
                print(f"⚠️ Server returned error status: {response.status_code}")
                physics_steps = 10
                
        except Exception as e:
            print(f"⚠️ Synchronous Pipeline Error: {e}")
            physics_steps = 10

        print(f"📡 MuJoCo data.ctrl: {np.array2string(data.ctrl, precision=4, suppress_small=True)}")

        # Step Physics Engine & Flush Graphics Pipeline
        for _ in range(physics_steps):
            mujoco.mj_step(model, data)
        viewer.sync()

        # Evaluate Completion Logic
        cube_height = data.xpos[cube_id][2]
        if cube_height - initial_cube_height > 0.04:
            print(f"📦 Final Cube Z: {cube_height:.4f} (Delta: {(cube_height - initial_cube_height):.4f}m)")
            print("🎉 GOAL ACCOMPLISHED: Red cube successfully picked up!")
            break

        # 7. Maintain visual loop stability
        elapsed = time.time() - step_start
        if elapsed < 0.16:
            print("sleeping...")
            time.sleep(0.16 - elapsed)

    viewer.close()
    print("👋 Exiting simulation bridge.")


if __name__ == "__main__":
    main()
