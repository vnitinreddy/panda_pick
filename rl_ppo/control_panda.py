"""
Interactive Cartesian (XYZ) IK Control for Franka Emika Panda.
Operating Mode: Position Control (Event-Driven IK)

This script launches MuJoCo viewer and allows the users to move
the arm through key controls. It allows to increment or decrement
x or y or z coordinate in steps with a step_size of 0.02. It then
calculates the inverse kinematics needed to move the arm to the new
position. Note that other coordinates also will change if needed to
step increment/decrement the given coordinate. The gripper open/close
can also be controlled through a key.

This script allows understanding the hand controls and forming an intution
on how to move the hand!

Run the script:
mjpython scripts/control_panda.py

While in viewer window, press the buttons for the required actions.
"""

import mujoco
import mujoco.viewer
import numpy as np
import time

# 1. Load Model & Data
model = mujoco.MjModel.from_xml_path("franka_emika_panda/scene.xml")
data = mujoco.MjData(model)

# Prevent the heavy arm from sagging under its own weight
for i in range(model.nbody):
    model.body_gravcomp[i] = 1.0

# 2. Identify End-Effector Reference
'''
The use_body flag makes your controller robust against different XML model layouts. 
It allows the script to function seamlessly regardless of whether the Panda robot's 
model marks its end-effector with an explicit <site> tag or relies directly on a 
link <body>.
'''
SITE_NAME = "attachment_site"
site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, SITE_NAME)

use_body = False
if site_id == -1:
    body_name = "hand" if mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "hand") != -1 else "link7"
    site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)
    use_body = True

n_arm_joints = min(7, model.nu)
ARM_ACTUATOR_IDS = list(range(n_arm_joints))

# 3. Controller Parameters
step_size = 0.02          # meters per key stroke (2 cm)
damping = 1e-3            # Damped Least-Squares stability factor
gripper_open = True

# GLFW key codes
KEY_W, KEY_S = 87, 83
KEY_A, KEY_D = 65, 68
KEY_R, KEY_F = 82, 70
KEY_G = 71
KEY_SPACE = 32
KEY_Q = 81

# MuJoCo gripper - TBD: Get from MuJoCo model?
GRIPPER = 7

mujoco.mj_forward(model, data)

def display_control_and_status():
    if use_body:
        ee_pos = data.xpos[site_id]
    else:
        ee_pos = data.site_xpos[site_id]
        
    grip = "OPEN" if gripper_open else "CLOSED"

    print("\033[2J\033[H")  # Clear terminal
    print("╔══════════════════════════════════════════╗")
    print("║   Panda Position IK (XYZ) Controller     ║")
    print("╠══════════════════════════════════════════╣")
    print("║  W/S: Move ±X   A/D: Move ±Y             ║")
    print("║  R/F: Move ±Z   G:   Toggle Gripper      ║")
    print("║  SPACE: Reset   Q:   Quit                ║")
    print("╠══════════════════════════════════════════╣")
    print(f"║ EE Pos (XYZ): [{ee_pos[0]:.3f}, {ee_pos[1]:.3f}, {ee_pos[2]:.3f}]    ║")
    print(f"║ Gripper:      {grip:>26s} ║")
    print("╚══════════════════════════════════════════╝")


def key_handler(key_pressed):
    global gripper_open

    delta_xyz = np.zeros(3)

    # Map Keys to Cartesian Movement
    if key_pressed == KEY_W:
        delta_xyz[0] = step_size
    elif key_pressed == KEY_S:
        delta_xyz[0] = -step_size
    elif key_pressed == KEY_A:
        delta_xyz[1] = step_size
    elif key_pressed == KEY_D:
        delta_xyz[1] = -step_size
    elif key_pressed == KEY_R:
        delta_xyz[2] = step_size
    elif key_pressed == KEY_F:
        delta_xyz[2] = -step_size

    # Solve IK ONLY when a key is pressed
    if np.any(delta_xyz):
        # 1. Get current Jacobian
        jac_p = np.zeros((3, model.nv))
        if use_body:
            mujoco.mj_jacBody(model, data, jac_p, None, site_id)
        else:
            mujoco.mj_jacSite(model, data, jac_p, None, site_id)

        J = jac_p[:, :n_arm_joints]

        # 2. Compute Damped Pseudo-Inverse
        inv_matrix = np.linalg.inv(J @ J.T + damping * np.eye(3))
        J_pinv = J.T @ inv_matrix

        # 3. Map cartesian step directly to joint angle steps
        delta_q = J_pinv @ delta_xyz

        # 4. Add the joint step to the position controller targets
        for i, act_id in enumerate(ARM_ACTUATOR_IDS):
            lo, hi = model.actuator_ctrlrange[act_id]
            # Add delta_q to the current target, constrained by physical limits
            data.ctrl[act_id] = np.clip(data.ctrl[act_id] + delta_q[i], lo, hi)
            
        display_control_and_status()

    # Toggle Gripper
    elif key_pressed == KEY_G:
        gripper_open = not gripper_open
        if model.nu > GRIPPER:
            lo, hi = model.actuator_ctrlrange[GRIPPER]
            data.ctrl[GRIPPER] = hi if gripper_open else lo
        display_control_and_status()

    # Reset
    elif key_pressed == KEY_SPACE:
        mujoco.mj_resetData(model, data)
        mujoco.mj_forward(model, data)
        for i in ARM_ACTUATOR_IDS:
            data.ctrl[i] = data.qpos[i]
        gripper_open = True
        display_control_and_status()

    # --- 2. Add Quit Handling ---
    elif key_pressed == KEY_Q:
        if viewer is not None and viewer.is_running():
            viewer.close()

# Initialize controller positions to match arm's starting qpos
for i in ARM_ACTUATOR_IDS:
    data.ctrl[i] = data.qpos[i]

if model.nu > 7:
    _, hi = model.actuator_ctrlrange[7]
    data.ctrl[7] = hi

# 4. Main Simulation Loop 
with mujoco.viewer.launch_passive(model, data, key_callback=key_handler) as viewer:
    display_control_and_status()
    
    while viewer.is_running():
        step_start = time.time()

        # The loop is now completely quiet. Physics steps forward smoothly
        # toward whatever targets were set by the last key press.
        mujoco.mj_step(model, data)
        viewer.sync()

        time_until_next_step = model.opt.timestep - (time.time() - step_start)
        if time_until_next_step > 0:
            time.sleep(time_until_next_step)

