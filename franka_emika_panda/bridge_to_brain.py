import os, sys, io, time, base64, requests, numpy as np, PIL.Image, mujoco, mujoco.viewer

# === CONFIG ===
SERVER_URL = "https://surface-shuffling-recede.ngrok-free.dev/predict"
HEADERS = {"ngrok-skip-browser-warning": "true"}
TASK_PROMPT = "Pick up the red cube."

# Control Constants
POSITION_GAIN = 10.0   # How fast the robot reacts to Cartesian deltas
EMA_ALPHA = 0.2        # Smoothing: lower = more fluid, higher = more responsive

# Initialize
model = mujoco.MjModel.from_xml_path("vla_task.xml")
data = mujoco.MjData(model)
renderer = mujoco.Renderer(model, height=224, width=224)

# FIX: Set PD gains for stiff, gravity-compensated movement
kp, kd = 3000.0, 2.0 * np.sqrt(3000.0)
for i in range(7):
    # Get the existing 10 parameters for this actuator
    params = model.actuator_biasprm[i].copy()
    
    # Update only the Gain (0), Kp (1), and Kd (2)
    params[0] = 0.0    # Gain
    params[1] = -kp    # Note: Menagerie models use negative Kp for spring stiffness
    params[2] = -kd    # Note: Menagerie models use negative Kd for damping
    
    # Assign the full 10-element array back
    model.actuator_biasprm[i] = params

def get_target_joint_positions(model, data, ee_delta):
    """Calculates target joint positions using Jacobian Pseudo-inverse."""
    # Get Jacobian for the 'pinch' site
    jacp = np.zeros((3, model.nv))
    jacr = np.zeros((3, model.nv))
    mujoco.mj_jacSite(model, data, jacp, jacr, mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "pinch"))
    
    # Use only the first 7 columns (arm joints)
    J = jacp[:, :7]
    
    # Pseudo-inverse for Differential IK: dq = J_pinv * dx
    # Damped least squares prevents instability near singularities
    J_pinv = J.T @ np.linalg.inv(J @ J.T + 0.01 * np.eye(3))
    
    # Calculate required joint change
    dq = J_pinv @ (ee_delta * POSITION_GAIN)
    return data.qpos[:7] + dq

def main():
    viewer = mujoco.viewer.launch_passive(model, data)
    smoothed_ctrl = data.qpos[:7].copy()
    target_marker_id = model.body("target_marker").id
    
    print("🚀 Running Differential IK Controller...")
    
    while viewer.is_running():
        # 1. Capture State
        renderer.update_scene(data, camera="vla_camera")
        img = PIL.Image.fromarray(renderer.render())
        buffered = io.BytesIO()
        img.save(buffered, format="JPEG")
        img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
        
        # 2. Get AI Action
        try:
            resp = requests.post(SERVER_URL, json={"image": img_str, "prompt": TASK_PROMPT}, headers=HEADERS, timeout=15)
            resp.raise_for_status() # This will catch 404s/500s immediately
            data_json = resp.json()
    
            # DEBUG: Inspect what the server is actually sending
            if "action" not in data_json:
                print(f"Full response: {data_json}")
                continue # Skip this step
                
            action = np.array(data_json["action"])

            action = np.array(resp.json()["action"]) # [dx, dy, dz, dr, dp, dy, gripper]
            
            # Update Visual Debugger
            target_pos = data.site('pinch').xpos + (action[:3] * 0.1)
            model.body_pos[target_marker_id] = target_pos
            mujoco.mj_forward(model, data)
            
            # 3. Calculate Targets via Differential IK
            target_qpos = get_target_joint_positions(model, data, action[:3])
            
            # 4. Smoothing (EMA Filter)
            smoothed_ctrl = (EMA_ALPHA * target_qpos) + ((1 - EMA_ALPHA) * smoothed_ctrl)
            
            # 5. Apply
            data.ctrl[:7] = smoothed_ctrl
            data.ctrl[7] = 0.04 * (1.0 - np.clip(action[6], 0, 1))
            
        except Exception as e:
            print(f"Controller Error: {e}")

        # 6. Step Physics
        for _ in range(20): 
            mujoco.mj_step(model, data)
        
        viewer.sync()
    
    viewer.close()

if __name__ == "__main__":
    main()