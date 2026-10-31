import mujoco
import mujoco.viewer
import numpy as np

model = mujoco.MjModel.from_xml_path("vla_task.xml")
data = mujoco.MjData(model)

# The ID of the site we want to move (the gripper)

site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "pinch")

if site_id == -1:
    print("Error: Could not find site 'pinch'. Check your panda.xml!")
    exit()

# The ID of the target we want to reach (the cube)
cube_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "cube")

with mujoco.viewer.launch_passive(model, data) as viewer:
    while viewer.is_running():
        # 1. Get current position of gripper and cube
        current_pos = data.site_xpos[site_id]
        target_pos = data.xpos[cube_id]
        
        # 2. Calculate the error (how far away are we?)
        error = target_pos - current_pos
        
        # 3. Simple Control: apply force proportional to the error
        # This is a very basic way to 'pull' the arm toward the cube
        data.ctrl[:3] = error * 10.0  # Adjusting the first 3 joints
        
        mujoco.mj_step(model, data)
        viewer.sync()
