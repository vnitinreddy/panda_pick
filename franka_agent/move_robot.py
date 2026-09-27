import mujoco
import mujoco.viewer
import numpy as np
import time

# 1. Load the environment
model = mujoco.MjModel.from_xml_path("vla_task.xml")
data = mujoco.MjData(model)

# 2. Launch the interactive viewer
with mujoco.viewer.launch_passive(model, data) as viewer:
    # Give the system a second to settle
    time.sleep(1)
    
    print("Moving the robot arm...")

    # 3. The Control Loop
    for i in range(1000):
        # Apply a simple sine wave to the first joint (Swing back and forth)
        # data.ctrl[0] is the base rotation joint
        data.ctrl[0] = np.sin(time.time() * 2.0) * 0.5
        
        # Lift the arm slightly using the second joint
        data.ctrl[1] = -0.5 

        # Step the physics
        mujoco.mj_step(model, data)

        # Update the viewer
        viewer.sync()
        
        # Slow down the loop so we can see it (approx 60fps)
        time.sleep(0.01)

print("Movement test complete.")
