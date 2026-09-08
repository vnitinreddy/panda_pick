import mujoco
import numpy as np
from PIL import Image

# Load the model you just verified
model = mujoco.MjModel.from_xml_path("vla_task.xml")
data = mujoco.MjData(model)
renderer = mujoco.Renderer(model, height=224, width=224)

# Step physics to settle the cube
# mujoco.mj_step(model, data)
# Run 100 steps of physics so the cube falls onto the table properly
for _ in range(100):
    mujoco.mj_step(model, data)

# Render the view from the VLA camera
renderer.update_scene(data, camera="vla_camera")
pixels = renderer.render()

# Save the image to check the quality
img = Image.fromarray(pixels)
img.save("vla_observation.png")
print("Observation saved as vla_observation.png")
