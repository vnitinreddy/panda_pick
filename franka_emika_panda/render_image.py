import mujoco
import cv2  # or matplotlib / PIL depending on how you save/show the image

# 1. Load model and data first
#model = mujoco.MjModel.from_xml_path("panda_site.xml")  # or your vla_task.xml path
model = mujoco.MjModel.from_xml_path("vla_task.xml")
data = mujoco.MjData(model)

# 2. Step physics once to populate spatial poses
mujoco.mj_step(model, data)

# 3. Initialize renderer
renderer = mujoco.Renderer(model, height=224, width=224)

# 4. Render and save
# (Replace 'front_camera' with your actual camera name in XML, or omit for free camera)
renderer.update_scene(data, camera="vla_camera")
rgb_img = renderer.render()

# Save image locally
cv2.imwrite("openvla_input.png", cv2.cvtColor(rgb_img, cv2.COLOR_RGB2BGR))
print("Saved openvla_input.png successfully!")

