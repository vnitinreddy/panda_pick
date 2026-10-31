import mujoco

model = mujoco.MjModel.from_xml_path("franka_emika_panda/scene.xml")

# Total control inputs expected by data.ctrl
print(f"Number of control inputs (nu): {model.nu}")

# Print actuator details
for i in range(model.nu):
    actuator_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
    ctrl_range = model.actuator_ctrlrange[i]
    print(f"Actuator {i} [{actuator_name}]: range = {ctrl_range}")
