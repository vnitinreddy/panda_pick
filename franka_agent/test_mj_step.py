import os
import time
import mujoco
import mujoco.viewer

ARM_JOINT_IDS = list(range(7))      # Joint IDs 0..6

def run_simulation(xml_path):
    model = mujoco.MjModel.from_xml_path(xml_path)
    data = mujoco.MjData(model)

    mujoco.mj_resetData(model, data)

    with mujoco.viewer.launch_passive(model, data) as viewer:

        i = 0
        while viewer.is_running():

            data.ctrl[0] = 0.0 + i * 0.1
            data.ctrl[1] = -0.5
            data.ctrl[2] = 1.0
            print("ctrl: ", data.ctrl[:7])
            for _ in range(10):
                mujoco.mj_step(model, data)
            i = i + 1
            viewer.sync()
            time.sleep(3)

if __name__ == "__main__":
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    XML_FILE_PATH = os.path.join(SCRIPT_DIR, "vla_task.xml")
    run_simulation(XML_FILE_PATH)