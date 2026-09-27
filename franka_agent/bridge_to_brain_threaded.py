import requests
import mujoco
import mujoco.viewer
import numpy as np
import PIL.Image
import io
import base64
import time
import threading
import sys

# === CONFIG ===
SERVER_URL = "https://surface-shuffling-recede.ngrok-free.dev/predict"
SERVER_GET_URL = "https://surface-shuffling-recede.ngrok-free.dev/"
ACTION_GAIN = 1.5
SCALE_MULTIPLIER = 15
SMOOTHING = 0.8
HEADERS = {"ngrok-skip-browser-warning": "true"}

# Define the explicit task instructions here!
TASK_PROMPT = "Pick up the red cube."

# === THREAD-SAFE SHARED STATE ===
pixels_lock = threading.Lock()
shared_pixels = None          # Main thread drops frames here
latest_action = np.zeros(7)   # Thread updates this; Main thread reads it

# Global MuJoCo variables
model = None
data = None
renderer = None
accomplished = False

action_lock = threading.Lock()
new_action_ready = False

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


def brain_thread():
    global shared_pixels, latest_action, new_action_ready
    print("🧠 Background Brain Thread: STARTED")
    
    # sleep to wait for viewer to load and run
    time.sleep(2)

    while True:
        img_to_upload = None
        
        # 1. Safely snatch the latest frame if the main thread provided one
        with pixels_lock:
            if shared_pixels is not None:
                img_to_upload = shared_pixels.copy() # .copy() avoids cross-thread memory corruption
                shared_pixels = None                # Consume it
        
        # 2. Process image and upload over the network in the background
        if img_to_upload is not None:
            try:
                # Encoding (Safe to perform inside a background thread)
                img = PIL.Image.fromarray(img_to_upload)
                buffered = io.BytesIO()
                img.save(buffered, format="JPEG")
                img_str = base64.b64encode(buffered.getvalue()).decode()
                
                print("📤 Posting image and task instruction to Cloud...")
                payload = {
                    "image": img_str,
                    "prompt": TASK_PROMPT  # <-- Passing the string over the network
		}
                response = requests.post(SERVER_URL, json=payload, headers=HEADERS, timeout=5)

                if response.status_code == 200:
                    new_act = np.array(response.json()["action"])
                    print(f"📡 Raw Action from Server: {np.array2string(new_act, precision=4, suppress_small=True)}")
		    # 🔍 DIAGNOSTIC CALIBRATION PRINT
                    print(f"🤖 Raw VLA Action Vector: {new_act}")
                    print(f"📊 Min/Max of predicted action: {new_act.min():.4f} / {new_act.max():.4f}")
                    with action_lock:
                        # Drop smoothing entirely if you want raw, definitive steps!
#latest_action *= (1 - SMOOTHING)
#latest_action += (new_act * SMOOTHING)
                        latest_action = new_act 
			#np.copyto(latest_action, new_act)
                        new_action_ready = True # <-- Set flag to alert main loop
                    # Apply smoothing formula to update global vector
                    #latest_action = (latest_action * (1 - SMOOTHING)) + (new_act * SMOOTHING)

                    print(f"📡 latest action: {np.array2string(latest_action, precision=4, suppress_small=True)}")
                else:
                    print(f"⚠️ Server returned error status: {response.status_code}")
 
            except Exception as e:
                print(f"⚠️ Brain Thread Error: {e}")
        
        # Prevent thread from resource-hogging your Mac when no frames are ready
        time.sleep(0.01)


def main():
    global model, data, renderer, shared_pixels, latest_action, accomplished, new_action_ready

    # Start network pipeline thread asynchronously
    t = threading.Thread(target=brain_thread, daemon=True)
    t.start()

    print("🚀 Starting Passive MuJoCo Viewer...")
    viewer = mujoco.viewer.launch_passive(model, data)
    
    cube_id = model.body("cube").id 
    initial_cube_height = data.xpos[cube_id][2] # Get current Z coordinate
    print(f"📦 Cube initialized on table at Z: {initial_cube_height:.4f}")

    while viewer.is_running():
        step_start = time.time()
        
        # 1. Render frame SAFELY on Main Thread (No OpenGL cross-threading)
        renderer.update_scene(data, camera="vla_camera")
        pixels = renderer.render()
        
        # 2. Hand off frame matrix to background thread
        with pixels_lock:
            shared_pixels = pixels

        # 3. Apply Control (Instantly uses the latest updated cloud vector)
        #data.ctrl[:7] = data.qpos[:7] + (latest_action[:7] * ACTION_GAIN)
        #data.ctrl[:7] = latest_action[:7]

	# 2. Check if a brand-new cloud action just arrived
        with action_lock:
            if new_action_ready == True:
                # Apply the target raw and clear!
                current_positions = np.array(data.ctrl[:7])
                # 2. Extract the raw delta from the server output
                vla_delta = np.array([-0.00289, 0.01252, -0.00361, 0.00049, 0.00524, 0.02572, 0.0])
                scaled_delta = vla_delta * SCALE_MULTIPLIER
                new_targets = current_positions + scaled_delta


#data.ctrl[:7] = latest_action[:7]
                data.ctrl[:7] = new_targets
                new_action_ready = False # Consume the flag
                print("🎬 Executing fresh server action step!")

                # OPTIONAL: Run MORE physics steps right now to force the arm
                # to distinctively jump/move to the target position instantly.
                physics_steps = 150
            else:
                # If no new data, let physics settle naturally without
                # slamming a static target control down over and over
                physics_steps = 10

        # 4. Step Physics Engine & Flush Graphics Pipeline
        for _ in range(physics_steps):
            mujoco.mj_step(model, data)
        viewer.sync()

        cube_height = data.xpos[cube_id][2]
        if cube_height - initial_cube_height > 0.04:
            print(f"📦 Final Cube Z: {cube_height:.4f} (Delta: {(cube_height - initial_cube_height):.4f}m)")
            print("🎉 GOAL ACCOMPLISHED: Red cube successfully picked up!")
            break # This breaks the MAIN loop, instantly dropping down to viewer.close()

        # 5. Cap local execution loop to ~60 FPS
        # 0.016 * 60 = 1 second
        elapsed = time.time() - step_start
        if elapsed < 0.16:
            time.sleep(0.16 - elapsed)

    viewer.close()
    print("👋 Exiting simulation bridge.")


if __name__ == "__main__":
    main()
