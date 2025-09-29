import base64
import os
import time
import datetime

import cv2
import numpy as np
import requests

# Import the robot-specific SDK/environment
from robot_env import RobotEnv

# --- 1. CONFIGURATION ---

# URL of the inference server
INFERENCE_SERVER_URL = "http://172.16.20.113:5001/infer"

# Robot control mode: 'eepose' for end-effector pose, 'joint' for joint angles
CONTROL_MODE = 'eepose'

# Directory to save periodic image logs
LOG_IMAGE_DIR = "./log_images_agilex"

# Mapping from server-expected camera names to environment camera names
CAMERA_MAPPING = {
    "cam_high": "realsense_0",
    "cam_left_wrist": "realsense_1",
    "cam_right_wrist": "realsense_2",
}

# --- 2. HELPER FUNCTIONS ---

def encode_image(img: np.ndarray) -> str:
    """Encodes an OpenCV image into a base64 PNG string."""
    _, buffer = cv2.imencode('.png', img)
    return base64.b64encode(buffer).decode('utf-8')

def setup_logging_directory():
    """Creates the logging directory if it doesn't exist."""
    if not os.path.exists(LOG_IMAGE_DIR):
        os.makedirs(LOG_IMAGE_DIR)
        print(f"Created log directory: {LOG_IMAGE_DIR}")

# --- 3. MAIN EXECUTION ---

def main():
    """Main function to connect to the robot, run the control loop, and handle shutdown."""
    setup_logging_directory()
    env = None
    
    try:
        # --- A. INITIALIZATION ---
        print("Initializing robot environment...")
        env = RobotEnv(
            realsense_serials=[0, 1, 2],
            arm_ip="can0+can1"
        )
        # Optional: Move robot to an initial position
        # initial_joint_command = [...]
        # env.control(initial_joint_command)
        time.sleep(2) # Wait for environment to stabilize
        print("Initialization complete!")

        # --- B. MAIN CONTROL LOOP ---
        print("Entering main control loop...")
        while True:
            time.sleep(1) # Loop frequency
            print("\n" + "="*50)

            # --- i. Get Observations ---
            print("1. Gathering robot state and images...")
            frames, state = env.update_obs_window()
            if state is None or not frames:
                print("Warning: No state or image data received. Skipping this cycle.")
                time.sleep(1)
                continue
            
            eef_pose_state = state["eef_pose"] # Right arm + Left arm

            # --- ii. Prepare Data for Server ---
            print("2. Preparing data for inference server...")
            encoded_images = {}
            timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            for server_name, env_name in CAMERA_MAPPING.items():
                img = frames.get(env_name)
                if img is not None and img.size > 0:
                    encoded_images[server_name] = encode_image(img)
                    log_path = os.path.join(LOG_IMAGE_DIR, f"{server_name}_{timestamp}.png")
                    cv2.imwrite(log_path, img)
                else:
                    print(f"Warning: Image for '{server_name}' is not available.")
            
            request_data = {
                "eef_pose": [eef_pose_state.tolist()],
                "instruction": "Pick up the pencil sharpener and place it to the left of the stapler.",
                "images": encoded_images
            }

            # --- iii. Send Request to Server ---
            print(f"3. Sending request to {INFERENCE_SERVER_URL}...")
            try:
                response = requests.post(INFERENCE_SERVER_URL, json=request_data, timeout=60)
                response.raise_for_status()
                result = response.json()
                print("...Success! Received response from server.")
            except requests.exceptions.RequestException as e:
                print(f"Error communicating with server: {e}. Retrying after 5s.")
                time.sleep(5)
                continue

            # --- iv. Parse and Execute Actions ---
            print("4. Parsing and executing actions...")
            actions = result.get(CONTROL_MODE, [])
            if not actions:
                print("No actions received from the model. Skipping.")
                continue

            # Execute a fixed number of steps from the returned plan
            for i, act in enumerate(np.array(actions)[:30]):
                action = np.array(act, dtype=np.float32)
                print(f"[Step {i+1}/{len(actions)}] Executing action: {np.round(action, 3)}")
                if CONTROL_MODE == 'eepose':
                    env.control_eef(action)
                elif CONTROL_MODE == 'joint':
                    # SDK might expect a different joint order (e.g., left then right)
                    if action.shape[0] == 14:
                        action = np.concatenate([action[7:], action[:7]]) # Example: Swap right and left arm joints
                    env.control(action)
                time.sleep(0.05) # Short pause between actions
            
            print("Action sequence execution complete.")

    except KeyboardInterrupt:
        print("\nProgram interrupted by user.")
    except Exception as e:
        print(f"\nAn unexpected error occurred: {e}")
    finally:
        # --- C. SHUTDOWN ---
        if env:
            print("Shutting down robot environment.")
            env.shutdown()
        print("Shutdown complete. Exiting program.")

if __name__ == "__main__":
    main()