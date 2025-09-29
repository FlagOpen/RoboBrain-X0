import base64
import io
import time
import datetime
import os

import cv2
import numpy as np
import requests
from PIL import Image

# Import the robot-specific SDK/environment
from robot_env import RobotEnv

# --- 1. CONFIGURATION ---

# URL of the inference server
INFERENCE_SERVER_URL = "http://172.16.20.166:8000/infer"

# Robot control mode: 'eepose' for end-effector pose, 'joint' for joint angles
CONTROL_MODE = 'eepose'

# Directory to save periodic image logs
LOG_IMAGE_DIR = "./log_images_r1lite_ros1"

# Mapping from server-expected camera names to environment camera names
CAMERA_MAPPING = {
    "cam_head": "image_right",
    "cam_left_wrist": "image_left_wrist",
    "cam_right_wrist": "image_right_wrist",
}

# --- 2. HELPER FUNCTIONS ---

def encode_image(img: np.ndarray) -> str:
    """Encodes a numpy array (BGR) into a base64 PNG string."""
    # Convert BGR (from OpenCV) to RGB for PIL
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(img_rgb)
    buffer = io.BytesIO()
    pil_image.save(buffer, format='PNG')
    return base64.b64encode(buffer.getvalue()).decode('utf-8')

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
        print("Initializing robot environment (ROS1)...")
        env = RobotEnv()
        time.sleep(2) # Wait for environment to stabilize
        print("Initialization complete!")

        # --- B. MAIN CONTROL LOOP ---
        print("Entering main control loop...")
        while True:
            # --- i. Get Observations ---
            print("\n" + "="*50)
            print("1. Gathering robot state and images...")
            frames, state = env.update_obs_window()
            if state is None or not frames:
                print("Warning: No state or image data received. Skipping this cycle.")
                time.sleep(1)
                continue
            
            eef_pose_state = state.get('eef_pose')
            qpos_state = state.get('qpos')

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
                "eef_pose": eef_pose_state,
                "state": qpos_state, # Some models may use joint state as well
                "instruction": "put the orange into the basket.",
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

            for i, act in enumerate(np.array(actions)):
                action = np.array(act, dtype=np.float32)
                print(f"[Step {i+1}/{len(actions)}] Executing action: {np.round(action, 3)}")

                if CONTROL_MODE == 'eepose':
                    # Optional: Apply scaling or clipping to gripper commands
                    # action[7] *= 0.9  # right gripper
                    # action[15] *= 0.9 # left gripper
                    env.control_eef(action)
                elif CONTROL_MODE == 'joint':
                    env.control(action)
                    
                time.sleep(0.05) # Short pause between actions
            
            print("Action sequence execution complete.")

    except KeyboardInterrupt:
        print("\nProgram interrupted by user.")
    except Exception as e:
        print(f"\nAn unexpected error occurred: {e}")
    finally:
        # --- C. SHUTDOWN ---
        # The original script did not have a shutdown call, but if your RobotEnv has one, call it here.
        # if env:
        #     env.shutdown()
        print("Program finished.")

if __name__ == "__main__":
    main()