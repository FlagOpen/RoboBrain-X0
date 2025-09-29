
import base64
import os
import time
import datetime
from typing import Tuple, Optional
 
import cv2
import numpy as np
import requests

# Import the robot-specific SDK
from a2d_sdk.robot import RobotDds, RobotController, CosineCamera

# --- 1. CONFIGURATION ---

# URL of the inference server
INFERENCE_SERVER_URL = "http://172.16.17.77:8000/infer"

# Robot control mode. 'eepose' for end-effector pose control.
CONTROL_MODE = 'eepose'

# Directory to save periodic image logs
LOG_IMAGE_DIR = "./log_images_a2d"

# Mapping from server-expected camera names to SDK camera names
CAMERA_MAPPING = {
    "cam_head": "head",
    "cam_left_wrist": "hand_left",
    "cam_right_wrist": "hand_right",
}

# --- 2. HELPER FUNCTIONS ---

def encode_image(img: np.ndarray) -> str:
    """Encodes an OpenCV image into a base64 PNG string."""
    _, buffer = cv2.imencode('.png', img)
    return base64.b64encode(buffer).decode('utf-8')

def get_and_encode_image(camera: CosineCamera, cam_sdk_name: str) -> Tuple[Optional[np.ndarray], Optional[str]]:
    """Captures an image from a specified camera and returns the raw image and its base64 encoded string."""
    try:
        img, _ = camera.get_latest_image(cam_sdk_name)
        if img is not None and img.size > 0:
            encoded_str = encode_image(img)
            return img, encoded_str
        print(f"Warning: Failed to get image from {cam_sdk_name}, or image is empty.")
        return None, None
    except Exception as e:
        print(f"Warning: Exception while capturing image from {cam_sdk_name}: {e}")
        return None, None

def setup_logging_directory():
    """Creates or clears the logging directory."""
    if not os.path.exists(LOG_IMAGE_DIR):
        os.makedirs(LOG_IMAGE_DIR)
        print(f"Created log directory: {LOG_IMAGE_DIR}")
    else:
        # Clear existing logs
        print(f"Clearing log directory: {LOG_IMAGE_DIR}")
        for filename in os.listdir(LOG_IMAGE_DIR):
            try:
                os.remove(os.path.join(LOG_IMAGE_DIR, filename))
            except Exception as e:
                print(f"Warning: Could not remove file {filename}: {e}")

# --- 3. MAIN EXECUTION ---

def main():
    """Main function to connect to the robot, run the control loop, and handle shutdown."""
    setup_logging_directory()

    robot_dds = None
    camera = None

    try:
        # --- A. INITIALIZATION ---
        print("Initializing robot system...")
        robot_dds = RobotDds()
        robot_controller = RobotController()

        camera_sdk_names = list(CAMERA_MAPPING.values())
        print(f"Initializing cameras: {camera_sdk_names}")
        camera = CosineCamera(camera_sdk_names)
        
        # Define initial arm joint positions for reset
        arm_initial_joint_position = [
            -1.075, 0.6108, 0.279, -1.284, 0.731, 1.495, -0.188,
             1.075, -0.6108, -0.279, 1.284, -0.731, -1.495, 0.188
        ]
        robot_dds.reset(arm_positions=arm_initial_joint_position)
        print("System initialization complete!")

        # --- B. MAIN CONTROL LOOP ---
        print("Entering main control loop...")
        while True:
            time.sleep(1) # Loop frequency
            print("\n" + "="*50)

            # --- i. Get Observations ---
            print("1. Gathering robot state and images...")
            try:
                motion_status = robot_controller.get_motion_status()
                left_cartesian = motion_status["frames"]["arm_left_link7"]
                right_cartesian = motion_status["frames"]["arm_right_link7"]
                gripper_states_raw, _ = robot_dds.gripper_states()
                
                # Combine state into a single list [right_arm_pose, right_gripper, left_arm_pose, left_gripper]
                eef_pose_state = [
                    right_cartesian["position"]["x"], right_cartesian["position"]["y"], right_cartesian["position"]["z"],
                    right_cartesian["orientation"]["quaternion"]["x"], right_cartesian["orientation"]["quaternion"]["y"],
                    right_cartesian["orientation"]["quaternion"]["z"], right_cartesian["orientation"]["quaternion"]["w"],
                    gripper_states_raw[1], # Right gripper
                    left_cartesian["position"]["x"], left_cartesian["position"]["y"], left_cartesian["position"]["z"],
                    left_cartesian["orientation"]["quaternion"]["x"], left_cartesian["orientation"]["quaternion"]["y"],
                    left_cartesian["orientation"]["quaternion"]["z"], left_cartesian["orientation"]["quaternion"]["w"],
                    gripper_states_raw[0], # Left gripper
                ]
            except (KeyError, IndexError) as e:
                print(f"Error getting robot state: {e}. Skipping this cycle.")
                continue

            # --- ii. Prepare Data for Server ---
            print("2. Preparing data for inference server...")
            encoded_images = {}
            timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            for server_name, sdk_name in CAMERA_MAPPING.items():
                raw_img, encoded_img = get_and_encode_image(camera, sdk_name)
                encoded_images[server_name] = encoded_img
                if raw_img is not None:
                    log_path = os.path.join(LOG_IMAGE_DIR, f"{server_name}_{timestamp}.png")
                    cv2.imwrite(log_path, raw_img)
            
            request_data = {
                "eef_pose": eef_pose_state,
                "instruction": "Put all the fruits into the basket.",
                "images": encoded_images
            }

            # --- iii. Send Request to Server ---
            print(f"3. Sending request to {INFERENCE_SERVER_URL}...")
            try:
                response = requests.post(INFERENCE_SERVER_URL, json=request_data, timeout=100)
                response.raise_for_status()
                result = response.json()
                print("...Success! Received response from server.")
            except requests.exceptions.RequestException as e:
                print(f"Error communicating with server: {e}. Retrying after 5s.")
                time.sleep(5)
                continue

            # --- iv. Parse and Execute Actions ---
            print("4. Parsing and executing actions...")
            if CONTROL_MODE == 'eepose':
                actions = result.get("eepose", [])
                if not actions:
                    print("No actions received from the model. Skipping.")
                    continue
                
                for i, act in enumerate(np.array(actions)[:20]): # Execute up to 20 steps
                    if act.shape[0] != 16:
                        print(f"Warning: Action dimension is {act.shape[0]}, expected 16. Skipping.")
                        continue
                    
                    # Deconstruct action into robot commands
                    right_pose_array, left_pose_array = act[0:8], act[8:16]
                    right_pose_dict = {"x": right_pose_array[0].item(), "y": right_pose_array[1].item(), "z": right_pose_array[2].item(), "qx": right_pose_array[3].item(), "qy": right_pose_array[4].item(), "qz": right_pose_array[5].item(), "qw": right_pose_array[6].item()}
                    left_pose_dict = {"x": left_pose_array[0].item(), "y": left_pose_array[1].item(), "z": left_pose_array[2].item(), "qx": left_pose_array[3].item(), "qy": left_pose_array[4].item(), "qz": left_pose_array[5].item(), "qw": left_pose_array[6].item()}
                    gripper_states = [left_pose_array[7].item(), right_pose_array[7].item()]

                    print(f"[Step {i+1}/{len(actions)}] Executing action...")
                    robot_controller.set_end_effector_pose_control(
                        lifetime=1.0,
                        control_group=["dual_arm"],
                        left_pose=left_pose_dict,
                        right_pose=right_pose_dict,
                    )
                    robot_dds.move_gripper(gripper_states)
                    time.sleep(0.1) # Short pause between actions
                
                print("Action sequence execution complete.")
            else:
                print(f"Error: Unsupported control mode '{CONTROL_MODE}'")
                break

    except KeyboardInterrupt:
        print("\nProgram interrupted by user.")
    except Exception as e:
        print(f"\nAn unexpected error occurred: {e}")
    finally:
        # --- C. SHUTDOWN ---
        print("Shutting down...")
        if robot_dds:
            print("Resetting robot to a safe position.")
            robot_dds.reset()
            time.sleep(2)
            robot_dds.shutdown()
        if camera:
            camera.close()
        print("Shutdown complete. Exiting program.")


if __name__ == "__main__":
    main()