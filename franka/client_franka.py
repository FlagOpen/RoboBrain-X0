import base64
import os
import time
import datetime

import cv2
import numpy as np
import pyrealsense2 as rs
import requests

# --- 1. CONFIGURATION ---

# URL of the inference server
INFERENCE_SERVER_URL = "http://172.16.17.77:8000/infer"

# URL of the local Franka ROS control server
FRANKA_CONTROL_SERVER_URL = "http://127.0.0.2:5000"

# Robot control mode (must match what the model outputs)
CONTROL_MODE = 'eepose'

# Directory to save periodic image logs
LOG_IMAGE_DIR = "./log_images_franka"

# --- 2. HARDWARE & HELPER CLASSES/FUNCTIONS ---

class MultiCameraRecorder:
    """A class to manage multiple RealSense cameras."""
    def __init__(self, serials):
        self.serials = serials
        self.pipelines = []
        self.profiles = []
        self.align = rs.align(rs.stream.color)

        for serial in self.serials:
            pipeline = rs.pipeline()
            config = rs.config()
            config.enable_device(serial)
            config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
            profile = pipeline.start(config)
            self.pipelines.append(pipeline)
            self.profiles.append(profile)
            print(f"Started camera with serial: {serial}")

    def record_frames(self) -> dict:
        """Capture frames from all connected cameras."""
        frames_dict = {}
        for i, pipeline in enumerate(self.pipelines):
            try:
                frames = pipeline.wait_for_frames()
                aligned_frames = self.align.process(frames)
                color_frame = aligned_frames.get_color_frame()
                if color_frame:
                    frames_dict[f'cam_{i}'] = np.asanyarray(color_frame.get_data(), dtype=np.uint8)
            except RuntimeError as e:
                print(f"Error capturing frame from camera {i}: {e}")
                frames_dict[f'cam_{i}'] = None
        return frames_dict

    def stop(self):
        """Stops all camera pipelines."""
        for pipeline in self.pipelines:
            pipeline.stop()

def get_franka_state():
    """Retrieves end-effector pose (quaternion) and gripper state from the control server."""
    pose_res = requests.post(f"{FRANKA_CONTROL_SERVER_URL}/getpos")
    pose = pose_res.json()['pose'] # [x, y, z, qx, qy, qz, qw]
    
    gripper_res = requests.post(f"{FRANKA_CONTROL_SERVER_URL}/get_gripper")
    gripper_width = gripper_res.json()['gripper']
    gripper_state = 1 if gripper_width > 0.7 else 0 # Normalize to 1 (open) or 0 (closed)
    
    return pose + [gripper_state]

def control_franka_pose(pose, gripper_state):
    """Sends pose and gripper commands to the control server."""
    # Send pose command
    requests.post(f"{FRANKA_CONTROL_SERVER_URL}/pose", json={"arr": pose})
    
    # Send gripper command
    gripper_pos = 270 if gripper_state > 0.5 else 30
    requests.post(f"{FRANKA_CONTROL_SERVER_URL}/move_gripper", json={"gripper_pos": gripper_pos})

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
    cameras = None

    try:
        # --- A. INITIALIZATION ---
        print("Initializing cameras...")
        # Find connected RealSense devices
        connected_devices = [d.get_info(rs.camera_info.serial_number) for d in rs.context().devices]
        if len(connected_devices) < 2:
            raise RuntimeError("Expected at least 2 RealSense cameras, but found fewer.")
        # Assign cameras based on a fixed order (e.g., by serial number)
        # IMPORTANT: You may need to adjust this order based on your setup.
        # Let's assume the first is 'front' and the second is 'wrist'.
        camera_serials = sorted(connected_devices) 
        cameras = MultiCameraRecorder(camera_serials)
        print(f"Initialized cameras: Front={camera_serials[0]}, Wrist={camera_serials[1]}")
        
        print("Moving robot to initial pose...")
        initial_pose = [0.305, 0.0, 0.481, 1.0, 0.0, 0.0, 0.0]
        control_franka_pose(initial_pose, 1) # Pose and open gripper
        time.sleep(5)
        print("Initialization complete!")

        # --- B. MAIN CONTROL LOOP ---
        print("Entering main control loop...")
        while True:
            # --- i. Get Observations ---
            print("\n" + "="*50)
            print("1. Gathering robot state and images...")
            
            # Get robot state
            eef_pose_state = get_franka_state()
            print(f"Current robot state: {np.round(eef_pose_state, 3)}")

            # Get images
            frames = cameras.record_frames()
            front_image = frames.get('cam_0')
            wrist_image = frames.get('cam_1')

            if front_image is None or wrist_image is None:
                print("Warning: Failed to capture one or more images. Skipping cycle.")
                time.sleep(1)
                continue

            # --- ii. Prepare Data for Server ---
            print("2. Preparing data for inference server...")
            encoded_front = encode_image(front_image)
            encoded_wrist = encode_image(wrist_image)

            # Log current images
            timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            cv2.imwrite(os.path.join(LOG_IMAGE_DIR, f"front_{timestamp}.png"), front_image)
            cv2.imwrite(os.path.join(LOG_IMAGE_DIR, f"wrist_{timestamp}.png"), wrist_image)

            request_data = {
                "eef_pose": eef_pose_state,
                "instruction": "Pick up the orange and place it into the plate.",
                "images": {
                    "cam_front": encoded_front,
                    "cam_wrist": encoded_wrist,
                }
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
            if CONTROL_MODE == 'eepose':
                actions = result.get("eepose", [])
                if not actions:
                    print("No actions received from the model. Skipping.")
                    continue

                for i, act in enumerate(actions):
                    action = np.array(act, dtype=np.float32)
                    if action.shape[0] != 8:
                        print(f"Warning: Action dimension is {action.shape[0]}, expected 8. Skipping.")
                        continue
                    
                    target_pose = action[:7]
                    target_gripper = action[7]
                    print(f"[Step {i+1}/{len(actions)}] Executing action...")
                    control_franka_pose(target_pose.tolist(), target_gripper.item())
                    time.sleep(0.05) # Short pause between actions

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
        if cameras:
            print("Stopping cameras.")
            cameras.stop()
        print("Shutdown complete. Exiting program.")

if __name__ == '__main__':
    main()