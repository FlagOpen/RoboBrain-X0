      
# from droid.droid.robot_env import RobotEnv
# from openpi_client import websocket_client_policy as _websocket_client_policy
import os
# import jax
import numpy as np
# from openpi_client import image_tools
import cv2
import requests
from PIL import Image
import base64
import time
import pyrealsense2 as rs
from pynput import keyboard



class Recorder:
    def __init__(self):

        self.connect_device = []
        for d in rs.context().devices:
            print('Found device: ', d.get_info(rs.camera_info.name), ' ', d.get_info(rs.camera_info.serial_number))
            self.connect_device.append(d.get_info(rs.camera_info.serial_number))
        assert len(self.connect_device) == 1
        
        self.pipeline = rs.pipeline()
        self.config = rs.config()
        self.config.enable_device(self.connect_device[0])
        self.config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
        self.config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
        self.align = rs.align(rs.stream.color)
        self.profile = self.pipeline.start(self.config)
        self.depth_scale = self.profile.get_device().first_depth_sensor().get_depth_scale()

    def record_frame(self):
        
        frames = self.pipeline.wait_for_frames()
        aligned_frames = self.align.process(frames)
        depth_frame = aligned_frames.get_depth_frame()
        color_frame = aligned_frames.get_color_frame()

        color_image = np.asanyarray(color_frame.get_data(), dtype=np.uint8)
        depth_image = np.asanyarray(depth_frame.get_data(), dtype=np.float32) * self.depth_scale * 1000
        
        return color_image, depth_image
    


class MultiRecorder:
    def __init__(self):
        self.align = rs.align(rs.stream.color)
        self.config = rs.config()
        self.config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
        self.config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
        self.connect_device = []
        for d in rs.context().devices:
            print('Found device: ', d.get_info(rs.camera_info.name), ' ', d.get_info(rs.camera_info.serial_number))
            # if d.get_info(rs.camera_info.serial_number) == "f1420629":
            #     continue
            # if d.get_info(rs.camera_info.serial_number) == "f1422292":
            #     continue
            self.connect_device.append(d.get_info(rs.camera_info.serial_number))
        print(self.connect_device)
        assert len(self.connect_device) == 2

        self.pipline_l515 = rs.pipeline()
        self.config.enable_device(self.connect_device[1])
        self.profile_l515 = self.pipline_l515.start(self.config)
        self.pipline_d435 = rs.pipeline()
        self.config.enable_device(self.connect_device[0])
        self.profile_d435 = self.pipline_d435.start(self.config)
        
        self.depth_scale_l515 = self.profile_l515.get_device().first_depth_sensor().get_depth_scale()
        self.depth_scale_d435 = self.profile_d435.get_device().first_depth_sensor().get_depth_scale()
        self.intr_l515 = None
        self.intr_d435 = None

    def record_frame(self):

        frames_front = self.pipline_l515.wait_for_frames()
        frames_wrist = self.pipline_d435.wait_for_frames()
        aligned_frames_front = self.align.process(frames_front)
        aligned_frames_wrist = self.align.process(frames_wrist)
        self.intr_l515 = aligned_frames_front.get_profile().as_video_stream_profile().get_intrinsics()
        self.intr_d435 = aligned_frames_wrist.get_profile().as_video_stream_profile().get_intrinsics()
    
        color_frame_front = aligned_frames_front.get_color_frame()
        color_frame_wrist = aligned_frames_wrist.get_color_frame()
        # depth_frame_front = aligned_frames_front.get_depth_frame()
        # depth_frame_wrist = aligned_frames_wrist.get_depth_frame()
        
        color_image_front = np.asanyarray(color_frame_front.get_data(), dtype=np.uint8)
        color_image_wrist = np.asanyarray(color_frame_wrist.get_data(), dtype=np.uint8)
        # depth_image_front = np.asanyarray(depth_frame_front.get_data(), dtype=np.float32) * self.depth_scale_l515 * 1000
        # depth_image_wrist = np.asanyarray(depth_frame_wrist.get_data(), dtype=np.float32) * self.depth_scale_d435 * 1000
        
        return color_image_front, color_image_wrist


def get_pose_quat():
    url = "http://127.0.0.2:5000/getpos"
    response = requests.post(url)
    cur_pose = response.json()['pose']
    
    return cur_pose

def get_pose_euler():
    url = "http://127.0.0.2:5000/getpos_euler"
    response = requests.post(url)
    cur_pose = response.json()['pose']
    
    return cur_pose

def get_joint():
    url = "http://127.0.0.2:5000/getq"
    response = requests.post(url)
    cur_joint = response.json()['q']
    
    return cur_joint    

def get_gripper():
    url = "http://127.0.0.2:5000/get_gripper"
    response = requests.post(url)
    cur_gripper = response.json()['gripper']
    gripper_open = 1 if cur_gripper > 0.7 else 0
    
    return gripper_open

def goto_pose(pose):
    url = "http://127.0.0.2:5000/pose"
    message = {
        "arr": pose
    }
    
    requests.post(url, json=message)
    
def goto_gripper(gripper):
    if gripper <= 0.3:
        message = {"gripper_pos": 30}
    else:
        message = {"gripper_pos": 270}
    
    url = "http://127.0.0.2:5000/move_gripper"
    requests.post(url, json=message)

def save_and_base64(image):
    # 保存图片
    # cv2.imwrite(save_path, image)
    # print(f"✅ 已保存到 {save_path}")

    # 转 base64
    success, buffer = cv2.imencode(".png", image)
    if not success:
        raise ValueError("❌ 图片编码失败")

    img_base64 = base64.b64encode(buffer).decode("utf-8")
    return img_base64

def get_pi0_input(obs, instruction):

    # wrist_image_base = save_and_base64(wrist_image, "cam_high.png")
    # left_image_base = save_and_base64(left_image, "cam_left_wrist.png")
    # right_image_base = save_and_base64(right_image, "cam_right_wrist.png")
    request_data = {
        "images": {
            "cam_wrist": obs["cam_wrist"],
            "cam_front": obs["cam_front"]
        },
        "eef_pose": obs["eef_pose"],
        "instruction": instruction,
    }
    return request_data



# def on_press(key):
#     try:
#         if hasattr(key, 'char') and key.char:
#             if key.char.lower() == 'c':
#                 print("关闭夹爪")
#                 goto_gripper(0)
#             elif key.char.lower() == 'o':
#                 print("打开夹爪")
#                 goto_gripper(1)
#     except AttributeError:
#         pass

# def on_release(key):
#     if key == keyboard.Key.esc:
#         print("退出程序")
#         return False


# listener = keyboard.Listener(on_press=on_press, on_release=on_release)
# listener.start()

def main():
    cameras = MultiRecorder()   
    # front_images = []
    # wrist_images = []
    
    
    robot_config = dict(
        max_timesteps=200
    )
    # instruction = "grasp the fruit that can keep doctor away"
    # instruction = "put the banana on the plate"
    # instruction = "pick up the peach. place the peach into the basket."
    instruction = "Pick up the orange into the plate."
   
    save_dir = "robobrain_data"
    os.makedirs(save_dir, exist_ok=True)
    goto_pose([0.3057233393192291, 0.0007003741338849068, 0.48132604360580444, 0.9999803304672241, 0.001191740040667355, 0.006135730072855949, 0.0005161708686500788])
    goto_gripper(1)
    time.sleep(5)
    
    close = False
    open = False
    
    
    while True:
        # for i in range(robot_config['max_timesteps']):
            # curr_obs = env.get_observation()
            # save  images
            # image_observations = curr_obs["image"]

        front_image, wrist_image = cameras.record_frame()
        
        # front_images.append(front_image.copy())
        # wrist_images.append(wrist_image.copy())
        
        pose = get_pose_quat()
        
        gripper = get_gripper()
        
        cur_robot_state = pose + [gripper]
        print("cur_robot_state:", cur_robot_state)
        
        time_str = int(time.time())
        
        front_image_path = os.path.join(save_dir, f"front_image.png")
        wrist_image_path = os.path.join(save_dir, f"wrist_image.png")
        # action_path = os.path.join(save_dir, f"action_{i}.npy")
        # state_path = os.path.join(save_dir, f"state_{i}.npy")
        print(front_image_path)
        cv2.imwrite(front_image_path, front_image)
        cv2.imwrite(wrist_image_path, wrist_image)
        # np.save(state_path, cur_robot_state)
        
        front_image = save_and_base64(front_image)
        wrist_image = save_and_base64(wrist_image)


        # obs = {
        #     "cam_front": front_image,
        #     "cam_wrist": wrist_image,
        #     "eef_pose": cur_robot_state,
        # }
        obs = {
            "cam_front": front_image,
            "cam_wrist": wrist_image,
            "eef_pose": cur_robot_state,
        }
        request_data = get_pi0_input(obs, instruction)
        # print(obs)
        # print(request_data)
        print("start infer")
        response = requests.post(
            # "http://172.16.16.239:5001/infer",
            "http://172.16.17.77:8000/infer",
            # "http://127.0.0.1:8000/infer",
            json=request_data
        )
        result=response.json()
        actions = result.get("eepose",[])
        # actions=np.array(actions)
        # print("actions:", actions)
        for i,act in enumerate(actions):
            # env.step(act)
            print("action:", act)
            goto_pose(act[:7]) 
            # goto_gripper(act[7])
            if  close==False and act[7] <= 0.7: 
                goto_gripper(0)
                close = True
                open = False
                print("close")
            
            elif open==False and act[7] > 0.7:
                goto_gripper(1)
                open = True
                close = False
                print("open")
            
            time.sleep(0.05)
                    

if __name__ == '__main__':
    main()

    