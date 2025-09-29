import base64
import io
import time
import numpy as np
import requests
import cv2
import datetime
from robot_env import RobotEnv
import ipdb
from PIL import Image

CONTROL_MODE = 'eepose'  # eepose
def encode_image(img: np.ndarray) -> str:
    """Encode OpenCV image as base64 PNG string."""
    _, buffer = cv2.imencode('.png', img)
    return base64.b64encode(buffer).decode('utf-8')

env = RobotEnv(
)

def encode_image_pil_simple(img: np.ndarray) -> str:
    """Encode numpy array as base64 PNG string using PIL."""
    # 转换为PIL图像（假设是RGB格式）
    pil_image = Image.fromarray(img.astype(np.uint8))
    
    # 保存到内存缓冲区并编码
    buffer = io.BytesIO()
    pil_image.save(buffer, format='PNG')
    return base64.b64encode(buffer.getvalue()).decode('utf-8')

def encode_image_from_frame_pil_simple(frames, cam_name):
    """简化版本的图像编码函数"""
    img = frames.get(cam_name)
    
    if img is not None and img.size > 0:
        try:
            return encode_image_pil_simple(img)
        except Exception as e:
            print(f"编码错误 ({cam_name}): {e}")
            return None
    else:
        print(f"警告：{cam_name} 图像为空")
        return None

try:
    while True:
        # cmd = input("\n按下 'c' 继续一次推理和控制，或按 Ctrl+C 退出：")
        # if cmd.strip().lower() != 'c':
        #     print("[!] 非法输入，输入 'c' 开始下一步。")
        #     continue

        frames, state = env.update_obs_window()

        # ipdb.set_trace()
        for name, img in frames.items():
            save_path = f"./latest_{name}.png"
            cv2.imwrite(save_path, img)
            print(f'[Saved] {save_path}')
        
        encoded_images = {
        # 'image_left': encode_image_from_frame(frames, "image_left"),
        'cam_head': encode_image_from_frame_pil_simple(frames, "image_right"),
        'cam_left_wrist': encode_image_from_frame_pil_simple(frames, "image_left_wrist"),
        'cam_right_wrist': encode_image_from_frame_pil_simple(frames, "image_right_wrist")
        }
        data = {
            "state": state['qpos'],           # shape: [1, 14]
            "eef_pose": state['eef_pose'],  # shape: [1, 14]
            # "instruction": "pick up the orange",
            "images": encoded_images,
            "instruction": 'put the orange into the basket.'
        }

        # response = requests.post("http://172.16.16.33:8003/infer", json=data, timeout=60)

        response = requests.post("http://172.16.20.166:8000/infer", json=data, timeout=60)
        print("[√] Response:", response.status_code)
        # print(response)
        result = response.json()
        
        print("[Response JSON]:", result)
        # ipdb.set_trace()
        
        if CONTROL_MODE == 'eepose':
            actions = result.get("eepose", [])
            if not actions:
                print("[!] 未返回动作，跳过控制")
                continue
            actions = np.array(actions)[:]

            # 获取完整动作序列并依次执行
            for i, act in enumerate(actions):
                action = np.array(act, dtype=np.float32)

                print(f"left gripper: {action[15]}, right gripper: {action[7]}")

                action[7] *= 0.9
                action[15] *= 0.9

                if action[7] < 70:
                    action[7] = 0
                    
                if action[15] < 70:
                    action[15] = 0


                # action[7]=70
                # action[15]=70


                print(f"[→ Step {i+1}] 执行动作: {action.round(3)}")
                env.control_eef(action)
                # env.control(action)
                time.sleep(0.01)  # 可根据实际需要调整间隔时间
                
        elif CONTROL_MODE == 'joint':
            actions = result.get("qpos", [])
            
            if not actions:
                print("[!] 未返回动作，跳过控制")
                continue
            actions = np.array(actions)[:]

            # 获取完整动作序列并依次执行
            for i, act in enumerate(actions):
                action = np.array(act, dtype=np.float32)
                print(f"[→ Step {i+1}] 执行动作: {action.round(3)}")
                env.control(action)
                time.sleep(0.05)  # 可根据实际需要调整间隔时间

except KeyboardInterrupt:
    print("\n[Main] Interrupted by user.")
finally:
    # env.shutdown()
    print("[Main] RobotEnv shut down.")
