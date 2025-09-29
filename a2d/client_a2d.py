import base64
import io
import time
import os
import numpy as np
import requests
import cv2
import datetime
from typing import Tuple, Union, Optional

# 从新机器人SDK导入
from a2d_sdk.robot import RobotDds, RobotController, CosineCamera

# --- 全局配置 ---
CONTROL_MODE = 'eepose'  
# INFERENCE_SERVER_URL = "http://172.16.16.77:5002/infer"
# INFERENCE_SERVER_URL = "http://172.16.16.33:8000/infer" 
INFERENCE_SERVER_URL = "http://172.16.17.77:8000/infer" 
LOG_IMAGE_DIR = "./Log_New"  # 保存周期性图像日志的文件夹

CAMERA_MAPPING = {
    "cam_head": "head",
    # "cam_high_fisheye": "head_center_fisheye",
    "cam_left_wrist": "hand_left",
    "cam_right_wrist": "hand_right",
}

def encode_image(img: np.ndarray) -> str:
    """将 OpenCV 图像编码为 base64 PNG 字符串。"""
    _, buffer = cv2.imencode('.png', img)
    return base64.b64encode(buffer).decode('utf-8')
def clear_log_directory():
    """清空 LOG_IMAGE_DIR 文件夹中的所有文件"""
    if os.path.exists(LOG_IMAGE_DIR):
        for filename in os.listdir(LOG_IMAGE_DIR):
            file_path = os.path.join(LOG_IMAGE_DIR, filename)
            try:
                if os.path.isfile(file_path):
                    os.remove(file_path)
                    print(f"已删除文件: {file_path}")
            except Exception as e:
                print(f"警告：无法删除文件 {file_path}: {e}")
    else:
        os.makedirs(LOG_IMAGE_DIR)
        print(f"📂 已创建文件夹: {LOG_IMAGE_DIR}")

def get_and_encode_image(camera: CosineCamera, cam_sdk_name: str) -> Tuple[np.ndarray, Optional[str]]:
    """
    从指定的摄像头获取图像，返回原始图像和编码后的字符串。
    """
    try:
        img, _ = camera.get_latest_image(cam_sdk_name)
        if img is not None and img.size > 0:
            encoded_str = encode_image(img)
            return img, encoded_str
        else:
            print(f"警告：无法获取 {cam_sdk_name} 的图像，或者图像为空。")
            return None, None
    except Exception as e:
        print(f"警告：获取 {cam_sdk_name} 图像时发生异常: {e}")
        return None, None

def main():
    """主程序：连接机器人，并进入主控制循环"""
    if not os.path.exists(LOG_IMAGE_DIR):
        os.makedirs(LOG_IMAGE_DIR)
        print(f"📂 已创建文件夹: {LOG_IMAGE_DIR}")
        # 清空 Log 文件夹
    print("🧹 清空日志文件夹...")
    clear_log_directory()

    robot_dds = None
    robot_controller = None
    camera = None

    try:
        # --- 1. 初始化机器人和相机 ---
        print("🤖 初始化机器人系统...")
        robot_dds = RobotDds()
        robot_controller = RobotController()
        
        # 从 CAMERA_MAPPING 获取所有需要使用的摄像头SDK名称
        camera_sdk_names = list(CAMERA_MAPPING.values())
        print(f"📷 正在初始化摄像头: {camera_sdk_names}")
        camera = CosineCamera(camera_sdk_names)
        # arm_initial_joint_position=[-1.63665295,  0.78416812,  0.61188424, -0.70639342,  1.04935575,
        # 1.44077671,  0.72583276,  1.77511871, -0.99129957, -1.53809536,
        # 0.63575584, -0.19526549, -1.14260852, -0.98163235]
        arm_initial_joint_position=[-1.075, 0.6108, 0.279, -1.284, 0.731, 1.495, -0.188,
                                    1.075, -0.6108, -0.279, 1.284, -0.731, -1.495, 0.188] ##和数采位置对齐
        
        robot_dds.reset(arm_positions=arm_initial_joint_position,
                        gripper_positions=[0.0,0.0],
                        hand_positions=robot_dds.hand_initial_joint_position,
                        waist_positions=robot_dds.waist_initial_joint_position,
                        head_positions=robot_dds.head_initial_joint_position
                        )
        print("✅ 系统初始化完成！")
        print("🚀 进入主控制循环...")

        # --- 2. 主控制循环 ---
        while True:
            time.sleep(1)
            print("\n" + "="*50)
            
            # --- 2.1. 获取状态和图像 ---
            try:
                # 获取手臂末端执行器位姿, 并整理为 "右+左" 顺序
                motion_status = robot_controller.get_motion_status()
                left_cartesian = motion_status["frames"]["arm_left_link7"]
                right_cartesian = motion_status["frames"]["arm_right_link7"]
                #a2d_sdk.gripper_states() 返回 ([左爪状态, 右爪状态], [时间戳]) 待确认
                gripper_states_raw, _ = robot_dds.gripper_states() 
                left_gripper_state = gripper_states_raw[0]
                right_gripper_state = gripper_states_raw[1]

                eef_pose_state = [
                    right_cartesian["position"]["x"], right_cartesian["position"]["y"], right_cartesian["position"]["z"],
                    # right_cartesian["orientation"]["euler"]["roll"], right_cartesian["orientation"]["euler"]["pitch"],right_cartesian["orientation"]["euler"]["yaw"],right_gripper_state,
                    right_cartesian["orientation"]["quaternion"]["x"], right_cartesian["orientation"]["quaternion"]["y"],right_cartesian["orientation"]["quaternion"]["z"],right_cartesian["orientation"]["quaternion"]["w"],right_gripper_state,
                    left_cartesian["position"]["x"], left_cartesian["position"]["y"],  left_cartesian["position"]["z"],
                    # left_cartesian["orientation"]["euler"]["roll"], left_cartesian["orientation"]["euler"]["pitch"],left_cartesian["orientation"]["euler"]["yaw"],left_gripper_state,
                    left_cartesian["orientation"]["quaternion"]["x"], left_cartesian["orientation"]["quaternion"]["y"], left_cartesian["orientation"]["quaternion"]["z"], left_cartesian["orientation"]["quaternion"]["w"],left_gripper_state
                ]
                

            except (KeyError, IndexError) as e:
                 print(f"❌ 获取机器人状态失败: {e}。跳过本轮循环。")
                 continue
            except Exception as e:
                 print(f"❌ 获取机器人状态时发生未知错误: {e}。跳过本轮循环。")
                 continue
            
            # 获取、编码并保存图像
            encoded_images = {}
            timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            for server_name, sdk_name in CAMERA_MAPPING.items():
                raw_img, encoded_img = get_and_encode_image(camera, sdk_name)
                encoded_images[server_name] = encoded_img
                if raw_img is not None:
                    print(f"✅ 已获取并编码图像: {server_name}")
                else:
                    print(f"⚠️ 无法获取图像: {server_name}")
                if raw_img is not None:
                    log_path = os.path.join(LOG_IMAGE_DIR, f"{server_name}_{timestamp}.png")
                    cv2.imwrite(log_path, raw_img)
                    latest_path = f"./{server_name}_latest.png"
                    cv2.imwrite(latest_path, raw_img)
                    print(f"[Saved] {latest_path}")

            # --- 2.2. 构造请求数据 ---
            data = {
                "eef_pose": eef_pose_state,         # shape: [1, 14] (右臂+右爪+左臂+左爪)
                "instruction": "Put all the fruits into the basket.",
                "images": encoded_images
            }


            # --- 2.3. 发送推理请求 ---
            try:
                print(f"🚀 正在发送请求到: {INFERENCE_SERVER_URL}")
                response = requests.post(INFERENCE_SERVER_URL, json=data, timeout=1000)
                print(f"[√] Response Status: {response.status_code}")
                if response.status_code != 200:
                    print(f"[!] 服务器返回错误: {response.text}")
                    continue
                result = response.json()
                print("[Response JSON]:", result)
                if "subtask" in result:
                    print("subtask: ", result["subtask"])
            except requests.exceptions.RequestException as e:
                print(f"[!] 请求推理服务器失败: {e}")
                time.sleep(5)
                continue

            # --- 2.4. 解析并执行动作 ---
            if CONTROL_MODE == 'eepose':
                actions = result.get("eepose", [])
                if not actions:
                    print("[!] 模型未返回有效动作，跳过控制")
                    continue
                
                actions = np.array(actions)[:20]

                for i, act in enumerate(actions):
                    action = np.array(act, dtype=np.float32)
                    if action.shape[0] != 16:
                        print(f"[!] 动作维度不正确 (应为16)，跳过此动作: {action}")
                        continue

                    right_pose_array, left_pose_array = action[0:8], action[8:16]
                    gripper_states = [left_pose_array[7].item(), right_pose_array[7].item()]
                    right_pose_dict = { "x": right_pose_array[0].item(), "y": right_pose_array[1].item(), "z": right_pose_array[2].item(), "qx": right_pose_array[3].item(), "qy": right_pose_array[4].item(), "qz": right_pose_array[5].item(), "qw": right_pose_array[6].item() }
                    left_pose_dict = { "x": left_pose_array[0].item(), "y": left_pose_array[1].item(), "z": left_pose_array[2].item(), "qx": left_pose_array[3].item(), "qy": left_pose_array[4].item(), "qz": left_pose_array[5].item(), "qw": left_pose_array[6].item() }
                    
                    print(f"[→ Step {i+1}/{len(actions)}] 执行动作...")
                    # robot_controller.set_end_effector_pose_control(
                    #     lifetime=1.0,
                    #     control_group=["dual_arm"],
                    #     left_pose=left_pose_dict,
                    #     right_pose=right_pose_dict,
                    # )
                    robot_controller.set_end_effector_pose_control(
                        lifetime=1.0,
                        control_group=["right_arm"],
                        right_pose=right_pose_dict,
                    )
                    # robot_controller.set_end_effector_pose_control(
                    #     lifetime=1.0,
                    #     control_group=["left_arm"],
                    #     left_pose=left_pose_dict,
                    # )
                    robot_dds.move_gripper(gripper_states)
                    if i%5 == 0:
                        time.sleep(0.1)
                
                print("✅ 动作序列执行完毕。")
            else:
                print(f"[!] 当前不支持的控制模式: {CONTROL_MODE}")
                break

    except KeyboardInterrupt:
        print("\n[Main] 用户手动中断程序。")
    except Exception as e:
        print(f"\n[Main] ❌ 程序执行时发生严重错误: {e}")
    finally:
        # --- 3. 安全关闭 ---
        if robot_dds:
            print("\n[Main] 重置机器人到安全位置...")
            robot_dds.reset()
            time.sleep(2)
            robot_dds.shutdown()
        if camera:
            camera.close()
        print("[Main] 程序已安全退出。")

if __name__ == "__main__":
    main()