import os
import time
import cv2
import numpy as np
from typing import List, Dict, Tuple
import rospy
import sys
import threading
from sensor_msgs.msg import JointState
from geometry_msgs.msg import Twist
from geometry_msgs.msg import PoseStamped
from hdas_msg.msg import motor_control
from std_msgs.msg import Bool
from std_msgs.msg import Float32
import pandas as pd
from scipy.spatial.transform import Rotation as R
from cv_bridge import CvBridge
from sensor_msgs.msg import CompressedImage

TOPIC_MAP = {
    "pose_left":         "/motion_control/pose_ee_arm_left",
    "pose_right":        "/motion_control/pose_ee_arm_right",
    "joint_left":        "/hdas/feedback_arm_left",
    "joint_right":       "/hdas/feedback_arm_right",
    "gripper_left":      "/hdas/feedback_gripper_left",
    "gripper_right":     "/hdas/feedback_gripper_right",
    "image_left":        "/hdas/camera_head/left_raw/image_raw_color/compressed",
    "image_right":       "/hdas/camera_head/right_raw/image_raw_color/compressed",
    "image_left_wrist":  "/hdas/camera_wrist_left/color/image_raw/compressed",
    "image_right_wrist": "/hdas/camera_wrist_right/color/image_raw/compressed",
}

def decode_image_series(topic: str, bridge: CvBridge):
    msg = rospy.wait_for_message(topic, CompressedImage, timeout=0.2)
    img = bridge.compressed_imgmsg_to_cv2(msg)  # BGR
    # return cv2.resize(img, (640, 480))
    return img



def _extract_pose_obj(msg):
    if hasattr(msg, "pose") and hasattr(msg.pose, "position") and hasattr(msg.pose, "orientation"):
        return msg.pose
    if hasattr(msg, "pose") and hasattr(msg.pose, "pose"):
        return msg.pose.pose
    return None
 
def decode_pose_series(topic: str):
    while True:
        msg = rospy.wait_for_message(topic, PoseStamped)
        p = _extract_pose_obj(msg)
        if p is not None: break
    val = [float(p.position.x), float(p.position.y), float(p.position.z),
           float(p.orientation.x), float(p.orientation.y),
           float(p.orientation.z), float(p.orientation.w)]
    return val

def decode_joint_series(topic: str):
    while True:
        msg = rospy.wait_for_message(topic, JointState)
        if hasattr(msg, "position"): break
    return list(msg.position)

class DualArmStateReader:
    def __init__(self):
        rospy.init_node('joint_state_sender', anonymous=True)
        self.left_joint_state_pub = rospy.Publisher('/motion_target/target_joint_state_arm_left', JointState, queue_size=10)
        self.right_joint_state_pub = rospy.Publisher('/motion_target/target_joint_state_arm_right', JointState, queue_size=10)
        self.torso_joint_state_pub_real = rospy.Publisher('/motion_target/target_joint_state_torso', JointState, queue_size=10)
        
        self.left_gripper_open = rospy.Publisher('/motion_target/target_position_gripper_left', JointState, queue_size=10)
        self.right_gripper_open = rospy.Publisher('/motion_target/target_position_gripper_right', JointState, queue_size=10)
        
        self.left_ee_pos = rospy.Publisher('/motion_target/target_pose_arm_left', PoseStamped, queue_size=10)
        self.right_ee_pos = rospy.Publisher('/motion_target/target_pose_arm_right', PoseStamped, queue_size=10)
        
        # self.acc_limit_pub = rospy.Publisher('/motion_target/chassis_acc_limit',Twist,queue_size=10)
        self.breaking_mode_pub = rospy.Publisher('/motion_target/brake_mode',Bool,queue_size=10)
        self.command = None
        time.sleep(1)

    def get_joint_states(self) -> dict:
        joint_left = decode_joint_series("/hdas/feedback_arm_left")

        gripper_left = decode_joint_series('/hdas/feedback_gripper_left')
        joint_left[-1] = gripper_left[-1]
        
        joint_right = decode_joint_series("/hdas/feedback_arm_right")    
        gripper_right = decode_joint_series('/hdas/feedback_gripper_right')
        joint_right[-1] = gripper_right[-1]
        
        joint_final = joint_right
        joint_final.extend(joint_left)
        
        return joint_final

    def get_eef_poses(self) -> dict:
        ee_left = decode_pose_series("/motion_control/pose_ee_arm_left")
        # euler_left = R.from_quat(ee_left[3:]).as_euler('xyz', degrees = False)   
        # ee_left = ee_left[:3]
        # ee_left.extend(euler_left)
        
        gripper_left = decode_joint_series('/hdas/feedback_gripper_left')
        ee_left.extend(gripper_left)
        
        
        ee_right = decode_pose_series("/motion_control/pose_ee_arm_right")
        # euler_right = R.from_quat(ee_right[3:]).as_euler('xyz', degrees = False)   
        # ee_right = ee_right[:3]
        # ee_right.extend(euler_right)
        
        gripper_right = decode_joint_series('/hdas/feedback_gripper_right')
        ee_right.extend(gripper_right)
        
        pose_final = ee_right
        pose_final.extend(ee_left)
        
        print(f'return_action: {pose_final}')
        
        return pose_final 
    
    def send_eef_commands(self, ee_left, ee_right, position_torso, gripper_left, gripper_right):
        # r_left = R.from_euler('xyz', ee_left[3:]).as_quat()
        r_left = ee_left[3:]
        # r_right = R.from_euler('xyz', ee_right[3:]).as_quat()
        r_right = ee_right[3:]
        left_ee_state = PoseStamped()
        left_ee_state.pose.position.x = ee_left[0]
        left_ee_state.pose.position.y = ee_left[1]
        left_ee_state.pose.position.z = ee_left[2]
        left_ee_state.pose.orientation.x = r_left[0]
        left_ee_state.pose.orientation.y = r_left[1]
        left_ee_state.pose.orientation.z = r_left[2]
        left_ee_state.pose.orientation.w = r_left[3]
        right_ee_state = PoseStamped()
        right_ee_state.pose.position.x = ee_right[0]
        right_ee_state.pose.position.y = ee_right[1]
        right_ee_state.pose.position.z = ee_right[2]
        right_ee_state.pose.orientation.x = r_right[0]
        right_ee_state.pose.orientation.y = r_right[1]
        right_ee_state.pose.orientation.z = r_right[2]
        right_ee_state.pose.orientation.w = r_right[3]
        torso_joint_state = JointState()
        torso_joint_state.position = position_torso
        
        left_open = JointState()
        right_open = JointState()
        left_open.position = [gripper_left]
        right_open.position = [gripper_right] 
        
        
        self.left_ee_pos.publish(left_ee_state)
        self.right_ee_pos.publish(right_ee_state)
        self.torso_joint_state_pub_real.publish(torso_joint_state)
        self.left_gripper_open.publish(left_open)
        self.right_gripper_open.publish(right_open)

    def send_breaking_mode(self,breaking_mode_signal):
        breaking_mode_msg = Bool()
        breaking_mode_msg.data = breaking_mode_signal[0]
        self.breaking_mode_pub.publish(breaking_mode_msg)

    def send_joint_commands(self, position_left, position_right, position_torso, gripper_left, gripper_right):
        left_joint_state = JointState()
        left_joint_state.position = position_left
        right_joint_state = JointState()
        right_joint_state.position = position_right
        torso_joint_state = JointState()
        torso_joint_state.position = position_torso
        
        left_open = JointState()
        right_open = JointState()
        left_open.position = [gripper_left]
        right_open.position = [gripper_right]

        self.left_joint_state_pub.publish(left_joint_state)
        self.right_joint_state_pub.publish(right_joint_state)
        self.torso_joint_state_pub_real.publish(torso_joint_state)
        self.left_gripper_open.publish(left_open)
        self.right_gripper_open.publish(right_open)


class RobotEnv:
    def __init__(
        self,
    ) -> None:
        self._arm: DualArmStateReader | None = DualArmStateReader()
        
        self.init_pose()
        time.sleep(3)

    def init_pose(self) -> None:
        init_eef_pose = [-2.2896037e-02, -3.3524475e-01, 3.3313975e-01, -5.8023985e-03, 5.3188358e-03, -1.0851217e-02, 99, -2.2927403e-02, 3.3490089e-01, 3.3418587e-01, 2.6812404e-02, -4.2551410e-04, -1.3829788e-02, 99]  # (14,)
        r_left = R.from_euler('xyz', init_eef_pose[3:6]).as_quat()
        r_right = R.from_euler('xyz', init_eef_pose[10:13]).as_quat()
        init_eef_pose = init_eef_pose[:3] + list(r_left) + [init_eef_pose[6]] + init_eef_pose[7:10] + list(r_right) + [init_eef_pose[13]]
        self.control_eef(init_eef_pose)
    
    # def 
    def update_obs_window(self) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
        bridge = CvBridge()
        if self._arm:
            frames = {}
            for topic_name in TOPIC_MAP:
                if 'image' in topic_name:
                    frames[topic_name] = decode_image_series(TOPIC_MAP[topic_name], bridge)
            state = {
                "qpos": self._arm.get_joint_states(),        # shape: (14,)
                "eef_pose": self._arm.get_eef_poses()  # shape: (14,)
            }
        else:
            state = None	
        return frames, state


    def control(self, action, wait: bool = True):
        if not self._arm:
            raise RuntimeError("Arm not initialised; pass arm_ip when constructing RobotEnv.")
            
        torso_joint = [-1.0339000225067139, 2.1635000705718994, 1.1513999700546265, 0.0]
        self._arm.send_joint_commands(action[7:13], action[:6], torso_joint, action[13], action[6])
    

    def control_eef(self, action, wait=True):
        if not self._arm:
            raise RuntimeError("Arm not initialised; pass arm_ip when constructing RobotEnv.")

        torso_joint = [-1.0339000225067139, 2.1635000705718994, 1.1513999700546265, 0.0]
        self._arm.send_eef_commands(action[8:15], action[:7], torso_joint, action[15], action[7])
        time.sleep(0.01)
        
    def shutdown(self):
        for _name, cam in self._cams:
            cam.stop()
            
if __name__ == "__main__":
    # 示例主函数入口
    env = RobotEnv(
    )

    try:
        while True:
            frames, state = env.update_obs_window()
            for name in frames:
                print(f"[Frame] {name}: shape={frames[name].shape}")
            if state is not None:
                print(f"[Arm State] {state.round(3)}")
            # time.sleep(0.1)
    except KeyboardInterrupt:
        print("\n[Main] Interrupted by user.")
    finally:
        # env.shutdown()
        print("[Main] RobotEnv shut down successfully.")
