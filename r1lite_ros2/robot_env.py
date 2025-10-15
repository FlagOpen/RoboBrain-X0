import time
import numpy as np
from typing import List, Dict, Tuple, Optional
import rclpy
from rclpy.node import Node
import threading
from sensor_msgs.msg import JointState
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import Bool
from scipy.spatial.transform import Rotation as R
from cv_bridge import CvBridge
from sensor_msgs.msg import CompressedImage
from rclpy.qos import qos_profile_sensor_data

TOPIC_MAP = {
    "/hdas/camera_head/left_raw/image_raw_color/compressed": "image_left",
    "/hdas/camera_head/right_raw/image_raw_color/compressed": "image_right",
    "/hdas/camera_wrist_left/color/image_rect_raw/compressed":  "image_left_wrist",
    "/hdas/camera_wrist_right/color/image_rect_raw/compressed":  "image_right_wrist"
}

qos_pub_profile = rclpy.qos.QoSProfile(
    reliability=rclpy.qos.ReliabilityPolicy.RELIABLE,  
    history=rclpy.qos.HistoryPolicy.KEEP_LAST,
    depth=10,
    durability=rclpy.qos.DurabilityPolicy.TRANSIENT_LOCAL
)
qos_sub_profile = rclpy.qos.QoSProfile(
    reliability=rclpy.qos.ReliabilityPolicy.BEST_EFFORT,  
    history=rclpy.qos.HistoryPolicy.KEEP_LAST,
    depth=10,
    durability=rclpy.qos.DurabilityPolicy.VOLATILE
)

class DualArmStateReader(Node):
    def __init__(self):
        super().__init__('joint_state_sender')
        
        # 创建发布者
        self.left_joint_state_pub = self.create_publisher(JointState, '/motion_target/target_joint_state_arm_left', qos_pub_profile)
        self.right_joint_state_pub = self.create_publisher(JointState, '/motion_target/target_joint_state_arm_right', qos_pub_profile)
        self.torso_joint_state_pub_real = self.create_publisher(JointState, '/motion_target/target_joint_state_torso', qos_pub_profile)
        self.left_gripper_open = self.create_publisher(JointState, '/motion_target/target_position_gripper_left', qos_pub_profile)
        self.right_gripper_open = self.create_publisher(JointState, '/motion_target/target_position_gripper_right', qos_pub_profile)
        self.left_ee_pos = self.create_publisher(PoseStamped, '/motion_target/target_pose_arm_left', qos_pub_profile)
        self.right_ee_pos = self.create_publisher(PoseStamped, '/motion_target/target_pose_arm_right', qos_pub_profile)
        self.breaking_mode_pub = self.create_publisher(Bool, '/motion_target/brake_mode', qos_pub_profile)
        
        self.bridge = CvBridge()
        
        # 存储最新消息的字典
        self.latest_messages = {}
        self.message_received = {}  # 记录是否至少收到过一次消息
        self.message_lock = threading.Lock()  # 用于线程安全的锁
        
        # 创建所有需要的订阅者
        self.setup_subscriptions()
        
        time.sleep(1)
    
    def setup_subscriptions(self):
        """设置所有需要的订阅者"""
        # 关节状态订阅者
        joint_topics = [
            "/hdas/feedback_arm_left",
            "/hdas/feedback_arm_right",
            "/hdas/feedback_gripper_left",
            "/hdas/feedback_gripper_right"
        ]
        
        for topic in joint_topics:
            self.latest_messages[topic] = None
            self.message_received[topic] = False
            self.create_subscription(
                JointState, 
                topic, 
                lambda msg, t=topic: self.joint_callback(msg, t),
                qos_sub_profile
            )
        
        # 位姿订阅者
        pose_topics = [
            "/motion_control/pose_ee_arm_left",
            "/motion_control/pose_ee_arm_right"
        ]
        
        for topic in pose_topics:
            self.latest_messages[topic] = None
            self.message_received[topic] = False
            self.create_subscription(
                PoseStamped,
                topic,
                lambda msg, t=topic: self.pose_callback(msg, t),
                qos_sub_profile
            )
        
        # 图像订阅者
        image_topics = [
            "/hdas/camera_head/left_raw/image_raw_color/compressed",
            "/hdas/camera_head/right_raw/image_raw_color/compressed",
            "/hdas/camera_wrist_left/color/image_rect_raw/compressed",
            "/hdas/camera_wrist_right/color/image_rect_raw/compressed"
        ]
        
        for topic in image_topics:
            self.latest_messages[topic] = None
            self.message_received[topic] = False
            self.create_subscription(
                CompressedImage,
                topic,
                lambda msg, t=topic: self.image_callback(msg, t),
                qos_sub_profile
            )
    
    def joint_callback(self, msg, topic):
        """关节状态回调函数"""
        with self.message_lock:
            self.latest_messages[topic] = msg
            self.message_received[topic] = True
    
    def pose_callback(self, msg, topic):
        """位姿回调函数"""
        with self.message_lock:
            self.latest_messages[topic] = msg
            self.message_received[topic] = True
    
    def image_callback(self, msg, topic):
        """图像回调函数"""
        with self.message_lock:
            self.latest_messages[topic] = msg
            self.message_received[topic] = True
    
    def _extract_pose_obj(self, msg):
        """从消息中提取位姿对象"""
        if hasattr(msg, "pose") and hasattr(msg.pose, "position") and hasattr(msg.pose, "orientation"):
            return msg.pose
        if hasattr(msg, "pose") and hasattr(msg.pose, "pose"):
            return msg.pose.pose
        return None
    
    def get_latest_joint_state(self, topic) -> Optional[List[float]]:
        """获取指定话题的最新关节状态"""
        with self.message_lock:
            if not self.message_received[topic]:
                return None
            if self.latest_messages[topic] is not None and hasattr(self.latest_messages[topic], "position"):
                return list(self.latest_messages[topic].position)
        
        return None
    
    def get_latest_pose(self, topic) -> Optional[List[float]]:
        """获取指定话题的最新位姿"""
        with self.message_lock:
            if not self.message_received[topic]:
                return None
                
            if self.latest_messages[topic] is not None:
                p = self._extract_pose_obj(self.latest_messages[topic])
                if p is not None:
                    return [float(p.position.x), float(p.position.y), float(p.position.z),
                            float(p.orientation.x), float(p.orientation.y),
                            float(p.orientation.z), float(p.orientation.w)]
        
        return None
    
    def get_latest_image(self, topic) -> Optional[np.ndarray]:
        """获取指定话题的最新图像"""
        with self.message_lock:
            if not self.message_received[topic]:
                return None
                
            if self.latest_messages[topic] is not None:
                try:
                    # 将消息转换为OpenCV图像
                    cv_image = self.bridge.compressed_imgmsg_to_cv2(self.latest_messages[topic])
                    return cv_image
                except Exception as e:
                    self.get_logger().error(f"Failed to convert image: {e}")
                    return None
        
        return None
    
    def get_all_images(self) -> Dict[str, Optional[np.ndarray]]:
        """获取所有图像"""
        images = {}
        image_topics = [
            "/hdas/camera_head/left_raw/image_raw_color/compressed",
            "/hdas/camera_head/right_raw/image_raw_color/compressed",
            "/hdas/camera_wrist_left/color/image_rect_raw/compressed",
            "/hdas/camera_wrist_right/color/image_rect_raw/compressed"
        ]
        
        for topic in image_topics:
            images[topic] = self.get_latest_image(topic)
        
        return images
    
    def get_joint_states(self) -> Optional[List[float]]:
        """获取所有关节状态"""
        joint_left = self.get_latest_joint_state("/hdas/feedback_arm_left")
        gripper_left = self.get_latest_joint_state('/hdas/feedback_gripper_left')
        
        joint_right = self.get_latest_joint_state("/hdas/feedback_arm_right")
        gripper_right = self.get_latest_joint_state('/hdas/feedback_gripper_right')
        
        if joint_left is None or gripper_left is None or joint_right is None or gripper_right is None:
            return None
        
        joint_left[-1] = gripper_left[0]
        joint_right[-1] = gripper_right[0]
        
        joint_final = joint_right
        joint_final.extend(joint_left)
        
        return joint_final
    
    def get_eef_poses(self) -> Optional[List[float]]:
        """获取所有末端执行器位姿"""
        ee_left = self.get_latest_pose("/motion_control/pose_ee_arm_left")
        gripper_left = self.get_latest_joint_state('/hdas/feedback_gripper_left')
        
        ee_right = self.get_latest_pose("/motion_control/pose_ee_arm_right")
        gripper_right = self.get_latest_joint_state('/hdas/feedback_gripper_right')
        
        if ee_left is None or gripper_left is None or ee_right is None or gripper_right is None:
            return None
        
        ee_left.extend(gripper_left)
        ee_right.extend(gripper_right)
        
        pose_final = ee_right
        pose_final.extend(ee_left)
        
        return pose_final
    
    def send_eef_commands(self, ee_left, ee_right, position_torso, gripper_left, gripper_right):
        """发送末端执行器命令"""

        r_left = ee_left[3:]
        r_right = ee_right[3:]

        # print(f'left_ee_state = ', ee_left[:3])
        # print(f'r_left = ', r_left)
        # print(f'right_ee_state = ', ee_right[:3])
        # print(f'r_right = ', r_right)
        # print(f'position_torso = ', position_torso)

        # print(f'gripper_left = ', gripper_left)
        # print(f'gripper_right = ', gripper_right)

        left_ee_state = PoseStamped()
        left_ee_state.header.stamp = self.get_clock().now().to_msg()
        left_ee_state.header.frame_id = "base_link"
        left_ee_state.pose.position.x = float(ee_left[0])
        left_ee_state.pose.position.y = float(ee_left[1])
        left_ee_state.pose.position.z = float(ee_left[2])
        left_ee_state.pose.orientation.x = float(r_left[0])
        left_ee_state.pose.orientation.y = float(r_left[1])
        left_ee_state.pose.orientation.z = float(r_left[2])
        left_ee_state.pose.orientation.w = float(r_left[3])
        
        right_ee_state = PoseStamped()
        right_ee_state.header.stamp = self.get_clock().now().to_msg()
        right_ee_state.header.frame_id = "base_link"
        right_ee_state.pose.position.x = float(ee_right[0])
        right_ee_state.pose.position.y = float(ee_right[1])
        right_ee_state.pose.position.z = float(ee_right[2])
        right_ee_state.pose.orientation.x = float(r_right[0])
        right_ee_state.pose.orientation.y = float(r_right[1])
        right_ee_state.pose.orientation.z = float(r_right[2])
        right_ee_state.pose.orientation.w = float(r_right[3])
        
        torso_joint_state = JointState()
        torso_joint_state.header.stamp = self.get_clock().now().to_msg()
        torso_joint_state.position = [float(x) for x in position_torso]
        
        left_open = JointState()
        left_open.header.stamp = self.get_clock().now().to_msg()
        right_open = JointState()
        right_open.header.stamp = self.get_clock().now().to_msg()
        left_open.position = [float(gripper_left)]
        right_open.position = [float(gripper_right)] 
        

        self.left_ee_pos.publish(left_ee_state)
        self.right_ee_pos.publish(right_ee_state)
        # self.torso_joint_state_pub_real.publish(torso_joint_state)
        self.left_gripper_open.publish(left_open)
        self.right_gripper_open.publish(right_open)
    
    def send_joint_commands(self, position_left, position_right, position_torso, gripper_left, gripper_right):
        """发送关节命令"""
        left_joint_state = JointState()
        left_joint_state.header.stamp = self.get_clock().now().to_msg()
        left_joint_state.position = position_left
        
        right_joint_state = JointState()
        right_joint_state.header.stamp = self.get_clock().now().to_msg()
        right_joint_state.position = position_right
        
        torso_joint_state = JointState()
        torso_joint_state.header.stamp = self.get_clock().now().to_msg()
        torso_joint_state.position = position_torso
        
        left_open = JointState()
        left_open.header.stamp = self.get_clock().now().to_msg()
        right_open = JointState()
        right_open.header.stamp = self.get_clock().now().to_msg()
        left_open.position = [gripper_left]
        right_open.position = [gripper_right]

        self.left_joint_state_pub.publish(left_joint_state)
        self.right_joint_state_pub.publish(right_joint_state)
        # self.torso_joint_state_pub_real.publish(torso_joint_state)
        self.left_gripper_open.publish(left_open)
        self.right_gripper_open.publish(right_open)


class RobotEnv:
    def __init__(self) -> None:
        rclpy.init()
        self._arm = DualArmStateReader()
        self.executor = rclpy.executors.MultiThreadedExecutor()
        self.executor.add_node(self._arm)
        
        # 在后台线程中运行执行器
        self.executor_thread = threading.Thread(target=self.executor.spin, daemon=True)
        self.executor_thread.start()
        
        # 等待初始数据
        self.wait_for_initial_data()
        
        self.init_pose()
        time.sleep(3)
    
    def wait_for_initial_data(self, timeout=5.0):
        """等待初始数据到达"""
        start_time = time.time()
        while time.time() - start_time < timeout:
            joint_states = self._arm.get_joint_states()
            eef_poses = self._arm.get_eef_poses()
            
            if joint_states is not None and eef_poses is not None:
                print("Initial data received")
                return True
            
            time.sleep(0.1)
        
        print("Timeout waiting for initial data")
        return False

    def init_pose(self) -> None:
        """初始化机器人位姿"""
        init_eef_pose = [-2.2896037e-02, -3.3524475e-01, 3.3313975e-01, -5.8023985e-03, 5.3188358e-03, -1.0851217e-02, 100, -2.2927403e-02, 3.3490089e-01, 3.3418587e-01, 2.6812404e-02, -4.2551410e-04, -1.3829788e-02, 100]  # (14,)
        r_left = R.from_euler('xyz', init_eef_pose[3:6]).as_quat()
        r_right = R.from_euler('xyz', init_eef_pose[10:13]).as_quat()
        init_eef_pose = init_eef_pose[:3] + list(r_left) + [init_eef_pose[6]] + init_eef_pose[7:10] + list(r_right) + [init_eef_pose[13]]
        # self.control_eef(init_eef_pose)
    
    def update_obs_window(self) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
        """更新观测窗口"""
        frames = {}
        state = None
        
        if self._arm:
            # 获取图像数据
            images = self._arm.get_all_images()
            for topic, img in images.items():
                if img is not None:
                    frames[TOPIC_MAP[topic]] = img
            
            # 获取状态数据
            joint_states = self._arm.get_joint_states()
            eef_poses = self._arm.get_eef_poses()
            
            if joint_states is not None and eef_poses is not None:
                state = {
                    "qpos": joint_states,
                    "eef_pose": eef_poses
                }
        
        return frames, state
    
    def control(self, action, wait: bool = True):
        action = [float(x) for x in action]  
        if not self._arm:
            raise RuntimeError("Arm not initialised; pass arm_ip when constructing RobotEnv.")

        torso_joint = [-0.8278999924659729, 1.7697999477386475, 0.3725000023841858, 0.0]
        self._arm.send_joint_commands(action[7:13], action[:6], torso_joint, action[13], action[6])
    
    def control_eef(self, action, wait=True):
        """控制末端执行器"""
        if not self._arm:
            raise RuntimeError("Arm not initialised")
            
        torso_joint = [-0.8295000195503235, 1.7687000036239624, 0.373199999332428, 0.0]
        self._arm.send_eef_commands(action[8:15], action[:7], torso_joint, action[15], action[7])
        
        if wait:
            time.sleep(0.01)
        
    def shutdown(self):
        """关闭机器人环境"""
        self.executor.shutdown()
        self._arm.destroy_node()
        rclpy.shutdown()


def main():
    """主函数"""
    env = RobotEnv()

    try:
        while rclpy.ok():
            frames, state = env.update_obs_window()
            
            # 打印图像信息
            for name, img in frames.items():
                if img is not None:
                    print(f"[Frame] {name}: shape={img.shape}")
                else:
                    print(f"[Frame] {name}: No image received")
            
            # 打印状态信息
            if state is not None:
                qpos = np.array(state["qpos"])
                eef_pose = np.array(state["eef_pose"])
                
                # 使用列表推导式格式化每个元素
                formatted_qpos = [f"{x:.3f}" for x in qpos]
                formatted_eef_pose = [f"{x:.3f}" for x in eef_pose]
                
                print(f"[Arm State - qpos] {formatted_qpos}")
                print(f"[Arm State - eef_pose] {formatted_eef_pose}")
            else:
                print("[Arm State] No state data received")
            
            time.sleep(0.1)  # 控制循环频率
            
    except KeyboardInterrupt:
        print("\n[Main] Interrupted by user.")
    finally:
        env.shutdown()
        print("[Main] RobotEnv shut down successfully.")


if __name__ == "__main__":
    main()