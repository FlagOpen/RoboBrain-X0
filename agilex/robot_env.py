import os
import time
import cv2
import numpy as np
from typing import List, Dict, Tuple
from piper_sdk import C_PiperInterface
# from pyorbbecsdk import Context, Pipeline, OBSensorType, OBFormat, Config
import pyrealsense2 as rs  

class RealsenseCamera:
    '''
    Supported resolutions for d405:
    1280x720: 5,15,30
    848x480: 5,15,30,60,90
    640x360: 5,15,30,60,90
    480x270: 5,15,30,60,90
    424x240: 5,15,30,60,90

    Supported resolutions for d455:
    1280x720: 5,15,30
    848x480: 5,15,30,60,90
    640x480: 5,15,30,60,90
    640x360: 5,15,30,60,90
    480x270: 5,15,30,60,90
    424x240: 5,15,30,60,90
    '''
    def __init__(self, index: int = 0, width=640, height=480, fps=30):
        DEPTH_RESOLUTION = (width, height)  
        COLOR_RESOLUTION = (width, height)
        DEPTH_FPS = fps
        COLOR_FPS = fps

        # First check available devices
        ctx = rs.context()
        devices = ctx.query_devices()
        if index >= len(devices):
            raise IndexError(f"RealSense device index {index} out of range. Found {len(devices)} devices.")

        # Configure depth and color streams
        print(f"Loading Intel Realsense Camera {index}")
        self.pipeline = rs.pipeline()
        config = rs.config()

        # Use device index
        config.enable_device(devices[index].get_info(rs.camera_info.serial_number))
        config.enable_stream(rs.stream.color, COLOR_RESOLUTION[0], COLOR_RESOLUTION[1], rs.format.bgr8, COLOR_FPS)
        # config.enable_stream(rs.stream.depth, DEPTH_RESOLUTION[0], DEPTH_RESOLUTION[1], rs.format.z16, DEPTH_FPS)

        # Start streaming
        try:
            self.pipeline.start(config)
            print(f"RealSense Camera {index} started successfully")
        except Exception as e:
            print(f"Failed to start RealSense Camera {index}: {e}")
            raise

        # Allow camera some time to warm up
        time.sleep(1)

        # Try to get a few frames to ensure the camera is working properly
        for i in range(5):
            try:
                frame = self.get_one_color_frame()
                if frame is not None:
                    print(f"RealSense Camera {index} ready")
                    break
            except Exception as e:
                print(f"RealSense Camera {index} warmup attempt {i+1} failed: {e}")
                if i == 4:  # Last attempt
                    raise
                time.sleep(0.5)

    def get_one_img(self):
        frames = self.pipeline.wait_for_frames()
        aligned_frames = self.align.process(frames)
        color_frame = aligned_frames.get_color_frame()
        depth_frame = aligned_frames.get_depth_frame()

        color_img = np.asanyarray(color_frame.get_data())
        depth_img = np.asanyarray(depth_frame.get_data())
        return color_img, depth_img

    # def get_one_color_frame(self):
    #     frames = self.pipeline.wait_for_frames()
    #     aligned_frames = self.align.process(frames)
    #     color_frame = aligned_frames.get_color_frame()
    #     return color_frame

    def get_one_color_frame(self) -> np.ndarray | None:
        try:
            frames = self.pipeline.wait_for_frames(5000)  # Increase timeout to 5 seconds
            if frames is None:
                return None
            color_frame = frames.get_color_frame()
            if color_frame is None:
                return None
            data = np.asanyarray(color_frame.get_data()).reshape((color_frame.get_height(), color_frame.get_width(), 3))
            return data
        except Exception as e:
            print(f"Error getting color frame: {e}")
            return None

    def get_one_depth_frame(self):
        frames = self.pipeline.wait_for_frames()
        aligned_frames = self.align.process(frames)
        depth_frame = aligned_frames.get_depth_frame()
        return depth_frame

    def release(self):
        self.pipeline.stop()
        cv2.destroyAllWindows()

class OrbbecCamera:
    def __init__(self, index: int = 0, width=640, height=480, fps=30):
        self.ctx = Context()
        device_list = self.ctx.query_devices()

        if index < 0 or index >= device_list.get_count():
            raise IndexError(f"Orbbec device index {index} out of range. Found {len(device_list)} devices.")

        self.device = device_list.get_device_by_index(index)
        # print(f"[OrbbecCamera] Using device index={index}, serial={self.device.get_serial_number()}")

        self.pipeline = Pipeline(self.device)
        profile_list = self.pipeline.get_stream_profile_list(OBSensorType.COLOR_SENSOR)
        self.color_profile = profile_list.get_video_stream_profile(width, height, OBFormat.RGB, fps)
        config = Config()
        config.enable_stream(self.color_profile)
        self.pipeline.start(config)
        for _ in range(3):
            self.get_one_color_frame()

    def get_one_color_frame(self) -> np.ndarray | None:
        frames = self.pipeline.wait_for_frames(60)
        if frames is None:
            return None
        color_frame = frames.get_color_frame()
        if color_frame is None:
            return None
        data = np.asanyarray(color_frame.get_data()).reshape((color_frame.get_height(), color_frame.get_width(), 3))
        return cv2.cvtColor(data, cv2.COLOR_RGB2BGR)

    def get_one_depth_frame(self) -> np.ndarray | None:
        frames = self.pipeline.wait_for_frames(100)
        if frames is None:
            return None
        depth_frame = frames.get_depth_frame()
        if depth_frame is None:
            return None
        data = np.asanyarray(depth_frame.get_data()).reshape((depth_frame.get_height(), depth_frame.get_width(), 1))
        return data

    def stop(self):
        self.pipeline.stop()


class DualArmStateReader:
    def __init__(self, can_left: str, can_right: str):
        self.can_left = can_left
        self.can_right = can_right
        self.piper_left = C_PiperInterface(self.can_left, False)
        self.piper_right = C_PiperInterface(self.can_right, False)
        self.connect()
        # self._initialize_arm(self.piper_left, name="left")
        # self._initialize_arm(self.piper_right, name="right")

    # def _initialize_arm(self, piper: C_PiperInterface, name="arm"):
    #     """切换模式并移动至初始位置"""
    #     initial_joints = [-1.15188663e-02,  5.79031351e-02, -4.69262936e-01,  1.42808183e-01,
    #                      1.13031626e+00, -6.81788497e-02,  6.62983961e-01]
    #     factor = 57324.840764  # 弧度转piper整数
    #     def encode(state):
    #         joints = [round(j * factor) for j in state[:6]]
    #         gripper = round(abs(state[6]) * 1000 * 100)
    #         return joints, gripper

    #     # 编码左右状态
    #     joints_left, gripper_left = encode(initial_joints)
    #     joints_right, gripper_right = encode(initial_joints)

    #     # 单臂控制
    #     piper.MotionCtrl_2(0x01, 0x01, 50, 0x00)
    #     piper.JointCtrl(*joints_left)
    #     piper.GripperCtrl(gripper_left, 1000, 0x01, 0)
    #     piper.MotionCtrl_2(0x01, 0x01, 50, 0x00)

    #     print(f"[{name}] Arm moved to initial pose.")


    def _enable_arm(self, piper: C_PiperInterface, name="arm"):
        piper.EnableArm(7)
        start_time = time.time()
        timeout = 5
        while time.time() - start_time < timeout:
            status = all([
                piper.GetArmLowSpdInfoMsgs().motor_1.foc_status.driver_enable_status,
                piper.GetArmLowSpdInfoMsgs().motor_2.foc_status.driver_enable_status,
                piper.GetArmLowSpdInfoMsgs().motor_3.foc_status.driver_enable_status,
                piper.GetArmLowSpdInfoMsgs().motor_4.foc_status.driver_enable_status,
                piper.GetArmLowSpdInfoMsgs().motor_5.foc_status.driver_enable_status,
                piper.GetArmLowSpdInfoMsgs().motor_6.foc_status.driver_enable_status,
            ])
            if status:
                print(f"[{name}] Arm enabled.")
                return
            time.sleep(0.5)
        raise TimeoutError(f"[{name}] EnableArm timeout after {timeout}s")

    def _initialize_arm(self, piper: C_PiperInterface, name="arm"):
        initial_joints = [-1.15188663e-02, 5.79031351e-02, -4.69262936e-01, 1.42808183e-01,
                          1.13031626e+00, -6.81788497e-02, 6.62983961e-01]
        factor = 57324.840764

        def encode(state):
            joints = [round(j * factor) for j in state[:6]]
            gripper = round(abs(state[6]) * 1000 * 100)
            return joints, gripper

        joints, gripper = encode(initial_joints)
        piper.MotionCtrl_2(0x01, 0x01, 20, 0x00)
        piper.JointCtrl(*joints)
        piper.GripperCtrl(gripper, 1000, 0x01, 0)
        # piper.MotionCtrl_2(0x01, 0x01, 20, 0x00)
        print(f"[{name}] Arm moved to initial pose.")
        time.sleep(0.5)

    def _initialize_dual_eef_pose(self, piper_left: C_PiperInterface, piper_right: C_PiperInterface):
        # Initial EEF Pose (first 7 dimensions for right arm, last 7 for left arm)
        initial_pose = [-0.98268986, -0.02801541, -0.18312784, -0.13389918,  0.7905487,
        0.5975816 ,  0.6517    , -0.9385504 , -0.14160725, -0.3147546 ,
        0.09481619,  0.77108294, -0.62963563,  0.693     ]

        def send_pose(piper: C_PiperInterface, pose: list, arm_name: str):
            x, y, z = pose[0:3]
            rx, ry, rz = pose[3:6]
            gripper = pose[6]

            # Convert units
            x, y, z = int(x * 1e6), int(y * 1e6), int(z * 1e6)
            rx, ry, rz = [int(a * 1000 * 360 / (2 * np.pi)) for a in (rx, ry, rz)]
            gripper = int(abs(gripper * 1000 * 100))

            piper.MotionCtrl_2(0x01, 0x00, 20, 0x00)
            piper.EndPoseCtrl(x, y, z, rx, ry, rz)
            piper.GripperCtrl(gripper, 1000, 0x01, 0)

            # piper.MotionCtrl_2(0x01, 0x01, 20, 0x00)
            print(f"[{arm_name}] EEF moved to initial pose.")

        # Right arm
        send_pose(piper_right, initial_pose[0:7], "right_arm")

        # Left arm
        send_pose(piper_left, initial_pose[7:14], "left_arm")


    def connect(self):
        self.piper_left.ConnectPort()
        self.piper_right.ConnectPort()
        print("[DualArm] Connected to both arms.")
        self._enable_arm(self.piper_left, name="left")
        self._enable_arm(self.piper_right, name="right")
        # self._initialize_dual_eef_pose(self.piper_left, self.piper_right)
        self._initialize_arm(self.piper_left, name="left")
        self._initialize_arm(self.piper_right, name="right")
        time.sleep(0.5)

        # self.connected = True

    def _get_joint_state(self, piper) -> np.ndarray:
        factor = 57324.840764
        joint_msg = piper.GetArmJointMsgs()
        gripper_msg = piper.GetArmGripperMsgs()
        joint_values = [
            joint_msg.joint_state.joint_1.real / factor,
            joint_msg.joint_state.joint_2.real / factor,
            joint_msg.joint_state.joint_3.real / factor,
            joint_msg.joint_state.joint_4.real / factor,
            joint_msg.joint_state.joint_5.real / factor,
            joint_msg.joint_state.joint_6.real / factor,
            gripper_msg.gripper_state.grippers_angle / 1000 / 100,
        ]
        return np.array(joint_values, dtype=np.float32)

    def _get_eef_pose(self, piper) -> np.ndarray:
        """从 Piper 接口读取末端位姿（eef pose）并标准化返回"""
        pose_msg = piper.GetArmEndPoseMsgs()
        gripper_msg = piper.GetArmGripperMsgs()

        # 转换单位并组合：位置（单位 m），姿态（单位 rad），夹爪（归一化）
        eef_pose = [
            pose_msg.end_pose.X_axis * 1e-6,  # mm -> m
            pose_msg.end_pose.Y_axis * 1e-6,
            pose_msg.end_pose.Z_axis * 1e-6,
            pose_msg.end_pose.RX_axis * np.pi / 180,  # deg -> rad
            pose_msg.end_pose.RY_axis * np.pi / 180,
            pose_msg.end_pose.RZ_axis * np.pi / 180,
            gripper_msg.gripper_state.grippers_angle / 1000 / 100  # 保持一致归一化
        ]
        return np.array(eef_pose, dtype=np.float32)

    def get_joint_states(self) -> dict:
        left_state = self._get_joint_state(self.piper_left)
        right_state = self._get_joint_state(self.piper_right)
        return {"left": left_state, "right": right_state}

    def get_eef_poses(self) -> dict:
        def read_pose(piper):
            pose_msg = piper.GetArmEndPoseMsgs()
            gripper_msg = piper.GetArmGripperMsgs()

            pose = [
                pose_msg.end_pose.X_axis * 1e-6,
                pose_msg.end_pose.Y_axis * 1e-6,
                pose_msg.end_pose.Z_axis * 1e-6,
                pose_msg.end_pose.RX_axis * np.pi / (180 * 1000),
                pose_msg.end_pose.RY_axis * np.pi / (180 * 1000),
                pose_msg.end_pose.RZ_axis * np.pi / (180 * 1000),
                gripper_msg.gripper_state.grippers_angle / 1000 / 100,
            ]
            return np.array(pose, dtype=np.float32)

        left_pose = read_pose(self.piper_left)
        right_pose = read_pose(self.piper_right)
        return {"left": left_pose, "right": right_pose}

    def send_eef_commands(self, left_pose: np.ndarray, right_pose: np.ndarray, speed: int = 20):
        def execute(piper, pose):
            pose = np.asarray(pose).flatten()
            piper.MotionCtrl_2(0x01, 0x00, speed, 0x00)
            piper.EndPoseCtrl(
                int(pose[0] * 1e6),
                int(pose[1] * 1e6),
                int(pose[2] * 1e6),
                int(pose[3] * 180 / np.pi * 1000),
                int(pose[4] * 180 / np.pi * 1000),
                int(pose[5] * 180 / np.pi * 1000),
            )
            piper.GripperCtrl(int(abs(pose[6]) * 1000 * 100 * 10), 1000, 0x01, 0)
            # piper.MotionCtrl_2(0x01, 0x00, speed, 0x00)
        # import ipdb; ipdb.set_trace()
        execute(self.piper_left, left_pose)
        execute(self.piper_right, right_pose)


    def send_joint_commands(self, left_state: np.ndarray, right_state: np.ndarray, speed: int = 20):
        factor = 57324.840764

        def encode(state):
            state = np.asarray(state).flatten()
            joints = [round(float(j) * factor) for j in state[:6]]
            gripper = round(float(abs(state[6])) * 1000 * 100)
            return joints, gripper

        joints_left, gripper_left = encode(left_state)
        joints_right, gripper_right = encode(right_state)

        self.piper_left.MotionCtrl_2(0x01, 0x01, speed, 0x00)
        self.piper_left.JointCtrl(*joints_left)
        self.piper_left.GripperCtrl(gripper_left, 1000, 0x01, 0)
        time.sleep(0.05)
        # self.piper_left.MotionCtrl_2(0x01, 0x01, speed, 0x00)

        self.piper_right.MotionCtrl_2(0x01, 0x01, speed, 0x00)
        self.piper_right.JointCtrl(*joints_right)
        self.piper_right.GripperCtrl(gripper_right, 1000, 0x01, 0)
        # self.piper_right.MotionCtrl_2(0x01, 0x01, speed, 0x00)
        time.sleep(0.05)


class PiperArm:
    def __init__(self, arm_ip: str = ""):
        self.arm = DualArmStateReader(can_left="can_left", can_right="can_right")

    def get_state(self) -> np.ndarray:
        states = self.arm.get_joint_states()
        return np.concatenate([states["right"], states["left"]], axis=0)

    def get_eef_pose(self) -> np.ndarray:
        """返回左右手末端执行器位姿（例如6DoF + gripper，共7维 * 2）"""
        poses = self.arm.get_eef_poses()  # 应该返回 {"left": np.array(7,), "right": np.array(7,)}
        return np.concatenate([poses["right"],poses["left"]], axis=0)  # shape: (14,)

    def execute(self, action: np.ndarray, wait: bool = True):
        right = action[:7]
        left = action[7:]
        self.arm.send_joint_commands(left, right, speed=20)

    def execute_eef(self, action: np.ndarray, wait: bool = True):
        right = action[:7]
        left = action[7:]
        self.arm.send_eef_commands(left, right)
        time.sleep(0.1)

class RobotEnv:
    def __init__(
        self,
        realsense_serials: List[int] | None = None,
        orbbec_serials: List[int] | None = None,
        arm_ip: str | None = None,
    ) -> None:
        self._cams: List[Tuple[str, OrbbecCamera]] = []

        # Check available RealSense devices
        if realsense_serials:
            ctx = rs.context()
            devices = ctx.query_devices()
            available_devices = len(devices)
            print(f"Found {available_devices} RealSense devices")

            for idx in realsense_serials:
                if idx >= available_devices:
                    print(f"Warning: RealSense device index {idx} not available (only {available_devices} devices found)")
                    continue
                try:
                    print(f"Initializing realsense_{idx}")
                    cam = RealsenseCamera(index=idx)
                    self._cams.append((f"realsense_{idx}", cam))
                except Exception as e:
                    print(f"Failed to initialize realsense_{idx}: {e}")

        for idx in orbbec_serials or []:
            try:
                print(f"Initializing orbbec_{idx}")
                cam = OrbbecCamera(index=idx)
                self._cams.append((f"orbbec_{idx}", cam))
            except Exception as e:
                print(f"Failed to initialize orbbec_{idx}: {e}")

        self._arm: PiperArm | None = PiperArm(arm_ip) if arm_ip else None

    def update_obs_window(self) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
        """
        返回：
        - frames: dict，相机名 -> 图像（BGR）
        - state: dict，包含 "qpos" 和 "eef_pose"，若未连接机械臂则为 None
        """
        frames: Dict[str, np.ndarray] = {}
        for name, cam in self._cams:
            img = cam.get_one_color_frame()
            if img is not None:
                frames[name] = img.copy()
                # cv2.imwrite(f"./{name}_latest.jpg", img)

        if self._arm:
            state = {
                "qpos": self._arm.get_state(),        # shape: (14,)
                "eef_pose": self._arm.get_eef_pose()  # shape: (14,)
            }
        else:
            state = None

        return frames, state


    def control(self, action: List[float] | np.ndarray, wait: bool = True):
        if not self._arm:
            raise RuntimeError("Arm not initialised; pass arm_ip when constructing RobotEnv.")
        self._arm.execute(action, wait=wait)

    def control_eef(self, action, wait=True):
        if not self._arm:
            raise RuntimeError("Arm not initialised; pass arm_ip when constructing RobotEnv.")
        self._arm.execute_eef(action, wait=wait)

    def shutdown(self):
        for _name, cam in self._cams:
            cam.stop()

if __name__ == "__main__":
    # Example main function entry
    orbbec_serials = [0, 1, 2]  # Replace with your actual device serial numbers
    env = RobotEnv(
        realsense_serials=[0,1,2],
        orbbec_serials=None,
        arm_ip="can0+can1"
    )

    try:
        while True:
            frames, state = env.update_obs_window()
            for name in frames:
                print(f"[Frame] {name}: shape={frames[name].shape}")
            if state is not None:
                print(f"[Arm State] {state.round(3)}")
            time.sleep(0.1)
    except KeyboardInterrupt:
        print("\n[Main] Interrupted by user.")
    finally:
        env.shutdown()
        print("[Main] RobotEnv shut down successfully.")