import os
import logging
import json
import numpy as np
import cv2
import h5py
from tqdm import tqdm
import jsonlines
from typing import Tuple, List, Dict, Any, Union
import sys

from .data_utils.pose_transform import euler_to_6d, compute_d6_axis_angle_deltas
from .action_token.action_chunk_to_fast_token import ActionChunkProcessor

FRAME_SAMPLE_INTERVAL = os.getenv("FRAME_SAMPLE_INTERVAL", 3)
ACTION_SAMPLE_INTERVAL = os.getenv("ACTION_SAMPLE_INTERVAL", 1)
PADDING = os.getenv("PADDING", 0)
DATA_VERSION = os.getenv("DATA_VERSION", 'data_test')
FRAME_SAMPLE_INTERVAL = int(FRAME_SAMPLE_INTERVAL)
ACTION_SAMPLE_INTERVAL = int(ACTION_SAMPLE_INTERVAL)
PADDING = int(PADDING)

_TOKENIZER_CACHE: dict[int, ActionChunkProcessor] = {}
def get_tokenizer(max_len: int) -> ActionChunkProcessor:
    """Return a cached ActionChunkProcessor (one per process).

    每个 Ray worker 进程各自维护 _TOKENIZER_CACHE，首次调用时才实例化。
    """
    tok = _TOKENIZER_CACHE.get(max_len)
    if tok is None:
        tok = ActionChunkProcessor(max_len=max_len)
        _TOKENIZER_CACHE[max_len] = tok
        logger.debug("Tokenizer initialised in PID %s (max_len=%s)", os.getpid(), max_len)
    return tok

# ----------------------------------------------------------------------------
# logging config
# ----------------------------------------------------------------------------

logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(message)s",
    level=logging.INFO,
)

logger = logging.getLogger(__name__)


min_gripper_value = {
    'company': [-0.00203000009059906, -0.0006300000241026282],
    'company2nd': [-0.0016799999866634607, -0.0028699999675154686]
}

def get_libero_dummy_action():
    """Get dummy/no-op action, used to roll out the simulation while the robot does nothing."""
    return np.array([0, 0, 0, 0, 0, 0, -1])

def find_last_duplicate_start(lst):
    if not lst:  # 处理空列表情况
        return -1
    
    last_element = lst[-1]
    # 从倒数第二个元素开始向前遍历
    for i in range(len(lst)-2, -1, -1):
        if lst[i] != last_element:
            # 找到第一个不同的元素，返回下一个索引
            if i + 1 == len(lst) - 1:
                return 0
            else:
                return i + 1
    return 0

class TrajectoryProcessor:
    def __init__(self, args):
        """初始化轨迹处理器
        
        Args:
            args: 包含各种配置参数的对象
        """
        self.args = args
        self._load_normalization_parameters()
        # self.action_tokenizer = get_tokenizer(self.args.max_len)  # 假设get_tokenizer已定义
        # 可以在这里初始化其他需要的属性

    def _load_normalization_parameters(self):
        """加载归一化参数"""
        with open(self.args.normal_path, 'r', encoding='utf-8') as f:
            norm_para = json.load(f)
        
        self.action_eepose_scale = np.array(norm_para["action.eepose"]['scale_']) # delta
        self.action_eepose_offset = np.array(norm_para["action.eepose"]['offset_'])
        self.action_qpos_scale = np.array(norm_para["action.qpos"]['scale_']) # delta
        self.action_qpos_offset = np.array(norm_para["action.qpos"]['offset_'])


    def json_fill(self, 
                raw_task: str, 
                task: str,
                image_path_list: List[str], 
                action_eepose_tokenizer_path, 
                state_eepose_path, 
                action_eepose_path,
                action_qpos_tokenizer_path,
                state_qpos_path,
                action_qpos_path
            ):
        """生成JSON条目
        
        Args:
            lan: 自然语言描述
            image_path_list: 图像路径列表
            action_token_path: 动作令牌路径
            
        Returns:
            生成的JSON字典
        """
        action_str_list = ['<action_token>'] * 1
        action_str = '<action_split>'.join(action_str_list)
        json_item = {
                "raw_task": raw_task,
                "task": task,
                "image": image_path_list,
                "action_eepose_token": action_eepose_tokenizer_path,
                "action_qpos_token": action_qpos_tokenizer_path,
                "state":{
                    "eepose": state_eepose_path,
                    "qpos": state_qpos_path,
                },
                "action":{
                    "eepose": action_eepose_path,
                    "qpos": action_qpos_path,
                },
                "conversations": [
                    {
                        "from": "human",
                        "value": f"According to the robot front image<image>, robot right wrist image<image> and robot left wrist image<image>, what action should the robot take to complete: {task}."
                    },
                    {
                        "from": "gpt",
                        "value": action_str
                    }
                ]
            }
        return json_item

    def transform(self, x: np.array, scale: np.array, offset: np.array, clip: bool = True) -> np.array:
        """数据转换（归一化）
        
        Args:
            x: 原始数据
            scale: 缩放因子
            offset: 偏移量
            clip: 是否裁剪到[-1, 1]范围
            
        Returns:
            转换后的数据
        """
        x_norm = x * scale + offset
        if clip:
            np.clip(x_norm, -1, 1, out=x_norm)  
        return x_norm

    def inverse_transform(self, x_norm: np.array, scale: np.array, offset: np.array) -> np.array:
        """逆转换（从归一化数据恢复原始数据）
        
        Args:
            x_norm: 归一化后的数据
            scale: 缩放因子
            offset: 偏移量
            
        Returns:
            原始数据
        """
        x_norm = np.asarray(x_norm)
        return (x_norm - offset) / scale

    def save_video_frames(
        self,
        video_paths: Union[str, List[str]],
        output_dir: str,
        start_frame: int = 0,
        end_frame: int = None,
        image_format: str = "jpg",
    ) -> Tuple[int, List[str]]:
        """Save frames of multiple videos and return number of frames written.

        Returns
        -------
        Tuple[int, List[str]]
            (frames_written, list_of_paths)
        """

        for video_path in video_paths:
            view_name = (
                f"{video_path.split('/')[-1].split('.')[0].split('_')[-2]}_"
                f"{video_path.split('/')[-1].split('.')[0].split('_')[-1]}"
            )

            if not os.path.exists(video_path):
                logger.warning("Video path does not exist and will be skipped: %s", video_path)
                return 0

            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                logger.warning("Unable to open video and will be skipped: %s", video_path)
                return 0
            
            cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

            for frame_idx in range(start_frame, end_frame + 1):
                ret, frame = cap.read()
                if not ret: break

                filename = f"{view_name}_{frame_idx}.{image_format}"
                output_path = os.path.join(output_dir, filename)
                if not os.path.exists(output_path):
                    resized_frame = cv2.resize(frame, (320, 240), interpolation=cv2.INTER_AREA)
                    cv2.imwrite(output_path, resized_frame)
            cap.release()

        return 1

    def compute_eepose_delta(self, chunk):
        # 计算间隔为ACTION_SAMPLE_INTERVAL的动作差值
        # 原始数组 shape: (91, 14)，间隔采样后每组包含30个样本点

        # 采样参数
        step = ACTION_SAMPLE_INTERVAL  # 间隔步长=3

        # 第一部分：前3列的差值计算
        # 采样范围: [3, 6, 9, ..., 90] 减去 [0, 3, 6, ..., 87]
        part1_start = chunk[step::step, :3]  # 从索引3开始，间隔3采样
        part1_end = chunk[:-step:step, :3]   # 从索引0开始到87，间隔3采样
        part1 = part1_start - part1_end

        # 第二部分：3-6列的角度差值计算（使用6D表示）
        # 采样范围: [0, 3, 6, ..., 90]（完整间隔采样）
        euler_angles = chunk[::step, 3:6]  # 从索引0开始，间隔3采样
        d6_angles = euler_to_6d(euler_angles)     # 转换为6D表示
        part2 = compute_d6_axis_angle_deltas(d6_angles)

        # 第三部分：第6列的直接采样（索引6）
        # 采样范围: [3, 6, 9, ..., 90]
        part3 = chunk[step::step, [6]]

        # 第四部分：7-10列的差值计算
        # 采样范围: [3, 6, 9, ..., 90] 减去 [0, 3, 6, ..., 87]
        part4_start = chunk[step::step, 7:10]
        part4_end = chunk[:-step:step, 7:10]
        part4 = part4_start - part4_end

        # 第五部分：10-13列的角度差值计算（使用6D表示）
        # 采样范围: [0, 3, 6, ..., 90]（完整间隔采样）
        euler_angles_2 = chunk[::step, 10:13]
        d6_angles_2 = euler_to_6d(euler_angles_2)
        part5 = compute_d6_axis_angle_deltas(d6_angles_2)

        # 第六部分：第13列的直接采样（索引13）
        # 采样范围: [3, 6, 9, ..., 90]
        part6 = chunk[step::step, [13]]

        # 拼接所有部分，得到最终的差值数组
        delta = np.concatenate(
            [part1, part2, part3, part4, part5, part6],
            axis=-1
        )
        return delta

    def compute_qpos_delta(self, chunk):
        # 计算间隔为ACTION_SAMPLE_INTERVAL的动作差值
        # 原始数组 shape: (91, 14)，间隔采样后每组包含30个样本点

        # 采样参数
        step = ACTION_SAMPLE_INTERVAL  # 间隔步长=3

        # 第一部分：前6列的差值计算
        # 采样范围: [3, 6, 9, ..., 90] 减去 [0, 3, 6, ..., 87]
        part1_start = chunk[step::step, :6]  # 从索引3开始，间隔3采样
        part1_end = chunk[:-step:step, :6]   # 从索引0开始到87，间隔3采样
        part1 = part1_start - part1_end

        # 第三部分：第6列的直接采样（索引6）
        # 采样范围: [3, 6, 9, ..., 90]
        part3 = chunk[step::step, [6]]

        # 第四部分：7-13列的差值计算
        # 采样范围: [3, 6, 9, ..., 90] 减去 [0, 3, 6, ..., 87]
        part4_start = chunk[step::step, 7:13]
        part4_end = chunk[:-step:step, 7:13]
        part4 = part4_start - part4_end

        # 第六部分：第13列的直接采样（索引13）
        # 采样范围: [3, 6, 9, ..., 90]
        part6 = chunk[step::step, [13]]

        # 拼接所有部分，得到最终的差值数组
        delta = np.concatenate(
            [part1, part3, part4, part6],
            axis=-1
        )
        return delta

    def compute_delta_v2(self, chunk):
        # 计算间隔为ACTION_SAMPLE_INTERVAL的动作差值
        # 原始数组 shape: (91, 14)，间隔采样后每组包含30个样本点

        # 采样参数
        step = ACTION_SAMPLE_INTERVAL  # 间隔步长=3

        delta_1step = np.concatenate(
            [
                (chunk[1:, :3] - chunk[:-1, :3]),
                compute_d6_axis_angle_deltas(euler_to_6d(chunk[:, 3:6])),
                chunk[1:, [6]],
                (chunk[1:, 7:10] - chunk[:-1, 7:10]),
                compute_d6_axis_angle_deltas(euler_to_6d(chunk[:, 10:13])),
                chunk[1:, [13]],
            ],
            axis=-1,
        )
        # 拼接所有部分，得到最终的差值数组
        N, M = delta_1step.shape
        # 计算结果数组的行数
        result_rows = (N + ACTION_SAMPLE_INTERVAL - 1) // ACTION_SAMPLE_INTERVAL  # 等同于向上取整除法
        # 初始化结果数组
        delta = np.zeros((result_rows, M))
        # 遍历数组，步长为3
        for i in range(result_rows):
            # 计算当前组的起始和结束索引
            start = i * ACTION_SAMPLE_INTERVAL
            end = start + ACTION_SAMPLE_INTERVAL
            # 对当前组的行进行求和
            delta[i] = delta_1step[start:end].sum(axis=0)
            delta[i][6], delta[i][13] = delta_1step[end-1][6], delta_1step[end-1][13]
        return np.array(delta)

    def process_single_item(self, task_info, sub_dir_path, sub_dir_path_image):
        """Process a single trajectory.

        The function is deliberately defensive: *every* IO operation is protected so that an
        isolated failure does not crash the entire job.
        """
        action_tokenizer = get_tokenizer(self.args.max_len)
        for item in tqdm(task_info[2]):
            try:
                self.total_traj += 1
                # ---------------------------------------------------------------------
                # directory setup
                # ---------------------------------------------------------------------
                front_mp4_path = item["data"]["high"]
                split_list = front_mp4_path.split("/")
                uuid = f"{split_list[-5]}_{split_list[-3]}_{split_list[-2]}"

                images_path = os.path.join(sub_dir_path_image, "images", uuid)
                action_token_path = os.path.join(sub_dir_path, "action_token", uuid)
                os.makedirs(images_path, exist_ok=True)
                os.makedirs(action_token_path, exist_ok=True)
                
                raw_task = item["raw_task"]
                task = item["task"]
                start = item["frame"]["start_frame"]
                end = item["frame"]["end_frame"]
                # ------------------------------------------------------------------
                # extract and write frames
                # ------------------------------------------------------------------
                left_wrist_mp4_path = item["data"]["left"]
                right_wrist_mp4_path = item["data"]["right"]

                result = self.save_video_frames(
                    video_paths=[front_mp4_path, left_wrist_mp4_path, right_wrist_mp4_path],
                    output_dir=images_path,
                    start_frame=start-1,
                    end_frame=end,
                    image_format="jpg",
                )

                if not result:
                    logger.warning("No frames were written for item %s – skipping.", uuid)
                    return []

                # ------------------------------------------------------------------
                # load state/action from HDF5
                # ------------------------------------------------------------------
                hdf5_path = item["data"]["state"]
                with h5py.File(hdf5_path, "r") as f:
                    if len(f.keys()) != 2:
                        logger.warning("State file has unexpected number of keys (%s) – skipping item %s", len(f.keys()), uuid)
                        return []
                    action = f["action"][:]
                    qpos = f["qpos"][:]
                # 消除夹爪偏移量
                action[:, 6] -= min_gripper_value[task_info[0]][0]
                action[:, 13] -= min_gripper_value[task_info[0]][1]

                qpos[:, 6] -= min_gripper_value[task_info[0]][0]
                qpos[:, 13] -= min_gripper_value[task_info[0]][1]
                # 过滤相邻不变的动作并记录原始索引
                filtered_action = []
                filtered_qpos = []
                original_indices = []  # 记录过滤后动作对应的原始索引

                # 保留第一帧
                # 基于过滤后的数据重新计算循环范围
                loop_start = max(0, start - 1)
                loop_end = max(loop_start, end)  # ensure non-negative

                self.original_sample_num += loop_end - loop_start
                filtered_action.append(action[loop_start])
                filtered_qpos.append(qpos[loop_start])
                original_indices.append(loop_start) 
                for i in range(loop_start+1, loop_end):
                    # 检查动作是否有变化
                    action_changed = not (np.array_equal(action[i], action[i-1]))
                    
                    if action_changed:
                        filtered_action.append(action[i])
                        filtered_qpos.append(qpos[i])
                        original_indices.append(i)  # 记录变化帧的原始索引

                # 转换为numpy数组
                filtered_action = np.array(filtered_action)
                filtered_qpos = np.array(filtered_qpos)
                original_indices = np.array(original_indices)

                self.wo_static_sample_num += len(filtered_action)
                # 更新结束索引为过滤后的长度
                data_end = len(filtered_action)
                if data_end == 0:
                    logger.warning("No valid action data after filtering for item %s", uuid)
                    return []

                # ------------------------------------------------------------------
                # Build JSON entries
                # ------------------------------------------------------------------
                json_entries: List[dict] = []
                if data_end < 0.5*self.args.chunk*ACTION_SAMPLE_INTERVAL:
                    construct_num = 0
                    self.filtered_traj += 1
                else:
                    if self.args.padding is None:
                        construct_num = max(1, data_end-self.args.chunk*ACTION_SAMPLE_INTERVAL)
                    else:
                        construct_num = max(1, data_end-self.args.chunk*ACTION_SAMPLE_INTERVAL+self.args.padding)


                for i in range(construct_num):
                    # 选择分块索引 - 用min确保不超出边界
                    index = [min(i + j, data_end - 1) for j in range(self.args.chunk*ACTION_SAMPLE_INTERVAL + 1)]

                    # 获取过滤后的数据块
                    action_chunk = filtered_action[index]
                    qpos_chunk = filtered_qpos[index]
                    # 获取对应的原始索引，用于匹配图像
                    original_index_chunk = original_indices[index]

                    try:
                        action_delta = self.compute_eepose_delta(action_chunk)
                        qpos_delta = self.compute_qpos_delta(qpos_chunk)
                        # 先逐项做差再求和有误差
                    except Exception as exc:
                        logger.error("Failed to build action delta for item %s/%s – %s", uuid, i, exc)
                        continue
                    
                    action_chunk = action_chunk[:-ACTION_SAMPLE_INTERVAL:ACTION_SAMPLE_INTERVAL]
                    action_chunk = np.concatenate([
                        action_chunk[:, :3],
                        euler_to_6d(action_chunk[:, 3:6]),
                        action_chunk[:, [6]],
                        action_chunk[:, 7:10],
                        euler_to_6d(action_chunk[:, 10:13]),
                        action_chunk[:, [13]]
                    ], axis=-1)
                    assert action_chunk.shape[1] == 20
                    qpos_chunk = qpos_chunk[:-ACTION_SAMPLE_INTERVAL:ACTION_SAMPLE_INTERVAL]

                    nor_action_eepose = self.transform(action_delta, self.action_eepose_scale, self.action_eepose_offset)
                    nor_action_qpos = self.transform(qpos_delta, self.action_qpos_scale, self.action_qpos_offset)
                    # 保持0为关闭，大于1为打开
                    base_idx = original_indices[i]
                    action_eepose_tokenizer_path = os.path.join(action_token_path, f"action_eepose_token_{base_idx}.npy")
                    state_eepose_path = os.path.join(action_token_path, f"state_eepose_{base_idx}.npy")
                    action_eepose_path = os.path.join(action_token_path, f"action_eepose_{base_idx}.npy")

                    state_qpos_path = os.path.join(action_token_path, f"state_qpos_{base_idx}.npy")
                    action_qpos_path = os.path.join(action_token_path, f"action_qpos_{base_idx}.npy")
                    action_qpos_tokenizer_path = os.path.join(action_token_path, f"action_qpos_token_{base_idx}.npy")
                    # 保存原始索引，用于后续验证或处理
                    action_eepose_token = action_tokenizer.process_action_chunk_to_fast_token(nor_action_eepose)
                    action_qpos_token = action_tokenizer.process_action_chunk_to_fast_token(nor_action_qpos)
                    # eepose
                    np.save(action_eepose_tokenizer_path, action_eepose_token)
                    np.save(state_eepose_path, action_chunk)
                    np.save(action_eepose_path, action_delta)
                    # joint
                    np.save(action_qpos_tokenizer_path, action_qpos_token)
                    np.save(state_qpos_path, qpos_chunk)
                    np.save(action_qpos_path, qpos_delta)
                    
                    image_path_list = [
                        os.path.join(images_path, f"{view}_{base_idx}.jpg")
                        for view in ["cam_high", "right_wrist", "left_wrist"]
                    ]

                    json_item = self.json_fill(
                        raw_task, 
                        task, 
                        image_path_list, 
                        action_eepose_tokenizer_path, 
                        state_eepose_path, 
                        action_eepose_path,
                        action_qpos_tokenizer_path,
                        state_qpos_path,
                        action_qpos_path,
                    )
                    json_entries.append(json_item)
                    #间隔采样，每隔3个取1个，保证10Hz

                try:
                    if json_entries[-1] in json_entries[0::FRAME_SAMPLE_INTERVAL]:
                        sampled_entries = json_entries[0::FRAME_SAMPLE_INTERVAL]
                    else:
                        sampled_entries = json_entries[0::FRAME_SAMPLE_INTERVAL]
                        sampled_entries.append(json_entries[-1])

                    for json_item in sampled_entries:
                        json_line = json.dumps(json_item)
                        self.json_file_writer.write(json_line+'\n')

                except:
                    print(f"过滤静止动作{data_end} < {0.5*self.args.chunk*ACTION_SAMPLE_INTERVAL}，跳过！")

            except Exception as exc:
                # 任何未处理的异常 - 报告并跳过此项目
                print(f"处理项目时发生异常 - {exc}")
                return []

    def process_task(self):
        self.filtered_traj = 0
        self.wo_static_sample_num = 0
        self.total_traj = 0
        self.original_sample_num = 0
        for task_i, task_info in enumerate(self.all_tasks):
            if task_i >= self.args.task_start_idx and task_i < self.args.task_end_idx:
                write_file_path1 = f"target_dtaa_path/{DATA_VERSION}/agilex_train_{task_info[0].split('.json')[0]}_{task_info[1].replace(' ', '_')}.jsonl"
                write_file_path2 = f"target_dtaa_path/{DATA_VERSION}_subtask/agilex_train_{task_info[0].split('.json')[0]}_{task_info[1].replace(' ', '_')}.jsonl"

                print(task_i)
                print(task_info[0]+'_'+task_info[1].replace(' ', '_'))
                sub_dir_path = os.path.join(self.args.root_path, task_info[0].split('.json')[0]+'_'+task_info[1].replace(' ', '_'))
                sub_dir_path_image = os.path.join(self.args.image_path,  task_info[0].split('.json')[0]+'_'+task_info[1].replace(' ', '_'))
                os.makedirs(sub_dir_path, exist_ok=True)
                self.json_file_writer = open(write_file_path1, 'w')
                self.process_single_item(task_info, sub_dir_path, sub_dir_path_image)
                self.json_file_writer.close()
                print('####')
        print(f"original_sample_num: {self.original_sample_num}\nwo_static_sample: {self.wo_static_sample_num}")
        print(f"total_traj: {self.total_traj}\nfiltered_traj: {self.filtered_traj}\n")
        print("Finshed")

    def run(self):
        """运行主处理流程"""
        json_dict: List[dict] = []
        os.makedirs(f"target_data_path/{DATA_VERSION}", exist_ok=True)
        self.all_tasks = []
        with open("RoboBrain-X0-Dataset/tree/main/agilex_company_annotations/agilex_company_annotations.json", "r") as f:
            tasks = json.load(f)
            self.all_tasks += [(data_source, _, tasks[_]) for _ in tasks]
        self.process_task()
        
       
# 使用示例
if __name__ == "__main__":
    # 这里通常会解析命令行参数
    import argparse
    
    parser = argparse.ArgumentParser(description="轨迹数据处理器参数")
    # 添加必要的命令行参数
    parser.add_argument("--root_path", type=str, default=f"target_data_path/{DATA_VERSION}/agilex_train_data_twin")
    parser.add_argument("--image_path", type=str, default=f"target_data_path/{DATA_VERSION}/agilex_train_data_twin")
    parser.add_argument("--normal_path", type=str, default="data_process/agilex/normal_stats_agilex_30Hz.json")
    parser.add_argument("--max_len", type=int, default=256)
    parser.add_argument("--padding", type=int, default=None)
    parser.add_argument("--chunk", type=int, default=30)
    parser.add_argument("--action", type=int, default=1)
    parser.add_argument("--task_start_idx", type=int, default=0)
    parser.add_argument("--task_end_idx", type=int, default=320)
    
    args = parser.parse_args()
    
    # 初始化并运行处理器
    processor = TrajectoryProcessor(args)
    processor.run()
