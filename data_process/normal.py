
# author: Chenghy
# date: 2025.09.02

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
import json
import h5py
import os
from tqdm import tqdm
import tap
import matplotlib.pyplot as plt
import sys
from typing import List

from .data_utils.pose_transform import euler_to_6d, compute_d6_axis_angle_deltas

ACTION_SAMPLE_INTERVAL = os.getenv("ACTION_SAMPLE_INTERVAL", 1)
ACTION_SAMPLE_INTERVAL = int(ACTION_SAMPLE_INTERVAL)

class Normalize_Arg(tap.Tap):
    data_json_path: str = "RoboBrain-X0-Dataset/tree/main/agilex_company_annotations/agilex_company_annotations.json" # agilex
    output_fig_path: str = "data_process/analysis_plots_agilex_all"
    normal_path: str = "data_process/agilex/normal_stats_agilex_30Hz.json"
    plot_variables: List[str] = [
        "action_eepose",
        "action_qpos",
        "state_eepose",
        "state_qpos",
    ]

match_dict = {
    "company":["RoboBrain_Robotic_AGXCompany", "RoboBrain_Robotic_AGXCompany2"],
    "company2nd": ["RoboBrain_Robotic_Agilex_2nd"]
}

# 创建一个反向映射，从source到key
source_to_key = {}
for key, sources in match_dict.items():
    for source in sources:
        source_to_key[source] = key

min_gripper_value = {
    'company': [-0.00203000009059906, -0.0006300000241026282],
    'company2nd': [-0.0016799999866634607, -0.0028699999675154686]
}



class AdvancedQuantileNormalizer(BaseEstimator, TransformerMixin):
    def __init__(self, lower_quantile=0.01, upper_quantile=0.99, 
                 target_range=(-1, 1), clip=True):
        """
        增强版分位数归一化器
        
        参数:
            lower_quantile: 下分位数(默认1%)
            upper_quantile: 上分位数(默认99%)
            target_range: 目标范围元组(默认[-1, 1])
            clip: 是否将超出范围的值裁剪到边界(默认True)
        """
        self.lower_quantile = lower_quantile
        self.upper_quantile = upper_quantile
        self.target_min, self.target_max = target_range
        self.clip = clip
        self.quantiles_low_ = None
        self.quantiles_high_ = None
        self.scale_ = None
        self.offset_ = None

    def fit(self, X, y=None):
        """计算各维度的分位数和缩放参数"""
        X = np.asarray(X)
        self.quantiles_low_ = np.quantile(X, self.lower_quantile, axis=0)
        self.quantiles_high_ = np.quantile(X, self.upper_quantile, axis=0)
        
        # 计算缩放参数
        self.scale_ = (self.target_max - self.target_min) / (
            self.quantiles_high_ - self.quantiles_low_ + 1e-8)  # 避免除零
        self.offset_ = self.target_min - self.quantiles_low_ * self.scale_
        
        return {
            "quantiles_low_": self.quantiles_low_,
            "quantiles_high_": self.quantiles_high_,
            "scale_": self.scale_,
            "offset_": self.offset_
        }

    def transform(self, X):
        """应用归一化"""
        X = np.asarray(X)
        X_norm = X * self.scale_ + self.offset_
        
        if self.clip:
            np.clip(X_norm, self.target_min, self.target_max, out=X_norm)
            
        return X_norm

    def inverse_transform(self, X_norm):
        """反归一化"""
        X_norm = np.asarray(X_norm)
        return (X_norm - self.offset_) / self.scale_

    def get_feature_names(self):
        """获取特征名称(用于pipeline)"""
        return [f"norm_dim_{i}" for i in range(len(self.quantiles_low_))]

def _to_serialisable(obj):
    """Convert numpy types to JSON-serialisable Python types."""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    raise TypeError(f"Type {type(obj)} not serialisable")

def _get_stats(data, name):
    """计算数据的统计信息"""
    if data.size == 0:
        return f"{name}: 数据为空"
    
    stats = {
        "mean": np.mean(data),
        "std": np.std(data),
        "min": np.min(data),
        "max": np.max(data),
        "q1": np.quantile(data, 0.01),
        "q99": np.quantile(data, 0.99),
        "num_samples": len(data)
    }
    
    return (f"{name}统计信息:\n"
            f"  - 均值: {stats['mean']:.6f}\n"
            f"  - 标准差: {stats['std']:.6f}\n"
            f"  - 最小值: {stats['min']:.6f}\n"
            f"  - 最大值: {stats['max']:.6f}\n"
            f"  - 1%分位数: {stats['q1']:.6f}\n"
            f"  - 99%分位数: {stats['q99']:.6f}\n"
            f"  - 样本数量: {stats['num_samples']}")

def _four_panel_plot(data_right, data_left, title_base, x_label, output_dir, stats_file):
    """生成一张大图(2x2)：右手含异常、右手去异常、左手含异常、左手去异常。"""
    if data_right.size == 0 and data_left.size == 0:
        print(f"{title_base}: 数据为空，跳过绘图。")
        return

    os.makedirs(output_dir, exist_ok=True)

    # 计算并保存统计信息
    stats_info = []
    stats_info.append(f"\n=== {title_base} ===")
    stats_info.append(_get_stats(data_right, "右手原始数据"))
    stats_info.append(_get_stats(data_left, "左手原始数据"))

    # 分位数裁剪
    def remove_outliers(x):
        if x.size == 0:
            return x
        low = np.quantile(x, 0.01)
        high = np.quantile(x, 0.99)
        return x[(x >= low) & (x <= high)]

    right_no = remove_outliers(data_right)
    left_no = remove_outliers(data_left)

    stats_info.append(_get_stats(right_no, "右手去除异常后数据"))
    stats_info.append(_get_stats(left_no, "左手去除异常后数据"))

    # 写入统计信息
    with open(stats_file, "a", encoding="utf-8") as f:
        f.write("\n".join(stats_info) + "\n")

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    ((ax_r_with, ax_r_no), (ax_l_with, ax_l_no)) = axes

    # 右手-含异常
    if data_right.size > 0:
        ax_r_with.hist(data_right, bins=100, color="#1f77b4", density=True)
        ax_r_with.axvline(np.mean(data_right), color='r', linestyle='dashed', linewidth=1)
    ax_r_with.set_title("Right (with outliers)")
    ax_r_with.set_xlabel(x_label)
    ax_r_with.set_ylabel("density")

    # 右手-去异常
    if right_no.size > 0:
        ax_r_no.hist(right_no, bins=100, color="#1f77b4", density=True)
        ax_r_no.axvline(np.mean(right_no), color='r', linestyle='dashed', linewidth=1)
    ax_r_no.set_title("Right (without outliers)")
    ax_r_no.set_xlabel(x_label)
    ax_r_no.set_ylabel("density")

    # 左手-含异常
    if data_left.size > 0:
        ax_l_with.hist(data_left, bins=100, color="#ff7f0e", density=True)
        ax_l_with.axvline(np.mean(data_left), color='r', linestyle='dashed', linewidth=1)
    ax_l_with.set_title("Left (with outliers)")
    ax_l_with.set_xlabel(x_label)
    ax_l_with.set_ylabel("density")

    # 左手-去异常
    if left_no.size > 0:
        ax_l_no.hist(left_no, bins=100, color="#ff7f0e", density=True)
        ax_l_no.axvline(np.mean(left_no), color='r', linestyle='dashed', linewidth=1)
    ax_l_no.set_title("Left (without outliers)")
    ax_l_no.set_xlabel(x_label)
    ax_l_no.set_ylabel("density")

    plt.suptitle(title_base)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plot_path = os.path.join(output_dir, f"{title_base.replace(' ', '_')}.png")
    plt.savefig(plot_path)
    plt.close(fig)
    print(f"分布图已保存至: {plot_path}")


def plot_selected_variables(action_eepose_delta, action_qpos_delta, state_eepose_abs, state_qpos_abs, variables_to_plot, output_dir):
    """按变量类别绘制：每个物理量一张大图(2x2)。
    variables_to_plot 支持: ["action_eepose", "action_qpos", "state_eepose", "state_qpos"]
    """
    os.makedirs(output_dir, exist_ok=True)
    stats_file = os.path.join(output_dir, "statistics.txt")
    
    # 清空统计文件
    with open(stats_file, "w", encoding="utf-8") as f:
        f.write("=== 数据统计信息 ===\n")

    def plot_action_eepose_xyz_gripper():
        if action_eepose_delta.size == 0:
            print("action_eepose: 数据为空，跳过绘图。")
            return
        
        # 位置增量
        mapping = {
            "delta_x": (0, 7),
            "delta_y": (1, 8),
            "delta_z": (2, 9),
        }
        for name, (idx_r, idx_l) in mapping.items():
            right = action_eepose_delta[:, idx_r]
            left = action_eepose_delta[:, idx_l]
            _four_panel_plot(right, left, f"action_eepose_{name}", name, output_dir, stats_file)
            
        # 6D旋转增量
        for i in range(3):
            right = action_eepose_delta[:, 3+i]
            left = action_eepose_delta[:, 10+i]
            _four_panel_plot(right, left, f"action_eepose_rotation_axis{i+1}", f"rotation_axis{i+1}", output_dir, stats_file)
            
        # 夹爪
        right = action_eepose_delta[:, 6]
        left = action_eepose_delta[:, 13]
        _four_panel_plot(right, left, "action_eepose_gripper", "gripper", output_dir, stats_file)

    def plot_state_eepose_xyz_gripper():
        if state_eepose_abs.size == 0:
            print("state_eepose: 数据为空，跳过绘图。")
            return
            
        # 位置
        mapping = {
            "x": (0, 10),
            "y": (1, 11),
            "z": (2, 12),
        }
        for name, (idx_r, idx_l) in mapping.items():
            right = state_eepose_abs[:, idx_r]
            left = state_eepose_abs[:, idx_l]
            _four_panel_plot(right, left, f"state_eepose_{name}", name, output_dir, stats_file)
            
        # 6D旋转
        for i in range(6):
            right = state_eepose_abs[:, 3+i]
            left = state_eepose_abs[:, 13+i]
            _four_panel_plot(right, left, f"state_eepose_6d_rotation_{i+1}", f"6d_rotation_{i+1}", output_dir, stats_file)
            
        # 夹爪
        right = state_eepose_abs[:, 9]
        left = state_eepose_abs[:, 19]
        _four_panel_plot(right, left, "state_eepose_gripper", "gripper", output_dir, stats_file)

    def plot_action_qpos_joints_gripper():
        if action_qpos_delta.size == 0:
            print("action_qpos: 数据为空，跳过绘图。")
            return
        # 关节增量
        for j in range(6):
            right = action_qpos_delta[:, j]
            left = action_qpos_delta[:, 7 + j]
            _four_panel_plot(right, left, f"action_qpos_joint_{j+1}", f"joint_{j+1}", output_dir, stats_file)
        # 夹爪
        _four_panel_plot(action_qpos_delta[:, 6], action_qpos_delta[:, 13], "action_qpos_gripper", "gripper", output_dir, stats_file)

    def plot_state_qpos_joints_gripper():
        if state_qpos_abs.size == 0:
            print("state_qpos: 数据为空，跳过绘图。")
            return
        # 关节位置
        for j in range(6):
            right = state_qpos_abs[:, j]
            left = state_qpos_abs[:, 7 + j]
            _four_panel_plot(right, left, f"state_qpos_joint_{j+1}", f"joint_{j+1}", output_dir, stats_file)
        # 夹爪
        _four_panel_plot(state_qpos_abs[:, 6], state_qpos_abs[:, 13], "state_qpos_gripper", "gripper", output_dir, stats_file)

    for var in variables_to_plot:
        if var == "action_eepose":
            plot_action_eepose_xyz_gripper()
        elif var == "state_eepose":
            plot_state_eepose_xyz_gripper()
        elif var == "action_qpos":
            plot_action_qpos_joints_gripper()
        elif var == "state_qpos":
            plot_state_qpos_joints_gripper()
        else:
            print(f"未知变量: {var}，已跳过。")


# 创建归一化器
normalizer = AdvancedQuantileNormalizer(
    lower_quantile=0.01,
    upper_quantile=0.99,
    target_range=(-1, 1)
)


error_list = []

def compute_eepose_delta(chunk):
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

def compute_eepose_delta_v2(chunk):
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

def compute_qpos_delta(chunk):
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

def process_data(data):
    """处理原始数据，返回处理后的数组和错误统计"""
    error_count = {
        "empty_file": 0,
        "nan_values": 0,
        "total_processed": 0
    }
    action_qpos_list = []
    action_eepose_list = []
    state_eepose_list = []
    state_qpos_list = []

    for key in tqdm(data.keys(), desc="处理任务"):
        items = data[key]
        if not items:
            continue

        for item in tqdm(items, leave=False, desc="处理数据"):
            raw_source = item['data']['high'].split('/')[-5]
            source = source_to_key[raw_source]
            start, end = item['frame']['start_frame'], item['frame']['end_frame'] + 1
            hdf5_path = item['data']['state']
        
            with h5py.File(hdf5_path, 'r') as f:
                if len(f.keys()) != 2:
                    temp = {
                        "path": item['data']['state'],
                        "frame": (start, end),
                        "none": True
                    }
                    error_list.append(temp)
                    error_count["empty_file"] += 1
                    continue
                state_eepose = f["action"][start:end]
                state_qpos = f["qpos"][start:end]
                action_eepose = state_eepose.copy()  # 使用深拷贝
                action_qpos = state_qpos.copy()      # 使用深拷贝

            if np.isnan(state_eepose).any():
                temp = {
                    "path": item['data']['state'],
                    "frame": (start, end),
                    "nan": True
                }
                error_list.append(temp)
                error_count["nan_values"] += 1
                continue
                
            error_count["total_processed"] += 1

            # ---------------------state_eepose: 绝对量。-------------------
            # 处理夹爪。
            state_eepose[:, 6] -= min_gripper_value[source][0]
            state_eepose[:, 13] -= min_gripper_value[source][1]

            if source in ['robomind', 'huaihai']:
                state_eepose[:, 6] /= 100
                state_eepose[:, 13] /= 100
            elif source in ['mengdi', 'xinlong']:
                state_eepose[:, 6] /= 10
                state_eepose[:, 13] /= 10
            
            # 处理orientation。
            # 右手: pos(0:3) + rpy(3:6)->6D + gripper(6)
            # 左手: pos(7:10) + rpy(10:13)->6D + gripper(13)
            state_eepose = np.concatenate([
                state_eepose[:, :3],
                euler_to_6d(state_eepose[:, 3:6]),
                state_eepose[:, [6]],
                state_eepose[:, 7:10],
                euler_to_6d(state_eepose[:, 10:13]),
                state_eepose[:, [13]]
            ], axis=-1)
            assert state_eepose.shape[1] == 20
        
            state_eepose_list.extend(state_eepose[:-ACTION_SAMPLE_INTERVAL:ACTION_SAMPLE_INTERVAL])  # (episode_len -1, 20)

            # ---------------------------state_qpos：绝对量。只处理夹爪-----------------
            state_qpos[:, 6] -= min_gripper_value[source][0]
            state_qpos[:, 13] -= min_gripper_value[source][1]
            if source in ['robomind', 'huaihai']:
                state_qpos[:, 6] /= 100
                state_qpos[:, 13] /= 100
            elif source in ['mengdi', 'xinlong']:
                state_qpos[:, 6] /= 10
                state_qpos[:, 13] /= 10
            assert state_qpos.shape[1] == 14
            
            state_qpos_list.extend(state_qpos[:-ACTION_SAMPLE_INTERVAL:ACTION_SAMPLE_INTERVAL])

            # ------------------action_eepose: 相对量。-------------------
            # 1. 处理夹爪
            action_eepose[:, 6] -= min_gripper_value[source][0]
            action_eepose[:, 13] -= min_gripper_value[source][1]
            if source in ['robomind', 'huaihai']:
                action_eepose[:, 6] /= 100
                action_eepose[:, 13] /= 100
            elif source in ['mengdi', 'xinlong']:
                action_eepose[:, 6] /= 10
                action_eepose[:, 13] /= 10

            delta_action_eepose = compute_eepose_delta(action_eepose)
            assert delta_action_eepose.shape[1] == 14 
            action_eepose_list.extend(delta_action_eepose)   # (episode_len -1, 14)


            # -------------- action_qpos: 处理夹爪 + diff----------------
            action_qpos[:, 6] -= min_gripper_value[source][0]
            action_qpos[:, 13] -= min_gripper_value[source][1]
            if source in ['robomind', 'huaihai']:
                action_qpos[:, 6] /= 100
                action_qpos[:, 13] /= 100
            elif source in ['mengdi', 'xinlong']:
                action_qpos[:, 6] /= 10
                action_qpos[:, 13] /= 10
            
            delta_action_qpos = compute_qpos_delta(action_qpos)
            assert delta_action_qpos.shape[1] == 14
            action_qpos_list.extend(delta_action_qpos)

    # 转换为numpy数组
    processed_data = {
        "action_eepose": np.array(action_eepose_list),
        "action_qpos": np.array(action_qpos_list),
        "state_eepose": np.array(state_eepose_list),
        "state_qpos": np.array(state_qpos_list)
    }

    return processed_data, error_count

def compute_normalization(processed_data):
    """计算归一化参数"""
    result_dict = {}
    
    for key, data in processed_data.items():
        if data.size > 0:
            norm_params = normalizer.fit(data)
            result_dict[key.replace("_", ".")] = {k: _to_serialisable(v) for k, v in norm_params.items()}
        else:
            print(f"警告: {key}为空，跳过归一化拟合")
    
    return result_dict

def save_error_statistics(error_count, output_dir):
    """保存错误统计信息"""
    stats_file = os.path.join(output_dir, "error_statistics.txt")
    total_errors = error_count["empty_file"] + error_count["nan_values"]
    total_items = error_count["total_processed"] + total_errors
    
    with open(stats_file, "w", encoding="utf-8") as f:
        f.write("=== 数据处理错误统计 ===\n")
        f.write(f"成功处理的数据项: {error_count['total_processed']}\n")
        f.write(f"空文件错误: {error_count['empty_file']}\n")
        f.write(f"NaN值错误: {error_count['nan_values']}\n")
        f.write(f"总错误数: {total_errors}\n")
        f.write(f"错误率: {(total_errors / total_items * 100):.2f}%\n")
    
    print(f"错误统计信息已保存至: {stats_file}")

def main():
    """主函数"""
    args = Normalize_Arg()
    
    # 1. 读取数据
    print("读取数据...")
    with open(args.data_json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    # 2. 处理数据
    print("处理数据...")
    processed_data, error_count = process_data(data)
    
    # 3. 保存错误统计
    print("保存错误统计...")
    if not os.path.exists(args.output_fig_path):
        os.makedirs(args.output_fig_path, exist_ok=True)
    save_error_statistics(error_count, args.output_fig_path)
    
    # 4. 计算归一化参数
    print("计算归一化参数...")
    norm_params = compute_normalization(processed_data)
    
    # 5. 保存归一化参数
    print("保存归一化参数...")
    with open(args.normal_path, "w", encoding="utf-8") as f:
        json.dump(norm_params, f, indent=4, ensure_ascii=False)
    print(f"归一化参数已保存至: {args.normal_path}")
    
    # 6. 绘图
    print("生成可视化图表...")
    try:
        plot_selected_variables(
            action_eepose_delta=processed_data["action_eepose"],
            action_qpos_delta=processed_data["action_qpos"],
            state_eepose_abs=processed_data["state_eepose"],
            state_qpos_abs=processed_data["state_qpos"],
            variables_to_plot=args.plot_variables,
            output_dir=args.output_fig_path,
        )
    except Exception as e:
        print(f"绘图发生异常: {e}")
        raise

if __name__ == "__main__":
    main()
