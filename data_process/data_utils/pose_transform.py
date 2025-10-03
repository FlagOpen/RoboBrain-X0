import numpy as np
from scipy.spatial.transform import Rotation as R

# -----------------------------------------------------------------------------
# 6D Rotation Representations <--> Euler Angles / Quaternions / Matrix
# -----------------------------------------------------------------------------

def euler_to_6d(euler_angles: np.ndarray, convention: str = 'xyz') -> np.ndarray:
    """
    将一批欧拉角转换为 6D 连续旋转表示。

    Args:
        euler_angles (np.ndarray): 形状为 (N, 3) 或 (3,) 的欧拉角数组。
        convention (str): 欧拉角旋转顺序，例如 'xyz', 'zyx'。
                          根据您的描述，LeRobot 使用 'xyz'。

    Returns:
        np.ndarray: 形状为 (N, 6) 或 (6,) 的 6D 旋转表示。
    """
    is_single = euler_angles.ndim == 1
    if is_single:
        euler_angles = euler_angles[np.newaxis, :]
        
    # 从欧拉角创建旋转对象
    rot = R.from_euler(convention, euler_angles)
    # 获取旋转矩阵
    matrix = rot.as_matrix()

    # 提取前两列并展平为6维向量
    d6 = np.concatenate([matrix[..., :, 0], matrix[..., :, 1]], axis=-1)
    
    return d6.flatten() if is_single else d6

def d6_to_matrix(d6: np.ndarray) -> np.ndarray:
    """
    将一批 6D 旋转表示转换为 3x3 旋转矩阵。
    使用 Gram-Schmidt 正交化来确保矩阵的正交性。

    Args:
        d6 (np.ndarray): 形状为 (N, 6) 或 (6,) 的 6D 旋转表示。

    Returns:
        np.ndarray: 形状为 (N, 3, 3) 或 (3, 3) 的旋转矩阵。
    """
    is_single = d6.ndim == 1
    if is_single:
        d6 = d6[np.newaxis, :]

    a1 = d6[:, 0:3]
    a2 = d6[:, 3:6]

    # 1. 标准化第一列 b1
    b1 = a1 / np.linalg.norm(a1, axis=1, keepdims=True)
    
    # 2. 计算第三列 b3
    b3_raw = np.cross(b1, a2, axis=1)
    b3 = b3_raw / np.linalg.norm(b3_raw, axis=1, keepdims=True)
    
    # 3. 重新计算第二列 b2 以确保正交
    b2 = np.cross(b3, b1, axis=1)
    
    matrix = np.stack([b1, b2, b3], axis=-1)
    
    return matrix[0] if is_single else matrix

def d6_to_euler(d6: np.ndarray, convention: str = 'xyz') -> np.ndarray:
    """
    将一批 6D 旋转表示转换为欧拉角。

    Args:
        d6 (np.ndarray): 形状为 (N, 6) 或 (6,) 的 6D 旋转表示。
        convention (str): 目标欧拉角旋转顺序。

    Returns:
        np.ndarray: 形状为 (N, 3) 或 (3,) 的欧拉角。
    """
    matrix = d6_to_matrix(d6)
    rot = R.from_matrix(matrix)
    euler = rot.as_euler(convention)
    return euler
    
def d6_to_quat(d6: np.ndarray) -> np.ndarray:
    """
    将一批 6D 旋转表示转换为四元数。

    Args:
        d6 (np.ndarray): 形状为 (N, 6) 或 (6,) 的 6D 旋转表示。

    Returns:
        np.ndarray: 形状为 (N, 4) 或 (4,) 的四元数 (xyzw)。
    """
    matrix = d6_to_matrix(d6)
    rot = R.from_matrix(matrix)
    quat = rot.as_quat()
    return quat

def quat_to_6d(quaternions: np.ndarray) -> np.ndarray:
    """
    将一批四元数转换为 6D 连续旋转表示。

    Args:
        quaternions (np.ndarray): 形状为 (N, 4) 或 (4,) 的四元数 (xyzw)。

    Returns:
        np.ndarray: 形状为 (N, 6) 或 (6,) 的 6D 旋转表示。
    """
    is_single = quaternions.ndim == 1
    if is_single:
        quaternions = quaternions[np.newaxis, :]
        
    rot = R.from_quat(quaternions)
    matrix = rot.as_matrix()
    d6 = np.concatenate([matrix[..., :, 0], matrix[..., :, 1]], axis=-1)

    return d6.flatten() if is_single else d6

# -----------------------------------------------------------------------------
# Convenience Wrappers for Euler <--> Quaternions
# -----------------------------------------------------------------------------

def euler_to_quat(euler_angles: np.ndarray, convention: str = 'xyz') -> np.ndarray:
    """
    将一批欧拉角转换为四元数。

    Args:
        euler_angles (np.ndarray): 形状为 (N, 3) 或 (3,) 的欧拉角。
        convention (str): 欧拉角旋转顺序。

    Returns:
        np.ndarray: 形状为 (N, 4) 或 (4,) 的四元数 (xyzw)。
    """
    rot = R.from_euler(convention, euler_angles)
    return rot.as_quat()

def quat_to_euler(quaternions: np.ndarray, convention: str = 'xyz') -> np.ndarray:
    """
    将一批四元数转换为欧拉角。

    Args:
        quaternions (np.ndarray): 形状为 (N, 4) 或 (4,) 的四元数 (xyzw)。
        convention (str): 目标欧拉角旋转顺序。

    Returns:
        np.ndarray: 形状为 (N, 3) 或 (3,) 的欧拉角。
    """
    rot = R.from_quat(quaternions)
    return rot.as_euler(convention)


def calculate_delta_6d_as_axis_angle(d6_t: np.ndarray, d6_t_plus_1: np.ndarray) -> np.ndarray:
    """
    计算两个 6D 姿态之间的 delta a ction，并以轴角（3D向量）形式返回。

    Args:
        d6_t (np.ndarray): t 时刻的 6D 姿态，形状 (N, 6) 或 (6,)
        d6_t_plus_1 (np.ndarray): t+1 时刻的 6D 姿态，形状 (N, 6) 或 (6,)

    Returns:
        np.ndarray: 轴角形式的 delta action，形状 (N, 3) 或 (3,)
    """
    # 1. 转换为旋转矩阵
    R_t = d6_to_matrix(d6_t)
    R_t_plus_1 = d6_to_matrix(d6_t_plus_1)

    # 2. 计算相对旋转 ΔR = R_{t+1} * R_t^{-1}
    # 对于批处理，使用 np.einsum 或循环。这里为了清晰，展示单个计算的逻辑
    # np.matmul (@) 和 .transpose(0, 2, 1) 可以很好地处理批处理
    if R_t.ndim == 3: # 批处理
        delta_R = R_t_plus_1 @ R_t.transpose(0, 2, 1)
    else: # 单个
        delta_R = R_t_plus_1 @ R_t.T
        
    # 3. 将 ΔR 转换为轴角
    delta_axis_angle = R.from_matrix(delta_R).as_rotvec()
    
    return delta_axis_angle

def calculate_delta_quat_as_axis_angle(quat_t: np.ndarray, quat_t_plus_1: np.ndarray) -> np.ndarray:
    """
    计算两个四元数之间的 delta action，并以轴角（3D向量）形式返回。
    包含关键的四元数对齐步骤。

    Args:
        quat_t (np.ndarray): t 时刻的四元数 (xyzw), 形状 (N, 4) 或 (4,)
        quat_t_plus_1 (np.ndarray): t+1 时刻的四元数 (xyzw), 形状 (N, 4) 或 (4,)

    Returns:
        np.ndarray: 轴角形式的 delta action，形状 (N, 3) 或 (3,)
    """
    is_single = quat_t.ndim == 1
    if is_single:
        quat_t = quat_t[np.newaxis, :]
        quat_t_plus_1 = quat_t_plus_1[np.newaxis, :]

    # 1. 对齐四元数
    # 计算点积
    dot_product = np.sum(quat_t * quat_t_plus_1, axis=1)
    # 如果点积为负，翻转 t+1 时刻的四元数
    quat_t_plus_1[dot_product < 0] = -quat_t_plus_1[dot_product < 0]
    
    # 2. 计算相对旋转 Δq = q_{t+1} * q_t^{-1}
    # scipy Rotation 会自动处理共轭和乘法
    q_t_rot = R.from_quat(quat_t)
    q_t_plus_1_rot = R.from_quat(quat_t_plus_1)
    
    delta_q_rot = q_t_plus_1_rot * q_t_rot.inv()
    
    # 3. 转换为轴角
    delta_axis_angle = delta_q_rot.as_rotvec()
    
    return delta_axis_angle.flatten() if is_single else delta_axis_angle
    
# -----------------------------------------------------------------------------

def compute_d6_axis_angle_deltas(d6_sequence: np.ndarray) -> np.ndarray:
    """
    对 6D 姿态序列沿时间维做“差分”，返回相邻帧之间的轴角（rotvec）。

    支持两种输入形状：
    - (T, 6): 单条序列，长度为 T
    - (B, T, 6): 批量序列，batch 大小为 B

    Returns:
        np.ndarray: 若输入为 (T, 6) 则返回 (T-1, 3);
                    若输入为 (B, T, 6) 则返回 (B, T-1, 3)。
    """
    if d6_sequence.ndim == 2:
        # 单序列 (T, 6)
        if d6_sequence.shape[0] < 2:
            return np.zeros((0, 3), dtype=d6_sequence.dtype)
        R_seq = d6_to_matrix(d6_sequence)  # (T, 3, 3)
        delta_R = R_seq[1:] @ np.transpose(R_seq[:-1], (0, 2, 1))
        deltas = R.from_matrix(delta_R).as_rotvec()  # (T-1, 3)
        return deltas
    elif d6_sequence.ndim == 3:
        # 批量序列 (B, T, 6)
        B, T, _ = d6_sequence.shape
        if T < 2:
            return np.zeros((B, 0, 3), dtype=d6_sequence.dtype)
        R_seq = d6_to_matrix(d6_sequence.reshape(-1, 6)).reshape(B, T, 3, 3)
        delta_R = R_seq[:, 1:, :, :] @ np.transpose(R_seq[:, :-1, :, :], (0, 1, 3, 2))
        deltas = R.from_matrix(delta_R.reshape(-1, 3, 3)).as_rotvec().reshape(B, T - 1, 3)
        return deltas
    else:
        raise ValueError("d6_sequence 应为 (T,6) 或 (B,T,6) 形状")


def compute_quat_axis_angle_deltas(quat_sequence: np.ndarray) -> np.ndarray:
    """
    对四元数姿态序列沿时间维做“差分”，返回相邻帧之间的轴角（rotvec）。
    自动进行相邻帧四元数的同向对齐（避免 180° 对称号翻转）。

    支持两种输入形状：
    - (T, 4): 单条序列，长度为 T
    - (B, T, 4): 批量序列，batch 大小为 B

    Returns:
        np.ndarray: 若输入为 (T, 4) 则返回 (T-1, 3);
                    若输入为 (B, T, 4) 则返回 (B, T-1, 3)。
    """
    if quat_sequence.ndim == 2:
        # 单序列 (T, 4)
        if quat_sequence.shape[0] < 2:
            return np.zeros((0, 3), dtype=quat_sequence.dtype)
        q_t = quat_sequence[:-1]
        q_tp1 = quat_sequence[1:]
        # 相邻对齐：若点积为负，则翻转后一帧
        dot = np.sum(q_t * q_tp1, axis=-1)
        q_tp1_aligned = np.where(dot[:, None] < 0, -q_tp1, q_tp1)
        rot_t = R.from_quat(q_t)
        rot_tp1 = R.from_quat(q_tp1_aligned)
        delta_rot = rot_tp1 * rot_t.inv()
        deltas = delta_rot.as_rotvec()
        return deltas
    elif quat_sequence.ndim == 3:
        # 批量序列 (B, T, 4)
        B, T, _ = quat_sequence.shape
        if T < 2:
            return np.zeros((B, 0, 3), dtype=quat_sequence.dtype)
        q_t = quat_sequence[:, :-1, :]  # (B, T-1, 4)
        q_tp1 = quat_sequence[:, 1:, :]  # (B, T-1, 4)
        dot = np.sum(q_t * q_tp1, axis=-1)  # (B, T-1)
        q_tp1_aligned = np.where(dot[..., None] < 0, -q_tp1, q_tp1)
        rot_t = R.from_quat(q_t.reshape(-1, 4))
        rot_tp1 = R.from_quat(q_tp1_aligned.reshape(-1, 4))
        delta_rot = rot_tp1 * rot_t.inv()
        deltas = delta_rot.as_rotvec().reshape(B, T - 1, 3)
        return deltas
    else:
        raise ValueError("quat_sequence 应为 (T,4) 或 (B,T,4) 形状")

if __name__ == '__main__':
    # --- 使用示例和验证 ---
    
    # 1. 定义一些样本欧拉角 (RPY)，使用 'xyz' 约定
    # 第一个样本，一个普通姿态
    # 第二个样本，接近万向节锁的姿态 (pitch 为 90 度)
    # 第三个样本，另一个姿态
    euler_original_batch = np.array([
        [0.1, 0.2, 0.3],
        [0.5, np.pi / 2, 0.0], 
        [-1.0, 0.5, 2.0]
    ])
    
    print("--- 批量处理示例 ---")
    print(f"原始欧拉角 (xyz):\n{euler_original_batch}\n")

    d6_batch = euler_to_6d(euler_original_batch, convention='xyz')
    print(f"转换后的 6D 表示:\n{d6_batch}\n")

    quat_batch = d6_to_quat(d6_batch)
    print(f"从 6D 恢复的四元数 (xyzw):\n{quat_batch}\n")

    d6_reconstructed_from_quat = quat_to_6d(quat_batch)
    print(f"从四元数重构的 6D 表示:\n{d6_reconstructed_from_quat}\n")

    matrix_batch = d6_to_matrix(d6_batch)
    print(f"从 6D 恢复的旋转矩阵 (第一个样本):\n{matrix_batch[0]}\n")

    euler_reconstructed_batch = d6_to_euler(d6_batch, convention='xyz')
    print(f"从 6D 恢复的欧拉角 (xyz):\n{euler_reconstructed_batch}\n")

    error = np.sqrt(np.mean((euler_original_batch - euler_reconstructed_batch)**2))
    print(f"原始欧拉角与“往返”转换后的欧拉角之间的均方根误差 (RMSE): {error:.12f}")
    assert np.allclose(euler_original_batch, euler_reconstructed_batch), "往返转换失败！"
    print("\n✅ 批量转换精度验证成功！\n")

    print("--- 单个向量处理示例 ---")
    euler_single = np.array([0.1, 0.2, 0.3])
    d6_single = euler_to_6d(euler_single)
    euler_reconstructed_single = d6_to_euler(d6_single)

    print(f"原始欧拉角: {euler_single}")
    print(f"转换后 6D: {d6_single}")
    print(f"恢复后欧拉角: {euler_reconstructed_single}")
    assert np.allclose(euler_single, euler_reconstructed_single), "单向量往返转换失败！"
    print("\n✅ 单向量转换精度验证成功！\n")

    # 2) 序列 diff 示例（6D / 四元数）
    print("--- 序列 diff 示例（单序列与批量） ---")

    # 单序列：构造已知增量，验证 diff 能恢复
    np.random.seed(0)
    T = 8
    base_inc = np.array([0.05, -0.02, 0.03])
    increments_single = np.tile(base_inc[None, :], (T - 1, 1))  # (T-1, 3)

    rotations = []
    rot = R.identity()
    rotations.append(rot)
    for inc in increments_single:
        rot = R.from_rotvec(inc) * rot
        rotations.append(rot)
    quat_seq_single = np.stack([r.as_quat() for r in rotations], axis=0)  # (T, 4)
    d6_seq_single = quat_to_6d(quat_seq_single)  # (T, 6)

    deltas_d6_single = compute_d6_axis_angle_deltas(d6_seq_single)
    deltas_quat_single = compute_quat_axis_angle_deltas(quat_seq_single)

    print(f"单序列 6D diff 首条: {deltas_d6_single[0]}")
    print(f"单序列 quat diff 首条: {deltas_quat_single[0]}")
    assert np.allclose(deltas_d6_single, increments_single, atol=1e-8)
    assert np.allclose(deltas_quat_single, increments_single, atol=1e-8)
    print("✅ 单序列 6D/quat diff 验证通过！\n")

    # 批量序列：两条不同增量的序列，验证批量接口
    B = 2
    T = 6
    inc0 = np.array([0.02, 0.01, -0.03])
    inc1 = np.array([-0.01, 0.04, 0.02])
    increments_batch = np.stack([
        np.tile(inc0[None, :], (T - 1, 1)),
        np.tile(inc1[None, :], (T - 1, 1)),
    ], axis=0)  # (B, T-1, 3)

    quat_seqs = []
    for b in range(B):
        rot = R.identity()
        quats = [rot.as_quat()]
        for inc in increments_batch[b]:
            rot = R.from_rotvec(inc) * rot
            quats.append(rot.as_quat())
        quat_seqs.append(np.stack(quats, axis=0))
    quat_seq_batch = np.stack(quat_seqs, axis=0)  # (B, T, 4)
    d6_seq_batch = quat_to_6d(quat_seq_batch.reshape(-1, 4)).reshape(B, T, 6)

    deltas_d6_batch = compute_d6_axis_angle_deltas(d6_seq_batch)
    deltas_quat_batch = compute_quat_axis_angle_deltas(quat_seq_batch)

    assert np.allclose(deltas_d6_batch, increments_batch, atol=1e-8)
    assert np.allclose(deltas_quat_batch, increments_batch, atol=1e-8)
    print("✅ 批量序列 6D/quat diff 验证通过！")