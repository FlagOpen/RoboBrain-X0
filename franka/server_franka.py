# -*- coding: utf-8 -*-
"""
RoboBrain Robotics API 服务 - Franka Panda

该服务提供一个HTTP接口，用于接收机器人状态和图像，并使用预训练的视觉语言模型进行推理，
返回预测的机器人动作序列。

支持两种操作模式：
1. 标准模式 (SUBTASK_MODE = False): 模型直接输出控制动作。
2. 子任务模式 (SUBTASK_MODE = True): 模型首先生成一个文本描述的子任务，然后输出相应的控制动作。

通过修改下面的 `SUBTASK_MODE` 和 `MODEL_PATH` 全局变量来切换模式和模型。

POST /infer 输入样例：
{
  "eef_pose": [0.1, 0.2, ..., 0.3],      # shape: [8]，当前末端执行器姿态 [x, y, z, qx, qy, qz, qw, gripper]
  "instruction": "请将桌上的苹果放入篮子",
  "images": {
      "cam_front": "<base64字符串>",
      "cam_wrist": "<base64字符串>"
  }
}
"""

import os
import sys
import torch
import h5py
import logging
import traceback
import json
import base64
import io
import numpy as np
import time
from flask import Flask, request, jsonify
from flask_cors import CORS
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from PIL import Image
from scipy.spatial.transform import Rotation as R

# 将外部代码库路径添加到sys.path
sys.path.append("/share/project/dumengfei/code/sim_data_process")
from pose_transform import add_delta_to_quat_pose
from action_token.action_chunk_to_fast_token import ActionChunkProcessor

# --- 服务配置 ---
# 是否启用子任务模式
SUBTASK_MODE = True  # 设置为 True 或 False 来切换模式

# 模型路径配置
# 根据 SUBTASK_MODE 选择不同的模型路径
if SUBTASK_MODE:
    MODEL_PATH = '/share/project/jiyuheng/ckpt/robotics_pretrain_modeltp1pp1_S6_subtask'
else:
    MODEL_PATH = '/share/project/jiyuheng/ckpt/robotics_pretrain_modeltp1pp1_S4_Franka_0924_transfer_9epoch'

CONFIG_PATH = MODEL_PATH
DEBUG = False

# 服务网络配置
SERVICE_CONFIG = {
    'host': '0.0.0.0',
    'port': 5002,
    'debug': False,
    'threaded': True,
    'max_content_length': 16 * 1024 * 1024
}

# --- 日志配置 ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- 全局变量 ---
app = Flask(__name__)
CORS(app)
model = None
processor = None
action_tokenizer = None

# --- 辅助函数与类 ---

_TOKENIZER_CACHE: dict[int, ActionChunkProcessor] = {}
def get_tokenizer(max_len: int) -> ActionChunkProcessor:
    """为每个进程缓存并返回一个ActionChunkProcessor实例"""
    tok = _TOKENIZER_CACHE.get(max_len)
    if tok is None:
        tok = ActionChunkProcessor(max_len=max_len)
        _TOKENIZER_CACHE[max_len] = tok
    return tok

def load_model():
    """加载并初始化模型和处理器"""
    global model, processor, action_tokenizer
    try:
        logger.info(f"开始加载模型: {MODEL_PATH} (Subtask Mode: {SUBTASK_MODE})")
        device_id = os.environ.get("EGL_DEVICE_ID", "0")
        device = f"cuda:{device_id}" if torch.cuda.is_available() else "cpu"

        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            MODEL_PATH,
            torch_dtype=torch.bfloat16,
            device_map=device,
            trust_remote_code=True,
        )
        processor = AutoProcessor.from_pretrained(MODEL_PATH, padding_side='left')
        model.eval()
        
        action_tokenizer = get_tokenizer(max_len=256)

        if torch.cuda.is_available():
            logger.info(f"模型已成功加载到 GPU: {torch.cuda.get_device_name()}")
        else:
            logger.info("模型已成功加载到 CPU")
        return True
    except Exception as e:
        logger.error(f"模型加载失败: {e}", exc_info=True)
        return False

def inverse_transform(x_norm, scale, offset):
    """根据均值和标准差对动作进行反归一化"""
    x_norm = np.asarray(x_norm)
    return (x_norm - offset) / scale

# 加载动作归一化统计数据
try:
    with open("/share/project/dumengfei/code/pretrain_data_process/real_data/franka/franka_data_pnp_0922/normal_stats_all_action10Hz_original.json", 'r') as f:
        action_stats = json.load(f)
except FileNotFoundError:
    logger.error("动作归一化统计文件未找到！服务可能无法正确执行反归一化。")
    action_stats = None

def decode_image_base64_to_pil(image_base64: str) -> Image:
    """将Base64编码的图片字符串解码为PIL Image对象"""
    try:
        image_data = base64.b64decode(image_base64)
        return Image.open(io.BytesIO(image_data)).convert('RGB')
    except Exception as e:
        logger.error(f"图片解码失败: {e}")
        raise ValueError("无效的Base64图片字符串")

def process_images(images_dict: dict) -> list:
    """处理输入的图像字典，返回一个PIL Image列表"""
    try:
        image_keys = ['cam_front', 'cam_wrist']
        processed_list = [decode_image_base64_to_pil(images_dict[k]).resize((320, 240)) for k in image_keys]
        # 保存图像用于调试
        for key, img in zip(image_keys, processed_list):
            img.save(f'/share/project/dumengfei/code/real_eval/image_log/franka_{key}.png')
        return processed_list
    except KeyError as e:
        raise ValueError(f"缺少必需的图像: {e}")
    except Exception as e:
        logger.error(f"处理图片时发生错误: {e}")
        raise ValueError("图片处理失败")

# --- Flask API 端点 ---

@app.route('/health', methods=['GET'])
def health_check():
    """健康检查端点，返回服务和模型状态"""
    if model is None or processor is None:
        return jsonify({"status": "error", "message": "模型未加载"}), 503
    
    gpu_info = {}
    if torch.cuda.is_available():
        gpu_info = {
            "name": torch.cuda.get_device_name(),
            "memory_allocated": f"{torch.cuda.memory_allocated() / 1024**3:.2f}GB",
            "memory_reserved": f"{torch.cuda.memory_reserved() / 1024**3:.2f}GB"
        }

    return jsonify({
        "status": "healthy",
        "model_loaded": True,
        "subtask_mode": SUBTASK_MODE,
        "model_path": MODEL_PATH,
        "gpu_info": gpu_info
    })

@app.route('/info', methods=['GET'])
def service_info():
    """提供服务元信息"""
    return jsonify({
        "service_name": "RoboBrain Robotics API for Franka",
        "version": "2.0.0",
        "subtask_mode": SUBTASK_MODE,
        "model_path": MODEL_PATH,
        "endpoints": {
            "/health": "GET",
            "/info": "GET",
            "/infer": "POST"
        }
    })

@app.route('/infer', methods=['POST'])
def infer_api():
    """核心推理API端点"""
    start_time = time.time()
    
    if model is None:
        return jsonify({"success": False, "error": "模型未加载，请检查服务状态"}), 503
    
    data = request.get_json()
    if not data or 'eef_pose' not in data or 'instruction' not in data or 'images' not in data:
        return jsonify({"success": False, "error": "请求数据不完整或格式错误"}), 400

    try:
        instruction = data['instruction']
        images = data['images']
        eef_pose = np.array(data['eef_pose'])
        images_pil = process_images(images)

        # --- Prompt 生成 ---
        if SUBTASK_MODE:
            # 子任务模式 Prompt
            prompt_template = (
                "You are controlling a Franka single-arm robot. Your task is to adjust the end effector (EEF) poses at 30Hz to complete a specified task. "
                "Your output must include two components: 1. Immediate sub-task: The specific action you will execute first to progress toward the overall task; 2. Control tokens: These will be decoded into a 30×7 action sequence to implement the sub-task. "
                "Each EEPose here includes 3 delta position(xyz) + 3 delta orientation(axis-angle) + 1 gripper(opening range)\n\n"
                "Your current visual inputs are robot front image"
            )
        else:
            # 标准模式 Prompt
            prompt_template = (
                "You are controlling a Franka single-arm robot. Your task is to adjust the end effector (EEF) poses at 30Hz to complete a specified task. "
                "You need to output control tokens that can be decoded into a 30×7 action sequence. The sequence has 30 consecutive actions, each with 7 dimensions. "
                "Each EEPose here includes 3 delta position(xyz) + 3 delta orientation(axis-angle) + 1 gripper(opening range)\n\n"
                "Your current visual inputs include: robot front image"
            )

        content = [
            {"type": "text", "text": prompt_template},
            {"type": "image", "image": f"data:image;base64,{images['cam_front']}"},
            {"type": "text", "text": " and robot wrist image"},
            {"type": "image", "image": f"data:image;base64,{images['cam_wrist']}"},
            {"type": "text", "text": f"\nYour overall task is: {instruction.lower()}."},
        ]
        
        messages = [{"role": "user", "content": content}]
        text_prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = processor(text=[text_prompt], images=images_pil, padding=True, return_tensors="pt").to(model.device)

        # --- 模型推理 ---
        gen_kwargs = {
            "max_new_tokens": 768, "do_sample": True, "temperature": 0.2,
            "pad_token_id": processor.tokenizer.pad_token_id, "eos_token_id": processor.tokenizer.eos_token_id,
            "repetition_penalty": 1.0, "use_cache": True,
        }
        with torch.no_grad():
            output_ids = model.generate(**inputs, **gen_kwargs)[0]
        
        input_length = inputs.input_ids.shape[1]
        output_tokens = output_ids[input_length:].detach().cpu().tolist()

        # --- 输出解析 ---
        subtask_result = "N/A"
        if SUBTASK_MODE:
            try:
                # 使用 <boa> (151665) token 分割子任务和动作
                boa_token = 151665
                split_index = output_tokens.index(boa_token)
                subtask_tokens = output_tokens[:split_index]
                action_tokens_raw = output_tokens[split_index + 1:]
                subtask_result = processor.tokenizer.decode(subtask_tokens, skip_special_tokens=True).strip()
                logger.info(f"解析到子任务: {subtask_result}")
            except ValueError:
                logger.warning("未找到 <boa> token，无法解析子任务。将整个输出视为动作。")
                action_tokens_raw = output_tokens
                subtask_result = "解析失败: 未找到 <boa> token"
        else:
            action_tokens_raw = output_tokens

        try:
            # 查找 <eoa> (151667) token 作为动作结束标志
            eoa_token = 151667
            end_index = action_tokens_raw.index(eoa_token)
            action_tokens_raw = action_tokens_raw[:end_index]
        except ValueError:
            logger.warning("未找到 <eoa> token，使用完整输出序列。")

        # 提取并解码动作
        action_ids = [t - 149595 for t in action_tokens_raw if 149595 <= t < 151643]
        actions_norm, _ = action_tokenizer._extract_actions_from_tokens([action_ids], action_horizon=30, action_dim=7)
        delta_actions = actions_norm[0]

        # --- 动作后处理 ---
        if delta_actions is None or action_stats is None:
             raise ValueError("动作解码失败或归一化统计数据未加载")

        scale = np.array(action_stats['action.eepose']['scale_'])
        offset = np.array(action_stats['action.eepose']['offset_'])
        delta_actions_denorm = inverse_transform(np.array(delta_actions), scale, offset)
        
        # 保存用于调试的动作日志
        with open(f'/share/project/dumengfei/code/real_eval/action_log/franka_action.json', 'w') as f:
            json.dump(delta_actions_denorm.tolist(), f)

        # 计算绝对姿态序列
        final_ee_actions = []
        current_eef_pose = eef_pose.copy()
        for i in range(30):
            current_eef_pose[:3] += delta_actions_denorm[i][:3]  # 位置更新
            current_eef_pose[3:7] = add_delta_to_quat_pose(current_eef_pose[3:7], delta_actions_denorm[i][3:6]) # 姿态更新
            current_eef_pose[7] = np.clip(delta_actions_denorm[i][6], 0, 1) # 夹爪更新
            final_ee_actions.append(current_eef_pose.tolist())

        processing_time = time.time() - start_time
        logger.info(f"推理完成, 耗时: {processing_time:.2f}秒. 模式: {'Subtask' if SUBTASK_MODE else 'Standard'}")
        
        response = {
            "success": True,
            "eepose": final_ee_actions,
            "processing_time": processing_time
        }
        if SUBTASK_MODE:
            response["subtask"] = subtask_result

        return jsonify(response)

    except Exception as e:
        logger.error(f"推理过程中发生严重错误: {e}", exc_info=True)
        return jsonify({"success": False, "error": str(e)}), 500

# --- 主程序入口 ---
if __name__ == '__main__':
    if not load_model():
        sys.exit(1)
    
    logger.info("RoboBrain Franka API 服务启动中...")
    logger.info(f"服务地址: http://{SERVICE_CONFIG['host']}:{SERVICE_CONFIG['port']}")
    logger.info(f"当前模式: {'Subtask' if SUBTASK_MODE else 'Standard'}")
    
    app.run(
        host=SERVICE_CONFIG['host'],
        port=SERVICE_CONFIG['port'],
        debug=SERVICE_CONFIG['debug'],
        threaded=SERVICE_CONFIG['threaded']
    )