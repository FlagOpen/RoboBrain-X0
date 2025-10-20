# -*- coding: utf-8 -*-
"""
RoboBrain Robotics API Service - Franka Panda

This service provides an HTTP interface to receive robot state and images, and uses a pre-trained vision-language model for inference,
returning predicted robot action sequences.

Supports two operation modes:
1. Standard Mode (SUBTASK_MODE = False): The model directly outputs control actions.
2. Subtask Mode (SUBTASK_MODE = True): The model first generates a text description of a subtask, then outputs corresponding control actions.

Switch modes and models by modifying the `SUBTASK_MODE` and `MODEL_PATH` global variables below.

POST /infer Input Example:
{
  "eef_pose": [0.1, 0.2, ..., 0.3],
  "instruction": "Please put the apple on the table into the basket",
  "images": {
      "cam_front": "<base64 string>",
      "cam_wrist": "<base64 string>"
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

from pathlib import Path
root = Path(__file__).parent.parent 
sys.path.append(str(root))
from data_process.action_token.action_chunk_to_fast_token import ActionChunkProcessor
from data_process.data_utils.pose_transform import add_delta_to_quat_pose

# --- Service Configuration ---
# Whether to enable subtask mode
SUBTASK_MODE = True  # Set to True or False to switch modes

# Model path configuration
# Choose different model paths based on SUBTASK_MODE
if SUBTASK_MODE:
    MODEL_PATH = ''
else:
    MODEL_PATH = ''

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

# --- Logging Configuration ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Global Variables ---
app = Flask(__name__)
CORS(app)
model = None
processor = None
action_tokenizer = None

# --- Helper Functions and Classes ---

_TOKENIZER_CACHE: dict[int, ActionChunkProcessor] = {}
def get_tokenizer(max_len: int) -> ActionChunkProcessor:
    """Cache and return an ActionChunkProcessor instance for each process"""
    tok = _TOKENIZER_CACHE.get(max_len)
    if tok is None:
        tok = ActionChunkProcessor(max_len=max_len)
        _TOKENIZER_CACHE[max_len] = tok
    return tok

def load_model():
    """Load and initialize the model and processor"""
    global model, processor, action_tokenizer
    try:
        logger.info(f"Loading model: {MODEL_PATH} (Subtask Mode: {SUBTASK_MODE})")
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
            logger.info(f"Model successfully loaded to GPU: {torch.cuda.get_device_name()}")
        else:
            logger.info("Model successfully loaded to CPU")
        return True
    except Exception as e:
        logger.error(f"Model loading failed: {e}", exc_info=True)
        return False

def inverse_transform(x_norm, scale, offset):
    """Denormalize actions based on mean and standard deviation"""
    x_norm = np.asarray(x_norm)
    return (x_norm - offset) / scale

# Load action normalization statistics
try:
    with open("", 'r') as f:
        action_stats = json.load(f)
except FileNotFoundError:
    logger.error("Action normalization statistics file not found! The service may not perform denormalization correctly.")
    action_stats = None

def decode_image_base64_to_pil(image_base64: str) -> Image:
    """将Base64编码的图片字符串解码为PIL Image对象"""
    try:
        image_data = base64.b64decode(image_base64)
        return Image.open(io.BytesIO(image_data)).convert('RGB')
    except Exception as e:
        logger.error(f"Image decoding failed: {e}")
        raise ValueError("Invalid Base64 image string")

def process_images(images_dict: dict) -> list:
    """处理输入的图像字典，返回一个PIL Image列表"""
    try:
        image_keys = ['cam_front', 'cam_wrist']
        processed_list = [decode_image_base64_to_pil(images_dict[k]).resize((320, 240)) for k in image_keys]
        # 保存图像用于调试
        for key, img in zip(image_keys, processed_list):
            img.save(f'image_log/franka_{key}.png')
        return processed_list
    except KeyError as e:
        raise ValueError(f"Missing required image: {e}")
    except Exception as e:
        logger.error(f"Error processing images: {e}")
        raise ValueError("Image processing failed")

# --- Flask API Endpoints ---

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint, returns service and model status"""
    if model is None or processor is None:
        return jsonify({"status": "error", "message": "Model not loaded"}), 503
    
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
    """Provide service metadata"""
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
    """Core inference API endpoint"""
    start_time = time.time()
    
    if model is None:
        return jsonify({"success": False, "error": "Model not loaded, please check service status"}), 503
    
    data = request.get_json()
    if not data or 'eef_pose' not in data or 'instruction' not in data or 'images' not in data:
        return jsonify({"success": False, "error": "Request data is incomplete or in the wrong format"}), 400

    try:
        instruction = data['instruction']
        images = data['images']
        eef_pose = np.array(data['eef_pose'])
        images_pil = process_images(images)

        # --- Prompt Generation ---
        if SUBTASK_MODE:
            # Subtask mode Prompt
            prompt_template = (
                "You are controlling a Franka single-arm robot. Your task is to adjust the end effector (EEF) poses at 30Hz to complete a specified task. "
                "Your output must include two components: 1. Immediate sub-task: The specific action you will execute first to progress toward the overall task; 2. Control tokens: These will be decoded into a 30×7 action sequence to implement the sub-task. "
                "Each EEPose here includes 3 delta position(xyz) + 3 delta orientation(axis-angle) + 1 gripper(opening range)\n\n"
                "Your current visual inputs are robot front image"
            )
        else:
            # Standard mode Prompt
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

        # --- Model Inference ---
        gen_kwargs = {
            "max_new_tokens": 768, "do_sample": True, "temperature": 0.2,
            "pad_token_id": processor.tokenizer.pad_token_id, "eos_token_id": processor.tokenizer.eos_token_id,
            "repetition_penalty": 1.0, "use_cache": True,
        }
        with torch.no_grad():
            output_ids = model.generate(**inputs, **gen_kwargs)[0]
        
        input_length = inputs.input_ids.shape[1]
        output_tokens = output_ids[input_length:].detach().cpu().tolist()

        # --- Output Parsing ---
        subtask_result = "N/A"
        if SUBTASK_MODE:
            try:
                # Use <boa> (151665) token to split subtask and action
                boa_token = 151665
                split_index = output_tokens.index(boa_token)
                subtask_tokens = output_tokens[:split_index]
                action_tokens_raw = output_tokens[split_index + 1:]
                subtask_result = processor.tokenizer.decode(subtask_tokens, skip_special_tokens=True).strip()
                logger.info(f"Parsed subtask: {subtask_result}")
            except ValueError:
                logger.warning("<boa> token not found, unable to parse subtask. Treating entire output as action.")
                action_tokens_raw = output_tokens
                subtask_result = "Parsing failed: <boa> token not found"
        else:
            action_tokens_raw = output_tokens

        try:
            # Find <eoa> (151667) token as action end flag
            eoa_token = 151667
            end_index = action_tokens_raw.index(eoa_token)
            action_tokens_raw = action_tokens_raw[:end_index]
        except ValueError:
            logger.warning("<eoa> token not found, using complete output sequence.")

        # Extract and decode actions
        action_ids = [t - 149595 for t in action_tokens_raw if 149595 <= t < 151643]
        actions_norm, _ = action_tokenizer._extract_actions_from_tokens([action_ids], action_horizon=30, action_dim=7)
        delta_actions = actions_norm[0]

        # --- Action Post-processing ---
        if delta_actions is None or action_stats is None:
             raise ValueError("Action decoding failed or normalization statistics not loaded")

        scale = np.array(action_stats['action.eepose']['scale_'])
        offset = np.array(action_stats['action.eepose']['offset_'])
        delta_actions_denorm = inverse_transform(np.array(delta_actions), scale, offset)
        
        # Save action log for debugging
        with open(f'action_log/franka_action.json', 'w') as f:
            json.dump(delta_actions_denorm.tolist(), f)

        # Calculate absolute pose sequence
        final_ee_actions = []
        current_eef_pose = eef_pose.copy()
        for i in range(30):
            current_eef_pose[:3] += delta_actions_denorm[i][:3]  # Position update
            current_eef_pose[3:7] = add_delta_to_quat_pose(current_eef_pose[3:7], delta_actions_denorm[i][3:6]) # Pose update
            current_eef_pose[7] = np.clip(delta_actions_denorm[i][6], 0, 1) # Gripper update
            final_ee_actions.append(current_eef_pose.tolist())

        processing_time = time.time() - start_time
        logger.info(f"Inference completed, time taken: {processing_time:.2f} seconds. Mode: {'Subtask' if SUBTASK_MODE else 'Standard'}")
        
        response = {
            "success": True,
            "eepose": final_ee_actions,
            "processing_time": processing_time
        }
        if SUBTASK_MODE:
            response["subtask"] = subtask_result

        return jsonify(response)

    except Exception as e:
        logger.error(f"Severe error during inference: {e}", exc_info=True)
        return jsonify({"success": False, "error": str(e)}), 500

# --- Main Program Entry ---
if __name__ == '__main__':
    if not load_model():
        sys.exit(1)
    
    logger.info("RoboBrain Franka API service starting...")
    logger.info(f"Service address: http://{SERVICE_CONFIG['host']}:{SERVICE_CONFIG['port']}")
    logger.info(f"Current mode: {'Subtask' if SUBTASK_MODE else 'Standard'}")
    
    app.run(
        host=SERVICE_CONFIG['host'],
        port=SERVICE_CONFIG['port'],
        debug=SERVICE_CONFIG['debug'],
        threaded=SERVICE_CONFIG['threaded']
    )