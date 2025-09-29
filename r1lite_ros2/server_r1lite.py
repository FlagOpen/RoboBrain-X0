# -*- coding: utf-8 -*-
"""
RoboBrain Robotics API Service - R1-Lite Robot

This service provides an HTTP interface for controlling the R1-Lite dual-arm robot.
It receives robot state and images, uses a pre-trained vision-language model for inference,
and returns a predicted sequence of actions.

It supports two operational modes:
1. Standard Mode (SUBTASK_MODE = False): The model directly outputs control actions.
2. Subtask Mode (SUBTASK_MODE = True): The model first generates a text description of a subtask
   and then outputs the corresponding control actions.

Switch modes and models by configuring the `SUBTASK_MODE` and associated path variables below.

POST /infer Input Example:
{
  "eef_pose": [0.1, 0.2, ..., 0.3],      # shape: [16], current right+left arm pose [pos_r, quat_r, grip_r, pos_l, quat_l, grip_l]
  "instruction": "Sort the items into the bins",
  "images": {
      "cam_head": "<base64_string>",
      "cam_right_wrist": "<base64_string>",
      "cam_left_wrist": "<base64_string>"
  }
}
"""
import os
import sys
import torch
import logging
import traceback
import json
import base64
import io
import re
import numpy as np
import time
from flask import Flask, request, jsonify
from flask_cors import CORS
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from PIL import Image
import cv2
from scipy.spatial.transform import Rotation as R

# Add external library path to sys.path
sys.path.append("/share/project/dumengfei/code/sim_data_process")
from pose_transform import add_delta_to_quat_pose
from action_token.action_chunk_to_fast_token import ActionChunkProcessor

# --- Service Configuration ---
# Enable/disable subtask mode
SUBTASK_MODE = True  # Set to True or False to switch modes

# Model and Statistics Path Configuration
if SUBTASK_MODE:
    MODEL_PATH = '/share/project/lizhiyu/data/ckpt/robotics_pretrain_modeltp1pp1_S6_subtask_r1lite_demo50'
    STATS_PATH = "/share/project/dumengfei/code/pretrain_data_process/real_data/r1lite/demo_0920/r1lite_normal_demo_0921_30Hz.json"
else:
    MODEL_PATH = '/share/project/jiyuheng/ckpt/robotics_pretrain_modeltp1pp1_S6_20'
    STATS_PATH = "/share/project/chenghy/data/r1lite/r1lite_normal_Afps1_Padding20_0906_temp.json"

CONFIG_PATH = MODEL_PATH
DEBUG = False

# Service Network Configuration
SERVICE_CONFIG = {
    'host': '0.0.0.0',
    'port': 5003,
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

# --- Helper Functions & Classes ---

_TOKENIZER_CACHE: dict[int, ActionChunkProcessor] = {}
def get_tokenizer(max_len: int) -> ActionChunkProcessor:
    """Cache and return an ActionChunkProcessor instance per process."""
    tok = _TOKENIZER_CACHE.get(max_len)
    if tok is None:
        tok = ActionChunkProcessor(max_len=max_len)
        _TOKENIZER_CACHE[max_len] = tok
    return tok

def load_model():
    """Load and initialize the model and processor."""
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
    """Denormalize actions based on scale and offset."""
    x_norm = np.asarray(x_norm)
    return (x_norm - offset) / scale

# Load action normalization statistics
try:
    logger.info(f"Loading action statistics from: {STATS_PATH}")
    with open(STATS_PATH, 'r') as f:
        action_stats = json.load(f)
except FileNotFoundError:
    logger.error(f"Action normalization statistics file not found at {STATS_PATH}! Service may not denormalize actions correctly.")
    action_stats = None

def decode_image_base64_to_pil(image_base64: str) -> Image:
    """Decode a Base64 encoded image string to a PIL Image object."""
    try:
        image_data = base64.b64decode(image_base64)
        image = Image.open(io.BytesIO(image_data)).convert('RGB')
        image_cv = np.array(image)
        return Image.fromarray(cv2.cvtColor(image_cv, cv2.COLOR_RGB2BGR))
    except Exception as e:
        logger.error(f"Image decoding failed: {e}")
        raise ValueError("Invalid Base64 image string")

def process_images(images_dict: dict) -> list:
    """Process the input image dictionary and return a list of resized PIL Images."""
    try:
        image_keys = ['cam_head', 'cam_right_wrist', 'cam_left_wrist']
        processed_list = [decode_image_base64_to_pil(images_dict[k]).resize((320, 240)) for k in image_keys]
        for key, img in zip(image_keys, processed_list):
            img.save(f'/share/project/dumengfei/code/real_eval/image_log/r1lite_{key}.png')
        return processed_list
    except KeyError as e:
        raise ValueError(f"Missing required image: {e}")
    except Exception as e:
        logger.error(f"An error occurred during image processing: {e}")
        raise ValueError("Image processing failed")

# --- Flask API Endpoints ---

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint."""
    if model is None:
        return jsonify({"status": "error", "message": "Model not loaded"}), 503
    return jsonify({"status": "healthy", "model_loaded": True, "subtask_mode": SUBTASK_MODE})

@app.route('/info', methods=['GET'])
def service_info():
    """Provides service metadata."""
    return jsonify({
        "service_name": "RoboBrain Robotics API for R1-Lite",
        "version": "1.0.0",
        "subtask_mode": SUBTASK_MODE,
        "model_path": MODEL_PATH,
        "endpoints": {"/health": "GET", "/info": "GET", "/infer": "POST"}
    })

@app.route('/infer', methods=['POST'])
def infer_api():
    """Main inference API endpoint."""
    start_time = time.time()
    
    if model is None:
        return jsonify({"success": False, "error": "Model not loaded"}), 503
    
    data = request.get_json()
    if not data or 'eef_pose' not in data or 'instruction' not in data or 'images' not in data:
        return jsonify({"success": False, "error": "Incomplete or malformed request data"}), 400

    try:
        instruction = data['instruction']
        images_dict = data['images']
        eef_pose = np.array(data['eef_pose'])
        images_pil = process_images(images_dict)

        # --- Prompt Generation ---
        if SUBTASK_MODE:
            prompt_template = (
                "You are controlling an r1lite dual-arm robot. Your task is to adjust the end effector (EEF) poses at 30Hz to complete a specified task. "
                "Your output must include two components: 1. Immediate sub-task: The specific action you will execute first to progress toward the overall task; 2. Control tokens: These will be decoded into a 30×14 action sequence to implement the sub-task. The action sequence has 30 consecutive actions, each with 14 dimensions. The first 7 dimensions control the right arm EEPose and the last 7 dimensions control the left arm EEPose. Each EEPose here includes 3 delta position(xyz) + 3 delta orientation(axis-angle) + 1 gripper(opening range)\n\nYour current visual inputs are: "
            )
        else:
            prompt_template = (
                "You are controlling an r1lite dual-arm robot. Your task is to adjust the end effector (EEF) poses at 30Hz to complete a specified task. "
                "You need to output control tokens that can be decoded into a 30×14 action sequence. "
                "The sequence has 30 consecutive actions, each with 14 dimensions. The first 7 dimensions control the right arm EEF, and the last 7 dimensions control the left arm EEF. "
                "Each EEPose here includes 3 delta position(xyz) + 3 delta orientation(axis-angle) + 1 gripper(opening range)\n\nYour current visual inputs are: "
            )

        content = [
            {"type": "text", "text": prompt_template},
            {"type": "text", "text": "robot front image"},
            {"type": "image", "image": f"data:image;base64,{images_dict['cam_head']}"},
            {"type": "text", "text": ", right wrist image"},
            {"type": "image", "image": f"data:image;base64,{images_dict['cam_right_wrist']}"},
            {"type": "text", "text": " and left wrist image"},
            {"type": "image", "image": f"data:image;base64,{images_dict['cam_left_wrist']}"},
            {"type": "text", "text": f"\nYour overall task is: {instruction.lower()}."},
        ]
        
        messages = [{"role": "user", "content": content}]
        text_prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = processor(text=[text_prompt], images=images_pil, padding=True, return_tensors="pt").to(model.device)

        # --- Model Inference ---
        gen_kwargs = {"max_new_tokens": 768, "do_sample": False, "temperature": 0.0, "pad_token_id": processor.tokenizer.pad_token_id, "eos_token_id": processor.tokenizer.eos_token_id}
        with torch.no_grad():
            output_tokens = model.generate(**inputs, **gen_kwargs)[0, inputs.input_ids.shape[1]:].detach().cpu().tolist()

        # --- Output Parsing ---
        subtask_result = "N/A"
        action_tokens_raw = output_tokens
        if SUBTASK_MODE:
            try:
                boa_token = 151665
                split_index = output_tokens.index(boa_token)
                subtask_tokens = output_tokens[:split_index]
                action_tokens_raw = output_tokens[split_index + 1:]
                subtask_result = processor.tokenizer.decode(subtask_tokens, skip_special_tokens=True).strip()
                logger.info(f"Parsed subtask: {subtask_result}")
            except ValueError:
                logger.warning("Could not find <boa> token. Treating entire output as actions.")
                subtask_result = "Parsing failed: <boa> token not found"

        try:
            eoa_token = 151667
            end_index = action_tokens_raw.index(eoa_token)
            action_tokens_raw = action_tokens_raw[:end_index]
        except ValueError:
            logger.warning("Could not find <eoa> token. Using full output sequence.")

        action_ids = [t - 149595 for t in action_tokens_raw if 149595 <= t < 151643]
        actions_norm, _ = action_tokenizer._extract_actions_from_tokens([action_ids], action_horizon=30, action_dim=14)
        delta_actions = actions_norm[0]
        
        # --- Action Post-processing ---
        if delta_actions is None or action_stats is None:
             raise ValueError("Action decoding failed or normalization stats not loaded")

        action_key = 'action.eepose' if SUBTASK_MODE else 'action.eepose.delta'
        scale = np.array(action_stats[action_key]['scale_'])
        offset = np.array(action_stats[action_key]['offset_'])
        delta_actions_denorm = inverse_transform(np.array(delta_actions), scale, offset)
        
        final_ee_actions = []
        current_eef_pose = eef_pose.copy()
        for i in range(30):
            # Right arm update
            current_eef_pose[:3] += delta_actions_denorm[i][:3]
            current_eef_pose[3:7] = add_delta_to_quat_pose(current_eef_pose[3:7], delta_actions_denorm[i][3:6])
            current_eef_pose[7] = delta_actions_denorm[i][6] # Gripper
            # Left arm update
            current_eef_pose[8:11] += delta_actions_denorm[i][7:10]
            current_eef_pose[11:15] = add_delta_to_quat_pose(current_eef_pose[11:15], delta_actions_denorm[i][10:13])
            current_eef_pose[15] = delta_actions_denorm[i][13] # Gripper
            
            final_ee_actions.append(current_eef_pose.tolist())

        processing_time = time.time() - start_time
        logger.info(f"Inference complete in {processing_time:.2f}s. Mode: {'Subtask' if SUBTASK_MODE else 'Standard'}")
        
        response = {"success": True, "eepose": final_ee_actions, "processing_time": processing_time}
        if SUBTASK_MODE:
            response["subtask"] = subtask_result

        return jsonify(response)

    except Exception as e:
        logger.error(f"A critical error occurred during inference: {e}", exc_info=True)
        return jsonify({"success": False, "error": str(e)}), 500

# --- Main Program Entry ---
if __name__ == '__main__':
    if not load_model():
        sys.exit(1)
    
    logger.info("RoboBrain R1-Lite API Service starting...")
    logger.info(f"Listening on http://{SERVICE_CONFIG['host']}:{SERVICE_CONFIG['port']}")
    logger.info(f"Current Mode: {'Subtask' if SUBTASK_MODE else 'Standard'}")
    
    app.run(
        host=SERVICE_CONFIG['host'],
        port=SERVICE_CONFIG['port'],
        debug=SERVICE_CONFIG['debug'],
        threaded=SERVICE_CONFIG['threaded']
    )