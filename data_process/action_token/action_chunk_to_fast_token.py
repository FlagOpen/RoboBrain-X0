#!/usr/bin/env python3
"""
Action Chunk to Fast Token 转换程序
"""
 
import numpy as np
import sys
import os
from typing import Dict, List
from scipy.fft import idct
# from action_token.tokenizer import FASTTokenizer
from transformers import AutoProcessor
script_dir = os.path.dirname(os.path.abspath(__file__))
tokenizer_path = os.path.join(script_dir, "fast")

class ActionChunkProcessor:
    """Action Chunk处理器"""
    
    def __init__(self, max_len: int = 256, fast_tokenizer_path: str = tokenizer_path):
        self.max_len = max_len
        
        # 初始化FAST tokenizer
        self.fast_tokenizer = AutoProcessor.from_pretrained(fast_tokenizer_path, trust_remote_code=True)
        # print(f"✓ FAST tokenizer加载成功")

        # 初始化FASTTokenizer
        # self.tokenizer = FASTTokenizer(max_len=max_len, fast_tokenizer_path=fast_tokenizer_path)
        # print(f"✓ FASTTokenizer初始化成功")
    
    def create_dummy_batch(self, batch_size: int = 2, action_horizon: int = 50, action_dim: int = 32) -> Dict:
        """创建dummy batch数据"""
        dummy_batch = {
            "prompt": [
                "Pick up the red block from the table",
                "Move the blue object to the left"
            ][:batch_size],
            "state": np.random.randn(batch_size, action_dim).astype(np.float32),
            "actions": np.random.randn(batch_size, action_horizon, action_dim).astype(np.float32),
        }
        return dummy_batch
    
    def process_action_chunk_to_fast_token(self, action_chunk: np.ndarray) -> List[List[int]]:
        """将action chunk转换为fast token"""
        fast_tokens = self.fast_tokenizer(action_chunk)
        return fast_tokens
    
    # def tokenize_with_context(self, prompt: str, state: np.ndarray, actions: np.ndarray):
    #     """使用上下文进行tokenization"""
    #     tokens, token_masks, ar_masks, loss_masks = self.tokenizer.tokenize(prompt, state, actions)
    #     return tokens, token_masks, ar_masks, loss_masks
    
    # def extract_actions_from_tokens(self, action_tokens: List[List[int]], action_horizon: int, action_dim: int) -> np.ndarray:
    #     """从tokens中提取动作（detokenize验证）"""
    #     # actions = self.tokenizer.extract_actions(tokens, action_horizon, action_dim)
    #     decode_actions, output_dims = self.fast_tokenizer.decode(action_tokens, time_horizon=action_horizon, action_dim=action_dim)
    #     return decode_actions, output_dims

    def extract_actions_from_tokens(self, action_tokens: List[List[int]], action_horizon: int, action_dim: int) -> np.ndarray:
        assert (
            action_horizon is not None and action_dim is not None
        ), "Tokenizer not initialized, call encode() once or pass in time_horizon and action_dim."

        decoded_actions = []
        output_dims = []
        for token in action_tokens:
            try:
                decoded_tokens = self.fast_tokenizer.bpe_tokenizer.decode(token)
                decoded_dct_coeff = np.array(list(map(ord, decoded_tokens))) + self.fast_tokenizer.min_token
                output_dim = len(decoded_dct_coeff)
                output_dims.append(output_dim)
                decoded_dct_coeff = decoded_dct_coeff.reshape(-1, action_dim)
                assert (
                    decoded_dct_coeff.shape
                    == (
                        action_horizon,
                        action_dim,
                    )
                ), f"Decoded DCT coefficients have shape {decoded_dct_coeff.shape}, expected ({action_horizon}, {action_dim})"
            except Exception as e:
                print(f"Error decoding tokens: {e}")
                print(f"Tokens: {token}")
                decoded_dct_coeff = np.zeros((action_horizon, action_dim))
            decoded_actions.append(idct(decoded_dct_coeff / self.fast_tokenizer.scale, axis=0, norm="ortho"))
        return np.stack(decoded_actions), output_dims

    def _extract_actions_from_tokens(self, action_tokens: List[List[int]], action_horizon: int, action_dim: int) -> np.ndarray:
        assert (
            action_horizon is not None and action_dim is not None
        ), "Tokenizer not initialized, call encode() once or pass in time_horizon and action_dim."

        decoded_actions = []
        output_dims = []
        for token in action_tokens:
            try:
                decoded_tokens = self.fast_tokenizer.bpe_tokenizer.decode(token)
                decoded_dct_coeff = np.array(list(map(ord, decoded_tokens))) + self.fast_tokenizer.min_token
                output_dim = len(decoded_dct_coeff)
                output_dims.append(output_dim)
                decoded_dct_coeff = decoded_dct_coeff.reshape(-1, action_dim)
                assert (
                    decoded_dct_coeff.shape
                    == (
                        action_horizon,
                        action_dim,
                    )
                ), f"Decoded DCT coefficients have shape {decoded_dct_coeff.shape}, expected ({action_horizon}, {action_dim})"
            except Exception as e:
                print(f"Error decoding tokens: {e}")
                print(f"Tokens: {token}")
                if len(decoded_dct_coeff.shape) == 2:
                    results = []
                    output_horizon = decoded_dct_coeff.shape[0]
                    needed_rows = 30
                    zero_rows_needed = max(0, needed_rows - output_horizon)
                    results.extend(decoded_dct_coeff[:needed_rows])
                    zero_vector = [0] * action_dim
                    results.extend([zero_vector for _ in range(zero_rows_needed)])
                    decoded_dct_coeff = np.array(results)
                else:
                    decoded_dct_coeff = np.zeros((action_horizon, action_dim))
            decoded_actions.append(idct(decoded_dct_coeff / self.fast_tokenizer.scale, axis=0, norm="ortho"))
        return np.stack(decoded_actions), output_dims

    def _extract_actions_from_tokens_v2(self, action_tokens: List[List[int]], action_horizon: int, action_dim: int) -> np.ndarray:
        assert (
            action_horizon is not None and action_dim is not None
        ), "Tokenizer not initialized, call encode() once or pass in time_horizon and action_dim."

        decoded_actions = []
        output_dims = []
        zero_nums = []
        for token in action_tokens:
            try:
                decoded_tokens = self.fast_tokenizer.bpe_tokenizer.decode(token)
                decoded_dct_coeff = np.array(list(map(ord, decoded_tokens))) + self.fast_tokenizer.min_token
                output_dim = len(decoded_dct_coeff)
                output_dims.append(output_dim)
                decoded_dct_coeff = decoded_dct_coeff.reshape(-1, action_dim)
                assert (
                    decoded_dct_coeff.shape
                    == (
                        action_horizon,
                        action_dim,
                    )
                ), f"Decoded DCT coefficients have shape {decoded_dct_coeff.shape}, expected ({action_horizon}, {action_dim})"
            except Exception as e:
                print(f"Error decoding tokens: {e}")
                print(f"Tokens: {token}")
                if len(decoded_dct_coeff.shape) == 2:
                    decoded_dct_coeff = decoded_dct_coeff
                else:
                    decoded_dct_coeff = np.zeros((action_horizon, action_dim))
            tmp_actions = idct(decoded_dct_coeff / self.fast_tokenizer.scale, axis=0, norm="ortho")
            if tmp_actions.shape == (action_horizon, action_dim):
                decoded_actions.append(tmp_actions)
                zero_nums.append(0)
            else:
                if action_horizon > tmp_actions.shape[0]:
                    delta_size = action_horizon - tmp_actions.shape[0] 
                    new_rows = np.zeros((delta_size, action_dim))
                    pad_actions = np.vstack((tmp_actions, new_rows))
                    decoded_actions.append(pad_actions)
                    zero_nums.append(delta_size)
                else:
                    truncate_actions = tmp_actions[:action_horizon, :]
                    decoded_actions.append(truncate_actions)
                    zero_nums.append(0)

            # decoded_actions.append(idct(decoded_dct_coeff / self.fast_tokenizer.scale, axis=0, norm="ortho"))
        return np.stack(decoded_actions), output_dims, zero_nums

    def validate_encode_decode(self, original_actions: np.ndarray, action_horizon: int, action_dim: int) -> Dict:
        """验证编码和解码的一致性"""
        print(f"🔍 验证编码解码一致性...")
        
        # 1. 直接使用FAST tokenizer编码
        fast_tokens = self.fast_tokenizer(original_actions[None])[0]
        
        # 2. 使用FAST tokenizer解码
        decoded_actions = self.fast_tokenizer.decode(
            [fast_tokens], 
            time_horizon=action_horizon, 
            action_dim=action_dim
        )[0]
        
        # 3. 计算误差
        mse_error = np.mean((original_actions - decoded_actions) ** 2)
        mae_error = np.mean(np.abs(original_actions - decoded_actions))
        
        print(f"  - MSE误差: {mse_error:.6f}")
        print(f"  - MAE误差: {mae_error:.6f}")
        
        return {
            "original_shape": original_actions.shape,
            "decoded_shape": decoded_actions.shape,
            "mse_error": mse_error,
            "mae_error": mae_error,
            "fast_tokens_length": len(fast_tokens)
        }
    
    def process_batch(self, dummy_batch: Dict) -> Dict:
        """处理整个batch的数据"""
        batch_size = len(dummy_batch["prompt"])
        results = {"processed_samples": []}
        
        for i in range(batch_size):
            sample_result = {"sample_id": i}
            
            try:
                # 1. 直接转换action chunk为fast token
                action_chunk = dummy_batch["actions"][i:i+1]
                fast_tokens = self.process_action_chunk_to_fast_token(action_chunk)
                sample_result["fast_tokens"] = fast_tokens[0]
                sample_result["fast_tokens_length"] = len(fast_tokens[0])
                
                # 2. 使用上下文进行tokenization
                tokens, token_masks, ar_masks, loss_masks = self.tokenize_with_context(
                    dummy_batch["prompt"][i],
                    dummy_batch["state"][i],
                    dummy_batch["actions"][i]
                )
                sample_result["context_tokens"] = tokens
                
                # 3. 从tokens中提取动作（detokenize验证）
                extracted_actions = self.extract_actions_from_tokens(
                    tokens, 
                    dummy_batch["actions"].shape[1], 
                    dummy_batch["actions"].shape[2]
                )
                sample_result["extracted_actions"] = extracted_actions
                
                # 4. 验证编码解码一致性
                validation_result = self.validate_encode_decode(
                    dummy_batch["actions"][i],
                    dummy_batch["actions"].shape[1],
                    dummy_batch["actions"].shape[2]
                )
                sample_result["validation"] = validation_result
                
                sample_result["success"] = True
                
            except Exception as e:
                sample_result["error"] = str(e)
                sample_result["success"] = False
            
            results["processed_samples"].append(sample_result)
        
        return results
    
    def print_results_summary(self, results: Dict):
        """打印处理结果摘要"""
        print("\n" + "="*50)
        print("🎯 处理结果摘要")
        print("="*50)
        
        successful_samples = 0
        total_fast_tokens = 0
        total_mse_error = 0
        total_mae_error = 0
        
        for sample in results["processed_samples"]:
            sample_id = sample["sample_id"]
            
            if sample.get("success", False):
                successful_samples += 1
                print(f"样本 {sample_id + 1}: ✅ 成功")
                print(f"  - Fast tokens长度: {sample['fast_tokens_length']}")
                print(f"  - 上下文tokens形状: {sample['context_tokens'].shape}")
                print(f"  - 提取动作形状: {sample['extracted_actions'].shape}")
                
                validation = sample["validation"]
                print(f"  - 编码解码MSE: {validation['mse_error']:.6f}")
                print(f"  - 编码解码MAE: {validation['mae_error']:.6f}")
                
                total_fast_tokens += sample['fast_tokens_length']
                total_mse_error += validation['mse_error']
                total_mae_error += validation['mae_error']
            else:
                print(f"样本 {sample_id + 1}: ❌ 失败")
        
        print(f"\n📊 统计: {successful_samples}/{len(results['processed_samples'])} 成功")
        if successful_samples > 0:
            print(f"平均Fast tokens长度: {total_fast_tokens/successful_samples:.1f}")
            print(f"平均MSE误差: {total_mse_error/successful_samples:.6f}")
            print(f"平均MAE误差: {total_mae_error/successful_samples:.6f}")


def main():
    """主函数"""
    print("🚀 Action Chunk to Fast Token 转换程序")
    
    try:
        # 初始化处理器
        processor = ActionChunkProcessor(max_len=256)
        
        # 创建dummy batch
        dummy_batch = processor.create_dummy_batch(batch_size=3, action_horizon=50, action_dim=32)
        
        # 处理batch
        results = processor.process_batch(dummy_batch)
        
        # 打印结果摘要
        processor.print_results_summary(results)
        
        print("\n🎉 程序执行完成！")
        
    except Exception as e:
        print(f"程序执行失败: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code) 
