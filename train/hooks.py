# train/hooks.py
import os
import pickle  # 用于序列化和反序列化 Python 对象
from packaging import version  # 用于比较版本号
import accelerate
import logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
def save_model(ckpt_file, unwrapped_nw, weight_dtype):
    """
    一个辅助函数，用于保存 LoRA 网络的权重。
    
    Args:
        ckpt_file (str): 检查点文件的保存路径。
        unwrapped_nw: 被 accelerator.unwrap_model() 解包后的 LoRA 网络模型。
        weight_dtype: 保存权重时使用的数据类型 (如 torch.float16)。
    """
    # 调用 LoRA network 对象自身的 save_weights 方法进行保存
    unwrapped_nw.save_weights(ckpt_file, weight_dtype, None)

def build_hooks(args, accelerator, zero_stage,training_state):
    """
    构建并向 Accelerator 注册自定义的保存和加载钩子。
    
    Args:
        args: 训练参数。
        accelerator: Accelerator 实例。
        zero_stage (int): DeepSpeed ZeRO 优化的阶段。
        training_state
    """
    # `accelerate` 0.16.0 will have better support for customized saving
    if version.parse(accelerate.__version__) >= version.parse("0.16.0"):
        # --- 根据是否为 DeepSpeed ZeRO Stage 3 定义不同的钩子逻辑 ---
        if zero_stage != 3:
            # --- 非 ZeRO-3 模式下的钩子 ---
            # 在这种模式下，主进程拥有完整的模型权重，可以手动保存。
            def save_model_hook(models, weights, output_dir):
                """
                在 `accelerator.save_state` 执行前调用的钩子，用于保存模型。
                
                Args:
                    models (list): Accelerator 准备的模型列表。
                    weights (list): Accelerator 准备的权重列表。
                    output_dir (str): 保存检查点的目录。
                """
                if accelerator.is_main_process:
                    # 1. 手动保存 LoRA 网络权重为 safetensors 文件
                    safetensor_save_path = os.path.join(output_dir, f"lora_diffusion_pytorch_model.safetensors")
                    # models[-1] 通常是我们最后 prepare 的 LoRA network
                    save_model(safetensor_save_path, accelerator.unwrap_model(models[-1]))
                    accelerator.print(f"\nsaving checkpoint: {safetensor_save_path}")
                    
                    # 2. 从 weights 列表中移除 LoRA 网络的权重
                    # 因为我们已经手动保存了，所以要防止 Accelerator 再存一遍，造成冗余。
                    if not args.use_deepspeed:
                        for _ in range(len(weights)):
                            weights.pop()
                    # 3. 保存数据采样器的状态
                    # 这对于精确恢复数据加载进度至关重要
                    _pos_start = training_state['batch_sampler'].sampler._pos_start
                    current_epoch = training_state['current_epoch']
                    with open(os.path.join(output_dir, "sampler_pos_start.pkl"), 'wb') as file:
                        pickle.dump([_pos_start, current_epoch], file)

            def load_model_hook(models, input_dir):
                """
                在 `accelerator.load_state` 执行前调用的钩子，用于加载自定义状态。
                
                Args:
                    models (list): 模型列表。
                    input_dir (str): 加载检查点的目录。
                """
                # 加载数据采样器的状态
                pkl_path = os.path.join(input_dir, "sampler_pos_start.pkl")
                if os.path.exists(pkl_path):
                    with open(pkl_path, 'rb') as file:
                        loaded_number, _ = pickle.load(file)
                         # 恢复采样器的起始位置。这里稍微回退一点是为了给 dataloader worker 预留缓冲，防止中断后数据加载出错。
                        batch_sampler = training_state['batch_sampler']
                        batch_sampler.sampler._pos_start = max(loaded_number - args.dataloader_num_workers * accelerator.num_processes * 2, 0)
                    logger.info(f"Load pkl from {pkl_path}. Get loaded_number = {loaded_number}.")
        else:
            # --- ZeRO-3 模式下的钩子 ---
            # 在这种模式下，模型参数被分片到各个 GPU，主进程没有完整的模型。
            # 因此，不能手动保存模型，必须依赖 `accelerator.save_state` 自身来整合和保存。
            def save_model_hook(models, weights, output_dir):
                """ZeRO-3 模式下的保存钩子。"""
                if accelerator.is_main_process:
                    # 在 ZeRO-3 模式下，此钩子唯一的工作就是保存数据采样器的状态。
                    # 模型的保存完全交给 Accelerator 处理。
                    _pos_start = training_state['batch_sampler'].sampler._pos_start
                    current_epoch = training_state['current_epoch']
                    with open(os.path.join(output_dir, "sampler_pos_start.pkl"), 'wb') as file:
                        pickle.dump([_pos_start, current_epoch], file)

            def load_model_hook(models, input_dir):
                """ZeRO-3 模式下的加载钩子，逻辑与非 ZeRO-3 模式相同。"""
                pkl_path = os.path.join(input_dir, "sampler_pos_start.pkl")
                if os.path.exists(pkl_path):
                    with open(pkl_path, 'rb') as file:
                        loaded_number, _ = pickle.load(file)
                        batch_sampler = training_state['batch_sampler']
                        batch_sampler.sampler._pos_start = max(loaded_number - args.dataloader_num_workers * accelerator.num_processes * 2, 0)
                    logger.info(f"Load pkl from {pkl_path}. Get loaded_number = {loaded_number}.")

        accelerator.register_save_state_pre_hook(save_model_hook)
        accelerator.register_load_state_pre_hook(load_model_hook)
