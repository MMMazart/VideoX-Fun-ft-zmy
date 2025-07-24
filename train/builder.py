# train/builder.py

import os
from transformers import AutoTokenizer
from models import AutoencoderKLWan, CLIPModel, WanT5EncoderModel, WanTransformer3DModel
# from models.wan_image_encoder import CLIPModel

from accelerate import Accelerator, AcceleratorState
from diffusers import FlowMatchEulerDiscreteScheduler
from omegaconf import OmegaConf
from transformers.utils import ContextManagers
import logging
from accelerate.utils import set_seed
import numpy as np
import torch
from utils.lora_utils import create_network # 导入 LoRA 网络创建工具
from .utils_train import filter_kwargs,get_logger
logger = get_logger(__name__)

def setup_third_party_logging(accelerator):
    """
    设置第三方库（如 datasets, transformers, diffusers）的日志级别。
    主进程输出更详细的警告信息，而其他进程只输出错误信息，以保持日志整洁。
    """
    import datasets, transformers, diffusers

    if accelerator.is_local_main_process:
        datasets.utils.logging.set_verbosity_warning()
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        datasets.utils.logging.set_verbosity_error()
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()

def init_tokenizer(args):
    """初始化并返回分词器 (Tokenizer)。"""
    # 从模型路径下的 'tokenizer' 子目录加载
    tokenizer_subpath = args['text_encoder_kwargs'].get('tokenizer_subpath', 'tokenizer')
    tokenizer = AutoTokenizer.from_pretrained(
        os.path.join(args.pretrained_model_name_or_path, tokenizer_subpath),
    )
    return tokenizer

def init_text_encoder(args):
    weight_dtype = torch.float16 if args.mixed_precision == "fp16" else torch.bfloat16 if args.mixed_precision == "bf16" else torch.float32
    def zero3_disable_context():
        """
        在 DeepSpeed ZeRO-3 模式下，提供一个上下文管理器来临时禁用权重分片。
        这对于加载不需要训练的、完整的模型至关重要。
        """
        if not AcceleratorState().deepspeed_plugin:
            return []
        return [AcceleratorState().deepspeed_plugin.zero3_init_context_manager(enable=False)]

    with ContextManagers(zero3_disable_context()):
        # 加载 Text encoder
        text_encoder = WanT5EncoderModel.from_pretrained(
            os.path.join(args.pretrained_model_name_or_path, args['text_encoder_kwargs'].get('text_encoder_subpath', 'text_encoder')),
            additional_kwargs=OmegaConf.to_container(args['text_encoder_kwargs']),
            low_cpu_mem_usage=True, # 启用低 CPU 内存使用模式
            torch_dtype=weight_dtype, # 设置权重的数据类型
        ).eval() # 设置为评估模式，因为我们不训练它

    return text_encoder

def init_vae(args):

    def zero3_disable_context():
        """为 VAE 提供与 Text Encoder 相同的 ZeRO-3 禁用上下文。"""
        if not AcceleratorState().deepspeed_plugin:
            return []
        return [AcceleratorState().deepspeed_plugin.zero3_init_context_manager(enable=False)]

    with ContextManagers(zero3_disable_context()): 
        #加载 VAE encoder
        vae = AutoencoderKLWan.from_pretrained(
            os.path.join(args.pretrained_model_name_or_path, args['vae_kwargs'].get('vae_subpath', 'vae')),
            additional_kwargs=OmegaConf.to_container(args['vae_kwargs']),
        ).eval()
    #这里要不要eval?,而且没有加weight_dtype
    return vae

def init_transformer3d(args):
    weight_dtype = torch.float16 if args.mixed_precision == "fp16" else torch.bfloat16 if args.mixed_precision == "bf16" else torch.float32
    transformer3d = WanTransformer3DModel.from_pretrained(
    os.path.join(args.pretrained_model_name_or_path, args['transformer_additional_kwargs'].get('transformer_subpath', 'transformer')),
    transformer_additional_kwargs=OmegaConf.to_container(args['transformer_additional_kwargs']),
    ).to(weight_dtype)  # 将模型转换为指定的权重类型
    return transformer3d

def init_clip_image_encoder(args):
    """根据训练模式，初始化并返回 CLIP 图像编码器。"""
    if args.train_mode != "normal":
    # 加载 Clip Image Encoder
        clip_image_encoder = CLIPModel.from_pretrained(
        os.path.join(args.pretrained_model_name_or_path, args['image_encoder_kwargs'].get('image_encoder_subpath', 'image_encoder'))
        ).eval()
    else:
        clip_image_encoder = None
    return clip_image_encoder

def init_lora(args,text_encoder,transformer3d):
    """创建 LoRA 网络并将其应用到 Text Encoder 和 Transformer 中。"""
    network = create_network(
        1.0,
        args.rank, # LoRA 网络的秩
        args.network_alpha, # LoRA 的 alpha 缩放因子
        text_encoder,
        transformer3d,
        neuron_dropout=None,
        skip_name=args.lora_skip_name,
    )
    # 将 LoRA 模块注入到目标模型层中
    network.apply_to(
        text_encoder,
        transformer3d,
        args.train_text_encoder and not args.training_with_video_token_length, # 是否训练 Text Encoder 的 LoRA 层
        True, # 是否训练 Transformer 的 LoRA 层
    )
    return network

def init_accelerator(args):
    """初始化 Hugging Face Accelerator。"""
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps, # 梯度累积步数
        mixed_precision=args.mixed_precision, # 混合精度类型 (fp16, bf16)
        log_with=args.report_to, # 日志记录目标 (如 tensorboard, wandb)
        project_config=args.accelerator_project_config, # 项目配置
    )
    return accelerator

def init_optimizer(args,network):
    """为 LoRA 网络的可训练参数初始化优化器。"""

    optimizer_cls = torch.optim.AdamW
    logger.info("Adding network parameters to optimizer")
    trainable_params = list(filter(lambda p: p.requires_grad, network.parameters()))
    trainable_params_optim = network.prepare_optimizer_params(args.learning_rate / 2, args.learning_rate, args.learning_rate)
    optimizer = optimizer_cls(
            trainable_params_optim,
            lr=args.learning_rate,
            betas=(args.adam_beta1, args.adam_beta2),
            weight_decay=args.adam_weight_decay,
            eps=args.adam_epsilon,
        )
    return optimizer

def load_vae_transformer(args,vae,transformer3d):
    """从指定的检查点路径加载 VAE 和 Transformer 的权重。"""
    if args.transformer_path is not None:
        logger.info(f"Loading Transformer from checkpoint: {args.transformer_path}")
        if args.transformer_path.endswith("safetensors"):
            from safetensors.torch import load_file, safe_open
            state_dict = load_file(args.transformer_path)
        else:
            state_dict = torch.load(args.transformer_path, map_location="cpu")
        state_dict = state_dict["state_dict"] if "state_dict" in state_dict else state_dict

        m, u = transformer3d.load_state_dict(state_dict, strict=False)
        logger.info(f"Transformer loaded. Missing keys: {len(m)}, Unexpected keys: {len(u)}")
        assert len(u) == 0, "Unexpected keys found in Transformer state_dict"

    if args.vae_path is not None:
        logger(f"Loading VAE from checkpoint: {args.vae_path}")
        if args.vae_path.endswith("safetensors"):
            from safetensors.torch import load_file, safe_open
            state_dict = load_file(args.vae_path)
        else:
            state_dict = torch.load(args.vae_path, map_location="cpu")
        state_dict = state_dict["state_dict"] if "state_dict" in state_dict else state_dict

        m, u = vae.load_state_dict(state_dict, strict=False)
        logger.info(f"VAE loaded. Missing keys: {len(m)}, Unexpected keys: {len(u)}")
        assert len(u) == 0, "Unexpected keys found in VAE state_dict"
    return vae,transformer3d

def init_models(args, config):
    """
    主初始化函数，协调所有组件的创建和设置。
    """      
    accelerator = init_accelerator(args)

    #这里后面好像要启动report_to日志
    deepspeed_plugin = accelerator.state.deepspeed_plugin

    if deepspeed_plugin is not None:
        zero_stage = int(deepspeed_plugin.zero_stage)
        logger.info(f"Using DeepSpeed Zero stage: {zero_stage}")
    else:
        zero_stage = 0
        logger.info("DeepSpeed is not enabled.")
    '''
    Stage 0：不使用 ZeRO，所有 GPU 复制完整的模型状态。
    Stage 1：优化器状态（如动量、方差）分片存储。
    Stage 2：额外将梯度分片存储。
    Stage 3：进一步将模型参数分片存储，实现最高级别的内存优化。
    在 ZeRO-3 模式下：

    所有要训练的模型必须在 accelerate.prepare() 之前就初始化好

    如果多个模型（比如 text_encoder、vae、transformer）都用一个 Accelerator.prepare()，容易冲突

    所以常见做法是：

    冻结的模型用 ContextManager(zero3_disable_context()) 包裹

    训练的模型（transformer3d）用 单独的 Accelerator.prepare()
    '''
    #这个逻辑暂未开启
    if zero_stage == 3:
        accelerator_transformer3d = Accelerator(
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            mixed_precision=args.mixed_precision,
            project_config=args.accelerator_project_config,
        )
    logger.info(accelerator.state, main_process_only=False)

    # 设置第三方库的日志
    setup_third_party_logging(accelerator)
    
    if args.seed is not None:
        set_seed(args.seed)
        # 为 numpy 和 torch 创建独立的、基于进程ID的随机数生成器
        rng = np.random.default_rng(np.random.PCG64(args.seed + accelerator.process_index))
        torch_rng = torch.Generator(accelerator.device).manual_seed(args.seed + accelerator.process_index)
    else:
        rng = None
        torch_rng = None

    # 根据混合精度设置确定权重的数据类型
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
        args.mixed_precision = accelerator.mixed_precision
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16
        args.mixed_precision = accelerator.mixed_precision 
 
    # 初始化噪声调度器
    noise_scheduler = FlowMatchEulerDiscreteScheduler(
        **filter_kwargs(FlowMatchEulerDiscreteScheduler, OmegaConf.to_container(config['scheduler_kwargs']))
    )

    # --- 按顺序初始化所有模型组件 ---
    tokenizer = init_tokenizer(args)    
    text_encoder = init_text_encoder(args)
    vae = init_vae(args)
    transformer3d = init_transformer3d(args)
    clip_image_encoder = init_clip_image_encoder(args)


    # --- 冻结所有基础模型的权重 ---
    # 这是 LoRA 训练的核心：只训练注入的 LoRA 层，而不改变原始模型
    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)
    transformer3d.requires_grad_(False)
    if clip_image_encoder:
        clip_image_encoder.requires_grad_(False)

    # 在冻结基础模型后，初始化 LoRA 网络
    network = init_lora(args,text_encoder,transformer3d)

    # 为 LoRA 网络的可训练参数创建优化器
    # 如果启用学习率缩放，则根据 batch size 和进程数量调整学习率 
    if args.scale_lr:
        args.learning_rate = (
        args.learning_rate * args.gradient_accumulation_steps * args.train_batch_size * accelerator.num_processes
        ) 
    optimizer = init_optimizer(args,network)
    
    return (
    accelerator,
    tokenizer,
    text_encoder,
    vae,
    transformer3d,
    clip_image_encoder,
    network,
    optimizer,
    noise_scheduler,
    zero_stage,
    torch_rng,
    weight_dtype
    )

