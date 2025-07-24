# lora_train/trainer.py

import torch
import math
import os
import pickle  # 用于序列化 Python 对象
import shutil  # 用于文件操作，如删除目录
from tqdm import tqdm  # 用于显示进度条
from builder import init_models, load_vae_transformer  # 自定义模块，用于初始化模型和加载 VAE/Transformer
from hooks import build_hooks  # 自定义模块，用于构建acclerate里的钩子函数
from diffusers.optimization import get_scheduler  # 从 diffusers 库导入学习率调度器
from data.dataset import ImageVideoDataset  # 自定义的数据集类
from data.samplers import AspectRatioImageVideoSampler, RandomSampler, ImageVideoSampler  # 自定义的数据采样器
from data.constants import ASPECT_RATIO_632  # 存储预定义宽高比的常量
import random
import numpy as np
from data.collate import build_collate_fn  # 自定义的 collate_fn，用于打包 batch
from .engine import TrainingEngine  # 封装单步训练逻辑的引擎类
import logging

# 获取一个日志记录器实例
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

class Trainer:
    """封装了整个 LoRA 训练流程的类。"""
    def __init__(self, args):
        """
        Trainer 的构造函数，负责初始化所有训练组件。
        
        Args:
            args: 包含所有训练配置的参数对象。
        """
        self.args = args
        self.config = args

        # --- 1. 初始化模型、优化器和 Accelerator ---
        # init_models 是一个关键函数，它会设置好分布式训练环境 (accelerator)
        # 并初始化所有需要的模型组件。
        (
            self.accelerator,
            self.tokenizer,
            self.text_encoder,
            self.vae,
            self.transformer3d,
            self.clip_image_encoder,
            self.network,
            self.optimizer,
            self.noise_scheduler,
            self.zero_stage,
            self.torch_rng,
            self.weight_dtype
        ) = init_models(self.args, self.config)
        # (可选) 从外部检查点加载 VAE 和 Transformer3D 的权重
        # self.vae,self.transformer3d = load_vae_transformer(args,self.vae,self.transformer3d)

        # 构建自定义钩子 (hooks)
        self.training_state = {'batch_sampler': None, 'current_epoch': None} 
        build_hooks(self.args,self.accelerator,self.zero_stage,self.training_state)

        # 如果启用梯度检查点，可以减少显存使用，但会增加计算时间
        if args.gradient_checkpointing:
            self.transformer3d.enable_gradient_checkpointing()

        # --- 2. 加载和准备数据 ---
        # 创建数据集实例
        train_dataset = ImageVideoDataset(
        args.train_data_meta, args.train_data_dir,
        video_sample_size=args.video_sample_size, video_sample_stride=args.video_sample_stride, video_sample_n_frames=args.video_sample_n_frames, video_sample_fps=args.video_sample_fps,
        video_repeat=args.video_repeat, 
        image_sample_size=args.image_sample_size,
        enable_inpaint=True if args.train_mode != "normal" else False,
        )

        def worker_init_fn(_seed):
            """为 DataLoader 的 worker 设置随机种子的工厂函数，确保数据加载的可复现性。"""
            _seed = _seed * 256
            def _worker_init_fn(worker_id):
                print(f"worker_init_fn with {_seed + worker_id}")
                np.random.seed(_seed + worker_id)
                random.seed(_seed + worker_id)
            return _worker_init_fn
        
        # 构建用于打包数据的 collate 函数
        collate_fn = build_collate_fn(args, sample_n_frames_bucket_interval=self.vae.config.temporal_compression_ratio)
        # 创建用于批处理采样器的随机数生成器
        batch_sampler_generator = torch.Generator().manual_seed(args.seed)

        if args.enable_bucket:
            # --- 分桶采样策略 (AspectRatio Bucketing) ---
            # 目的是将相似宽高比的图片/视频分到同一个 batch，以减少填充，提高 GPU 利用率
            aspect_ratio_sample_size = {key : [x / 632 * args.video_sample_size for x in ASPECT_RATIO_632[key]] for key in ASPECT_RATIO_632.keys()}
            batch_sampler = AspectRatioImageVideoSampler(
                sampler=RandomSampler(train_dataset, generator=batch_sampler_generator), dataset=train_dataset.dataset, 
                batch_size=args.train_batch_size, train_folder = args.train_data_dir, drop_last=True,
                aspect_ratios=aspect_ratio_sample_size,
            )
            
            train_dataloader = torch.utils.data.DataLoader(
                train_dataset,
                batch_sampler=batch_sampler, # 使用分桶采样器
                collate_fn=collate_fn,
                persistent_workers=True if args.dataloader_num_workers != 0 else False,
                num_workers=args.dataloader_num_workers,
                worker_init_fn=worker_init_fn(args.seed + self.accelerator.process_index)
            )
        else:
            # --- 普通采样策略 ---
            batch_sampler = ImageVideoSampler(RandomSampler(train_dataset, generator=batch_sampler_generator), train_dataset, args.train_batch_size)
            
            train_dataloader = torch.utils.data.DataLoader(
            train_dataset,
            batch_sampler=batch_sampler,  # 使用普通采样器
            collate_fn = collate_fn,
            persistent_workers=True if args.dataloader_num_workers != 0 else False,
            num_workers=args.dataloader_num_workers,
            worker_init_fn=worker_init_fn(args.seed + self.accelerator.process_index)
            )
        self.train_dataset = train_dataset
        self.train_dataloader = train_dataloader
        self.batch_sampler = batch_sampler

        # --- 3. 计算训练总步数 ---
        overrode_max_train_steps = False
        # 每个 epoch 的更新步数 = 数据加载器长度 / 梯度累积步数
        self.num_update_steps_per_epoch = math.ceil(len(self.train_dataloader) / args.gradient_accumulation_steps)
        if args.max_train_steps is None:
            # 如果未指定总步数，则根据 epoch 数量计算
            args.max_train_steps = args.num_train_epochs * self.num_update_steps_per_epoch
            overrode_max_train_steps = True

        # --- 4. 设置学习率调度器 ---
        self.lr_scheduler = get_scheduler(
            name=args.lr_scheduler,
            optimizer=self.optimizer,
            num_warmup_steps=args.lr_warmup_steps,
            num_training_steps=args.max_train_steps,
        )

        # --- 5. 使用 Accelerator 准备核心组件 ---
        # `accelerator.prepare` 会自动处理模型、优化器和数据加载器在分布式环境下的封装
        self.network, self.optimizer, self.train_dataloader, self.lr_scheduler = self.accelerator.prepare(
            self.network, self.optimizer, self.train_dataloader, self.lr_scheduler
        )

        # 将不需要训练的模型移动到正确的设备和数据类型
        self.vae.to(self.accelerator.device, dtype=self.weight_dtype)
        self.transformer3d.to(self.accelerator.device, dtype=self.weight_dtype)
        if not args.enable_text_encoder_in_dataloader:
            self.text_encoder.to(self.accelerator.device)
        # if args.zero_stage == 3:
        #     transformer3d = self.accelerator_transformer3d.prepare(
        #     transformer3d
        # )
            
        # --- 6. 重新计算训练步数和周期数 ---
        # 在 `accelerator.prepare` 之后，Dataloader 的长度可能会因分布式采样而改变，所以需要重新计算
        self.num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
        if overrode_max_train_steps:
            args.max_train_steps = args.num_train_epochs * self.num_update_steps_per_epoch        
        # 根据最终的总步数，反向计算所需的总 epoch 数
        args.num_train_epochs = math.ceil(args.max_train_steps / self.num_update_steps_per_epoch)

        # --- 7. 初始化实验跟踪器 (如 TensorBoard, W&B) ---
        if self.accelerator.is_main_process:
            tracker_config = dict(vars(args))
            tracker_config.pop("validation_prompts", None) # 移除不需要跟踪的配置
            self.accelerator.init_trackers(args.tracker_project_name, tracker_config)

    def _load_pkl(checkpoint_path):
        """从 pickle 文件中加载 epoch 数，用于断点续训。"""
        pkl_path = os.path.join(checkpoint_path, "sampler_pos_start.pkl")
        if os.path.exists(pkl_path):
            with open(pkl_path, 'rb') as file:
                _, first_epoch = pickle.load(file)
                print(f"Load pkl from {pkl_path}. Get first_epoch = {first_epoch}.")
        else:
            first_epoch = None
        return first_epoch
    def _resume_from_checkpoint(self):

        """处理从检查点恢复训练的逻辑。"""
        if not self.args.resume_from_checkpoint:
            return 0, 0 # # 如果不恢复，则从第0步、第0个epoch开始;initial_global_step, first_epoch
        # 如果指定为 "latest"，则自动查找最新的检查点
        if self.args.resume_from_checkpoint == "latest":
            dirs = [d for d in os.listdir(self.args.output_dir) if d.startswith("checkpoint")]
            if not dirs:
                self.accelerator.print("No 'latest' checkpoint found. Starting new training run.")
                return 0, 0
            # 按步数排序，找到最大的那个
            path = sorted(dirs, key=lambda x: int(x.split("-")[1]))[-1]
        else:
            path = os.path.basename(self.args.resume_from_checkpoint)

        
        checkpoint_path = os.path.join(self.args.output_dir, path)
        if not os.path.exists(checkpoint_path):
            self.accelerator.print(f"Checkpoint '{path}' does not exist. Starting new training run.")
            self.args.resume_from_checkpoint = None
            return 0, 0

        self.accelerator.print(f"Resuming from checkpoint {checkpoint_path}")
        self.accelerator.load_state(checkpoint_path)
        # 从检查点目录名中解析全局步数
        global_step = int(path.split("-")[1])  
        # 尝试从 pkl 文件中加载 epoch 数
        _, first_epoch = self._load_pkl(checkpoint_path)
        if first_epoch is None:
            # 如果 pkl 不存在，则根据全局步数估算
            first_epoch = global_step // self.num_update_steps_per_epoch
        
        # 如果不是 ZeRO Stage 3，需要手动加载 LoRA 网络的权重
        if self.args.zero_stage != 3:
            from safetensors.torch import load_file, safe_open
            state_dict = load_file(os.path.join(os.path.join(self.args.output_dir, path), "lora_diffusion_pytorch_model.safetensors"))
            # 解包模型并加载权重
            m, u = self.accelerator.unwrap_model(self.network).load_state_dict(state_dict, strict=False)
            print(f"missing keys: {len(m)}, unexpected keys: {len(u)}")

        return global_step, first_epoch        

    def _save_model(self, ckpt_file, unwrapped_nw, weight_dtype):
        """保存 LoRA 网络的权重。"""
        unwrapped_nw.save_weights(ckpt_file, dtype=weight_dtype)

    def _save_checkpoint(self,global_step):
        """保存检查点的核心逻辑。"""
        # 只在主进程上执行保存操作
        if not self.accelerator.is_main_process:
            return

        ## --- 管理检查点数量，防止占满磁盘 ---
        if self.args.checkpoints_total_limit is not None:
            checkpoints = sorted(
                [d for d in os.listdir(self.args.output_dir) if d.startswith("checkpoint")],
                key=lambda x: int(x.split("-")[1])
            )
            # 如果检查点数量超过限制，则删除最旧的
            if len(checkpoints) >= self.args.checkpoints_total_limit:
                num_to_remove = len(checkpoints) - self.args.checkpoints_total_limit + 1
                for ckpt_to_remove in checkpoints[:num_to_remove]:
                    ckpt_to_remove_path = os.path.join(self.args.output_dir, ckpt_to_remove)
                    shutil.rmtree(ckpt_to_remove_path) 

            if self.args.save_state:
                # 保存完整的训练状态（模型、优化器、调度器等），用于恢复训练
                accelerator_save_path = os.path.join(self.args.output_dir, f"checkpoint-{global_step}")
                self.accelerator.save_state(accelerator_save_path)
                logger.info(f"Saved state to {accelerator_save_path}")
            else:
                # 如果不保存完整状态，则只保存 LoRA 权重
                safetensor_save_path = os.path.join(self.args.output_dir, f"checkpoint-{global_step}.safetensors")
                unwrapped_network = self.accelerator.unwrap_model(self.network)
                self._save_model(safetensor_save_path, unwrapped_network,self.weight_dtype)
                logger.info(f"Saved safetensor to {safetensor_save_path}")  

    def _save_checkpoint_end(self,global_step):
        """在训练结束时保存最终的模型。"""
        if not self.accelerator.is_main_process:
            return
        
        # 保存 LoRA 权重
        safetensor_save_path = os.path.join(self.args.output_dir, f"checkpoint-{global_step}.safetensors")
        unwrapped_network = self.accelerator.unwrap_model(self.network)
        self._save_model(safetensor_save_path, unwrapped_network, self.weight_dtype)
        logger.info(f"Saved final model to {safetensor_save_path}")
        # 如果需要，也保存最终的训练状态
        if self.args.save_state:
            accelerator_save_path = os.path.join(self.args.output_dir, f"checkpoint-{global_step}")
            self.accelerator.save_state(accelerator_save_path)
            logger.info(f"Saved state to {accelerator_save_path}")
    
    def train(self):
        """主训练循环。"""
        # 实例化训练引擎，它封装了单步的 forward 和 backward 逻辑
        engine = TrainingEngine(
            self.args, self.weight_dtype, self.accelerator, self.tokenizer,
            self.text_encoder, self.vae, self.transformer3d, self.clip_image_encoder, self.noise_scheduler
        )
        args = self.args
        total_batch_size = self.args.train_batch_size * self.accelerator.num_processes * self.args.gradient_accumulation_steps
        
        # --- 打印训练信息 ---
        logger.info("***** Running training *****")
        logger.info(f"  Num examples = {len(self.train_dataset)}")
        logger.info(f"  Num Epochs = {args.num_train_epochs}")
        logger.info(f"  Instantaneous batch size per device = {args.train_batch_size}")
        logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
        logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
        logger.info(f"  Total optimization steps = {args.max_train_steps}")        
        
        # 尝试从检查点恢复，获取起始步数和 epoch
        initial_global_step, first_epoch = self._resume_from_checkpoint()
        global_step = initial_global_step

        # 设置 TQDM 进度条
        progress_bar = tqdm(
            range(initial_global_step, args.max_train_steps),
            disable=not self.accelerator.is_local_main_process,
            desc="Training Steps",
        )

        # 获取需要训练的参数
        trainable_params = list(filter(lambda p: p.requires_grad, self.network.parameters()))
        
        # --- 开始训练循环 ---
        for epoch in range(first_epoch, args.num_train_epochs):
            train_loss = 0.0
            # 为每个 epoch 设置不同的随机种子，以保证数据采样的随机性
            self.batch_sampler.sampler.generator = torch.Generator().manual_seed(args.seed + epoch)
            
            for step, batch in enumerate(self.train_dataloader): 
                if epoch==first_epoch and step==0:
                    print("健全性检查")

                # 使用 `accelerator.accumulate` 来自动处理梯度累积
                with self.accelerator.accumulate(self.transformer3d):
                    # 调用引擎执行单步训练，计算损失
                    loss = engine.train_step(batch,self.torch_rng)

                    # 收集所有进程的损失并计算平均值
                    avg_loss = self.accelerator.gather(loss.repeat(args.train_batch_size)).mean()
                    train_loss += avg_loss.item() / self.args.gradient_accumulation_steps

                    # 执行反向传播
                    self.accelerator.backward(loss)  

                    # 当梯度同步时（即累积了足够的步数后）                  
                    if self.accelerator.sync_gradients:
                        # (可选) 裁剪梯度，防止梯度爆炸
                        self.accelerator.clip_grad_norm_(trainable_params, args.max_grad_norm)
                    
                    # 更新模型权重
                    self.optimizer.step()
                    # 更新学习率
                    self.lr_scheduler.step()
                    # 清空梯度
                    self.optimizer.zero_grad()    

                # --- 日志记录和模型保存 ---
                # 只有在梯度同步（即完成一次优化步骤）后才执行
                if self.accelerator.sync_gradients:
                    progress_bar.update(1)
                    global_step += 1
                    
                    # 使用 accelerator 记录损失到跟踪器 (TensorBoard/W&B)
                    self.accelerator.log({"train_loss": train_loss}, step=global_step)
                    train_loss = 0.0 # 重置训练损失

                    # 按预设的步数保存检查点
                    if global_step % args.checkpointing_steps == 0:
                        #training_state给钩子函数提供访问路径
                        self.training_state['batch_sampler'] = self.batch_sampler
                        self.training_state['current_epoch'] = epoch
                        self._save_checkpoint(global_step)
                # 更新进度条的后缀信息
                logs = {"step_loss": loss.detach().item(), "lr": self.lr_scheduler.get_last_lr()[0]}
                progress_bar.set_postfix(**logs)

                # 如果达到最大训练步数，则跳出循环
                if global_step >= args.max_train_steps:
                    break

            # 如果内层循环已跳出，外层也应跳出，原本代码有bug，这里修复了
            if global_step >= args.max_train_steps:
                break
        # --- 训练结束 ---
        # 等待所有进程执行完毕
        self.accelerator.wait_for_everyone()
        # 保存最终的模型
        self.training_state['batch_sampler'] = self.batch_sampler
        self.training_state['current_epoch'] = epoch
        self._save_checkpoint_end(global_step)
        # 结束训练，关闭所有跟踪器 
        self.accelerator.end_training()
        
        return
