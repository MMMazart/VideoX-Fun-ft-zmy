# train/engine.py

import torch
import torch.nn.functional as F
import numpy as np
from einops import rearrange
from PIL import Image
import torchvision.transforms.functional as TF
from utils.discrete_sampler import DiscreteSampling
from diffusers.training_utils import compute_density_for_timestep_sampling,compute_loss_weighting_for_sd3
import math
import losses

class TrainingEngine:
    """
    封装单次训练步骤的逻辑，包括前向传播、损失计算和反向传播。
    """
    def __init__(
        self,
        args,
        weight_dtype,
        accelerator,
        tokenizer,
        text_encoder,
        vae,
        transformer3d,
        clip_image_encoder,
        noise_scheduler,
    ):
        """
        构造函数，接收并存储所有必要的组件。
        """
        self.args = args
        self.weight_dtype = weight_dtype
        self.accelerator = accelerator
        self.tokenizer = tokenizer
        self.text_encoder = text_encoder
        self.vae = vae
        self.transformer3d = transformer3d
        self.clip_image_encoder = clip_image_encoder
        self.noise_scheduler = noise_scheduler
        # 初始化一个用于离散时间步采样的工具
        self.idx_sampling = DiscreteSampling(args.train_sampling_steps, uniform_sampling=args.uniform_sampling)

    def _batch_encode_vae(self, pixel_values):
        """
        [辅助函数] 通过小批量（mini-batch）的方式对像素值进行VAE编码，以节省显存。
        这对于处理高分辨率视频数据尤其重要。
        """
        pixel_values = rearrange(pixel_values, "b f c h w -> b c f h w")
        latents = []
        # 将一个大批次切分成多个小批次进行处理
        for i in range(0, pixel_values.shape[0], self.args.vae_mini_batch):
            chunk = pixel_values[i : i + self.args.vae_mini_batch]
            # VAE 编码得到一个分布
            latent_dist = self.vae.encode(chunk)[0]
            # 从分布中采样得到潜变量
            latent_sample = latent_dist.sample()
            latents.append(latent_sample)
        return torch.cat(latents, dim=0)
    
    def _get_sigmas(self, timesteps, n_dim, dtype):
        """[辅助函数] 根据给定的时间步（timesteps），从噪声调度器中查找对应的 sigma 值。"""
        sigmas = self.noise_scheduler.sigmas.to(device=self.accelerator.device, dtype=dtype)
        schedule_timesteps = self.noise_scheduler.timesteps.to(self.accelerator.device)
        timesteps = timesteps.to(self.accelerator.device)
        # 找到 timesteps 在总时间表中的索引
        step_indices = [(schedule_timesteps == t).nonzero().item() for t in timesteps]
        sigma = sigmas[step_indices].flatten()
        # 扩展 sigma 的维度，使其能够与潜变量张量进行广播操作（例如，从 (B,) 扩展到 (B, 1, 1, 1, 1)）
        while len(sigma.shape) < n_dim:
            sigma = sigma.unsqueeze(-1)
        return sigma
    
    def train_step(self, batch, torch_rng, rng=None):
        """
        执行单次训练迭代的核心函数。

        返回:
            torch.Tensor: 计算出的损失值。
        """
        args = self.args        
        # --- 1. 数据准备：获取潜变量 (latents) 和条件 (prompt_embeds) ---
        pixel_values = batch["pixel_values"].to(self.weight_dtype)
        # [显存优化] 如果启用低显存模式，在需要时将模型移至GPU，用完后移回CPU
        if args.low_vram:
            self.vae.to(self.accelerator.device)
            if args.train_mode != "normal":
                self.clip_image_encoder.to(self.accelerator.device)
            if not args.enable_text_encoder_in_dataloader:
                self.text_encoder.to("cpu")
        with torch.no_grad():
            #这里没有要inpaint的逻辑
            # 将像素值通过 VAE 编码为潜变量
            latents = self._batch_encode_vae(pixel_values)
            # [显存优化] VAE 使用完毕，将其移回CPU以释放显存
            if args.low_vram:
                self.vae.to('cpu')
                if not args.enable_text_encoder_in_dataloader: 
                    self.text_encoder.to(self.accelerator.device)
                torch.cuda.empty_cache()

            # 获取文本嵌入
            if args.enable_text_encoder_in_dataloader:
                prompt_embeds = batch['encoder_hidden_states'].to(device=latents.device)
            else:
                prompt_ids = self.tokenizer(
                    batch['text'], 
                    padding="max_length", 
                    max_length=args.tokenizer_max_length, 
                    truncation=True, 
                    add_special_tokens=True, 
                    return_tensors="pt"
                )
                text_input_ids = prompt_ids.input_ids
                prompt_attention_mask = prompt_ids.attention_mask

                seq_lens = prompt_attention_mask.gt(0).sum(dim=1).long()
                prompt_embeds = self.text_encoder(text_input_ids.to(latents.device), attention_mask=prompt_attention_mask.to(latents.device))[0]
                prompt_embeds = [u[:v] for u, v in zip(prompt_embeds, seq_lens)]
            
            # [显存优化] Text Encoder 使用完毕，移回CPU
            if args.low_vram and not args.enable_text_encoder_in_dataloader:
                    self.text_encoder.to('cpu')
                    torch.cuda.empty_cache() 

            # --- 2. 扩散过程：采样时间步并添加噪声 ---             
            bsz, channel, num_frames, height, width = latents.size()
            noise = torch.randn(latents.size(), device=latents.device, generator=torch_rng, dtype=self.weight_dtype)
            # 时间步采样
            if not args.uniform_sampling:
                # 使用加权采样，更关注某些特定的噪声水平
                u = compute_density_for_timestep_sampling(
                    weighting_scheme=args.weighting_scheme,
                    batch_size=bsz,
                    logit_mean=args.logit_mean,
                    logit_std=args.logit_std,
                    mode_scale=args.mode_scale,
                )
                indices = (u * self.noise_scheduler.config.num_train_timesteps).long()
            else:
                # 均匀采样
                indices = self.idx_sampling(bsz, generator=torch_rng, device=latents.device)
                indices = indices.long().cpu()
            timesteps = self.noise_scheduler.timesteps[indices].to(device=latents.device)
            
            # --- 添加噪声 (基于 Flow Matching 的 ODE 路径) ---
            # 获取与时间步对应的 sigma 值
            sigmas = self._get_sigmas(timesteps, n_dim=latents.ndim, dtype=latents.dtype)
            # 根据公式 z_t = (1-sigma_t)*z_0 + sigma_t*z_1 生成带噪潜变量
            noisy_latents = (1.0 - sigmas) * latents + sigmas * noise
            target = noise - latents
            target_shape = (self.vae.latent_channels, num_frames, width, height)
            seq_len = math.ceil(
                (target_shape[2] * target_shape[3]) /
                (self.accelerator.unwrap_model(self.transformer3d).config.patch_size[1] * self.accelerator.unwrap_model(self.transformer3d).config.patch_size[2]) *
                target_shape[1]
            )

            # --- 3. 模型前向传播，进行预测 ---
            # 使用自动混合精度，加速计算并减少显存
            with torch.cuda.amp.autocast(dtype=self.weight_dtype):
                noise_pred = self.transformer3d(
                    x=noisy_latents,
                    context=prompt_embeds,
                    t=timesteps,
                    seq_len=seq_len,
                    y=None,
                    clip_fea=None, 
                )
            # --- 4. 损失计算 ---
            # 计算每个时间步的损失权重
            loss = losses.calculate_loss(
                noise_pred=noise_pred, 
                target=target, 
                sigmas=sigmas, 
                args=self.args
            )

            # weighting = compute_loss_weighting_for_sd3(weighting_scheme=args.weighting_scheme, sigmas=sigmas)
            # loss = self._custom_mse_loss(noise_pred.float(), target.float(), weighting.float())
            # loss = loss.mean()
            # if args.motion_sub_loss and noise_pred.size()[1] > 2:
            #     gt_sub_noise = noise_pred[:, 1:, :].float() - noise_pred[:, :-1, :].float()
            #     pre_sub_noise = target[:, 1:, :].float() - target[:, :-1, :].float()
            #     sub_loss = F.mse_loss(gt_sub_noise, pre_sub_noise, reduction="mean")
            #     loss = loss * (1 - args.motion_sub_loss_ratio) + sub_loss * args.motion_sub_loss_ratio
            # 5. 反向传播
            self.accelerator.backward(loss)
            return loss
