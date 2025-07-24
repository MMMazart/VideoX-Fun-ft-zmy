# data/collate.py
import numpy as np
import torch
import random
import transforms
# 从我们新建的模块中导入
from .transforms import VideoDataTransforms,ImageDataTransforms
from .constants import ASPECT_RATIO_512, ASPECT_RATIO_632, ASPECT_RATIO_RANDOM_CROP_PROB
from PIL import Image
from .utils_data import get_logger
#获取一个日志记录器实例
logger = get_logger(__name__)

def build_collate_fn(args, sample_n_frames_bucket_interval=1):
    """
    构建一个清晰、职责分离的 collate_fn。
    """
    def collate_fn(samples):
        # 定义图像和视频的变换流程
        video_transforms = transforms.Compose(
            [   # 将视频帧的短边缩放到指定尺寸
                transforms.Resize(min(args.video_sample_size)),
                transforms.CenterCrop(args.video_sample_size),
                # 将像素值从 [0, 1] 标准化到 [-1, 1]
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True),
            ]
        )
        image_transforms   = transforms.Compose([
            transforms.Resize(min(args.image_sample_size)),
            transforms.CenterCrop(args.image_sample_size),
            transforms.ToTensor(), # 将 PIL Image 转换为 [C, H, W] 的 Tensor，并将值缩放到 [0, 1]
            transforms.Normalize([0.5, 0.5, 0.5],[0.5, 0.5, 0.5])
        ])

        # 过滤掉加载失败的样本
        samples = [sample for sample in samples if sample is not None]
        if not samples:
            logger.info("Warning: Skipping a batch because all samples failed to load.")
            return None
        # 基于批次中的第一个样本来确定整个批次的数据类型
        pixel_value     = samples[0]["pixel_values"]
        data_type       = samples[0]["data_type"]
        first_sample_shape    = np.shape(pixel_value)  #f, h, w, c

        # 初始化输出字典
        new_samples = {
            "pixel_values": [],
            "text": []
        }
        for sample in samples:
            pixel_values = sample['pixel_values']
            if data_type=='image':
                transformed_pixels = image_transforms(pixel_values).unsqueeze(0)
            else:
                # 对视频数据进行预处理：转为 Tensor, 交换维度 (F,H,W,C) -> (F,C,H,W), 归一化
                pixel_values = torch.from_numpy(pixel_values).permute(0, 3, 1, 2).contiguous()
                pixel_values = pixel_values / 255.
                transformed_pixels = video_transforms(pixel_values)
            # 将处理后的数据和文本添加到输出字典
            new_samples["pixel_values"].append(transformed_pixels)
            new_samples["text"].append(sample["text"])
        new_samples["pixel_values"] = torch.stack(new_samples["pixel_values"])
        
        return new_samples
    
    # 2. 返回最终的 collate_fn 函数
    def collate_fn_bucket(samples):
        # 1. 实例化自定义的数据变换器类
        # 这些类可能包含了更复杂的逻辑，比如动态调整裁剪尺寸
        image_transforms = ImageDataTransforms(args,sample_n_frames_bucket_interval)
        video_transforms = VideoDataTransforms(args,sample_n_frames_bucket_interval)

        # 步骤 1: 安全地过滤掉加载失败的样本
        samples = [sample for sample in samples if sample is not None]
        if not samples:
            logger.info("Warning: Skipping a batch because all samples failed to load.")
            return None
        
        # 步骤 2：确定整个批次的属性（如数据类型和形状）
        pixel_value     = samples[0]["pixel_values"]
        data_type       = samples[0]["data_type"]

        if data_type == 'video':
            first_sample_shape = np.shape(pixel_value) # (F, H, W, C)
        else: #image下pixel_value本来是PIL属性
            arr = np.array(pixel_value)
            first_sample_shape = (1,) + arr.shape  # => (1, H, W, C)

        if data_type == 'image':
            active_transformer = image_transforms
        elif data_type == 'video':
            active_transformer = video_transforms

        # 步骤 3: 确定整个批次的目标尺寸 (Target Size)
        # 目标 token 长度 = 视频帧数 * token_sample_size^2
        target_token_length = (args.video_sample_n_frames
                               * args.token_sample_size
                               * args.token_sample_size)

        # 初始化输出字典
        new_samples = {
            "target_token_length": target_token_length,
            "pixel_values": [],
            "text": []
        }

        #动态调整策略，用于视频数据
        if data_type=='video' and args.random_hw_adapt and args.training_with_video_token_length:
            # 动态根据当前 batch 中各视频的最小均值尺寸选择帧数
            means = []
            for ex in samples:
                _, hh, ww, _ = np.shape(ex["pixel_values"])
                means.append((hh + ww) / 2)  # 计算每个视频的平均边长
            local_min_size = np.min(means)   # 找到批次中的最小值
            # 将这个最小值传递给变换器，变换器内部可能会用它来动态调整采样帧数或裁剪大小
            active_transformer.local_min_size = local_min_size

        # 步骤 4: 遍历批次中的每个样本，进行处理
        for sample in samples:
            # 确保图像数据是 Numpy 数组格式，并有伪帧维度
            if isinstance(sample["pixel_values"], Image.Image):
                sample["pixel_values"] = np.array(sample["pixel_values"])
                sample["pixel_values"] = np.expand_dims(np.array(sample["pixel_values"]), 0)

            # 根据配置决定是使用批次中第一个样本的形状还是当前样本的形状
            if not args.random_ratio_crop:
                # 如果不进行随机比例裁剪，则所有样本都参照第一个样本的形状进行处理
                origin_shape = first_sample_shape
            else:
                # 否则，每个样本根据自己的原始形状进行处理
                origin_shape = np.shape(sample["pixel_values"])

            transformed_pixels = active_transformer(sample, origin_shape)

 
            new_samples["pixel_values"].append(transformed_pixels)
            new_samples["text"].append(sample["text"])

        batch_video_length = active_transformer.batch_video_length    
        new_samples["pixel_values"] = torch.stack([sample[:batch_video_length] for sample in new_samples["pixel_values"]])    
        return new_samples
    
    if args.enable_bucket:
        return collate_fn_bucket
    else:
        return collate_fn 
