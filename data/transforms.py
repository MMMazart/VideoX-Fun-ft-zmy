# data/transforms.py
import random
from typing import Dict, List, Tuple

import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image
# 从常量模块导入预定义的宽高比字典和概率分布
from .constants import ASPECT_RATIO_632,ASPECT_RATIO_RANDOM_CROP_PROB,ASPECT_RATIO_RANDOM_CROP_632
from torchvision import transforms
from .samplers import get_closest_ratio_key

# =====================================================================================
# 1. 辅助函数
# =====================================================================================

def get_random_downsample_ratio(sample_size, image_ratio=[],
                            all_choices=False, rng=None):
    """
    根据给定的样本大小（sample_size），计算并返回一个随机的下采样率。
    这是一种数据增强技术，通过模拟不同分辨率的输入来提高模型的泛化能力。
    """
    def _create_special_list(length):
        """创建一个加权的概率列表，使得第一个元素（通常是1.0，即不下采样）被选中的概率最高。"""
        if length == 1:
            return [1.0]
        # 第一个元素的概率为 90%
        first_element = 0.90
        # 剩余 10% 的概率均分给其他元素
        remaining_sum = 0.10
        other = remaining_sum / (length - 1)
        return [first_element] + [other] * (length - 1)
    # 根据输入尺寸大小，确定可选的下采样率列表
    if sample_size >= 1536:
        number_list = [1, 1.25, 1.5, 2, 2.5, 3] + image_ratio
    elif sample_size >= 1024:
        number_list = [1, 1.25, 1.5, 2] + image_ratio
    elif sample_size >= 768:
        number_list = [1, 1.25, 1.5] + image_ratio
    elif sample_size >= 512:
        number_list = [1] + image_ratio
    else:
        number_list = [1]

    if all_choices:
        return number_list

    # 根据上面创建的加权概率，随机选择一个下采样率
    probs = np.array(_create_special_list(len(number_list)))
    if rng is None:
        return np.random.choice(number_list, p=probs)
    else:
        # 如果提供了随机数生成器 (rng)，则使用它以保证可复现性
        return rng.choice(number_list, p=probs)

class BaseDataTransforms:
    """
    一个封装了图像和视频通用数据变换逻辑的**基类**。
    它处理动态的裁剪、缩放和归一化。子类（VideoDataTransforms, ImageDataTransforms）
    需要根据各自的类型（视频或图像）来定义具体的尺寸计算逻辑。
    """
    def __init__(self, args, sample_n_frames_bucket_interval):
        self.args = args
        self.sample_n_frames_bucket_interval = sample_n_frames_bucket_interval
        # 以下这些属性将在子类中被具体赋值
        self.aspect_ratio_random_crop_sample_size = {} # 存储随机裁剪的目标尺寸
        self.aspect_ratio_sample_size = {}             # 存储标准裁剪的目标尺寸
        self.batch_video_length = 0                    # 记录批次内视频的统一帧长

    def _get_random_crop_transform(self, origin_shape: Tuple[int, int, int, int]) -> transforms.Compose:
        """根据预设的多种宽高比，随机选择一种并生成对应的裁剪变换。"""
        keys = list(self.aspect_ratio_random_crop_sample_size.keys())
        probs = ASPECT_RATIO_RANDOM_CROP_PROB # 使用预定义的概率分布
        rc_key = np.random.choice(keys, p=probs) # 随机选择一个宽高比的键
        random_sample_size = self.aspect_ratio_random_crop_sample_size[rc_key]
        # 确保目标尺寸是16的倍数，这对于后续的 VAE 处理可能是必要的
        random_sample_size = [int(x / 16) * 16 for x in random_sample_size]

        f, h, w, c = origin_shape
        th, tw = random_sample_size
        
        # 为了在缩放时保持原始宽高比，避免图像拉伸
        if th / tw > h / w:
            nh, nw = int(th), int(w / h * th)
        else:
            nh, nw = int(h / w * tw), int(tw)
        # 返回一个包含“缩放 -> 中心裁剪 -> 归一化”的变换流
        return transforms.Compose([
            transforms.Resize([nh, nw]),
            transforms.CenterCrop([int(x) for x in random_sample_size]),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True),
        ])

    def _get_standard_crop_transform(self, origin_shape: Tuple[int, int, int, int]) -> transforms.Compose:
        """根据最接近的宽高比，生成标准的缩放和中心裁剪变换。"""
        f, h, w, c = origin_shape
        # 找到与当前样本(h, w)最接近的预设宽高比
        key = get_closest_ratio_key(h, w, ratios=self.aspect_ratio_sample_size)
        closest_size = self.aspect_ratio_sample_size[key]
        closest_size = [int(x / 16) * 16 for x in closest_size]
        
        # 保持原始宽高比进行缩放
        if closest_size[0] / h > closest_size[1] / w:
            resize_size = closest_size[0], int(w * closest_size[0] / h)
        else:
            resize_size = int(h * closest_size[1] / w), closest_size[1]

        return transforms.Compose([
            transforms.Resize(resize_size, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.CenterCrop(closest_size),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True),
        ])

    def _adjust_batch_video_length(self, frame_num: int):
        """根据分桶策略调整视频的帧长，确保批次内所有视频帧数一致。"""
        # 帧长不能超过批次内已经确定的最小帧长
        batch_video_length = int(min(self.batch_video_length, frame_num))
        # 将帧长对齐到预设的间隔（bucket interval），例如，如果间隔是4，23会变成21
        batch_video_length = (batch_video_length - 1) // self.sample_n_frames_bucket_interval * self.sample_n_frames_bucket_interval + 1
        if batch_video_length <= 0:
            batch_video_length = 1
        self.batch_video_length = batch_video_length

    def __call__(self, example: Dict, origin_shape: Tuple[int, int, int, int]) -> torch.Tensor:
        """对单个样本应用变换。"""
        # 将 Numpy 数组转为 PyTorch Tensor，并进行维度换位和归一化
        pixel_values = torch.from_numpy(example["pixel_values"]).permute(0, 3, 1, 2).contiguous()
        pixel_values = pixel_values / 255.
        
        # 根据配置选择使用随机裁剪还是标准裁剪
        if self.args.random_ratio_crop:
            transform = self._get_random_crop_transform(origin_shape)
        else:
            transform = self._get_standard_crop_transform(origin_shape)
            
        transformed_pixels = transform(pixel_values)
        
        # 调整帧长以适应批次
        self._adjust_batch_video_length(len(example["pixel_values"]))

        return transformed_pixels


class VideoDataTransforms(BaseDataTransforms):
    """专门用于视频数据的变换类，继承自 BaseDataTransforms。"""
    def __init__(self, args, sample_n_frames_bucket_interval):
        super().__init__(args, sample_n_frames_bucket_interval)

         # --- 视频特有的、复杂的初始化逻辑 ---
        if args.random_hw_adapt:
            # 如果启用了自适应宽高策略
            if args.training_with_video_token_length:
                # 保持 Token 总量恒定策略
                target_token_length = (args.video_sample_n_frames * args.token_sample_size * args.token_sample_size)
                length_to_frame_num = self._get_length_to_frame_num(target_token_length)
                # self.local_min_size 是在 collate_fn 中根据当前批次的实际内容动态设置的
                local_min_size = self.local_min_size 
                # 基于批次中的最小分辨率，选择一个合适的采样尺寸
                choice_list = [length for length in list(length_to_frame_num.keys()) if length < local_min_size * 1.25]
                if not choice_list:
                    choice_list = list(length_to_frame_num.keys())

                local_video_sample_size = np.random.choice(choice_list)
                random_downsample_ratio = args.video_sample_size / local_video_sample_size
                # 根据选择的采样尺寸，从映射表中查出对应的帧数
                self.batch_video_length = length_to_frame_num[local_video_sample_size]
            else:
                # 如果不保持 Token 恒定，则只进行随机下采样
                random_downsample_ratio = get_random_downsample_ratio(args.video_sample_size, image_ratio=[])
                self.batch_video_length = args.video_sample_n_frames + sample_n_frames_bucket_interval
        else:
            # 如果不启用自适应，则不进行下采样
            random_downsample_ratio = 1
            self.batch_video_length = args.video_sample_n_frames + sample_n_frames_bucket_interval

        #根据计算出的下采样率，确定最终用于分桶的基准尺寸 (base_size)
        base_size = args.video_sample_size / random_downsample_ratio
        if args.random_ratio_crop:
            self.aspect_ratio_random_crop_sample_size = {
                key: [x / 632 * base_size for x in ASPECT_RATIO_RANDOM_CROP_632[key]]
                for key in ASPECT_RATIO_RANDOM_CROP_632
            }
        else:
            self.aspect_ratio_sample_size = {
                key: [x / 632 * base_size for x in ASPECT_RATIO_632[key]]
                for key in ASPECT_RATIO_632
            }

    def _get_length_to_frame_num(self, token_length: int) -> Dict[int, int]:
        """[内部辅助函数] 计算一个映射表：{空间尺寸: 帧数}，以保证总Token数接近目标值。"""
        args = self.args
        # 生成一个可选的空间尺寸列表
        sample_sizes = [args.image_sample_size]
        if args.image_sample_size > args.video_sample_size:
            sample_sizes = list(range(args.video_sample_size, args.image_sample_size + 1, 128))
            if sample_sizes[-1] != args.image_sample_size:
                sample_sizes.append(args.image_sample_size)
        
        # 对于每个可能的空间尺寸，计算出要达到 target_token_length 所需的帧数
        return {
            size: min(token_length / size / size, args.video_sample_n_frames) // self.sample_n_frames_bucket_interval * self.sample_n_frames_bucket_interval + 1
            for size in sample_sizes
        }


class ImageDataTransforms(BaseDataTransforms):
    """专门用于图像数据的变换类，继承自 BaseDataTransforms。"""
    def __init__(self, args, sample_n_frames_bucket_interval):
        super().__init__(args, sample_n_frames_bucket_interval)

        # --- 图像特有的初始化逻辑 ---
        if not args.random_hw_adapt:
            random_downsample_ratio = 1
        else:
            # 图像的下采样率计算可能需要考虑与视频尺寸的比例
            ratio_list = [args.image_sample_size / args.video_sample_size]
            random_downsample_ratio = get_random_downsample_ratio(args.image_sample_size, image_ratio=ratio_list)
        
        # 暂议，这里图像的batch_video_length应该是1，[:20]也不会报错
        self.batch_video_length = args.video_sample_n_frames + sample_n_frames_bucket_interval

        # 使用 `args.image_sample_size` 和计算出的下采样率，设置分桶尺寸
        base_size = args.image_sample_size / random_downsample_ratio
        if args.random_ratio_crop:
            self.aspect_ratio_random_crop_sample_size = {
                key: [x / 632 * base_size for x in ASPECT_RATIO_RANDOM_CROP_632[key]]
                for key in ASPECT_RATIO_RANDOM_CROP_632
            }
        else:
            self.aspect_ratio_sample_size = {
                key: [x / 632 * base_size for x in ASPECT_RATIO_632[key]]
                for key in ASPECT_RATIO_632
            }