# data/datasets.py
import os
import json
import csv
import random
import gc
from contextlib import contextmanager

import cv2
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
from decord import VideoReader
from func_timeout import func_timeout, FunctionTimedOut
from .utils import get_logger
#获取一个日志记录器实例
logger = get_logger(__name__)
# 定义视频读取的超时时间
VIDEO_READER_TIMEOUT = 20

@contextmanager
def VideoReader_contextmanager(*args, **kwargs):
    """
    一个上下文管理器，旨在确保 decord.VideoReader 的资源被正确、及时地释放。
    `decord` 在某些情况下可能存在内存泄漏风险，此管理器通过显式删除和垃圾回收来增强其健壮性。
    """
    # 初始化 VideoReader。设置 num_threads=0 是为了避免与 PyTorch DataLoader 的多进程（num_workers > 0）机制发生冲突或死锁。
    # 这里的 `*args` 和 `**kwargs` 使得这个管理器可以接受 VideoReader 的所有原生参数。
    vr = VideoReader(*args, **kwargs, num_threads=0)
    try:
        # yield 将 VideoReader 对象返回给 with 语句块
        yield vr
    finally:
        # with 语句块结束后，无论成功或异常，这里的代码都会被执行
        # decord 的 VideoReader 需要手动关闭，这里通过 del 和强制垃圾回收 (gc.collect) 来确保资源被释放
        del vr
        gc.collect()

def get_video_reader_batch(video_reader, batch_index):
    """从 VideoReader 对象中根据索引高效地获取一批帧。"""
    # get_batch 是 decord 的一个优化功能，可以一次性读取多个帧
    frames = video_reader.get_batch(batch_index).asnumpy()
    return frames

def resize_frame(frame, target_short_side):
    """
    按比例缩放单帧图像，使其短边等于目标尺寸 (target_short_side)，同时保持宽高比。
    """
    h, w, _ = frame.shape
    if h < w:
        # 当高度是短边时
        if target_short_side >= h:
            return frame # 如果目标尺寸不小于当前短边，无需缩放
        new_h = target_short_side
        new_w = int(w * (new_h / h)) # 根据比例计算新宽度
    else:
        # 当宽度是短边或相等时
        if target_short_side >= w:
            return frame # 如果目标尺寸不小于当前短边，无需缩放
        new_w = target_short_side
        new_h = int(h * (new_w / w)) # 根据比例计算新高度
    
    # 问：确保新尺寸是偶数，以避免后续处理中的问题，这里需要吗？
    # 答：是的，这是一个很好的实践。许多视频编解码器和深度学习操作（特别是涉及卷积和池化的下采样/上采样）
    # 要求或在处理偶数尺寸的输入时效率更高、效果更好。强制为偶数可以提高模型的健壮性。
    new_h = new_h if new_h % 2 == 0 else new_h - 1
    new_w = new_w if new_w % 2 == 0 else new_w - 1
    
    return cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)



class ImageVideoDataset(Dataset):
    """
    一个可以同时处理图像和视频数据的 PyTorch 数据集类。
    """
    def __init__(
            self,
            ann_path, data_root=None,
            video_sample_size=512, 
            video_sample_fps=16,
            video_sample_n_frames=81,
            image_sample_size=512,
            video_repeat=0,
            text_drop_ratio=0.1,
            # enable_bucket=False,
            enable_inpaint=False,
            # (新增) 一个统一的尺寸参数，用于视频帧的初步缩放
            pre_transform_short_side=512,
        ):
        self.data_root = data_root
        self.video_sample_fps = video_sample_fps
        self.video_sample_n_frames = video_sample_n_frames
        self.text_drop_ratio = text_drop_ratio
        self.pre_transform_short_side = pre_transform_short_side

        # 保证 sample_size 是一个 (height, width) 元组
        self.video_sample_size = tuple(video_sample_size) if not isinstance(video_sample_size, int) else (video_sample_size, video_sample_size)
        self.image_sample_size  = tuple(image_sample_size) if not isinstance(image_sample_size, int) else (image_sample_size, image_sample_size)
        
        #设置统一的图像和视频的尺寸，对视频帧进行最小边缩放（resize）处理。
        self.larger_side_of_image_and_video = max(min(self.image_sample_size), min(self.video_sample_size))


        logger.info(f"Loading annotations from {ann_path} ...")
        self.dataset = self._load_annotations(ann_path, video_repeat)
        self.length = len(self.dataset)
        logger.info(f"Data scale: {self.length}")

        # [优化] 在初始化时预加载并缓存所有数据的尺寸
        # 防止后续bucket sampling获取文件尺寸反复执行I/O操作，可能很慢
        logger.info("Caching data dimensions...")
        self.dimensions_cache = [self._get_dimensions_from_file(i) for i in range(self.length)]
        logger.info("Caching complete.")        



    def _load_annotations(self, ann_path, video_repeat):

        """从 CSV 或 JSON 文件加载标注，并根据 video_repeat 平衡数据。"""
        if ann_path.endswith('.csv'):
            with open(ann_path, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                full_dataset = list(reader)
        elif ann_path.endswith('.json'):
            with open(ann_path, 'r', encoding='utf-8') as f:
                full_dataset = json.load(f)
        else:
            raise ValueError(f"Unsupported annotation file format: {ann_path}")

        # 如果 video_repeat > 0，根据 video_repeat 参数复制视频数据以实现平衡
        if video_repeat > 0:
            videos = [data for data in full_dataset if data.get('type') == 'video']
            images = [data for data in full_dataset if data.get('type') != 'video']
            repeated_videos = []
            for _ in range(video_repeat):
                repeated_videos.extend(videos)
            # 将图像和视频数据合并
            final_dataset = images + repeated_videos

            return final_dataset
            
        return full_dataset

    def _get_dimensions_from_file(self, idx):
        """
        [内部方法] 高效地只读取单个数据项的尺寸，而不加载完整的数据。
        此方法仅在 __init__ 的缓存构建阶段被调用。
        """
        data_info = self.dataset[idx]
        path = os.path.join(self.data_root, data_info['file_path'])
        data_type = data_info.get('type', 'image')
        try:
            if data_type == 'video':
                # 使用上下文管理器安全地打开视频，只为读取宽高
                with VideoReader_contextmanager(path) as vr:
                    return vr.height, vr.width
            else:  # image
                with Image.open(path) as img:
                    width, height = img.size
                return height, width
        except Exception as e:
            logger.info(f"Warning: Could not get dimensions for item {idx} ({path}): {e}")
            return 0, 0  # 返回 (0, 0) 标记为无效数据，方便后续处理

    def get_dimensions(self, idx):
        """[公共接口] 为 Sampler 提供从缓存中快速获取数据尺寸的方法。"""
        return self.dimensions_cache[idx]

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        """
        [核心方法] 根据索引获取一个数据项。
        实现了完整的错误处理逻辑，如果任何步骤失败，则返回 None。
        这要求后续的 collate_fn 能够处理和过滤掉 None 值。
        """
        idx = idx % self.length # 确保索引在有效范围内
        try:
            # 从缓存中快速获取尺寸
            height, width = self.get_dimensions(idx)
            # 如果缓存的尺寸是无效的 (0, 0)，说明这个数据有问题，直接抛出异常跳过
            if height == 0 or width == 0:
                raise ValueError("Invalid dimensions (0, 0)")

            # 调用内部方法加载实际的像素数据和文本
            pixel_values, text, data_type = self._load_item(idx)

            # 以一定概率随机丢弃文本，这是为了实现 Classifier-Free Guidance 训练
            if random.random() < self.text_drop_ratio:
                text = ''

            # 构建并返回一个包含所有信息的字典
            sample = {
                "pixel_values": pixel_values,  # PIL Image 或 Numpy Array
                "text": text,
                "data_type": data_type,
                "height": height,
                "width": width,
                "idx": idx
            }
            return sample
        except Exception as e:
            # 如果加载过程中出现任何异常，打印错误信息并返回 None
            # collate_fn 会负责过滤掉这些无效的 None 样本
            file_path = self.dataset[idx].get('file_path', 'N/A')
            logger.info(f"Error: Failed to load item {idx} ({file_path}): {e}")
            return None

    def _load_item(self, idx):
        """[内部方法] 根据索引实际从磁盘加载原始的图像或视频数据。"""
        data_info = self.dataset[idx]
        if self.data_root is None:
            path = data_info['file_path']
        else:
            path = os.path.join(self.data_root, data_info['file_path'])
        text = data_info['text']
        data_type = data_info.get('type', 'image')

        if data_type == 'video':
            with VideoReader_contextmanager(path,num_threads=2) as video_reader:
                video_length = len(video_reader)  # 获取视频总帧数
                if video_length == 0:
                    raise ValueError("Video has no frames.")
                
                # --- 计算要采样的帧数和起止点 ---
                # 目标采样时长
                video_sample_time = self.video_sample_n_frames / self.video_sample_fps
                # 根据视频实际帧率，计算出这段时长对应的帧数
                video_duration_in_frames = video_sample_time * video_reader.get_avg_fps() #时长*帧数
                # 最终采样的片段长度不能超过视频本身的总长度
                clip_length = min(video_length, int(video_duration_in_frames))   #实际采样帧数
                if clip_length == 0:
                    raise ValueError("Calculated clip length is zero.")
                
                # 随机选择一个起始帧
                start_idx = random.randint(0, video_length - clip_length)
                # 使用 np.linspace 在选定的片段内均匀采样出 self.video_sample_n_frames 数量的帧
                batch_index = np.linspace(start_idx, start_idx + clip_length - 1, self.video_sample_n_frames, dtype=int)

                # --- 带超时地读取视频帧，增加健壮性 ---
                try:
                    sample_args = (video_reader, batch_index)
                    # 使用 func_timeout 包装视频读取函数，如果超过指定时间则抛出 FunctionTimedOut 异常
                    raw_frames = func_timeout(
                        VIDEO_READER_TIMEOUT, get_video_reader_batch, args=sample_args
                    )
                    # --- 对读取的每一帧进行预缩放 ---
                    resized_frames = []
                    for i in range(len(raw_frames)):
                        frame = raw_frames[i]
                        resized_frame = resize_frame(frame, self.larger_side_of_image_and_video)
                        resized_frames.append(resized_frame)
                    pixel_values = np.array(resized_frames)
                except FunctionTimedOut:
                    raise ValueError(f"Timeout reading video: {path}")
                except Exception as e:
                    raise ValueError(f"Failed to extract frames from video. Error is {e}.")

                # 进行初步的帧缩放  代采用
                # resized_frames = [resize_frame(frame, self.pre_transform_short_side) for frame in raw_frames]
                # pixel_values = np.stack(resized_frames) # 返回 (F, H, W, C) 的 Numpy 数组

            return pixel_values, text, 'video'
        else:  # image
            # 对于图像，只打开并转换为 RGB 格式，返回原始的PIL格式
            # 将复杂的图像变换（如 random crop, flip）推迟到 collate_fn 中进行，
            image = Image.open(path).convert('RGB')
  
            return image, text, 'image'