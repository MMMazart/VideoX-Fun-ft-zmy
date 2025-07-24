# data/samplers.py
import os
from collections import defaultdict
from typing import Dict, List, Sized, Optional, Iterator, Any, Dataset

import numpy as np
import torch
from torch.utils.data import BatchSampler, Sampler
# 从我们创建的常量文件中导入预设的宽高比字典
from .constants import ASPECT_RATIO_512
from .utils_data import get_logger,get_closest_ratio_key

#获取一个日志记录器实例
logger = get_logger(__name__)

# =====================================================================================
# 简单的采样器 (用于 enable_bucket=False)
# =====================================================================================

class ImageVideoSampler(BatchSampler):
    """
    一个简单的批次采样器，其主要功能是将数据按类型（图像或视频）分组。
    它确保每个生成的批次中只包含一种类型的数据。
    它不按宽高比分桶。
    """

    def __init__(self,
                 sampler: Sampler,
                 dataset: Dataset,
                 batch_size: int,
                 drop_last: bool = False
                ) -> None:
        if not isinstance(sampler, Sampler):
            raise TypeError('sampler should be an instance of ``Sampler``, '
                            f'but got {sampler}')
        if not isinstance(batch_size, int) or batch_size <= 0:
            raise ValueError('batch_size should be a positive integer value, '
                             f'but got batch_size={batch_size}')
        self.sampler = sampler
        self.dataset = dataset
        self.batch_size = batch_size
        self.drop_last = drop_last

        # 为图像和视频分别创建一个桶
        self.bucket = {'image':[], 'video':[]}

    def __iter__(self):
        """迭代器，用于生成批次。"""
        # 从上游采样器（如 RandomSampler）中获取索引
        for idx in self.sampler:
            # 从数据集中查询该索引对应的数据类型,并将索引放入对应的bucket中
            content_type = self.dataset.dataset[idx].get('type', 'image')
            self.bucket[content_type].append(idx)

            # 当视频桶满了，就 yield 这个批次，并清空桶
            if len(self.bucket['video']) == self.batch_size:
                bucket = self.bucket['video']
                yield bucket.copy()
                bucket.clear()
             #当图像桶满了，也做同样的操作
            elif len(self.bucket['image']) == self.batch_size:
                bucket = self.bucket['image']
                yield bucket.copy()
                bucket.clear()
        '''
        原版代码没有这个逻辑
        # 在迭代结束时，处理剩余的不完整批次（如果 drop_last=False）
        if not self.drop_last:
            if self.bucket['video']:
                yield self.bucket['video'].copy()
            if self.bucket['image']:
                yield self.bucket['image'].copy()
        '''

class RandomSampler(Sampler[int]):
    """
    一个可复现的随机采样器。
    它接受一个 `generator` 对象，使得在设置相同种子时，每次运行的随机顺序都一样。
    这是保证实验可复现性的重要一环。
    """
    data_source: Sized
    replacement: bool

    def __init__(self, data_source: Sized, replacement: bool = False,
                 num_samples: Optional[int] = None, generator=None) -> None:
        self.data_source = data_source
        self.replacement = replacement
        self._num_samples = num_samples
        self.generator = generator
        # for resume
        self._pos_start = 0 # 用于断点续训时记录上一次的起始位置

        if not isinstance(self.replacement, bool):
            raise TypeError(f"replacement should be a boolean value, but got replacement={self.replacement}")

        if self._num_samples is not None and not isinstance(self._num_samples, int) or self._num_samples <= 0:
            raise ValueError(f"num_samples should be a positive integer value, but got num_samples={self._num_samples}")

        if self.generator is None:
            # 如果未提供生成器，则创建一个默认的
            seed = int(torch.empty((), dtype=torch.int64).random_().item())
            self.generator = torch.Generator()
            self.generator.manual_seed(seed)
        else:
            self.generator = self.generator
    
    @property
    def num_samples(self) -> int:
        """返回总采样数。"""
        if self._num_samples is None:
            return len(self.data_source)
        return self._num_samples

    def __iter__(self) -> Iterator[int]:
        n = len(self.data_source)

        if self.replacement:
            # 有放回抽样：分批次生成随机索引，每批 32 个，加速小批量生成
            for _ in range(self.num_samples // 32):
                yield from torch.randint(high=n, size=(32,), dtype=torch.int64, generator=self.generator).tolist()
            # 生成剩余的 num_samples % 32 个
            yield from torch.randint(high=n, size=(self.num_samples % 32,), dtype=torch.int64, generator=self.generator).tolist()
        else:
            # # 无放回抽样：num_samples // n 次完整遍历，每次都打乱一次
            # yield from torch.randperm(n, generator=generator).tolist()
            # 这是原始代码，比较复杂：无放回抽样：num_samples // n 次完整遍历，每次都打乱一次
            for _ in range(self.num_samples // n):
                xx = torch.randperm(n, generator=self.generator).tolist()  # 打乱索引列表
                # 从 _pos_start 开始依次 yield，循环结束后重置 _pos_start
                if self._pos_start >= n:
                    self._pos_start = 0
                logger.info("xx top 10", xx[:10], self._pos_start)  # 调试信息：显示打乱后的前 10
                for idx in range(self._pos_start, n):
                    yield xx[idx]
                    # 更新下次迭代开始的位置（环回）
                    self._pos_start = (self._pos_start + 1) % n
                self._pos_start = 0
            # 最后 yield 剩余的 num_samples % n 个
            rem = torch.randperm(n, generator=self.generator).tolist()[:self.num_samples % n]
            yield from rem

    def __len__(self) -> int:
        return self.num_samples

# =====================================================================================
#   高级采样器 (用于 enable_bucket=True 时)
# =====================================================================================
class AspectRatioBatchSamplerBase(BatchSampler):
    """
    一个高效的、基于宽高比的分桶批次采样器的**基类**。
    它实现了流式处理（streaming）的分桶逻辑，避免了一次性将所有索引加载到内存中。

    此采样器旨在将具有相似宽高比的数据（图像或视频）组合到同一个批次中，
    从而最小化因填充（padding）而浪费的计算资源，加速训练。

    """

    def __init__(
        self,
        sampler: Sampler,
        dataset,
        batch_size: int,
        aspect_ratios_dict: Dict[str, List[float]] = ASPECT_RATIO_512,
        drop_last: bool = True,
    ):
        
        """
        初始化 AspectRatioBucketSampler。

        Args:
            sampler (Sampler): 一个上游采样器，用于提供原始的数据索引序列（例如，随机顺序的）。
            dataset: 实现了 `__len__` 和 `get_dimensions(idx)` 方法的数据集对象。
            batch_size (int): 每个批次的大小。
            aspect_ratios_dict (Dict[str, List[float]]): 用于分桶的宽高比字典。
            drop_last (bool): 如果为 True，则丢弃最后一个不完整的批次。
        """
        # --- 参数校验 ---
        if not isinstance(sampler, Sampler):
            raise TypeError('sampler should be an instance of ``Sampler``, '
                            f'but got {sampler}')
        if not isinstance(batch_size, int) or batch_size <= 0:
            raise ValueError('batch_size should be a positive integer value, '
                             f'but got batch_size={batch_size}')
        # --- 核心属性初始化 ---
        self.dataset = dataset  # 这个 dataset 必须提供一个快速的 get_dimensions 方法
        self.batch_size = batch_size
        self.aspect_ratios_dict = aspect_ratios_dict
        self.drop_last = drop_last
        
        #  为每个 ratio 建立一个空桶
        self._aspect_ratio_buckets = {ratio: [] for ratio in aspect_ratios_dict}
        # # 当前还可用的桶 keys
        self.current_available_bucket_keys = list(aspect_ratios_dict.keys())

    def __iter__(self):
        '''
        流式处理
        '''
        # 从上游采样器 (self.sampler) 中逐个获取数据索引
        for idx in self.sampler:
            height, width = self.dataset.get_dimensions(idx)

            # 跳过无效数据（例如，无法读取尺寸的文件）
            if height == 0 or width == 0:
                logger.info(f"Warning: Skipping item {idx} with invalid dimensions ({height}x{width}).")
                continue

            # 调用辅助函数，找到最接近当前样本宽高比的预设比例，并获取其键
            closest_ratio_key = get_closest_ratio_key(height, width, self.aspect_ratios_dict)
            
            # (可选的过滤逻辑) 如果该比例不在可用列表中，则跳过
            if closest_ratio_key not in self.current_available_bucket_keys:
                continue

            # 根据找到的键，选择正确的桶
            bucket = self._aspect_ratio_buckets[closest_ratio_key]
            bucket.append(idx)

            # 如果桶已满，就把它作为一个完整的批次 yield 出去
            # DataLoader 会接收这个 yield 的列表
            if len(bucket) == self.batch_size:
                yield bucket.copy()
                bucket.clear()
                #防止后面清空桶时，外部拿到的列表也被清空。
                # yield bucket[:]
                # del bucket[:]
        '''
        # --- 循环结束后，处理剩余的、不完整的批次 ---
        # 如果 drop_last 设置为 False，则需要把所有桶里剩下的数据也打包成批次
        if not self.drop_last:
            # 遍历所有的桶
            for bucket in self._aspect_ratio_buckets.values():
                # 如果桶里还有数据
                if bucket:
                    # 就把这个不完整的批次也 yield 出去
                    yield bucket.copy()
        '''
class AspectRatioImageSampler(AspectRatioBatchSamplerBase):
    """只分图像的高宽比批采样器。"""
    def __init__(
        self,
        sampler: Sampler[int],
        dataset: Any,
        batch_size: int,
        aspect_ratios_dict: Dict[float, Any],
        drop_last: bool = False,
    ):
        super().__init__(sampler, dataset, batch_size, aspect_ratios_dict, drop_last)


class AspectRatioVideoSampler(AspectRatioBatchSamplerBase):
    """只分视频的高宽比批采样器。"""
    def __init__(
        self,
        sampler: Sampler[int],
        dataset: Any,
        batch_size: int,
        aspect_ratios_dict: Dict[float, Any],
        drop_last: bool = False,
    ):
        super().__init__(sampler, dataset, batch_size, aspect_ratios_dict, drop_last)

class AspectRatioImageVideoSampler(BatchSampler):
    """
    同时分图像和视频的高宽比批采样器。
    """
    def __init__(
        self,
        sampler: Sampler,
        dataset: Any,
        batch_size: int,
        aspect_ratios_dict: Dict[float, Any],
        drop_last: bool = False,
        train_folder: Optional[str] = None
    ) -> None:
        # --- 参数校验 ---
        if not isinstance(sampler, Sampler):
            raise TypeError('sampler should be an instance of ``Sampler``, '
                            f'but got {sampler}')
        if not isinstance(batch_size, int) or batch_size <= 0:
            raise ValueError('batch_size should be a positive integer value, '
                             f'but got batch_size={batch_size}')
        # --- 核心属性初始化 ---
        self.sampler = sampler
        self.dataset = dataset
        self.batch_size = batch_size
        self.aspect_ratios_dict = aspect_ratios_dict
        self.drop_last = drop_last
        # image/video 各自 buckets
        ## 这是一个两层的字典，第一层按'image'/'video'分类，第二层按宽高比分类
        self.buckets = {
            'image': {r: [] for r in aspect_ratios_dict},
            'video': {r: [] for r in aspect_ratios_dict}
        }
        self.current_available_bucket_keys = list(aspect_ratios_dict.keys())

    def __iter__(self):
        # 从上游采样器（如 RandomSampler）中逐个获取数据索引
        for idx in self.sampler:
            content_type = self.dataset[idx].get('type', 'image')
            height, width = self.dataset.get_dimensions(idx)
            # 跳过无效数据（例如，无法读取尺寸的文件）
            if height == 0 or width == 0:
                print(f"Warning: Skipping item {idx} with invalid dimensions ({height}x{width}).")
                continue 

            closest_ratio_key = get_closest_ratio_key(height, width, self.aspect_ratios_dict)
            if closest_ratio_key not in self.current_available_bucket_keys:
                continue
            
            #    将索引放入正确的桶中
            #    首先根据 content_type 选择 'image' 或 'video' 的桶集合，
            #    然后根据 closest_ratio_key 选择最终的那个桶（列表）。
            bucket = self.buckets[content_type][closest_ratio_key]
            bucket.append(idx)
            #如果桶已满，就把它作为一个完整的批次 yield 出去
            if len(bucket) == self.batch_size:
                yield bucket.copy()
                bucket.clear()
        # --- 循环结束后，处理所有桶中剩余的、不完整的批次 ---
        if not self.drop_last:
            for type_buckets  in self.buckets.values():
                for bucket in type_buckets .values():
                    if bucket:
                        yield bucket.copy()


# class AspectRatioBucketSampler(BatchSampler):
#     """
#     一个高效的、基于宽高比的分桶批次采样器 (Batch Sampler)。

#     此采样器旨在将具有相似宽高比的数据（图像或视频）组合到同一个批次中，
#     从而最小化因填充（padding）而浪费的计算资源，加速训练。

#     核心策略:
#     1.  **预分桶 (Pre-bucketing)**: 在初始化阶段（__init__），采样器会遍历整个数据集一次，
#         利用`dataset.get_dimensions()`方法获取每个数据项的尺寸，并将其索引（index）
#         预先分配到对应的宽高比桶（bucket）中。
#     2.  **高效迭代 (__iter__)**: 在每个训练周期（epoch）开始时，迭代器只需对每个桶内部的
#         索引进行随机排序，然后组合成批次即可。这个过程不涉及任何文件I/O，速度非常快。
#     """

#     def __init__(
#         self,
#         dataset,
#         batch_size: int,
#         aspect_ratios_dict: Dict[str, List[float]] = ASPECT_RATIO_512,
#         drop_last: bool = False,
#         seed: int = 42,
#     ):
#         """
#         初始化 AspectRatioBucketSampler。

#         Args:
#             dataset: 实现了 `__len__` 和 `get_dimensions(idx)` 方法的数据集对象。
#             batch_size (int): 每个批次的大小。
#             aspect_ratios_dict (Dict[str, List[float]], optional):
#                 用于分桶的宽高比字典。默认为 `ASPECT_RATIO_512`。
#             drop_last (bool, optional): 如果为 True，则丢弃最后一个不完整的批次。默认为 True。
#             seed (int, optional): 用于可复现随机性的种子。默认为 42。
#         """
#         # BatchSampler 的 __init__ 不需要调用，因为我们自己实现了所有逻辑
#         self.dataset = dataset
#         self.batch_size = batch_size
#         self.aspect_ratios_dict = aspect_ratios_dict
#         self.drop_last = drop_last
#         # 使用Numpy的随机数生成器，以保证可复现的随机性
#         self.rng = np.random.default_rng(seed)

#         # [核心步骤] 在初始化时调用内部分桶方法，完成所有数据项的预分桶
#         print("Pre-bucketing data based on aspect ratio...")
#         self.buckets = self._create_buckets()
#         print(f"Created {len(self.buckets)} buckets.")

#         # 预先计算总批次数，供外部（如tqdm进度条）调用
#         self.num_batches = self._calculate_num_batches()
#         print(f"Total batches: {self.num_batches}")


#     def _create_buckets(self) -> Dict[str, List[int]]:
#         """
#         [内部方法] 遍历数据集，将每个数据项的索引放入对应的宽高比桶中。
#         """
#         # buckets for each aspect ratio 
#         buckets = {ratio: [] for ratio in aspect_ratios}
#         # [str(k) for k, v in aspect_ratios] 
#         current_available_bucket_keys = list(aspect_ratios.keys())

#         for i in range(len(self.dataset)):
#             height, width = self.dataset.get_dimensions(i)

#             # 跳过无效数据（例如，无法读取尺寸的文件）
#             if height == 0 or width == 0:
#                 print(f"Warning: Skipping item {i} with invalid dimensions ({height}x{width}).")
#                 continue

#             closest_ratio = get_closest_ratio_key(height, width, self.aspect_ratios_dict)
#             if closest_ratio not in self.current_available_bucket_keys:
#                 continue            
            

#             buckets[closest_ratio].append(idx)
#             # yield a batch of indices in the same aspect ratio group
#             if len(bucket) == self.batch_size:
#                 yield bucket[:]
#                 del bucket[:]
        
#         # 为了保证每个epoch的起始状态一致，对桶的键进行排序
#         sorted_buckets = dict(sorted(buckets.items()))
#         return sorted_buckets

#     def _calculate_num_batches(self) -> int:
#         """
#         [内部方法] 根据分桶结果和drop_last设置，计算一个周期内的总批次数。
#         """
#         num_batches = 0
#         for bucket in self.buckets.values():
#             if self.drop_last:
#                 num_batches += len(bucket) // self.batch_size
#             else:
#                 num_batches += (len(bucket) + self.batch_size - 1) // self.batch_size
#         return num_batches

#     def __len__(self) -> int:
#         """返回一个周期（epoch）中的总批次数。"""
#         return self.num_batches

#     def __iter__(self):
#         """
#         创建并返回一个批次索引的迭代器。
#         """
#         all_batches = []
#         for bucket in self.buckets.values():
#             self.rng.shuffle(bucket)
#             for i in range(0, len(bucket), self.batch_size):
#                 batch = bucket[i : i + self.batch_size]
#                 if len(batch) == self.batch_size or (not self.drop_last and len(batch) > 0):
#                     all_batches.append(batch)

#         # 对所有批次进行全局随机排序，增加训练的随机性
#         self.rng.shuffle(all_batches)
#         yield from all_batches