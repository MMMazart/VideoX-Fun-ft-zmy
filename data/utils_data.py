
import logging
from typing import Dict, List
def get_logger(name = None):
    logger = logging.getLogger(name or __name__)
    logger.setLevel(logging.INFO)
    return logger

def get_closest_ratio_key(height: float, width: float, ratios_dict: Dict[str, List[float]]) -> str:
    """
    计算给定高宽的最接近的预设宽高比，并返回其在字典中的键。

    Args:
        height (float): 图像或视频帧的高度。
        width (float): 图像或视频帧的宽度。
        ratios_dict (Dict[str, List[float]]): 预设的宽高比字典，键是字符串形式的比例（如 "0.5"），值是[高, 宽]。

    Returns:
        str: 在字典中代表最接近的宽高比的键（字符串形式）。
    """
    aspect_ratio = height / width
    # 将字典的键（字符串）转换为浮点数进行比较
    float_keys = {float(k): k for k in ratios_dict.keys()}
    closest_float_key = min(float_keys.keys(), key=lambda r: abs(r - aspect_ratio))
    return float_keys[closest_float_key]

