import torch
import torch.nn.functional as F
from diffusers.training_utils import compute_loss_weighting_for_sd3

def custom_mse_loss(noise_pred, target, weighting=None, threshold=50):
    """
    自定义的均方误差（MSE）损失。
    支持可选的逐样本加权（weighting）和误差阈值（threshold），
    后者可以看作是一种简化的 Huber Loss，能减少极端误差对梯度的影响。
    """
    noise_pred, target, weighting = noise_pred.float(), target.float(), weighting.float()
    diff = noise_pred - target
    mse_loss = F.mse_loss(noise_pred, target, reduction='none')
    
    # 创建一个掩码，只保留误差在阈值内的部分
    mask = (diff.abs() <= threshold).float()
    masked_loss = mse_loss * mask
    
    # 如果提供了权重，则应用到损失上
    if weighting is not None:
        # 确保权重可以广播到损失张量的形状
        while len(weighting.shape) < len(masked_loss.shape):
            weighting = weighting.unsqueeze(-1)
        masked_loss = masked_loss * weighting
        
    return masked_loss.mean()

def calculate_motion_loss(noise_pred, target):
    """
    计算运动损失（相邻帧之间的差异损失）。
    这会鼓励模型学习更连贯的视频运动。
    """
    # 注意：视频的帧维度通常是第2个维度 (B, C, F, H, W)
    # 计算预测值在时间维度上的帧间差异
    #这里是[:, 1:, :]还是，有待商榷
    pred_diff = noise_pred[:, :, 1:] - noise_pred[:, :, :-1]
    # 计算目标值在时间维度上的帧间差异
    target_diff = target[:, :, 1:] - target[:, :, :-1]
    
    # 对帧间差异计算 MSE 损失
    motion_loss = F.mse_loss(pred_diff.float(), target_diff.float(), reduction="mean")
    return motion_loss

def calculate_loss(noise_pred, target, sigmas, args):
    """
    计算总训练损失的主函数。
    它整合了加权MSE损失和可选的运动损失。

    Args:
        noise_pred (torch.Tensor): 模型预测的噪声。
        target (torch.Tensor): 真实的目标噪声。
        sigmas (torch.Tensor): 当前时间步对应的 sigma 值。
        args: 包含所有配置的参数对象。

    Returns:
        torch.Tensor: 计算出的最终总损失。
    """
    # 1. 计算每个时间步的损失权重
    weighting = compute_loss_weighting_for_sd3(
        weighting_scheme=args.weighting_scheme, 
        sigmas=sigmas
    )
    
    # 2. 计算主要的加权 MSE 损失
    main_loss = custom_mse_loss(
        noise_pred, 
        target, 
        weighting=weighting, 
        threshold=args.loss_threshold if hasattr(args, 'loss_threshold') else 50
    )

    # 3. 如果启用了运动损失，则计算并加权求和
    #    注意：检查帧维度 (dim=2) 是否大于1
    if args.motion_sub_loss and noise_pred.size()[1] > 2:
        motion_loss = calculate_motion_loss(noise_pred, target)
        # 将主损失和运动损失加权求和
        total_loss = main_loss * (1 - args.motion_sub_loss_ratio) + motion_loss * args.motion_sub_loss_ratio
    else:
        total_loss = main_loss
        
    return total_loss