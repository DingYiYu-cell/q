import os
import torch
from torch.utils.tensorboard import SummaryWriter
from utils import logger
class Visualizer:
    def __init__(self, log_dir):
        """
        初始化记录器喵
        log_dir: 存放日志的路径（建议放在 save_path 目录下）
        """
        self.writer = SummaryWriter(log_dir=log_dir)
        logger.info(f" TensorBoard 日志将保存至: {log_dir} ")

    def log_scalars(self, main_tag, tag_scalar_dict, global_step):
        """
        记录多个数值（比如同时记录训练 Loss 和 验证 Loss）喵
        """
        for tag, scalar_value in tag_scalar_dict.items():
            self.writer.add_scalar(f"{main_tag}/{tag}", scalar_value, global_step)

    def log_image(self, tag, image_tensor, global_step):
        """
        记录图片（方便看分割效果）喵
        """
        self.writer.add_image(tag, image_tensor, global_step)

    def close(self):
        """用完记得关门喵"""
        self.writer.close()
        
        
        
class get_dice():
    def __init__(self):
        pass 
    def get_dice(pred, target, threshold=0.5):
        # 1. 先把输出变成 0 或 1 的二值图喵
        pred = (torch.sigmoid(pred) > threshold).float()
        target = target.float()
    
        # 2. 计算交集和并集喵
        smooth = 1e-6  # 防止除以 0 的小补丁喵
        intersection = (pred * target).sum()
        dice = (2. * intersection + smooth) / (pred.sum() + target.sum() + smooth)
        return dice.item()