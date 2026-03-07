import os
from torch.utils.tensorboard import SummaryWriter

class Visualizer:
    def __init__(self, log_dir):
        """
        初始化记录器喵
        log_dir: 存放日志的路径（建议放在 save_path 目录下）
        """
        self.writer = SummaryWriter(log_dir=log_dir)
        print(f"📈 TensorBoard 日志将保存至: {log_dir} 喵")

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