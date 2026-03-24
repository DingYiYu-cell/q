import torch
import torch.nn as nn
import torch.nn.functional as F
import os
from torch.utils.data import DataLoader
from MyDataset import MyDataset
from AttResUNet import AttResUNet
from config import config
import sys
from torch.utils.data import random_split
from utils import logger
from visualizer import Visualizer, get_dice

# ==========================================
# 0. 系统检查
# ==========================================
class SystemValidator:
    def __init__(self):
        """仅做逻辑检查，路径由外部传入"""
        pass

    def _fail(self, msg):
        """报错并强行退出"""
        logger.error(f"环境报错: {msg}")
        sys.exit(1)

    def check_env(self, data_root, save_path):
        logger.info("启动前自检...")
        
        # 1. 检查 CUDA (GPU)
        if not torch.cuda.is_available():
            self._fail("未检测到 CUDA 设备。请确认已开启 GPU 实例，或检查驱动配置。")
        
        gpu_name = torch.cuda.get_device_name(0)
        logger.info(f"GPU 环境: {gpu_name}")

        # 2. 检查数据集路径
        if not os.path.exists(data_root):
            self._fail(f"数据集目录不存在: {data_root}")
        logger.info(f"数据集路径: {data_root}")

        # 3. 检查模型保存路径 (不存在则自动创建)
        if not os.path.exists(save_path):
            try:
                os.makedirs(save_path)
                logger.info(f"提示: 已创建保存目录: {save_path}")
            except Exception as e:
                self._fail(f"无法创建保存目录 {save_path}: {e}")
        logger.info(f"保存路径: {save_path}")

        logger.info("[SUCCESS] 检查通过，环境就绪！")
        logger.info("="*40 + "\n")
SystemValidator().check_env(config["data_root"], config["save_path"])


# ==========================================
# 1. 训练流程 (含数据集划分与自动验证)
# ==========================================
class DiceLoss(nn.Module):
    def __init__(self, smooth=1e-6):
        super().__init__()
        self.smooth = smooth

    def forward(self, pred, target):
        pred = pred.view(-1)
        target = target.view(-1)
        
        intersection = (pred * target).sum()
        dice = (2. * intersection + self.smooth) / (pred.sum() + target.sum() + self.smooth)
        
        return 1 - dice


# ==========================================
# 新增：计算 F1 Score 和 IoU 的函数
# ==========================================
def get_f1_score(pred, target, threshold=0.5):
    """
    计算二分类 F1 Score
    pred: 模型输出的概率图 (B,1,H,W) 或 (B,H,W)，取值 [0,1]
    target: 真实标签 (B,1,H,W) 或 (B,H,W)，取值 {0,1}
    threshold: 二值化阈值
    """
    # 确保维度一致
    if pred.dim() == 4:
        pred = pred.squeeze(1)
    if target.dim() == 4:
        target = target.squeeze(1)
    
    # 二值化
    pred_bin = (pred > threshold).float()
    target_bin = target.float()
    
    # 计算混淆矩阵元素
    tp = (pred_bin * target_bin).sum().item()
    fp = (pred_bin * (1 - target_bin)).sum().item()
    fn = ((1 - pred_bin) * target_bin).sum().item()
    
    # 防止除零
    precision = tp / (tp + fp + 1e-7)
    recall = tp / (tp + fn + 1e-7)
    f1 = 2 * precision * recall / (precision + recall + 1e-7)
    
    return f1

def get_iou(pred, target, threshold=0.5):
    """
    计算二分类 IoU (Jaccard Index)
    pred: 模型输出的概率图 (B,1,H,W) 或 (B,H,W)，取值 [0,1]
    target: 真实标签 (B,1,H,W) 或 (B,H,W)，取值 {0,1}
    threshold: 二值化阈值
    """
    if pred.dim() == 4:
        pred = pred.squeeze(1)
    if target.dim() == 4:
        target = target.squeeze(1)
    
    pred_bin = (pred > threshold).float()
    target_bin = target.float()
    
    intersection = (pred_bin * target_bin).sum().item()
    union = (pred_bin + target_bin).sum().item() - intersection  # 并集 = 正样本总数 - 交集
    
    iou = intersection / (union + 1e-7)
    return iou

viz = Visualizer(log_dir=os.path.join(config["save_path"], "tf-logs"))  # 实例化Tensorboard记录器

# 1. 实例化并划分数据集 (8:1:1)
full_dataset = MyDataset(data_root=config["data_root"])

train_size = int(0.7 * len(full_dataset))
val_size = int(0.2 * len(full_dataset))
test_size = len(full_dataset) - train_size - val_size

train_dataset, val_dataset, test_dataset = random_split(
    full_dataset, [train_size, val_size, test_size],
    generator=torch.Generator().manual_seed(config["global_seed"])
)

# 2. 建立 DataLoader
train_loader = DataLoader(train_dataset, batch_size=config["batch"], shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=config["batch"], shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=config["batch"], shuffle=False)

logger.info(f"数据划分完成：训练集 {len(train_dataset)}，验证集 {len(val_dataset)}，测试集 {len(test_dataset)}")

# 3. 初始化模型与优化器
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"当前使用的设备: {device}")

model = AttResUNet(in_channels=1, out_channels=1).to(device)
criterion_bce = nn.BCELoss()
criterion_dice = DiceLoss()


optimizer = torch.optim.AdamW(model.parameters(), lr=config["lr"])
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config["num_epochs"], eta_min=1e-7)

# 用于保存最佳模型的变量
best_val_loss = float('inf')
best_val_dice = 0.0

logger.info("开始训练...")

for epoch in range(config["num_epochs"]):
    # --- 训练阶段 ---
    model.train()
    train_loss = 0.0
    avg_train_loss = 0.0
    
    for step, (images, masks) in enumerate(train_loader):
        images, masks = images.to(device), masks.to(device)
        
        # 前向传播
        
        outputs = model(images)
        loss_bce = criterion_bce(outputs, masks)
        loss_dice = criterion_dice(outputs, masks)
        loss = 0.6*loss_bce + 1.2*loss_dice
        
        # 反向传播与优化
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        train_loss += loss.item()
        
        if step % 5 == 0:
            logger.info(f"Epoch [{epoch+1}/{config['num_epochs']}], Step [{step}/{len(train_loader)}], Train Loss: {loss.item():.4f}")
    
    avg_train_loss = train_loss / len(train_loader)
    logger.info(f'Current Train Epoch :  Avg LOSS:{avg_train_loss}')
    
    # --- 验证阶段 ---
    model.eval()
    epoch_val_loss = 0.0
    epoch_val_dice = 0.0
    epoch_val_f1 = 0.0      # 新增：累计 F1
    epoch_val_iou = 0.0      # 新增：累计 IoU
    
    with torch.no_grad():
        for images, masks in val_loader:
            images, masks = images.to(device), masks.to(device)
            outputs = model(images)
            
            # 损失
            v_bce_loss = criterion_bce(outputs, masks)
            v_dice_loss = criterion_dice(outputs, masks)
            v_loss = 0.5 * v_bce_loss + 1.5 * v_dice_loss
            epoch_val_loss += v_loss.item()
            
            # 指标
            epoch_val_dice += get_dice(outputs, masks)                # 原有 Dice（默认阈值 0.5? 需查看 get_dice 实现）
            epoch_val_f1 += get_f1_score(outputs, masks, threshold=0.5)
            epoch_val_iou += get_iou(outputs, masks, threshold=0.5)
    
    avg_val_loss = epoch_val_loss / len(val_loader)
    avg_val_dice = epoch_val_dice / len(val_loader)
    avg_val_f1 = epoch_val_f1 / len(val_loader)
    avg_val_iou = epoch_val_iou / len(val_loader)
    
    # 打印多个阈值下的 Dice（原有）
    logger.info(f"Threshold 0.4 Dice: {get_dice(outputs, masks, threshold=0.4):.4f}")
    logger.info(f"Threshold 0.5 Dice: {get_dice(outputs, masks, threshold=0.5):.4f}")
    logger.info(f"Threshold 0.6 Dice: {get_dice(outputs, masks, threshold=0.6):.4f}")
    logger.info(f"Threshold 0.7 Dice: {get_dice(outputs, masks, threshold=0.7):.4f}")
    logger.info(f"Threshold 0.8 Dice: {get_dice(outputs, masks, threshold=0.8):.4f}")
    logger.info(f"Threshold 0.9 Dice: {get_dice(outputs, masks, threshold=0.9):.4f}")
    
    # 新增：打印 F1 和 IoU
    logger.info(f"F1 Score (th=0.5): {avg_val_f1:.4f}")
    logger.info(f"IoU (th=0.5): {avg_val_iou:.4f}")
    
    # 当前学习率
    current_lr = optimizer.param_groups[0]['lr']
    logger.info(f"Current Learning Rate: {current_lr:.6f}")
    
    # TensorBoard 记录
    viz.log_scalars("Loss", {"train": avg_train_loss, "val": avg_val_loss}, epoch)
    viz.log_scalars("Dice", {"val": avg_val_dice}, epoch)
    viz.log_scalars("F1", {"val": avg_val_f1}, epoch)      # 新增
    viz.log_scalars("IoU", {"val": avg_val_iou}, epoch)    # 新增
    
    logger.info(f"===> Epoch [{epoch+1}/{config['num_epochs']}] Avg Train Loss: {avg_train_loss:.4f}  | Avg Val Loss: {avg_val_loss:.4f} | Avg Val Dice: {avg_val_dice:.4f} | F1: {avg_val_f1:.4f} | IoU: {avg_val_iou:.4f}")
    
    # 保存性能最好的模型（基于 Dice）
    if avg_val_dice > best_val_dice:
        best_val_dice = avg_val_dice
        best_val_loss = avg_val_loss
        checkpoint = {
            "state_dict": model.state_dict(),
            "config": config
        }
        torch.save(checkpoint, "att_res_unet_best.pth")
        logger.info(f"发现更优验证集表现，模型已更新保存~")
    
    scheduler.step()
    logger.info("-" * 30)

viz.close()
logger.info(f"训练完成！最优验证集 Loss 为: {best_val_loss:.4f}, 最优验证集 Dice 为：{best_val_dice:.4f}")