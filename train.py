import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import sys
from torch.utils.data import DataLoader, random_split
from MyDataset import MyDataset
from AttResUNet import AttResUNet
from config import config
from utils import logger
from visualizer import Visualizer, get_dice

# ==========================================
# 1. 损失函数定义 (针对极不平衡样本优化)
# ==========================================
class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, pred, target):
        # 增加数值稳定性，防止 log(0) 导致 NaN
        pred = torch.clamp(pred, 1e-6, 1.0 - 1e-6)
        target = target.view(-1)
        pred = pred.view(-1)
        
        loss = -self.alpha * (1 - pred)**self.gamma * target * torch.log(pred) - \
               (1 - self.alpha) * pred**self.gamma * (1 - target) * torch.log(1 - pred)
        return loss.mean()

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
# 2. 评价指标计算函数
# ==========================================
def get_metrics(pred, target, threshold=0.5):
    """同时计算 F1 和 IoU"""
    pred_bin = (pred > threshold).float().view(-1)
    target_bin = target.float().view(-1)
    
    tp = (pred_bin * target_bin).sum().item()
    fp = (pred_bin * (1 - target_bin)).sum().item()
    fn = ((1 - pred_bin) * target_bin).sum().item()
    
    precision = tp / (tp + fp + 1e-7)
    recall = tp / (tp + fn + 1e-7)
    f1 = 2 * precision * recall / (precision + recall + 1e-7)
    
    intersection = tp
    union = (pred_bin + target_bin).sum().item() - intersection
    iou = intersection / (union + 1e-7)
    
    return f1, iou

# ==========================================
# 3. 数据准备
# ==========================================
viz = Visualizer(log_dir=os.path.join(config["save_path"], "tf-logs"))
full_dataset = MyDataset(data_root=config["data_root"])

#!切分数据集
train_size = int(0.7 * len(full_dataset))
val_size = int(0.2 * len(full_dataset))
test_size = len(full_dataset) - train_size - val_size

train_dataset, val_dataset, test_dataset = random_split(
    full_dataset, [train_size, val_size, test_size],
    generator=torch.Generator().manual_seed(config["global_seed"])
)
# ==========================================
# 4. 训练入口函数封装
# ==========================================
def run_training(test_params):
    

    # 使用 Optuna 建议的 batch_size
    train_loader = DataLoader(train_dataset, batch_size=test_params['batch_size'], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=test_params['batch_size'], shuffle=True)

    # ==========================================
    # 4. 模型初始化
    # ==========================================
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AttResUNet(in_channels=1, out_channels=1).to(device)

    criterion_focal = FocalLoss(alpha=0.25, gamma=2.0)
    criterion_dice = DiceLoss()

    # 使用 Optuna 建议的参数
    optimizer = torch.optim.AdamW(model.parameters(), lr=test_params['lr'])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=test_params['cos_maxt'], eta_min=1e-7)

    best_val_dice = 0.0

    # ==========================================
    # 5. 训练主循环
    # ==========================================
    logger.info(f"开始训练任务 | 注入参数: {test_params}")

    for epoch in range(config["num_epochs"]):
        # --- 训练阶段 ---
        model.train()
        total_train_loss = 0.0
        
        for step, (images, masks) in enumerate(train_loader):
            images, masks = images.to(device), masks.to(device)
            
            outputs = model(images)
            
            # 复合损失：0.8 Focal + 1.2 Dice
            l_f = criterion_focal(outputs, masks)
            l_d = criterion_dice(outputs, masks)
            loss = 0.8 * l_f + 1.2 * l_d
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_train_loss += loss.item()
            
            current_lr = optimizer.param_groups[0]['lr']#!从torch内部维护的字典捕获当前学习率
            if step % 5 == 0:
                logger.info(
                    f"Epoch [{epoch+1}/{config['num_epochs']}] "
                    f"Step [{step}/{len(train_loader)}] | "
                    f"Loss: {loss.item():.4f} | "
                    f"LR: {current_lr:.8f}"
                )
        
        avg_train_loss = total_train_loss / len(train_loader)
        
        # --- 验证阶段 ---
        model.eval()
        val_loss, val_dice, val_f1, val_iou = 0.0, 0.0, 0.0, 0.0
        
        with torch.no_grad():
            for images, masks in val_loader:
                images, masks = images.to(device), masks.to(device)
                outputs = model(images)
                
                # 损失计算与训练对齐
                v_l_f = criterion_focal(outputs, masks)
                v_l_d = criterion_dice(outputs, masks)
                v_loss = 0.8 * v_l_f + 1.2 * v_l_d
                val_loss += v_loss.item()
                
                # 指标计算
                val_dice += get_dice(outputs, masks, threshold=0.5)
                f1, iou = get_metrics(outputs, masks, threshold=0.5)
                val_f1 += f1
                val_iou += iou
        
        # 计算均值
        avg_val_loss = val_loss / len(val_loader)
        avg_val_dice = val_dice / len(val_loader)
        avg_val_f1 = val_f1 / len(val_loader)
        avg_val_iou = val_iou / len(val_loader)
        
        # TensorBoard 记录
        viz.log_scalars("Loss", {"train": avg_train_loss, "val": avg_val_loss}, epoch)
        viz.log_scalars("Metrics", {"Dice": avg_val_dice, "F1": avg_val_f1, "IoU": avg_val_iou}, epoch)
        viz.log_scalars("Learning Rate", {"Learning Rate": current_lr}, epoch)
        
        logger.info(f"===> Epoch {epoch+1}: Train Loss: {avg_train_loss:.4f} | Val Dice: {avg_val_dice:.4f} | F1: {avg_val_f1:.4f} | IoU: {avg_val_iou:.4f}")
        
        # 保存最优模型
        if avg_val_dice > best_val_dice:
            best_val_dice = avg_val_dice
            checkpoint = {"state_dict": model.state_dict(), "config": config}
            torch.save(checkpoint, os.path.join(config["save_path"], "att_res_unet_best.pth"))
            logger.info(f"验证集表现提升，模型已更新。")
        
        scheduler.step()

    viz.close()
    
    # 计算综合得分 (Dice + F1 + IoU) / 3
    composite_score = (avg_val_dice + avg_val_f1 + avg_val_iou) / 3
    logger.info(f"Trial 结束。综合得分: {composite_score:.4f} (Dice: {avg_val_dice:.4f}, F1: {avg_val_f1:.4f}, IoU: {avg_val_iou:.4f})")
    
    # 清理显存避免堆积
    del model, optimizer, train_loader, val_loader
    torch.cuda.empty_cache()

    return composite_score
