import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import sys
import csv
import copy
from torch.utils.data import DataLoader, random_split
from MyDataset import MyDataset
from AttResUNet import AttResUNet
from config import config
from utils import logger
from visualizer import Visualizer, get_dice

# 全局唯一守门员
global_best_score = -float('inf')

class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0):
        super().__init__()
        self.alpha = alpha; self.gamma = gamma
    def forward(self, pred, target):
        pred = torch.clamp(pred, 1e-6, 1.0 - 1e-6)
        target = target.view(-1); pred = pred.view(-1)
        loss = -self.alpha * (1 - pred)**self.gamma * target * torch.log(pred) - \
               (1 - self.alpha) * pred**self.gamma * (1 - target) * torch.log(1 - pred)
        return loss.mean()

class DiceLoss(nn.Module):
    def __init__(self, smooth=1e-6):
        super().__init__()
        self.smooth = smooth
    def forward(self, pred, target):
        pred = pred.view(-1); target = target.view(-1)
        intersection = (pred * target).sum()
        dice = (2. * intersection + self.smooth) / (pred.sum() + target.sum() + self.smooth)
        return 1 - dice

def get_metrics(pred, target, threshold=0.5):
    pred_bin = (pred > threshold).float().view(-1)
    target_bin = target.float().view(-1)
    tp = (pred_bin * target_bin).sum().item()
    fp = (pred_bin * (1 - target_bin)).sum().item()
    fn = ((1 - pred_bin) * target_bin).sum().item()
    f1 = 2 * tp / (2 * tp + fp + fn + 1e-7)
    iou = tp / (tp + fp + fn + 1e-7)
    return f1, iou

viz = Visualizer(log_dir=os.path.join(config["save_path"], "tf-logs"))
full_dataset = MyDataset(data_root=config["data_root"])
train_size = int(0.7 * len(full_dataset))
val_size = int(0.2 * len(full_dataset))
test_size = len(full_dataset) - train_size - val_size
train_dataset, val_dataset, test_dataset = random_split(full_dataset, [train_size, val_size, test_size], generator=torch.Generator().manual_seed(config["global_seed"]))

def run_training(test_params):
    global global_best_score
    
    # ✅ 1. 注入参数显示
    logger.debug(f"第{test_params['trial_num']}个Trial | 开始训练任务 | 注入参数: {test_params}")
    
    train_loader = DataLoader(train_dataset, batch_size=test_params['batch_size'], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=test_params['batch_size'], shuffle=False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AttResUNet(in_channels=1, out_channels=1).to(device)
    
    criterion_focal = FocalLoss(); criterion_dice = DiceLoss()
    optimizer = getattr(torch.optim, test_params['opt_class'])(model.parameters(), lr=test_params['lr'])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=test_params['cos_maxt'], eta_min=1e-7)

    trial_best_score = -float('inf'); trial_best_weights = None; trial_best_metrics = {}
    trial_history = []

    for epoch in range(config["num_epochs"]):
        # --- 训练阶段 ---
        model.train()
        total_train_loss = 0.0
        for step, (images, masks) in enumerate(train_loader):
            images, masks = images.to(device), masks.to(device)
            out = model(images)
            loss = test_params['alpha'] * criterion_focal(out, masks) + test_params['beta'] * criterion_dice(out, masks)
            optimizer.zero_grad(); loss.backward(); optimizer.step()
            total_train_loss += loss.item()

            # ✅ 2. 恢复 Epoch 内部的多点实时输出
            current_lr = optimizer.param_groups[0]['lr']
            if step % 5 == 0:
                logger.info(
                    f"Trial [{test_params['trial_num']}/{test_params['total_trials']}]"
                    f"Epoch [{epoch+1}/{config['num_epochs']}] "
                    f"Step [{step}/{len(train_loader)}] | "
                    f"Loss: {loss.item():.4f} | "
                    f"LR: {current_lr:.8f}"
                )

        # --- 验证阶段 ---
        model.eval()
        sum_dice, sum_f1, sum_iou = 0.0, 0.0, 0.0
        with torch.no_grad():
            for images, masks in val_loader:
                images, masks = images.to(device), masks.to(device)
                out = model(images)
                sum_dice += get_dice(out, masks)
                f1, iou = get_metrics(out, masks); sum_f1 += f1; sum_iou += iou
        
        avg_dice, avg_f1, avg_iou = sum_dice/len(val_loader), sum_f1/len(val_loader), sum_iou/len(val_loader)
        current_score = (avg_dice + avg_f1 + avg_iou) / 3
        
        trial_history.append({
            "epoch": epoch + 1,
            "train_loss": total_train_loss / len(train_loader),
            "val_dice": avg_dice,
            "val_f1": avg_f1,
            "val_iou": avg_iou,
            "score": current_score
        })

        # ✅ 3. Epoch 总结打印
        logger.info(f"One Epoch 結束===> Epoch {epoch+1}: Train Loss: {total_train_loss/len(train_loader):.4f} | Val Dice: {avg_dice:.4f} | F1: {avg_f1:.4f} | IoU: {avg_iou:.4f}<===")
        
        if current_score > trial_best_score:
            trial_best_score = current_score
            trial_best_weights = copy.deepcopy(model.state_dict())
            trial_best_metrics = {"dice": avg_dice, "f1": avg_f1, "iou": avg_iou, "score": trial_best_score}
        scheduler.step()

    # --- 全局最强保存 ---
    if trial_best_score > global_best_score:
        global_best_score = trial_best_score
        save_name = f"model_trial_{test_params['trial_num']}.pth"
        torch.save({"state_dict": trial_best_weights, "metrics": trial_best_metrics}, save_name)
        logger.debug(f"🔥 全場最強更新！保存为{save_name}.pth | Score: {global_best_score:.4f}")
        # 🚀 新增：突破纪录时导出 Origin 绘图专用的 CSV 文件
        csv_name = f"origin_data_trial_{test_params['trial_num']}.csv"
        if trial_history:
            keys = trial_history[0].keys()
            with open(csv_name, 'w', newline='') as f:
                dict_writer = csv.DictWriter(f, fieldnames=keys)
                dict_writer.writeheader()
                dict_writer.writerows(trial_history)
    else:
        logger.debug('***********本局未找到更优模型**************')
    del model; torch.cuda.empty_cache()
    return trial_best_score