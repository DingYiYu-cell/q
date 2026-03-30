import optuna
import wandb
import torch
import torch.nn as nn
from AttResUNet import AttResUNet
from train import FocalLoss,DiceLoss,train_loader,val_loader,get_dice,device
# ... 其他原有 import 保持不变 ...

def objective(trial):
    
    # ==========================================
    # A. 定义待搜索的超参数 (Hyperparameter Sampling)
    # ==========================================
    # 搜索学习率：1e-5 到 1e-2
    lr = trial.suggest_float("lr", 1e-5, 1e-2, log=True)
    # 搜索 Focal Loss 的 Gamma：1.5 到 4.0
    gamma = trial.suggest_float("gamma", 1.5, 4.0)
    # 搜索复合损失的权重比例 (Dice 的权重)
    dice_weight = trial.suggest_float("dice_weight", 0.5, 2.0)
    
    # ==========================================
    # B. 初始化 WandB (每个 Trial 一个 Run)
    # ==========================================
    run = wandb.init(
        project="Crack_Detection_Optuna",
        config={
            "lr": lr,
            "gamma": gamma,
            "dice_weight": dice_weight,
            "trial_id": trial.number
        },
        reinit=True, # 允许在一个进程内多次初始化
        group="Optuna_Search"
    )

    # ==========================================
    # C. 模型与优化器初始化
    # ==========================================
    model = AttResUNet(in_channels=1, out_channels=1).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    # 注意：这里 Epoch 建议设少一点（如 10-20），快速试错
    num_epochs = 15 
    
    criterion_focal = FocalLoss(alpha=0.25, gamma=gamma)
    criterion_dice = DiceLoss()
    
    best_dice_in_trial = 0.0

    # ==========================================
    # D. 训练循环 (简化版)
    # ==========================================
    for epoch in range(num_epochs):
        model.train()
        for images, masks in train_loader:
            images, masks = images.to(device), masks.to(device)
            outputs = model(images)
            
            l_f = criterion_focal(outputs, masks)
            l_d = criterion_dice(outputs, masks)
            loss = 0.8 * l_f + dice_weight * l_d
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # 验证逻辑 (复用你原来的代码)
        model.eval()
        val_dice = 0.0
        with torch.no_grad():
            for images, masks in val_loader:
                images, masks = images.to(device), masks.to(device)
                outputs = model(images)
                val_dice += get_dice(outputs, masks)
        
        avg_val_dice = val_dice / len(val_loader)
        
        # 记录到 WandB
        wandb.log({"trial_dice": avg_val_dice, "epoch": epoch})
        
        # ==========================================
        # E. Optuna 剪枝 (Pruning)
        # 如果当前这组参数跑了几轮发现没戏，直接掐断省时间
        # ==========================================
        trial.report(avg_val_dice, epoch)
        if trial.should_prune():
            run.finish()
            raise optuna.exceptions.TrialPruned()
            
        best_dice_in_trial = max(best_dice_in_trial, avg_val_dice)

    run.finish()
    return best_dice_in_trial

# ==========================================
# 6. 启动搜索主程序
# ==========================================
if __name__ == "__main__":
    # 如果没联网，可以开启离线模式：os.environ["WANDB_MODE"] = "offline"
    wandb.login()
    
    study = optuna.create_study(
        direction="maximize", 
        pruner=optuna.pruners.MedianPruner() # 自动剪枝策略
    )
    
    # 跑 30 次尝试，寻找最优解
    study.optimize(objective, n_trials=30)

    print("\n" + "="*40)
    print(f"搜索结束！最优参数: {study.best_params}")
    print(f"最高验证集 Dice: {study.best_value:.4f}")