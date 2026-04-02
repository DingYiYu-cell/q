import optuna
import train
import datetime
import os
import torch

total_trials = 3
def objective(trial):
    test_params = {
        'opt_class': trial.suggest_categorical('opt_class', ['AdamW','RMSprop','SGD']),
        'lr': trial.suggest_float('lr', 1e-5, 1e-3, log=True),
        'batch_size': trial.suggest_int('batch', 10, 50, log=True),
        'cos_maxt': trial.suggest_int('cos_maxt', 5, 10),
        'alpha': trial.suggest_float('alpha', 0.7, 1.3),
        'beta': trial.suggest_float('beta', 0.7, 1.3),
        'trial_num': trial.number,
        'total_trials':total_trials,
    }
    return train.run_training(test_params,trial=trial)

if __name__ == "__main__":
    study = optuna.create_study(
    direction="maximize",
    pruner=optuna.pruners.MedianPruner(
        n_startup_trials=1,  # 跑完 2 个 Trial 后才开启剪枝检查
        n_warmup_steps=2,    # 每个 Trial 跑完 2 个 Epoch 就开始考核，不准混日子
        interval_steps=1     # 每个 Epoch 都检查一次
    )
)
    study.optimize(objective, n_trials=total_trials)
    

    best_trial_num = study.best_trial.number
    best_model_name = f"model_trial_{best_trial_num}.pth"
    csv_name = f"origin_data_trial_{best_trial_num}.csv"
    # 检查文件是否存在，防止中途夭折导致读取失败
    if not os.path.exists(best_model_name) and os.path.exist(csv_name):
        print("❌ 错误：没找到 best_model.pth，可能所有 Trial 都跑得太烂了或者中途崩了。")
    else:
        # 修正拼写错误：map_vars -> map_location
        best_ckpt = torch.load(best_model_name, map_location=torch.device('cpu'))
        m = best_ckpt["metrics"]

        # 严格还原你原来的输出报告格式！！
        print("\n" + "🚀" * 30)
        print("📢  Optuna 調參任務最終清算報告")
        print("🚀" * 30 + "\n")

        print(f"{'='*60}")
        print(f"📌 狀態標記: 🔥 [THE BEST ONE]")
        print(f"⏰ 存儲時間: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"📂 存儲位置: {os.getcwd()}")
        print(f"📄 存儲名稱: {best_model_name} AND {csv_name}")
        print(f"📈 綜合得分: {m['score']:.4f} (Dice: {m['dice']:.4f}, F1: {m['f1']:.4f}, IoU: {m['iou']:.4f})")
        print(f"⚙️ 注入參數: {study.best_trial.params}")
        print(f"{'='*60}\n")