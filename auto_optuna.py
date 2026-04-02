import optuna
import train
import datetime
import os
import torch

total_trials = 150
def objective(trial):
    test_params = {
        'opt_class': trial.suggest_categorical('opt_class', ['AdamW','RMSprop','SGD','RAdam','NAdam']),
        'lr': trial.suggest_float('lr', 1e-5, 1e-3, log=True),
        'batch_size': trial.suggest_int('batch', 10, 100, log=True),
        'cos_maxt': trial.suggest_int('cos_maxt', 5, 10),
        'alpha': trial.suggest_float('alpha', 0.5, 1.5),
        'beta': trial.suggest_float('beta', 0.5, 1.5),
        'trial_num': trial.number,
        'total_trials':total_trials,
    }
    return train.run_training(test_params)

if __name__ == "__main__":
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=total_trials)
    

    best_trial_num = study.best_trial.number
    best_model_name = f"model_trial_{best_trial_num}.pth"

    # 检查文件是否存在，防止中途夭折导致读取失败
    if not os.path.exists("best_model.pth"):
        print("❌ 错误：没找到 model.pth，可能所有 Trial 都跑得太烂了或者中途崩了。")
    else:
        # 修正拼写错误：map_vars -> map_location
        best_ckpt = torch.load("best_model.pth", map_location=torch.device('cpu'))
        m = best_ckpt["metrics"]

        # 严格还原你原来的输出报告格式！！
        print("\n" + "🚀" * 30)
        print("📢  Optuna 調參任務最終清算報告")
        print("🚀" * 30 + "\n")

        print(f"{'='*60}")
        print(f"📌 狀態標記: 🔥 [THE BEST ONE]")
        print(f"⏰ 存儲時間: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"📂 存儲位置: {os.getcwd()}")
        print(f"📄 存儲名稱: {best_model_name}")
        print(f"📈 綜合得分: {m['score']:.4f} (Dice: {m['dice']:.4f}, F1: {m['f1']:.4f}, IoU: {m['iou']:.4f})")
        print(f"📉 參數數量: 24.74 M")
        print(f"⚙️ 注入參數: {study.best_trial.params}")
        print(f"{'='*60}\n")