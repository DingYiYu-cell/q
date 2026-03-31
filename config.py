import torch
# ==========================================
# 超参数
# ==========================================
# data_root = "/root/Dataset/Pituitary tumor"
# batch = 20  
# num_epochs = 50 
# lr = 1e-4
# global_seed = 42

config= {
    "data_root": 'E:/Desktop/train_outputpro',
    "batch":20,
    "num_epochs":50,
    # "ctrl_Cos_nums":15,交给optuna管理
    #"lr":1e-4, 交给optuna管理
    "global_seed":42,
    "save_path": 'E:/Desktop/save_path',
    }#初始条件写入字典
torch.manual_seed(config['global_seed'])#固定全局种子