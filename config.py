# ==========================================
# 超参数
# ==========================================
data_root = "/root/Dataset/Pituitary tumor"
batch = 20  
num_epochs = 50 
lr = 1e-4
global_seed = 42
config= {
    "data_root":data_root,
    "batch":batch,
    "num_epochs":num_epochs,
    "lr":lr,
    "global_seed":global_seed
    }#初始条件写入字典
torch.manual_seed(global_seed)#固定全局种子