import torch
import torch.nn as nn
import torch.nn.functional as F  # 修正：必须导入 F 才能使用 interpolate
import os
from torch.utils.data import DataLoader
import MyDataset
import AttResUNet
from config import config
import sys
# ==========================================
# 0. 系统检查
# ==========================================
class SystemValidator:
    def __init__(self):
        """仅做逻辑检查，路径由外部传入"""
        pass

    def _fail(self, msg):
        """报错并强行退出"""
        print(f"\n🚨 [环境报错]: {msg}")
        sys.exit(1)

    def check_env(self, data_root, save_path):
        print("="*40)
        print("🔍 启动前自检...")
        
        # 1. 检查 CUDA (GPU)
        if not torch.cuda.is_available():
            self._fail("未检测到 CUDA 设备。请确认已开启 GPU 实例，或检查驱动配置。")
        
        gpu_name = torch.cuda.get_device_name(0)
        print(f"✅ GPU 环境: {gpu_name}")

        # 2. 检查数据集路径
        if not os.path.exists(data_root):
            self._fail(f"数据集目录不存在: {data_root}")
        print(f"✅ 数据集路径: {data_root}")

        # 3. 检查模型保存路径 (不存在则自动创建)
        if not os.path.exists(save_path):
            try:
                os.makedirs(save_path)
                print(f"📂 提示: 已创建保存目录: {save_path}")
            except Exception as e:
                self._fail(f"无法创建保存目录 {save_path}: {e}")
        print(f"✅ 保存路径: {save_path}")

        print("🚀 [SUCCESS] 检查通过，环境就绪！")
        print("="*40 + "\n")
SystemValidator().check_env(config["data_root"], config["save_path"])
        


# ==========================================
# 3. 训练流程 (含数据集划分与自动验证)
# ==========================================
from torch.utils.data import random_split

# 1. 实例化并划分数据集 (8:1:1)
full_dataset = MyDataset(data_root=config["data_root"])

train_size = int(0.8 * len(full_dataset))
val_size = int(0.1 * len(full_dataset))
test_size = len(full_dataset) - train_size - val_size

train_dataset, val_dataset, test_dataset = random_split(
    full_dataset, [train_size, val_size, test_size],
    generator=torch.Generator().manual_seed(config["global_seed"])
)

# 2. 建立 DataLoader
train_loader = DataLoader(train_dataset, batch_size=config["batch"], shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=config["batch"], shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=config["batch"], shuffle=False)

print(f"✅ 数据划分完成：训练集 {len(train_dataset)}，验证集 {len(val_dataset)}，测试集 {len(test_dataset)}")

# 3. 初始化模型与优化器
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🚀 当前使用的设备: {device}")

model = AttResUNet(in_channels=1, out_channels=1).to(device)
criterion = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"])
# --- 新增：余弦退火策略 ---
# T_max 通常设置为总的 epoch 数，表示学习率从最大降到最小所需的周期
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config["config["num_epochs"]"], eta_min=1e-6)
# eta_min 是学习率能降到的最小值，建议设置一个较小的数（如 1e-6）而不是 0

# 用于保存最佳模型的变量
best_val_loss = float('inf')

print("开始训练...")

for epoch in range(config["num_epochs"]):
    # --- 训练阶段 ---
    model.train()
    epoch_train_loss = 0.0
    
    for step, (images, masks) in enumerate(train_loader):
        images, masks = images.to(device), masks.to(device)
        
        # 前向传播
        outputs = model(images)
        loss = criterion(outputs, masks)
        
        # 反向传播与优化
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        epoch_train_loss += loss.item()
        
        if step % 5 == 0:
            print(f"Epoch [{epoch+1}/{config["num_epochs"]}], Step [{step}/{len(train_loader)}], Train Loss: {loss.item():.4f}")

    # --- 验证阶段 ---
    model.eval() # 切换为评估模式
    epoch_val_loss = 0.0
    
    with torch.no_grad(): # 验证时不计算梯度，节省内存和时间
        for images, masks in val_loader:
            images, masks = images.to(device), masks.to(device)
            outputs = model(images)
            v_loss = criterion(outputs, masks)
            epoch_val_loss += v_loss.item()
    # --- 新增：更新学习率 ---
    # 获取当前学习率用于打印查看
    current_lr = optimizer.param_groups[0]['lr']
    print(f"📡 Current Learning Rate: {current_lr:.6f}")
    # 计算本轮平均损失
    avg_train_loss = epoch_train_loss / len(train_loader)
    avg_val_loss = epoch_val_loss / len(val_loader)
    
    print(f"===> Epoch [{epoch+1}/{config["num_epochs"]}] Avg Train Loss: {avg_train_loss:.4f} | Avg Val Loss: {avg_val_loss:.4f}")

    # --- 保存性能最好的模型 ---
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        checkpoint={
            "state_dict":model.state_dict(),
            "config":config
        }#打包模型参数以及初始条件
        torch.save(checkpoint, "att_res_unet_best.pth")#一并封装保存
        print(f"⭐ 发现更优验证集表现，模型已更新保存！")
    print("-" * 30)

print(f"训练完成！最优验证集 Loss 为: {best_val_loss:.4f}")