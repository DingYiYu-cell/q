import torch
import re
import torch.nn as nn
import torch.nn.functional as F  # 修正：必须导入 F 才能使用 interpolate
import os
import torchvision
from torch.utils.data import Dataset, DataLoader
from PIL import Image

"""
                            _ooOoo_
                           o8888888o
                           88" . "88
                           (| -_- |)
                           O\  =  /O
                        ____/`---'\____
                      .'  \\|     |//  `.
                     /  \\|||  :  |||//  \
                    /  _||||| -:- |||||-  \
                    |   | \\\  -  /// |   |
                    | \_|  ''\---/''  |   |
                    \  .-\__  `-`  ___/-. /
                  ___`. .'  /--.--\  `. . __
               ."" '<  `.___\_<|>_/___.'  >'"".
              | | :  `- \`.;`\ _ /`;.`/ - ` : | |
              \  \ `-.   \_ __\ /__ _/   .-` /  /
         ======`-.____`-.___\_____/___.-`____.-'======
                            `=---='
        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
                 佛祖保佑       永無BUG       水豚加持
                 指標正常       汎化超常      畢業順利
"""
# ==========================================
# 超参数
# ==========================================
data_root = "/root/Dataset/Pituitary tumor"
batch = 20  
num_epochs = 50 
lr = 1e-4
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

    def check_env(self, dataset_path, save_path):
        print("="*40)
        print("🔍 启动前自检...")
        
        # 1. 检查 CUDA (GPU)
        if not torch.cuda.is_available():
            self._fail("未检测到 CUDA 设备。请确认已开启 GPU 实例，或检查驱动配置。")
        
        gpu_name = torch.cuda.get_device_name(0)
        print(f"✅ GPU 环境: {gpu_name}")

        # 2. 检查数据集路径
        if not os.path.exists(dataset_path):
            self._fail(f"数据集目录不存在: {dataset_path}")
        print(f"✅ 数据集路径: {dataset_path}")

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
SystemValidator().check_env(dataset_path='',save_path='')
        

# ==========================================
# 1. 自定义数据类
# ==========================================
class MyDataset(Dataset):
    def __init__(self, data_root):
        super().__init__()
        self.base_dir = data_root
        if not os.path.exists(self.base_dir):
            raise FileNotFoundError(f"找不到目录: {self.base_dir}")

        all_files = os.listdir(self.base_dir)

        def sort_key(x):
            match = re.search(r'enh_(\d+)', x)
            return int(match.group(1)) if match else 0

        self.img_files = sorted(
            [f for f in all_files if f.startswith("enh_") and f.endswith(".png") and "_mask" not in f],
            key=sort_key
        )
        self.mask_files = sorted(
            [f for f in all_files if f.startswith("enh_") and f.endswith("_mask.png")],
            key=sort_key
        )

        if len(self.img_files) != len(self.mask_files):
            print(f"⚠️ 警告：原图({len(self.img_files)})与掩码({len(self.mask_files)})数量不一致！")
        else:
            print(f"✅ 成功加载！共找到 {len(self.img_files)} 对有效的图像-掩码对。")

        self.transform = torchvision.transforms.Compose([
            torchvision.transforms.Resize((256, 256)), 
            torchvision.transforms.ToTensor()
        ])

    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, idx):
        img_path = os.path.join(self.base_dir, self.img_files[idx])
        mask_path = os.path.join(self.base_dir, self.mask_files[idx])
        img = Image.open(img_path).convert("L")
        mask = Image.open(mask_path).convert("L")
        return self.transform(img), self.transform(mask)

# ==========================================
# 2. 模型组件
# ==========================================
class ResBlock(nn.Module):#残差卷积块 类
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.mp = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_channels)
        )
        self.shortcut = nn.Sequential()#恒等映射
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.mp(x)
        out += self.shortcut(x)
        return self.relu(out)#两条线路会和后再ReLU

class AttentionGate(nn.Module):#注意力门 类
    def __init__(self, F_g, F_l, F_int):
        super().__init__()
        self.W_g = nn.Sequential(
            nn.Conv2vd(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )
        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        return x * psi

class EncoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.res_block = ResBlock(in_channels, out_channels)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        f = self.res_block(x)
        p = self.pool(f)
        return f, p

class DecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DecoderBlock, self).__init__()
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        # 修正：这里直接传参数，不再使用不匹配的关键词 gating_channels
        self.att_gate = AttentionGate(in_channels, out_channels, out_channels // 2)
        self.res_block = ResBlock(in_channels + out_channels, out_channels)

    def forward(self, x, skip):
        g = self.upsample(x)
        # 修正：尺寸检查逻辑
        if g.size()[2:] != skip.size()[2:]:
            g = F.interpolate(g, size=skip.size()[2:], mode='bilinear', align_corners=True)
        
        s = self.att_gate(g, skip) # 修正：直接传参
        d = torch.cat([s, g], dim=1)
        return self.res_block(d)

class AttResUNet(nn.Module):
    def __init__(self, in_channels=1, out_channels=1):
        super().__init__()
        self.enc1 = EncoderBlock(in_channels, 64)
        self.enc2 = EncoderBlock(64, 128)
        self.enc3 = EncoderBlock(128, 256)
        self.bottleneck = ResBlock(256, 512)
        self.dec3 = DecoderBlock(512, 256)
        self.dec2 = DecoderBlock(256, 128)
        self.dec1 = DecoderBlock(128, 64)
        self.final = nn.Conv2d(64, out_channels, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        f1, p1 = self.enc1(x)
        f2, p2 = self.enc2(p1)
        f3, p3 = self.enc3(p2)
        b = self.bottleneck(p3)
        d3 = self.dec3(b, f3)
        d2 = self.dec2(d3, f2)
        d1 = self.dec1(d2, f1)
        return self.sigmoid(self.final(d1))

# ==========================================
# 3. 训练流程 (含数据集划分与自动验证)
# ==========================================
from torch.utils.data import random_split

# 1. 实例化并划分数据集 (8:1:1)
full_dataset = MyDataset(data_root=data_root)

train_size = int(0.8 * len(full_dataset))
val_size = int(0.1 * len(full_dataset))
test_size = len(full_dataset) - train_size - val_size

train_dataset, val_dataset, test_dataset = random_split(
    full_dataset, [train_size, val_size, test_size],
    generator=torch.Generator().manual_seed(42)
)

# 2. 建立 DataLoader
train_loader = DataLoader(train_dataset, batch_size=batch, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch, shuffle=False)

print(f"✅ 数据划分完成：训练集 {len(train_dataset)}，验证集 {len(val_dataset)}，测试集 {len(test_dataset)}")

# 3. 初始化模型与优化器
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🚀 当前使用的设备: {device}")

model = AttResUNet(in_channels=1, out_channels=1).to(device)
criterion = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=lr)
# --- 新增：余弦退火策略 ---
# T_max 通常设置为总的 epoch 数，表示学习率从最大降到最小所需的周期
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=1e-6)
# eta_min 是学习率能降到的最小值，建议设置一个较小的数（如 1e-6）而不是 0

# 用于保存最佳模型的变量
best_val_loss = float('inf')

print("开始训练...")

for epoch in range(num_epochs):
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
            print(f"Epoch [{epoch+1}/{num_epochs}], Step [{step}/{len(train_loader)}], Train Loss: {loss.item():.4f}")

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
    
    print(f"===> Epoch [{epoch+1}/{num_epochs}] Avg Train Loss: {avg_train_loss:.4f} | Avg Val Loss: {avg_val_loss:.4f}")

    # --- 保存性能最好的模型 ---
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        torch.save(model.state_dict(), "att_res_unet_best.pth")
        print(f"⭐ 发现更优验证集表现，模型已更新保存！")
    print("-" * 30)

print(f"训练完成！最优验证集 Loss 为: {best_val_loss:.4f}")