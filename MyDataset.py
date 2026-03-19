import os
import re
import torch
import torchvision
from torch.utils.data import Dataset
from PIL import Image
import numpy as np

class MyDataset(Dataset):
    def __init__(self, data_root):
        super().__init__()
        self.base_dir = data_root
        if not os.path.exists(self.base_dir):
            raise FileNotFoundError(f"找不到目录: {self.base_dir}")

        all_files = os.listdir(self.base_dir)

        # 排序逻辑：提取文件名开头的数字
        def sort_key(x):
            match = re.search(r'^(\d+)', x)
            return int(match.group(1)) if match else 0

        # 1. 提取原图文件名列表
        self.img_files = sorted(
            [f for f in all_files if f.endswith(".png") and not f.endswith("_mask.png") and re.match(r'^\d+', f)],
            key=sort_key
        )
        
        # 2. 提取掩码图文件名列表
        self.mask_files = sorted(
            [f for f in all_files if f.endswith("_mask.png")],
            key=sort_key
        )

        # 验证配对
        self.validate_pairs()

        # 定义缩放操作，确保图片和掩码尺寸一致喵
        self.resize = torchvision.transforms.Resize((224, 224))
        self.to_tensor = torchvision.transforms.ToTensor()

    def validate_pairs(self):
        if len(self.img_files) != len(self.mask_files):
            print(f"⚠️ 警告：原图({len(self.img_files)})与掩码({len(self.mask_files)})数量不一致！")
        
        mismatches = 0
        for i in range(min(len(self.img_files), len(self.mask_files))):
            img_num = self.img_files[i].replace(".png", "")
            mask_num = self.mask_files[i].replace("_mask.png", "")
            if img_num != mask_num:
                mismatches += 1
        
        if mismatches == 0 and len(self.img_files) == len(self.mask_files):
            print(f"✅ 成功加载！共找到 {len(self.img_files)} 对完美的图像-掩码对。")
        elif mismatches > 0:
            print(f"❌ 严重错误：有 {mismatches} 对文件名序号无法匹配！")

    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, index):
        # 读取原图
        img_path = os.path.join(self.base_dir, self.img_files[index])
        image = Image.open(img_path).convert('L')
        image = self.resize(image)
        image = self.to_tensor(image)

        # 读取掩码图（对应 self.mask_files）喵
        mask_path = os.path.join(self.base_dir, self.mask_files[index])
        
        # 1. 转为灰度图
        mask_img = Image.open(mask_path).convert('L')
        # 2. 缩放到和原图一样的大小
        mask_img = self.resize(mask_img)
        # 3. 转为 numpy 矩阵进行二值化处理
        mask_np = np.array(mask_img)
        # 4. 只要有颜色(红色)的地方全部变成 1.0，其余背景 0.0 喵
        mask_binary = (mask_np > 0).astype(np.float32)
        
        # 5. 转回 Tensor 并增加通道维度 (1, H, W)
        mask_tensor = torch.from_numpy(mask_binary).unsqueeze(0)

        return image, mask_tensor