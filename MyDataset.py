# ==========================================
# 1. 自定义数据类
# ==========================================
from torch.utils.data import Dataset
import os
import re
import torchvision
from PIL import Image
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
            torchvision.transforms.Resize((224, 224)), 
            torchvision.transforms.ToTensor()
        ])

    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, idx):
        img_path = os.path.join(self.base_dir, self.img_files[idx])
        mask_path = os.path.join(self.base_dir, self.mask_files[idx])
        img = Image.open(img_path)
        mask = Image.open(mask_path)
        return self.transform(img), self.transform(mask)