import torch
import cv2
import numpy as np
import os
from AttResUNet import AttResUNet

def predict():
    # --- 1. 参数配置 ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    weight_path = "model_trial_33.pth"  # 替换为你的权重文件名
    img_path = "test2.png"      # 替换为你要测试的图片路径
    output_path = "result2.png"       # 输出结果路径
    input_size = (224, 224)         # 保持与训练时的输入尺寸一致
    threshold = 0.1                  # 分割阈值

    # --- 2. 加载模型 ---
    # 根据你的 AttResUNet 定义：in=1, out=1
    model = AttResUNet(in_channels=1, out_channels=1).to(device)
    
    # 根据你的保存逻辑：torch.save({"state_dict": ...}, ...)
    checkpoint = torch.load(weight_path, map_location=device)
    if 'state_dict' in checkpoint:
        model.load_state_dict(checkpoint['state_dict'])
    else:
        model.load_state_dict(checkpoint) # 兼容直接保存 state_dict 的情况
    
    model.eval()
    print(f"✅ 模型加载成功: {weight_path}")

    # --- 3. 图像预处理 ---
    # 读取图像并转为单通道灰度图（符合模型 in_channels=1）
    src_img = cv2.imread(img_path)
    if src_img is None:
        print(f"❌ 找不到图片: {img_path}")
        return
    
    gray_img = cv2.cvtColor(src_img, cv2.COLOR_BGR2GRAY)
    h, w = gray_img.shape
    
    # 缩放至模型输入尺寸并归一化
    input_img = cv2.resize(gray_img, input_size)
    input_tensor = input_img.astype(np.float32) / 255.0
    input_tensor = torch.from_numpy(input_tensor).unsqueeze(0).unsqueeze(0).to(device)

    # --- 4. 执行推理 ---
    with torch.no_grad():
        output = model(input_tensor)
        # 你的模型输出通常经过 Sigmoid，若没有，请取消下行注释
        # output = torch.sigmoid(output)
        
        pred = output.squeeze().cpu().numpy()

    # --- 5. 后处理与保存 ---
    # 将概率图转为二值图
    binary_mask = (pred > threshold).astype(np.uint8) * 255
    
    # 还原回原图尺寸
    result_mask = cv2.resize(binary_mask, (w, h), interpolation=cv2.INTER_NEAREST)

    # 保存结果
    cv2.imwrite(output_path, result_mask)
    print(f"🚀 分割完成！结果已保存至: {output_path}")

    # (可选) 生成叠加图，方便查看效果
    overlay = src_img.copy()
    overlay[result_mask > 0] = [0, 255, 0] # 将分割区域染成绿色
    combined = cv2.addWeighted(src_img, 0.7, overlay, 0.3, 0)
    cv2.imwrite("overlay_result.png", combined)

if __name__ == "__main__":
    predict()