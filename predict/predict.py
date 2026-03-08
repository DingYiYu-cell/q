import os
# 1. 解决 OMP 报错（两个指挥官打架的问题）喵！
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
import sys
import torch
import cv2
import numpy as np



# --- 2. 自动定位文件夹路径喵 ---
current_file = os.path.abspath(__file__)
current_dir = os.path.dirname(current_file) # predict 文件夹
q_dir = os.path.dirname(current_dir)         # q 文件夹
main_dir = os.path.dirname(q_dir)           # main 文件夹

# 把所有相关目录都加到 Python 的搜索名单里喵
sys.path.append(current_dir)
sys.path.append(q_dir)
sys.path.append(main_dir)

# 3. 尝试导入模型类喵
try:
    from AttResUNet import AttResUNet
    print("✅ 成功召唤 AttResUNet 家族喵！")
except ImportError as e:
    print(f"❌ 找不到模型零件喵：{e}")
    sys.exit()

def predict():
    # --- 4. 文件路径配置喵 ---
    img_name = 'test_crack.png' 
    img_path = os.path.join(current_dir, img_name)
    
    model_name = 'att_res_unet_best.pth'
    model_path = os.path.join(q_dir, model_name)
    
    img_size = (256, 256) # 考研 5500 词汇要记，这个尺寸也要记死喵
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 5. 加载模型与权重喵
    model = AttResUNet().to(device)
    try:
        checkpoint = torch.load(model_path, map_location=device)
        if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
            model.load_state_dict(checkpoint['state_dict'])
        else:
            model.load_state_dict(checkpoint)
        print(f"✅ 成功加载权重：{model_path} 喵！")
    except Exception as e:
        print(f"❌ 加载权重失败了喵：{e}")
        return

    model.eval()

    # --- 6. 核心修复：解决中文路径读取喵 ---
    print(f"🔍 正在尝试读取：{img_path}")
    try:
        # 使用 numpy 绕过 OpenCV 对中文路径的解析喵
        raw_data = np.fromfile(img_path, dtype=np.uint8)
        src_img = cv2.imdecode(raw_data, cv2.IMREAD_COLOR)
    except Exception as e:
        print(f"❌ 读取过程崩溃了喵：{e}")
        src_img = None

    if src_img is None:
        print(f"❌ 读图失败！请确认 '{img_name}' 确实在 predict 文件夹里喵！")
        print(f"📂 当前文件夹下的文件有：{os.listdir(current_dir)}")
        return
    
    # 7. 预处理：变灰度图 + 缩放喵
    gray_img = cv2.cvtColor(src_img, cv2.COLOR_BGR2GRAY)
    resized_img = cv2.resize(gray_img, img_size)
    img_data = resized_img.astype(np.float32) / 255.0
    img_tensor = torch.from_numpy(img_data).unsqueeze(0).unsqueeze(0).to(device)

    # 8. 推理时间喵！
    print("🚀 正在进行模型推理...")
    with torch.no_grad():
        output = model(img_tensor)
        output = output.cpu().numpy()[0, 0]

    # --- 9. 核心修复：强行生成图片（解决中文保存问题）喵 ---
    print("🎨 正在生成结果图片...")
    prob_out_path = os.path.join(current_dir, 'prob_result.png')
    result_out_path = os.path.join(current_dir, 'result.png')

    try:
        # 概率图保存（带中文路径保护喵）
        prob_map = (output * 255).astype(np.uint8)
        prob_res = cv2.resize(prob_map, (src_img.shape[1], src_img.shape[0]))
        _, buf1 = cv2.imencode('.png', prob_res)
        buf1.tofile(prob_out_path)
        print(f"✨ 成功强行生成概率图：{prob_out_path}")

        # 二值图保存（阈值设为 0.5 确保一定有东西喵）
        threshold = 0.5 
        prediction = (output > threshold).astype(np.uint8) * 255
        final_res = cv2.resize(prediction, (src_img.shape[1], src_img.shape[0]))
        _, buf2 = cv2.imencode('.png', final_res)
        buf2.tofile(result_out_path)
        print(f"✨ 成功强行生成二值图：{result_out_path}")
        
    except Exception as e:
        print(f"❌ 保存图片时夭折了喵：{e}")

    print(f"🏁 任务完成！如果文件夹还是空的，请查看上方报错喵~！~!")

if __name__ == '__main__':
    predict()