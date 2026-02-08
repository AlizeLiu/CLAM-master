import os
import numpy as np
import h5py
import openslide
import cv2
import torch
import torchstain
from PIL import Image

# --- 配置区域 ---
SLIDE_PATH = r"J:\Work\CLAM-master\toy_example\macenko_demo_1.svs"
H5_PATH = r"J:\Work\CLAM-master\toy_test\patches\macenko_demo_1.h5"
REF_PATH = r"J:\Work\CLAM-master\macenko_simple\reference.png"


# ----------------

def debug_deep_fixed():
    print("--- 🔬 开始深度调试 (维度修复版) ---")

    # 1. 准备参考图
    ref_img = cv2.cvtColor(cv2.imread(REF_PATH), cv2.COLOR_BGR2RGB)
    ref_img = cv2.resize(ref_img, (224, 224))

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print(f"1. 设备: {device}")

    # [修复] 参考图转 (C, H, W)
    T_tensor = torch.from_numpy(ref_img).permute(2, 0, 1).to(device)
    normalizer = torchstain.normalizers.MacenkoNormalizer(backend='torch')
    normalizer.fit(T_tensor)
    print("   ✅ Normalizer 初始化成功 (CHW mode)")

    # 2. 读取测试图
    wsi = openslide.open_slide(SLIDE_PATH)
    with h5py.File(H5_PATH, 'r') as f:
        coords = f['coords'][:]

    # 找一张深色的图
    print("2. 寻找深色测试图...")
    test_img = None
    for i in range(200):
        idx = np.random.randint(len(coords))
        img = wsi.read_region(tuple(coords[idx]), 0, (256, 256)).convert('RGB')
        img = img.resize((224, 224))
        arr = np.array(img)
        if arr.mean() < 180:
            test_img = arr
            print(f"   ✅ 找到 Patch {idx}, Mean={arr.mean():.2f}")
            break

    if test_img is None:
        print("   ❌ 未找到合适图片，随机使用一张")
        test_img = arr

    # 3. 执行标准化
    try:
        # [修复] 输入图转 (C, H, W)
        img_tensor = torch.from_numpy(test_img).permute(2, 0, 1).to(device)

        result = normalizer.normalize(I=img_tensor, stains=False)

        # 处理返回值 (torchstain 可能返回元组)
        if isinstance(result, tuple):
            norm_tensor = result[0]
        else:
            norm_tensor = result

        # [修复] 输出图转回 (H, W, C)
        norm_np = norm_tensor.permute(1, 2, 0).cpu().numpy().astype(np.uint8)

        # 4. 计算差异
        diff = np.abs(test_img.astype(np.float32) - norm_np.astype(np.float32))
        mae = np.mean(diff)

        print(f"\n📊 结果分析:")
        print(f"   原图均值: {test_img.mean():.2f}")
        print(f"   新图均值: {norm_np.mean():.2f}")
        print(f"   平均像素差异 (MAE): {mae:.4f}")

        if mae > 1.0:
            print("\n🎉 成功！检测到像素变化，标准化已生效。")
        else:
            print("\n⚠️ 警告：像素变化极小，可能是参考图与原图风格过于接近。")

    except Exception as e:
        print(f"\n❌ 依然报错: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    debug_deep_fixed()