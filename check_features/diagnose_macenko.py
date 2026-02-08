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

def diagnose():
    print(f"--- 开始诊断 ---")
    print(f"1. 检查参考图像: {REF_PATH}")
    if not os.path.exists(REF_PATH):
        print("错误：找不到参考图像文件！")
        return

    ref_img = cv2.imread(REF_PATH)
    ref_img = cv2.cvtColor(ref_img, cv2.COLOR_BGR2RGB)
    print(f"   参考图像尺寸: {ref_img.shape}")
    print(f"   参考图像平均亮度: {np.mean(ref_img):.2f} (理想值应在 150-200 之间)")

    # 检查参考图是否有足够的颜色差异
    ref_std = np.std(ref_img)
    print(f"   参考图像对比度(std): {ref_std:.2f}")
    if ref_std < 20:
        print("警告：参考图像对比度过低，可能导致算法失败！建议更换对比度更高的图片。")

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print(f"2. 初始化算法 (Device: {device})...")

    try:
        T = torch.from_numpy(ref_img).to(device)
        normalizer = torchstain.normalizers.MacenkoNormalizer(backend='torch')
        normalizer.fit(T)
        print("Macenko 初始化成功！参考图似乎没问题。")
    except Exception as e:
        print(f"致命错误：Macenko 初始化失败！说明参考图片不合格。")
        print(f"   错误信息: {e}")
        return

    print(f"3. 检查切片 Patch...")
    wsi = openslide.open_slide(SLIDE_PATH)
    with h5py.File(H5_PATH, 'r') as f:
        coords = f['coords'][:]
        patch_size = f['coords'].attrs.get('patch_size', 256)

    print(f"   共 {len(coords)} 个坐标。开始抽样检测前 10 个...")

    # 随机抽 10 个来看看
    indices = np.random.choice(len(coords), 10, replace=False)

    for i, idx in enumerate(indices):
        coord = coords[idx]
        print(f"\n--- Patch {idx} (坐标: {coord}) ---")

        # 读取
        img_pil = wsi.read_region(tuple(coord), 0, (int(patch_size), int(patch_size))).convert('RGB')
        img_pil = img_pil.resize((224, 224))
        img_np = np.array(img_pil)

        # 计算统计量
        img_gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
        mean_val = np.mean(img_gray)
        white_ratio = np.sum(img_gray > 220) / img_gray.size
        std_val = np.std(img_gray)

        print(f"   数据统计: Mean={mean_val:.1f}, WhiteRatio={white_ratio:.2f}, Std={std_val:.1f}")

        # 模拟过滤逻辑
        if mean_val > 240 or white_ratio > 0.90:
            print("结果: 被判定为背景 (Filtered as Background)")
        else:
            print("结果: 通过背景过滤，尝试标准化...")
            try:
                img_tensor = torch.from_numpy(img_np).to(device)
                normalizer.normalize(I=img_tensor, stains=False)
                print("成功: 标准化完成！")
            except Exception as e:
                print(f"失败: 算法计算出错 -> {e}")


if __name__ == "__main__":
    diagnose()