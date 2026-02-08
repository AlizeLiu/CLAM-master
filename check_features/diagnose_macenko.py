import os
import numpy as np
import h5py
import openslide
import cv2
import torch
import torchstain
from PIL import Image

# --- 配置区域 ---
# 请确保这里的文件路径是你想要测试的那个文件
SLIDE_PATH = r"J:\Work\CLAM-master\toy_example\macenko_demo_1.svs"
H5_PATH = r"J:\Work\CLAM-master\toy_test\patches\macenko_demo_1.h5"
REF_PATH = r"J:\Work\CLAM-master\macenko_simple\reference.png"


# ----------------

def diagnose():
    print(f"--- === ---")

    # 1. 准备参考图
    if not os.path.exists(REF_PATH):
        print("❌ 错误：找不到参考图像！")
        return

    ref_img = cv2.imread(REF_PATH)
    ref_img = cv2.cvtColor(ref_img, cv2.COLOR_BGR2RGB)

    # 强制参考图也为 224x224
    if ref_img.shape[0] != 224 or ref_img.shape[1] != 224:
        ref_img = cv2.resize(ref_img, (224, 224))

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print(f"1. 设备: {device}")

    try:
        T = torch.from_numpy(ref_img).to(device)
        normalizer = torchstain.normalizers.MacenkoNormalizer(backend='torch')
        normalizer.fit(T)
        print("   ✅ Macenko 初始化成功")
    except Exception as e:
        print(f"   ❌ 初始化失败: {e}")
        return

    # 2. 读取切片
    try:
        wsi = openslide.open_slide(SLIDE_PATH)
    except Exception as e:
        print(f"❌ 无法打开切片: {e}")
        return

    with h5py.File(H5_PATH, 'r') as f:
        coords = f['coords'][:]
        patch_size_h5 = f['coords'].attrs.get('patch_size', 256)

    print(f"2. 开始检测 Patch (H5 Patch Size: {patch_size_h5})")

    # 随机抽 10 个
    indices = np.random.choice(len(coords), 10, replace=False)

    for i, idx in enumerate(indices):
        coord = coords[idx]
        print(f"\n--- Patch {idx} (坐标: {coord}) ---")

        try:
            # [A] 读取原始 Patch
            img_pil = wsi.read_region(tuple(coord), 0, (int(patch_size_h5), int(patch_size_h5))).convert('RGB')
            img_np = np.array(img_pil)

            # [B] 核心修复：强制 Resize 到 224x224 (和你主程序保持一致)
            # 解决 'size 172032' 报错
            if img_np.shape[0] != 224 or img_np.shape[1] != 224:
                img_np = cv2.resize(img_np, (224, 224))

            # [C] 严格背景过滤 (Mean > 210 即丢弃)
            # 解决 'kthvalue' 报错
            img_gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
            mean_val = np.mean(img_gray)

            if mean_val > 210:
                print(f"   ⚠️ [正确跳过] 背景太白 (Mean={mean_val:.1f})")
                continue

            # [D] 标准化
            img_tensor = torch.from_numpy(img_np).to(device)
            normalizer.normalize(I=img_tensor, stains=False)
            print(f"   ✅ [成功] 标准化完成 (Mean={mean_val:.1f})")

        except Exception as e:
            print(f"   ❌ 失败: {e}")


if __name__ == "__main__":
    diagnose()