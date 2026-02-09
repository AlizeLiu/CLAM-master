import os
import numpy as np
import h5py
import openslide
import cv2
import torch
import torchstain
import matplotlib.pyplot as plt
from PIL import Image
from tqdm import tqdm

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


class MacenkoNormalizer:
    def __init__(self, target_path):
        print(f"正在加载参考图像: {target_path}")
        target = cv2.imread(target_path)
        if target is None:
            raise ValueError(f"无法读取参考图像: {target_path}")
        target = cv2.cvtColor(target, cv2.COLOR_BGR2RGB)

        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        print(f"🔥 当前运行设备: {self.device}")

        # 强制 Resize 到 224
        if target.shape[0] != 224 or target.shape[1] != 224:
            target = cv2.resize(target, (224, 224))

        # --- 核心修正 1: 转为 (C, H, W) 格式 ---
        # torchstain 的 torch 后端要求输入形状为 (C, H, W)
        self.target_tensor = torch.from_numpy(target).permute(2, 0, 1).to(self.device)

        self.normalizer = torchstain.normalizers.MacenkoNormalizer(backend='torch')
        self.normalizer.fit(self.target_tensor)

    def process(self, img_pil):
        try:
            # 统一 Resize
            img_pil = img_pil.resize((224, 224), Image.BICUBIC)
            img_np = np.array(img_pil)

            # 背景过滤
            img_gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
            mean_val = np.mean(img_gray)
            if mean_val > 210:
                return None, False

            # --- 核心修正 2: 输入转为 (C, H, W) ---
            img_tensor = torch.from_numpy(img_np).permute(2, 0, 1).to(self.device)

            # 执行标准化
            # result 通常返回 (norm_tensor, H, E)
            result = self.normalizer.normalize(I=img_tensor, stains=False)

            if isinstance(result, tuple):
                norm_tensor = result[0]
            else:
                norm_tensor = result

            # --- 核心修正 3: 输出处理 ---
            # torchstain normalize 后在 torch 后端下可能返回 (C, H, W)
            if isinstance(norm_tensor, torch.Tensor):
                # 如果是 (C, H, W)，需要 permute 回 (H, W, C) 以供 PIL 使用
                if norm_tensor.ndimension() == 3 and norm_tensor.shape[0] == 3:
                    norm_tensor = norm_tensor.permute(1, 2, 0)
                norm_np = norm_tensor.cpu().numpy()
            else:
                norm_np = norm_tensor

            # 确保数值范围并转为 uint8
            norm_np = np.clip(norm_np, 0, 255).astype(np.uint8)
            return Image.fromarray(norm_np), True

        except Exception as e:
            # print(f"  [错误] 算法计算失败: {e}")
            return None, False


def plot_histogram(ax, img_np, title):
    colors = ('r', 'g', 'b')
    for i, color in enumerate(colors):
        hist = cv2.calcHist([img_np], [i], None, [256], [0, 256])
        ax.plot(hist, color=color, alpha=0.7, linewidth=1)
        ax.set_xlim([0, 256])
    ax.set_title(title, fontsize=9)
    ax.grid(True, alpha=0.2)
    ax.set_yticklabels([])


def visualize_patches(slide_path, h5_path, target_ref_path, output_dir, num_samples=5):
    slide_name = os.path.basename(slide_path)
    os.makedirs(output_dir, exist_ok=True)

    try:
        wsi = openslide.open_slide(slide_path)
    except Exception as e:
        print(f"无法打开切片: {e}")
        return

    normalizer = MacenkoNormalizer(target_ref_path)
    ref_img = Image.open(target_ref_path).resize((224, 224))

    with h5py.File(h5_path, 'r') as f:
        coords = f['coords'][:]
        patch_level = f['coords'].attrs.get('patch_level', 0)
        h5_patch_size = f['coords'].attrs.get('patch_size', 256)

    indices = np.random.choice(len(coords), num_samples, replace=False)
    valid_samples = []

    for idx in indices:
        coord = coords[idx]
        try:
            img_pil = wsi.read_region(tuple(coord), patch_level, (int(h5_patch_size), int(h5_patch_size))).convert(
                'RGB')
            # 预处理 resize 到 224
            img_pil = img_pil.resize((224, 224))
            norm_pil, success = normalizer.process(img_pil)

            # 如果成功则展示标准化图，否则展示原图
            valid_samples.append({
                'orig': img_pil,
                'norm': norm_pil if success else img_pil,
                'status': "成功" if success else "跳过"
            })
        except Exception as e:
            print(f"处理 patch {idx} 失败: {e}")
            continue

    if not valid_samples:
        print("\n❌ 错误：未处理任何可用的 Patch。")
        return

    # --- 绘图 ---
    fig, axes = plt.subplots(len(valid_samples), 5, figsize=(18, 4 * len(valid_samples)))
    if len(valid_samples) == 1: axes = [axes]

    for i, item in enumerate(valid_samples):
        orig_np = np.array(item['orig'])
        norm_np = np.array(item['norm'])

        axes[i][0].imshow(item['orig'])
        axes[i][0].set_title(f"原始图像 (Patch {i})", fontsize=10)
        axes[i][0].axis('off')

        axes[i][1].imshow(ref_img)
        axes[i][1].set_title("参考标准", fontsize=10)
        axes[i][1].axis('off')

        axes[i][2].imshow(item['norm'])
        axes[i][2].set_title(f"Macenko 结果 ({item['status']})", fontsize=10, color='blue', fontweight='bold')
        axes[i][2].axis('off')

        plot_histogram(axes[i][3], orig_np, "原始 RGB 分布")
        plot_histogram(axes[i][4], norm_np, "标准化 RGB 分布")

    plt.tight_layout()
    save_path = os.path.join(output_dir, f"check_{slide_name}.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\n🎉 校验图已生成: {save_path}")
    plt.close()


if __name__ == "__main__":
    # 路径配置（请根据实际环境检查）
    SLIDE_PATH = r"J:\Work\CLAM-master\toy_example\macenko_demo_1.svs"
    H5_PATH = r"J:\Work\CLAM-master\toy_test\patches\macenko_demo_1.h5"
    REF_PATH = r"J:\Work\CLAM-master\macenko_simple\reference.png"
    OUTPUT_DIR = r"J:\Work\CLAM-master\normalization_check"

    visualize_patches(SLIDE_PATH, H5_PATH, REF_PATH, OUTPUT_DIR)