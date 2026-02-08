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

        # 强制使用 GPU
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        print(f"🔥 当前运行设备: {self.device}")

        # 强制将参考图也 Resize 到 224 (虽然 fit 不需要，但保持一致性好)
        if target.shape[0] != 224 or target.shape[1] != 224:
            target = cv2.resize(target, (224, 224))

        self.target_tensor = torch.from_numpy(target).to(self.device)
        self.normalizer = torchstain.normalizers.MacenkoNormalizer(backend='torch')
        self.normalizer.fit(self.target_tensor)

    def process(self, img_pil):
        try:
            # --- 核心修复 1: 强制 Resize ---
            # 无论原始是 256x256 还是边缘图像，一律变为 224x224
            img_pil = img_pil.resize((224, 224), Image.BICUBIC)
            img_np = np.array(img_pil)

            # --- 核心修复 2: 严厉的背景过滤 ---
            img_gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
            mean_val = np.mean(img_gray)

            # 之前 230 太松了，导致 Mean=215 的 Patch 报错
            # 现在：只要平均亮度 > 210 (偏白)，直接跳过
            if mean_val > 210:
                # print(f"  [过滤] 背景太白 (Mean: {mean_val:.1f})")
                return None, False

            # 转移到 GPU 计算
            img_tensor = torch.from_numpy(img_np).to(self.device)
            norm, _, _ = self.normalizer.normalize(I=img_tensor, stains=False)

            if isinstance(norm, torch.Tensor):
                norm = norm.cpu().numpy()

            return Image.fromarray(norm.astype(np.uint8)), True

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

    if len(coords) < num_samples:
        print(f"可用 patch 数量不足 {num_samples}，仅随机抽取 {len(coords)} 个。")
        num_samples = len(coords)

    print(f"切片 {slide_name} 共有 {len(coords)} 个 patch，随机抽取 {num_samples} 个进行展示...")

    indices = np.random.choice(len(coords), num_samples, replace=False)
    valid_samples = []

    for idx in indices:
        coord = coords[idx]
        try:
            img_pil = wsi.read_region(tuple(coord), patch_level, (int(h5_patch_size), int(h5_patch_size))).convert('RGB')
            img_pil = img_pil.resize((224, 224))
            norm_pil, success = normalizer.process(img_pil)
            # 始终保留这 5 个随机样本，标准化失败则用原图兜底
            valid_samples.append({'orig': img_pil, 'norm': norm_pil if success else img_pil})
        except Exception as e:
            print(f"读取或处理 patch 失败: {e}")
            valid_samples.append({'orig': img_pil if 'img_pil' in locals() else None, 'norm': img_pil if 'img_pil' in locals() else None})
            continue

    if len(valid_samples) == 0:
        print("\n❌ 错误：未找到任何可用的 Patch。")
        return

    print(f"\n✅ 随机抽取到 {len(valid_samples)} 个样本，正在绘图...")

    # --- 绘图 ---
    fig, axes = plt.subplots(len(valid_samples), 5, figsize=(16, 3.5 * len(valid_samples)))
    plt.subplots_adjust(hspace=0.4, wspace=0.2)

    if len(valid_samples) == 1:
        axes = [axes]

    for i, item in enumerate(valid_samples):
        orig = item['orig']
        norm = item['norm']

        ax_row = axes[i] if len(valid_samples) > 1 else axes

        ax_row[0].imshow(orig)
        ax_row[0].set_title("原始图像", fontsize=10)
        ax_row[0].axis('off')

        ax_row[1].imshow(ref_img)
        ax_row[1].set_title("参考标准", fontsize=10)
        ax_row[1].axis('off')

        ax_row[2].imshow(norm)
        ax_row[2].set_title("Macenko 标准化", fontsize=10, color='blue', fontweight='bold')
        ax_row[2].axis('off')

        plot_histogram(ax_row[3], np.array(orig), "原始分布")
        plot_histogram(ax_row[4], np.array(norm), "标准化分布")

    save_path = os.path.join(output_dir, f"check_{slide_name}.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\n🎉 校验图已生成: {save_path}")
    plt.close()


if __name__ == "__main__":
    # 路径配置
    SLIDE_PATH = r"J:\Work\CLAM-master\toy_example\macenko_demo_1.svs"
    H5_PATH = r"J:\Work\CLAM-master\toy_test\patches\macenko_demo_1.h5"
    REF_PATH = r"J:\Work\CLAM-master\macenko_simple\reference.png"
    OUTPUT_DIR = r"J:\Work\CLAM-master\normalization_check"

    visualize_patches(SLIDE_PATH, H5_PATH, REF_PATH, OUTPUT_DIR)