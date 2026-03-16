import os
import h5py
import torch
import numpy as np
import glob
import argparse
from hilbertcurve.hilbertcurve import HilbertCurve
from tqdm import tqdm


def compute_and_save_hilbert_indices(h5_dir, save_dir, p):
    """
    读取 CLAM 生成的 .h5 坐标文件，计算 Hilbert 排序索引，并保存为 .pt 文件。
    """
    os.makedirs(save_dir, exist_ok=True)
    h5_files = glob.glob(os.path.join(h5_dir, '*.h5'))

    if len(h5_files) == 0:
        print(f"Error: 在 {h5_dir} 中没有找到 .h5 文件！请检查路径。")
        return

    # 初始化 2维的 Hilbert 曲线
    hilbert_curve = HilbertCurve(p, 2)
    grid_max = (1 << p) - 1  # 例如 p=10 时，网格最大值为 1023

    print(f"开始处理 {len(h5_files)} 个 WSI 坐标文件...")
    print(f"Hilbert 曲线阶数 (p): {p}, 离散网格分辨率: {grid_max + 1}x{grid_max + 1}")

    for h5_path in tqdm(h5_files):
        slide_id = os.path.splitext(os.path.basename(h5_path))[0]
        save_path = os.path.join(save_dir, f"{slide_id}_hilbert.pt")

        # 避免重复计算，支持断点续传
        if os.path.exists(save_path):
            continue

        try:
            with h5py.File(h5_path, 'r') as f:
                coords = f['coords'][:]  # shape: (num_patches, 2)
        except Exception as e:
            print(f"读取 {slide_id} 失败: {e}")
            continue

        num_patches = len(coords)
        if num_patches == 0:
            torch.save(torch.tensor([], dtype=torch.long), save_path)
            continue

        x_coords = coords[:, 0]
        y_coords = coords[:, 1]

        x_shifted = x_coords - np.min(x_coords)
        y_shifted = y_coords - np.min(y_coords)

        max_range = max(np.max(x_shifted), np.max(y_shifted))

        if max_range > 0:
            grid_x = (x_shifted / max_range * grid_max).astype(int)
            grid_y = (y_shifted / max_range * grid_max).astype(int)
        else:
            grid_x, grid_y = x_shifted.astype(int), y_shifted.astype(int)

        distances = []
        for x, y in zip(grid_x, grid_y):
            x_c = min(max(x, 0), grid_max)
            y_c = min(max(y, 0), grid_max)
            dist = hilbert_curve.distance_from_point([x_c,y_c])
            distances.append(dist)

        sorted_indices = np.argsort(distances)

        torch.save(torch.tensor(sorted_indices, dtype=torch.long), save_path)

    print(f"Hilbert 排序索引生成完毕，保存: {save_dir}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='生成 Hilbert 空间重排索引')

    # 核心输入输出参数
    parser.add_argument('--h5_dir', type=str, required=True,
                        help='CLAM 提取出的含有 coords 的 .h5 文件夹路径')
    parser.add_argument('--save_dir', type=str, required=True,
                        help='保存 Hilbert 索引的输出文件夹路径')

    # 可选的高级参数
    parser.add_argument('--p', type=int, default=10,
                        help='Hilbert 曲线的阶数。p=10 代表 1024x1024 的网格分辨率 (默认: 10)')

    args = parser.parse_args()

    compute_and_save_hilbert_indices(args.h5_dir, args.save_dir, args.p)