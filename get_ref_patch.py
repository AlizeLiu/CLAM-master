import os
import h5py
import openslide
import numpy as np
from PIL import Image
import argparse


def get_reference_patch(wsi_path, h5_path, save_name='reference.png'):
    """
    从 SVS 中随机抽取一张确认为组织的 Patch 作为参考图
    """
    print(f"正在处理 SVS: {wsi_path}")

    # 1. 打开 SVS
    wsi = openslide.open_slide(wsi_path)

    # 2. 打开对应的 H5 坐标文件
    if not os.path.exists(h5_path):
        print("错误：找不到对应的 .h5 文件，请先运行 create_patches.py")
        return

    with h5py.File(h5_path, 'r') as f:
        coords = f['coords'][:]
        patch_level = f['coords'].attrs['patch_level']
        patch_size = f['coords'].attrs['patch_size']

    print(f"找到 {len(coords)} 个可用坐标。")

    # 3. 随机抽取逻辑
    # 我们多抽几张（比如 5 张），你自己从中选一张最顺眼的
    indices = np.random.choice(len(coords), 5, replace=False)

    # save directory is the macenko_simple folder
    save_dir = 'macenko_simple'
    os.makedirs(save_dir, exist_ok=True)

    for i, idx in enumerate(indices):
        coord = coords[idx]
        x, y = coord

        # 从 SVS 读取高分辨率原图 (Level 0)
        # 注意：这里假设你的 h5 是基于 level 0 坐标的，CLAM 默认如此
        patch = wsi.read_region((x, y), patch_level, (patch_size, patch_size)).convert('RGB')

        # 保存
        out_name = os.path.join(save_dir, f"ref_candidate_{i}.png")
        patch.save(out_name)
        print(f"已保存候选参考图: {out_name}")

    print("\n完成，选择一张染色细胞最清晰的，")
    print("用于 extract_features.py。")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='提取染色参考图')
    parser.add_argument('--wsi_path', type=str, required=True, help='TCGA SVS 文件的路径')
    parser.add_argument('--h5_path', type=str, required=True,
                        help='该 SVS 对应的 .h5 坐标文件路径 (在 patches 文件夹里)')
    args = parser.parse_args()

    get_reference_patch(args.wsi_path, args.h5_path)