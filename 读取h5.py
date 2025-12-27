import h5py
import numpy as np


def read_all_datasets(file_path, max_preview=50):
    """
    读取HDF5文件中的所有数据集名称、大小及内容。

    参数：
        file_path (str): HDF5文件路径。
        max_preview (int): 每个数据集打印的最大数据数量，默认值为5。
    """
    # 打开HDF5文件
    with h5py.File(file_path, 'r') as file:
        print(f"Reading HDF5 file: {file_path}\n")

        # 遍历文件中所有的数据集

        def print_dataset_info(name, obj):
            if not isinstance(obj, h5py.Dataset):
                return

            print(f"Dataset Name: {name}")
            print(f"- Dimensions: {obj.shape}")
            print(f"- Data Type: {obj.dtype}")

            data = obj[...]
            # 预览：1 维数据直接切片，≥2 维数据保留行列结构
            if obj.ndim == 1:
                preview = data[:max_preview]
            else:
                preview = data[:max_preview, ...]  # 前 max_preview 行

            # 打印预览
            print(f"- Data Preview :\n{preview}")
            print()

        # 递归访问文件结构，打印所有数据集信息
        file.visititems(print_dataset_info)


# 替换为你的 HDF5 文件路径
h5_file_path = r" "
read_all_datasets(h5_file_path)

