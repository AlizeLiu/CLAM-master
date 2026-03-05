import torch
import os
from tqdm import tqdm
import argparse

# ================= 配置区域 =================
# 修改此处，或者通过命令行参数传入 --dir
DEFAULT_PT_PATH = r'J:\Work\CLAM-master\toy_test\feature_univ1\pt_files'
EXPECTED_DIM = 1024  # 固定UNI_v1 的特征维度


# ===========================================

def check_all_pt_files(directory, expected_dim):
    if not os.path.isdir(directory):
        print(f"❌ 错误：路径不存在 -> {directory}")
        return

    files = [f for f in os.listdir(directory) if f.endswith('.pt')]
    total_files = len(files)

    if total_files == 0:
        print("⚠️  警告：该目录下没有找到 .pt 文件。")
        return

    print(f"🔎 开始校验 {total_files} 个文件...")
    print(f"📂 目标目录: {directory}")
    print(f"📏 预期维度: {expected_dim}")
    print("-" * 50)

    passed_count = 0
    failed_files = []
    corrupted_files = []

    # 使用 tqdm 显示进度条
    for filename in tqdm(files, desc="Checking", unit="file"):
        file_path = os.path.join(directory, filename)

        try:
            # 尝试加载文件
            features = torch.load(file_path, map_location='cpu')  # 映射到CPU以节省显存

            # 1. 检查是否为空或格式错误
            if not isinstance(features, torch.Tensor):
                print(f"\n❌ 格式错误: {filename} 不是 Tensor 类型")
                corrupted_files.append(filename)
                continue

            # 2. 检查维度 (Shape: [N, 1024])
            if len(features.shape) != 2:
                print(f"\n❌ 形状错误: {filename} 形状为 {features.shape}, 期望是 [N, {expected_dim}]")
                failed_files.append(filename)
                continue

            if features.shape[1] != expected_dim:
                print(f"\n❌ 维度不匹配: {filename} 第二维度是 {features.shape[1]}, 期望 {expected_dim}")
                failed_files.append(filename)
                continue

            # 3. (可选) 检查数据类型，防止 float16/32 混用导致后续报错
            # if features.dtype != torch.float32:
            #     print(f"⚠️ 类型警告: {filename} 是 {features.dtype}")

            passed_count += 1

        except Exception as e:
            print(f"\n☠️  文件损坏或无法读取: {filename} | 错误信息: {e}")
            corrupted_files.append(filename)

    # ================= 总结报告 =================
    print("\n" + "=" * 50)
    print(f"✅ 校验通过: {passed_count}/{total_files}")

    if len(failed_files) > 0:
        print(f"❌ 维度/形状异常: {len(failed_files)} 个")
        print(f"   列表: {failed_files}")

    if len(corrupted_files) > 0:
        print(f"☠️  文件损坏: {len(corrupted_files)} 个 (建议删除后重新提取)")
        print(f"   列表: {corrupted_files}")

    if passed_count == total_files:
        print("\n🎉 完美！所有文件的维度和完整性均验证通过。")
    else:
        print("\n⚠️  请检查上述报错文件。")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Batch Check PT Files')
    parser.add_argument('--dir', type=str, default=DEFAULT_PT_PATH, help='Path to the pt_files folder')
    args = parser.parse_args()

    check_all_pt_files(args.dir, EXPECTED_DIM)