import torch
import os

#检查UNI_V1生成的pt文件的维度和数据类型


# 替换为你 E 盘下保存 pt 文件的路径
pt_path = r''

# 获取文件夹下第一个文件
files = [f for f in os.listdir(pt_path) if f.endswith('.pt')]
if files:
    sample_file = files[0]
    features = torch.load(os.path.join(pt_path, sample_file))

    print(f"检查文件: {sample_file}")
    print(f"特征张量形状 (Shape): {features.shape}")
    print(f"数据类型 (Dtype): {features.dtype}")

    # 验证维度
    if features.shape[1] == 1024:
        print("✅ 维度校验通过：1024 ")
    else:
        print(f"❌ 维度异常：预期 1024，实际得到 {features.shape[1]}")
else:
    print("文件夹内没有找到 .pt 文件，请检查路径。")