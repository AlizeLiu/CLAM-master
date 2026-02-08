import time
import os
import argparse
from functools import partial

import torch
import torch.nn as nn
import timm
from torchvision import transforms
from timm.data import resolve_data_config, create_transform
from timm.layers import SwiGLUPacked
from torch.utils.data import DataLoader
from PIL import Image
import torchstain  # 必须安装: pip install torchstain
import cv2  # 必须安装: pip install opencv-python
import h5py
import openslide
from tqdm import tqdm
import numpy as np

from utils.file_utils import save_hdf5
from dataset_modules.dataset_h5 import Dataset_All_Bags, Whole_Slide_Bag_FP
from timm.models import create_model

# 注意：如果 get_encoder 报错，我们可以直接在这里定义 Virchow 加载逻辑

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')


class StainNormTransform(object):
    """
    Macenko 染色标准化 Transform (最终维度修复版)
    集成: 强制尺寸对齐 + 严格背景过滤 + HWC转CHW维度适配
    """

    def __init__(self, target_path, device='cuda'):
        print(f"Initializing Macenko Normalizer with reference: {target_path}")
        target = cv2.imread(target_path)
        if target is None:
            raise ValueError(f"Could not read reference image at {target_path}")
        target = cv2.cvtColor(target, cv2.COLOR_BGR2RGB)

        self.device = torch.device(device) if torch.cuda.is_available() else torch.device('cpu')

        # [关键] 参考图也必须转为 (C, H, W) 格式
        # target shape: (H, W, C) -> (C, H, W)
        target_tensor = torch.from_numpy(target).permute(2, 0, 1).to(self.device)

        self.normalizer = torchstain.normalizers.MacenkoNormalizer(backend='torch')
        self.normalizer.fit(target_tensor)

    def is_background(self, img_np):
        """基于严格阈值的背景检测"""
        img_gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
        mean_val = np.mean(img_gray)
        if mean_val > 210:  # 亮度阈值
            return True
        white_ratio = np.sum(img_gray > 210) / img_gray.size
        if white_ratio > 0.70:  # 白色占比阈值
            return True
        return False

    def __call__(self, img):
        try:
            # 1. 强制尺寸对齐
            if img.size != (224, 224):
                img = img.resize((224, 224), Image.BICUBIC)

            img_np = np.array(img)

            # 2. 强力背景过滤
            if self.is_background(img_np):
                return img

            # 3. Macenko 标准化 (维度修复核心)
            # numpy (H, W, C) -> tensor (H, W, C) -> permute (C, H, W)
            img_tensor = torch.from_numpy(img_np).permute(2, 0, 1).to(self.device)

            # normalize 返回的是 (norm, H, E) 或者 norm，视版本而定
            # 加上 stains=False 通常只返回归一化后的图
            norm, _, _ = self.normalizer.normalize(I=img_tensor, stains=False)

            # 4. 结果转回
            if isinstance(norm, torch.Tensor):
                # (C, H, W) -> permute (H, W, C) -> cpu -> numpy
                norm = norm.permute(1, 2, 0).cpu().numpy()

            norm = norm.astype(np.uint8)
            return Image.fromarray(norm)

        except Exception as e:
            # 兜底：返回 Resize 后的原图
            if img.size != (224, 224):
                img = img.resize((224, 224), Image.BICUBIC)
            return img


def compute_w_loader(output_path, loader, model, model_name, verbose=0):
    if verbose > 0:
        print(f'processing a total of {len(loader)} batches')

    mode = 'w'
    for count, data in enumerate(tqdm(loader)):
        with torch.inference_mode():
            batch = data['img'].to(device, non_blocking=True)
            coords = data['coord'].numpy().astype(np.int32)

            with torch.autocast(device_type='cuda', dtype=torch.float16):
                output = model(batch)

                if model_name == 'virchow':
                    # Virchow 特殊处理 (2560维)
                    class_token = output[:, 0]
                    patch_tokens = output[:, 1:]
                    features = torch.cat([class_token, patch_tokens.mean(1)], dim=-1)
                elif model_name in ['uni_v1', 'UNI', 'h-optimus-0']:
                    # UNI 和 H-optimus-0 官方输出直接就是 [B, 1024/1536]
                    features = output
                elif model_name == 'Prov-GigaPath':
                    features = output[:, 0] if len(output.shape) == 3 else output
                else:
                    features = output

            features = features.cpu().numpy().astype(np.float32)
            asset_dict = {'features': features, 'coords': coords}
            save_hdf5(output_path, asset_dict, attr_dict=None, mode=mode)
            mode = 'a'
    return output_path


def load_virchow():
    print("Loading Virchow using Official Configuration...")

    # 按照官方文档：必须指定 mlp_layer 和 act_layer 才能正确初始化架构并对齐权重
    model = timm.create_model(
        "hf-hub:paige-ai/Virchow",
        pretrained=True,
        mlp_layer=SwiGLUPacked,
        act_layer=torch.nn.SiLU
    )

    model = model.to(device)
    model.eval()

    # 按照官方文档：使用 resolve_data_config 自动获取模型预期的预处理配置
    config = resolve_data_config(model.pretrained_cfg, model=model)
    img_transforms = create_transform(**config)

    print("Virchow model and transforms initialized successfully.")
    return model, img_transforms


def load_prov_giga_path():
    print("Loading Prov-GigaPath using Official Configuration...")
    # Gigapath 的架构是巨大的，本质上是 ViT-Giant (patch 14)
    # 官方模型 ID  "hf_hub:prov-gigapath/prov-gigapath"
    model = timm.create_model(
        "hf_hub:prov-gigapath/prov-gigapath",
        pretrained=True
    )

    model = model.to(device)
    model.eval()

    # Gigapath 的标准预处理：224x224, ImageNet 归一化
    config = resolve_data_config(model.pretrained_cfg, model=model)
    img_transforms = create_transform(**config)

    print("Prov-GigaPath model and transforms initialized successfully.")
    return model, img_transforms


def load_uni():
    print("Loading MahmoodLab UNI using Official Configuration...")
    # 按照官方文档：必须传 init_values=1e-5 才能正确加载 LayerScale 参数
    # dynamic_img_size=True 允许处理略微偏离 224 的尺寸
    model = timm.create_model(
        "hf-hub:MahmoodLab/uni",
        pretrained=True,
        init_values=1e-5,
        dynamic_img_size=True
    )
    model = model.to(device)
    model.eval()

    # 使用官方推荐的 resolve_data_config 方式获取 transforms
    config = resolve_data_config(model.pretrained_cfg, model=model)
    img_transforms = create_transform(**config)

    print("UNI model and transforms initialized successfully.")
    # UNI 的特征维度是 1024
    return model, img_transforms


def load_h_optimus():
    print("Loading Bioptimus H-optimus-0 using Official Configuration...")
    # 按照官方文档：必须传 init_values=1e-5
    model = timm.create_model(
        "hf-hub:bioptimus/H-optimus-0",
        pretrained=True,
        init_values=1e-5,
        dynamic_img_size=False
    )
    model = model.to(device)
    model.eval()

    # 官方文档指定的特定归一化参数，这对于病理图像特征的准确性至关重要
    img_transforms = transforms.Compose([
        transforms.Resize(224, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=(0.707223, 0.578729, 0.703617),
            std=(0.211883, 0.230117, 0.177517)
        ),
    ])

    print("H-optimus-0 model and transforms initialized successfully.")
    return model, img_transforms


parser = argparse.ArgumentParser(description='Feature Extraction')
parser.add_argument('--data_h5_dir', type=str, default=None)
parser.add_argument('--data_slide_dir', type=str, default=None)
parser.add_argument('--slide_ext', type=str, default='.svs')
parser.add_argument('--csv_path', type=str, default=None)
parser.add_argument('--feat_dir', type=str, default=None)
# 增加 virchow 选项
parser.add_argument('--model_name', type=str, default='resnet50_trunc',
                    choices=['resnet50_trunc', 'uni_v1', 'conch_v1', 'virchow', 'Prov-GigaPath', 'h-optimus-0'])
parser.add_argument('--batch_size', type=int, default=128)  # 建议先从 128 开始试
parser.add_argument('--no_auto_skip', default=False, action='store_true')
parser.add_argument('--target_patch_size', type=int, default=224)

# --- 新增参数：染色标准化参考图 ---
parser.add_argument('--target_ref', type=str, default=None,
                    help='Path to the reference image for Macenko normalization. If None, normalization is skipped.')

args = parser.parse_args()

if __name__ == '__main__':
    print('initializing dataset')
    if args.csv_path is None:
        raise NotImplementedError

    # 1. 初始化数据集
    bags_dataset = Dataset_All_Bags(args.csv_path)

    # 2. 建立文件夹
    os.makedirs(args.feat_dir, exist_ok=True)
    os.makedirs(os.path.join(args.feat_dir, 'pt_files'), exist_ok=True)
    os.makedirs(os.path.join(args.feat_dir, 'h5_files'), exist_ok=True)

    # 3. 加载模型
    if args.model_name == 'virchow':
        model, img_transforms = load_virchow()
    elif args.model_name == 'Prov-GigaPath':
        model, img_transforms = load_prov_giga_path()
    elif args.model_name == 'h-optimus-0':
        model, img_transforms = load_h_optimus()
    elif args.model_name == 'uni_v1' or args.model_name == 'UNI':
        model, img_transforms = load_uni()
    else:
        from models import get_encoder

        model, img_transforms = get_encoder(args.model_name, target_img_size=args.target_patch_size)
        model = model.to(device)

    # --- 4. 插入染色标准化逻辑 ---
    if args.target_ref is not None:
        print(f"\n[INFO] Enabling Macenko Stain Normalization...")
        if not os.path.exists(args.target_ref):
            raise FileNotFoundError(f"Reference image not found at {args.target_ref}")

        # 实例化自定义的 Normalizer
        stain_norm = StainNormTransform(args.target_ref)

        # 将 Normalizer 插入到现有的 Transform 序列的最前端
        # 确保顺序：Raw Image (PIL) -> Stain Norm -> Resize/ToTensor/Normalize
        if isinstance(img_transforms, transforms.Compose):
            # 如果是 Compose 对象，拆开 list 插在最前面
            new_transforms_list = [stain_norm] + img_transforms.transforms
            img_transforms = transforms.Compose(new_transforms_list)
        else:
            # 如果是单个 Transform 或者 Sequential，直接打包
            img_transforms = transforms.Compose([stain_norm, img_transforms])

        print("[INFO] Transforms updated with Stain Normalization.")
    else:
        print("\n[INFO] Skipping Stain Normalization (no target_ref provided).")
    # ---------------------------

    model.eval()
    total = len(bags_dataset)
    # 染色标准化计算量大，建议 num_workers 保持 0 或较小值以防内存溢出，或者根据 CPU 核心数调整
    loader_kwargs = {'num_workers': 0, 'pin_memory': True} if device.type == "cuda" else {}

    # --- 计数器逻辑 ---
    processed_count = 0
    max_per_run = 100  # 设定本次运行处理的上限
    # -----------------

    print(f"开始特征提取任务，目标处理: {max_per_run} 张切片")

    for bag_candidate_idx in tqdm(range(total)):
        # 获取当前切片的 ID
        slide_id = bags_dataset[bag_candidate_idx].split(args.slide_ext)[0]

        # 检查是否已经存在（断点续传）
        dest_files = os.listdir(os.path.join(args.feat_dir, 'pt_files'))
        if not args.no_auto_skip and (slide_id + '.pt' in dest_files):
            # 如果已经跑过了，直接跳过，不计入 processed_count
            continue

        # 如果已经处理够了 80 张新的，就跳出循环退出程序
        if processed_count >= max_per_run:
            print(f"\n已完成本次设定的 {max_per_run} 张任务，正在安全关闭...")
            break

        # 开始处理当前的切片
        bag_name = slide_id + '.h5'
        h5_file_path = os.path.join(args.data_h5_dir, 'patches', bag_name)
        slide_file_path = os.path.join(args.data_slide_dir, slide_id + args.slide_ext)

        # 检查文件是否存在
        if not os.path.exists(h5_file_path):
            print(f"跳过 {slide_id}: 未找到 patch h5 文件")
            continue
        if not os.path.exists(slide_file_path):
            print(f"跳过 {slide_id}: 未找到 slide 文件")
            continue

        output_path = os.path.join(args.feat_dir, 'h5_files', bag_name)
        time_start = time.time()

        try:
            wsi = openslide.open_slide(slide_file_path)
            # 使用包含 StainNorm 的 img_transforms 初始化 Dataset
            dataset = Whole_Slide_Bag_FP(file_path=h5_file_path, wsi=wsi, img_transforms=img_transforms)
            loader = DataLoader(dataset=dataset, batch_size=args.batch_size, **loader_kwargs)

            # 提取特征
            output_file_path = compute_w_loader(output_path, loader=loader, model_name=args.model_name, model=model,
                                                verbose=1)

            # 将 h5 转换为 pt
            with h5py.File(output_file_path, "r") as file:
                features = file['features'][:]
            features = torch.from_numpy(features)
            torch.save(features, os.path.join(args.feat_dir, 'pt_files', slide_id + '.pt'))

            # 成功处理完一张，计数器加 1
            processed_count += 1
            time_elapsed = time.time() - time_start
            print(f'\nProgress: {processed_count}/{max_per_run} | Slide {slide_id} took {time_elapsed:.2f}s')

            # 每张跑完清空一下显存缓存
            torch.cuda.empty_cache()

        except Exception as e:
            print(f"处理切片 {slide_id} 时出错: {e}")
            continue

    print(f"本次任务结束。共新处理切片: {processed_count} 张。")