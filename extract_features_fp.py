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
import h5py
import openslide
from tqdm import tqdm
import numpy as np

from utils.file_utils import save_hdf5
from dataset_modules.dataset_h5 import Dataset_All_Bags, Whole_Slide_Bag_FP
from timm.models import create_model

# 注意：如果 get_encoder 报错，我们可以直接在这里定义 Virchow 加载逻辑

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')


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
                    choices=['resnet50_trunc', 'uni_v1', 'conch_v1', 'virchow','Prov-GigaPath','h-optimus-0'])
parser.add_argument('--batch_size', type=int, default=128)  # 建议先从 128 开始试
parser.add_argument('--no_auto_skip', default=False, action='store_true')
parser.add_argument('--target_patch_size', type=int, default=224)
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

    model.eval()
    total = len(bags_dataset)
    loader_kwargs = {'num_workers': 0, 'pin_memory': True} if device.type == "cuda" else {}

    # --- 计数器逻辑 ---
    processed_count = 0
    max_per_run = 80  # 设定本次运行处理的上限
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

        output_path = os.path.join(args.feat_dir, 'h5_files', bag_name)
        time_start = time.time()

        try:
            wsi = openslide.open_slide(slide_file_path)
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