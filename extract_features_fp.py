import time
import os
import argparse
from functools import partial

import torch
import torch.nn as nn
import timm
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
                elif model_name == 'uni_v1' or model_name == 'UNI':
                    # UNI 官方输出已经是 [B, 1024]，直接使用即可
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




parser = argparse.ArgumentParser(description='Feature Extraction')
parser.add_argument('--data_h5_dir', type=str, default=None)
parser.add_argument('--data_slide_dir', type=str, default=None)
parser.add_argument('--slide_ext', type=str, default='.svs')
parser.add_argument('--csv_path', type=str, default=None)
parser.add_argument('--feat_dir', type=str, default=None)
# 增加 virchow 选项
parser.add_argument('--model_name', type=str, default='resnet50_trunc',
                    choices=['resnet50_trunc', 'uni_v1', 'conch_v1', 'virchow','Prov-GigaPath'])
parser.add_argument('--batch_size', type=int, default=128)  # 建议先从 128 开始试
parser.add_argument('--no_auto_skip', default=False, action='store_true')
parser.add_argument('--target_patch_size', type=int, default=224)
args = parser.parse_args()

if __name__ == '__main__':
    print('initializing dataset')
    if args.csv_path is None:
        raise NotImplementedError

    bags_dataset = Dataset_All_Bags(args.csv_path)
    os.makedirs(args.feat_dir, exist_ok=True)
    os.makedirs(os.path.join(args.feat_dir, 'pt_files'), exist_ok=True)
    os.makedirs(os.path.join(args.feat_dir, 'h5_files'), exist_ok=True)
    dest_files = os.listdir(os.path.join(args.feat_dir, 'pt_files'))

    # 根据选择加载模型
    if args.model_name == 'virchow':
        model, img_transforms = load_virchow()
    elif args.model_name == 'Prov-GigaPath':
        model, img_transforms = load_prov_giga_path()
    elif args.model_name == 'uni_v1' or args.model_name == 'UNI':
        model, img_transforms = load_uni()
    else:
        from models import get_encoder

        model, img_transforms = get_encoder(args.model_name, target_img_size=args.target_patch_size)
        model = model.to(device)

    model.eval()
    total = len(bags_dataset)
    loader_kwargs = {'num_workers': 0, 'pin_memory': True} if device.type == "cuda" else {}

    for bag_candidate_idx in tqdm(range(total)):
        slide_id = bags_dataset[bag_candidate_idx].split(args.slide_ext)[0]
        bag_name = slide_id + '.h5'
        h5_file_path = os.path.join(args.data_h5_dir, 'patches', bag_name)
        slide_file_path = os.path.join(args.data_slide_dir, slide_id + args.slide_ext)

        if not args.no_auto_skip and slide_id + '.pt' in dest_files:
            continue

        output_path = os.path.join(args.feat_dir, 'h5_files', bag_name)
        time_start = time.time()
        wsi = openslide.open_slide(slide_file_path)
        dataset = Whole_Slide_Bag_FP(file_path=h5_file_path, wsi=wsi, img_transforms=img_transforms)

        loader = DataLoader(dataset=dataset, batch_size=args.batch_size, **loader_kwargs)
        output_file_path = compute_w_loader(output_path, loader=loader,model_name=args.model_name, model=model, verbose=1)

        # 将 h5 转换为 CLAM 需要的 pt 文件
        with h5py.File(output_file_path, "r") as file:
            features = file['features'][:]

        features = torch.from_numpy(features)
        torch.save(features, os.path.join(args.feat_dir, 'pt_files', slide_id + '.pt'))

        print('\nSlide {} processed. Feature shape: {}'.format(slide_id, features.shape))