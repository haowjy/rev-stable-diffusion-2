# Copyright (c) Facebook, Inc. and its affiliates.
# Modified by Jimmy Yao from https://github.com/microsoft/GenerativeImage2Text/blob/main/generativeimage2text/train.py

from pynvml import *

def get_gpu_utilization():
    nvmlInit()
    handle = nvmlDeviceGetHandleByIndex(0)
    info = nvmlDeviceGetMemoryInfo(handle)
    
    return info

def print_gpu_utilization():
    info = get_gpu_utilization()
    print(f"GPU memory occupied: {info.used//1024**2} MB.")
    
print_gpu_utilization()


from generativeimage2text.train import (get_image_transform, get_transform_image_norm, 
                                        get_inception_train_transform, 
                                        get_data, collate_fn, recursive_to_device)
from generativeimage2text.common import Config
from generativeimage2text.torch_common import load_state_dict

from transformers import BertTokenizer

import torch
import logging

from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
from torch.cuda.amp import autocast
    
from samgit.model import sam_image_crops, get_samgit_model

import cv2
import torchvision.transforms as transforms
from PIL import Image


def get_small_transform_vit_default(cfg, crop=24):
    default_normalize = transforms.Normalize(
            mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    normalize = get_transform_image_norm(cfg, default_normalize) # cfg is clip
    transform = get_inception_train_transform(
        bgr2rgb=True,
        crop_size=crop,
        normalize=normalize,
        small_scale=cfg.input_small_scale,
        no_color_jitter=cfg.no_color_jitter,
        no_flip=cfg.no_flip,
        no_aspect_dist=cfg.no_aspect_dist,
        resize_crop=cfg.resize_crop,
        max_size=cfg.train_max_size,
        interpolation=cfg.interpolation or Image.BILINEAR,
    )
    return transform

def prepare_model(pretrained_path=None, sam_model_type="vit_h", sam_checkpoint = "../models/sam_vit_h_4b8939.pth", half=True):
    cfg = {
        'crop_region_extend_in_datatransform': 4,
        'data_normalize': 'clip',
        'train_crop_size': 224,
        'input_small_scale': 0.8,
        'no_color_jitter': True,
        'no_flip': True,
        'no_aspect_dist': True,
        'interpolation': 'bicubic',
        'min_size_range32': [160, 224], # in pretraining, it is multi-scale from 160 to 224; while for fine-tuning, it is single scale
        # 'min_size_range32': [224], # in pretraining, it is multi-scale from 160 to 224; while for fine-tuning, it is single scale
        'patch_size': 16,
        'train_transform': 'vitp',
    }
    cfg = Config(cfg, {})
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
    
    param = {
        "top_n_bbox": 4,
    }
    model = get_samgit_model(tokenizer, param)
    
    # from pre-trained model
    if pretrained_path is not None:
        checkpoint = torch.load(pretrained_path)['model']
        load_state_dict(model, checkpoint)
        
    # if half:
    #     model.half()
    model.train()
    model.cuda()

    # preprocess object crops with sam
    sam = sam_model_registry[sam_model_type](checkpoint=sam_checkpoint).half()
    # if half:
    #     sam.half()
    sam.eval()
    sam.requires_grad_(False)
    sam.to(device='cuda')
    mask_generator = SamAutomaticMaskGenerator(
        model=sam,
        points_per_side=4,
        pred_iou_thresh=0.95,
        stability_score_thresh=0.95,
    )
    
    return model, mask_generator, tokenizer, cfg, param

def forward(model, mask_generator, tokenizer, cfg, param, image_files, captions, prefixs=None, half=True):
    if prefixs is None:
        prefixs = [''] * len(captions)

    all_data = []
    
    image_transform = get_image_transform(cfg)
    # resize to 24x24
    small_transform = get_small_transform_vit_default(cfg, crop=24)
    
    for image_file, prefix, target in zip(image_files, prefixs, captions):
        data = get_data(image_file, prefix, target,
                        tokenizer, image_transform)
        all_data.append(data)
    data = collate_fn(all_data)
    data = recursive_to_device(data, 'cuda')
    
    logging.info(f"GPU memory occupied: {get_gpu_utilization().used//1024**2} MB.")
    
    with autocast(dtype=torch.float16 if half else torch.float32):
        batch_img_crops = []
        for img_name in image_files:
            image = cv2.imread(img_name)
            crops = sam_image_crops(image, mask_generator, param.get('top_n_bbox', 4))
            new_crops = []
            for crop in crops:
                if 0 in crop.shape:
                    continue
                crop = Image.fromarray(crop)
                new_crops.append(small_transform(crop))

            if len(new_crops) == 0:
                batch_img_crops.append(torch.zeros((0, 3, 24, 24), dtype=torch.float16))
            else:
                crops = torch.stack(new_crops)
                crops.half()
                batch_img_crops.append(crops)
            
        loss_dict = model(data, *batch_img_crops) # hard-coded to cuda
    return loss_dict

import os
import pandas as pd
from tqdm import tqdm

from torch.utils.data import DataLoader, Dataset, RandomSampler

class DiffusionDBDataset(Dataset):
    def __init__(self, df, db_base_path):
        self.df = df
        self.db_base_path = db_base_path
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image_file = os.path.join(self.db_base_path, row["new_image_name"])
        prompt = row["prompt"]
        return image_file, prompt
    
import logging
logging.basicConfig(filename='train.info.log', encoding='utf-8', level=logging.INFO)

import numpy as np
import random

# set seeds
np.random.seed(42)
torch.manual_seed(42)
random.seed(42)

BATCH_SIZE = 128
WORKERS = 2

LR = 5e-5

SAVE_STEPS = 200

PRETRAINED_PATH = "output/GIT_BASE/snapshot/model.pt"
MODEL_NAME = "SAMGIT"
    
def main():
    # get data
    DIFFUSIONDB_IMGS_PATH = "/mnt/d/AiStuff/data/diffusionDB2M-f2/images"
    df = pd.read_csv(os.path.join(DIFFUSIONDB_IMGS_PATH, "diffusiondb-filtered.csv"))
    # first 100 for eval
    df_eval = df[:100]
    # rest for training
    df_train = df[100:]

    # 1 epoch
    model, mask_generator, tokenizer, cfg, param = prepare_model(pretrained_path=PRETRAINED_PATH, half=True)
    model.train()

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
    scaler = torch.cuda.amp.GradScaler(enabled=True)
    
    train_dataset = DiffusionDBDataset(df_train, DIFFUSIONDB_IMGS_PATH)
    train_random_sampler = RandomSampler(train_dataset)
    train_loader = DataLoader(train_dataset, sampler=train_random_sampler, batch_size=BATCH_SIZE, num_workers=WORKERS)

    eval_dataset = DiffusionDBDataset(df_eval, DIFFUSIONDB_IMGS_PATH)
    eval_loader = DataLoader(eval_dataset, batch_size=BATCH_SIZE, num_workers=WORKERS)

    train_losses = []

    pbar = tqdm(train_loader)
    for batch in pbar:
        image_files, captions = batch
        
        logging.info(f"memory_allocated: {torch.cuda.memory_allocated()/(1024**2)} MB, max_memory_allocated: {torch.cuda.max_memory_allocated()/(1024**2)} MB")
        loss_dict = forward(model, mask_generator, tokenizer, cfg, param, image_files, captions, half=True)
        loss = sum(loss_dict.values())
        
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        optimizer.zero_grad()
        
        train_losses.append(loss.detach().cpu().numpy())
        logging.info(f"[step {pbar.n + 1}] loss: {loss.detach().cpu().numpy():.4f}")
        pbar.set_description(f"[step {pbar.n + 1}] loss: {loss.detach().cpu().numpy():.4f}")
        
        if pbar.n + 1 % SAVE_STEPS == 0:
            torch.save({
                'epoch': 1,
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'losses': [train_losses],
                }, f"output/{MODEL_NAME}/epoch{1}-step{pbar.n+1}/model.pt")
        
    print("EVAL")
    model.eval()
    eval_losses = []

    pbar = tqdm(eval_loader)
    for batch in pbar:
        image_files, captions = batch
        
        loss = forward(model, mask_generator, tokenizer, cfg, param, image_files, captions)
        eval_losses.append(loss.detach().cpu().numpy())
    
    # save model dict and loss

    torch.save({
                'epoch': 1,
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'losses': [train_losses],
                }, f"output/{MODEL_NAME}/epoch{1}/model.pt")
    

if __name__ == "__main__":
    main()