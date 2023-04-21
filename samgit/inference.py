# Copyright (c) Facebook, Inc. and its affiliates.
# Modified by Jimmy Yao from https://github.com/microsoft/GenerativeImage2Text/blob/main/generativeimage2text/inference.py

from generativeimage2text.process_image import load_image_by_pil
from generativeimage2text.torch_common import load_state_dict

from transformers import BertTokenizer

import torch

from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
from torch.cuda.amp import autocast
    
from samgit.model import sam_image_crops, get_samgit_model

import cv2
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize
from PIL import Image

def get_image_transform(crop_size=224):
    trans = [
            Resize(crop_size, interpolation=Image.BICUBIC),
            CenterCrop(crop_size),
            lambda image: image.convert("RGB"),
        ]
    trans.extend([
        ToTensor(),
        Normalize(
            (0.48145466, 0.4578275, 0.40821073),
            (0.26862954, 0.26130258, 0.27577711), # clip defaults
        ),
    ])
    transforms = Compose(trans)
    return transforms

def prepare_model(pretrained_path, tokenizer_pretrained="bert-base-uncased", sam_model_type="vit_h", sam_checkpoint = "../models/sam_vit_h_4b8939.pth", half=True):
    tokenizer = BertTokenizer.from_pretrained(tokenizer_pretrained, do_lower_case=True)
    
    param = {
        "top_n_bbox": 4,
    }
    model = get_samgit_model(tokenizer, param)
    checkpoint = torch.load(pretrained_path)['model']
    load_state_dict(model, checkpoint)
    model.cuda()
    # if half:
    #     model.half()
    model.train()
    model.cuda()
    model.eval()
    
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
    
    return model, mask_generator, tokenizer, param

from generativeimage2text.train import (collate_fn, recursive_to_device)

def forward(model, mask_generator, tokenizer, param, image_files, prefix='', half=True, main_crop_size=224):
    
    if isinstance(image_files, str):
        image_files = [image_files]
    
    image_transform = get_image_transform(crop_size=main_crop_size)
    
    all_data = []
    for image_file in image_files:
        if prefix != '':
            max_text_len = 40
            prefix_encoding = tokenizer(prefix,
                                        padding='do_not_pad',
                                        truncation=True,
                                        add_special_tokens=False,
                                        max_length=max_text_len)
            payload = prefix_encoding['input_ids']
            if len(payload) > max_text_len - 2:
                payload = payload[-(max_text_len - 2):]
            input_ids = [tokenizer.cls_token_id] + payload
            data = {
                'image': image_transform(load_image_by_pil(image_file)),
                'prefix': torch.tensor(input_ids),
            }
        else:
            data = {
                'image': image_transform(load_image_by_pil(image_file)),
            }
        all_data.append(data)
    data = collate_fn(all_data)
    data = recursive_to_device(data, 'cuda')
    
    small_transform = get_image_transform(crop_size=24)
    
    with torch.no_grad():
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

            result = model(data, *batch_img_crops) # hard-coded to cuda
            
    return result

def generate(result, tokenizer):
    captions = tokenizer.batch_decode(result['predictions'], skip_special_tokens=True)
    return captions

import os
import pandas as pd

def main():
    model, mask_generator, tokenizer, param = prepare_model('../output/SAMGIT/epoch1/model.pt')
    
    PART1_IMGS_PATH = "../datasets/sampleeval"
    df = pd.read_csv(os.path.join("prompts.csv"))
    img_paths = [os.path.join(PART1_IMGS_PATH, df.iloc[idx]['imgId']+".png") for idx in range(len(df))]
    
    result = forward(model, mask_generator, tokenizer, param, image_files=img_paths)
    print(result['predictions'][0].tolist())

# TODO: https://github.com/salaniz/pycocoevalcap - submit to coco metrics
