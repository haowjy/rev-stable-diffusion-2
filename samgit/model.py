
import torch
import torch.nn as nn
from generativeimage2text.layers.decoder import (create_projecton_layer)

from generativeimage2text.model import get_image_encoder
from generativeimage2text.layers.decoder import (TransformerDecoderTextualHead,
                             GeneratorWithBeamSearch)

from samgit.decoder import SAMGitCaptioningModel

def sam_image_crops(image, mask_generator, top_n_bbox):
    masks = mask_generator.generate(image)
    sorted_anns = sorted(masks, key=(lambda x: x['predicted_iou']), reverse=True)[:top_n_bbox]
    
    crops = []
    for ann in sorted_anns:
        x0, y0, x1, y1 = ann['bbox']
        if x0 > x1: 
            x0, x1 = x1, x0
        if y0 > y1: 
            y0, y1 = y1, y0

        crops.append(image[y0:y1, x0:x1, :])
    return crops


class ImgObjEncoder(nn.Module):
    def __init__(self, image_encoder, top_n_bbox=4, visual_feature_size=768, textual_feature_size=768, visual_projection_type='linearLn'):
        super().__init__()
        self.image_encoder = image_encoder
        self.top_n_bbox = top_n_bbox
        self.object_projection = create_projecton_layer(
            visual_projection_type='linearLn',
            visual_feature_size=visual_feature_size,
            textual_feature_size=textual_feature_size,
        )
        self.visual_feature_size = visual_feature_size
        self.textual_feature_size = textual_feature_size
    
    def forward(self, batch, batch_image_crops: list[torch.Tensor]):
        # batch_image_crops = [batch, top_n_bbox, 3, 24, 24]
        # batch_size = len(batch)
        image_features = self.image_encoder(batch)
        # print("image_features", image_features.shape)
        batch_image_crop_features = []
        for b in range(len(batch_image_crops)): # batch
            # print(self.image_encoder)
            # print(batch_image_crops[b].device)
            image_crop_features = self.image_encoder(batch_image_crops[b].to(device='cuda')) # idk why but it cuda has to be here
            # print("image_crop_features", image_crop_features.shape)
            img_feature_size = image_crop_features.shape[1]
            
            image_crop_features = image_crop_features.flatten(0, 1) # [n_bbox*dim, 768]
            # print("image_crop_features", image_crop_features.shape)
            
            # add padding
            top_n = len(batch_image_crops[b])
            img_feature_size * self.top_n_bbox
            # print("image_crop_features", image_crop_features.shape)
            if top_n < self.top_n_bbox:
                padding = torch.zeros((self.top_n_bbox - top_n)*img_feature_size, self.visual_feature_size, device=batch_image_crop_features[0].device)
                image_crop_features = torch.cat([padding, image_crop_features], dim=0)
            # print("image_crop_features", image_crop_features.shape)
            batch_image_crop_features.append(image_crop_features)
            
        batch_image_crop_features = torch.stack(batch_image_crop_features, dim=0)
        
        img_obj_features = torch.cat([image_features, batch_image_crop_features], dim=1)
        
        return img_obj_features


def get_samgit_model(tokenizer, param):
    image_encoder = get_image_encoder(
        param.get('image_encoder_type', 'CLIPViT_B_16'),
        input_resolution=param.get('test_crop_size', 224),
    )
    
    img_obj_encoder = ImgObjEncoder(
        image_encoder,
        param.get('top_n_bbox', 4),
        visual_feature_size=param.get('visual_feature_size', 768),
        textual_feature_size=768,
        visual_projection_type=param.get('visual_projection_type', 'linearLn'),
    )
    text_decoder = TransformerDecoderTextualHead(
        visual_feature_size=param.get('visual_feature_size', 768),
        vocab_size=30522,
        hidden_size=768,
        num_layers=6,
        attention_heads=12,
        feedforward_size=768* 4,
        max_caption_length=1024,
        mask_future_positions=True,
        padding_idx=0,
        decoder_type='bert_en',
        visual_projection_type='linearLn',
    )
    decoder = GeneratorWithBeamSearch(
        eos_index=tokenizer.sep_token_id,
        #max_steps=40,
        max_steps=1024,
        beam_size=4,
        length_penalty=0.6,
    )
    
    model = SAMGitCaptioningModel(
        img_obj_encoder,
        text_decoder,
        decoder=decoder,
        sos_index=tokenizer.cls_token_id,
        eos_index=tokenizer.sep_token_id,
        tokenizer=tokenizer,
        use_history_for_infer=True,
        loss_type='smooth',
        num_image_with_embedding=param.get('num_image_with_embedding')
    )
    return model