# Copyright (c) Facebook, Inc. and its affiliates.
# Modified by Jimmy Yao from https://github.com/microsoft/GenerativeImage2Text/blob/main/generativeimage2text/layers/decoder.py

import torch
from torch import nn

from generativeimage2text.layers.decoder import (SmoothLabelCrossEntropyLoss, convert2valid)

import logging
from pprint import pformat
import functools

from torch.cuda.amp import autocast

        
class SAMGitCaptioningModel(nn.Module):
    def __init__(
        self,
        visual,
        textual,
        sos_index=1,
        eos_index=2,
        decoder=None,
        #use_masked_as_input_for_train=False,
        loss_type=None,
        context_not_share_embedding=False,
        scst=False,
        tokenizer=None,
        scst_temperature=1.,
        use_history_for_infer=False,
        pooling_images=None,
        num_image_with_embedding=0,
    ):
        super().__init__()
        self.image_encoder = visual
        self.textual = textual
        self.padding_idx = self.textual.padding_idx

        # These boundary indices are needed for beam search.
        self.sos_index = sos_index
        self.eos_index = eos_index
        self.decoder = decoder
        self.scst = scst
        self.tokenizer = tokenizer

        if self.scst:
            raise NotImplementedError
            #from .utils_caption_evaluate import (
                    #ScstRewardCriterion)
            #self.scst_criterion = ScstRewardCriterion(
                #cider_cached_tokens='data/coco_caption/gt/coco-train-words.p',
                #baseline_type='greedy',
            #)
            #self.scst_fwd_times = 0
            #self.scst_temperature = scst_temperature
        if loss_type is None:
            self.loss = nn.CrossEntropyLoss(ignore_index=self.padding_idx)
        elif loss_type == 'smooth':
            self.loss = SmoothLabelCrossEntropyLoss(ignore_index=self.padding_idx)
        else:
            raise NotImplementedError(loss_type)
        #self.use_masked_as_input_for_train = use_masked_as_input_for_train

        self.verbose = {'num_has_image': 0, 'num_no_image': 0}
        self.context_not_share_embedding = context_not_share_embedding
        if context_not_share_embedding:
            self.context_embedding = self.textual.embedding.clone()
            # check whether the parameters are shared or not. it should not
            # share
        self.use_history_for_infer = use_history_for_infer
        self.pooling_images = pooling_images

        if num_image_with_embedding:
            logging.info('creating temperal embedding')
            self.img_temperal_embedding = nn.ParameterList(
                nn.Parameter(torch.zeros(1, 1, self.textual.visual_feature_size)) for _ in range(num_image_with_embedding)
            )
        self.num_image_with_embedding = num_image_with_embedding

    def forward(self, batch, batch_img_crops):
        result = self.forward_one(batch, batch_img_crops, return_info=False)
        return result

        # shape: (batch_size, channels, height, width)
    def forward_one(self, batch, batch_img_crops, return_info=False):
        # shape: (batch_size, max_caption_length, vocab_size)
        if 'image' in batch:
            if isinstance(batch['image'], (list, tuple)):
                features = [self.image_encoder(im, batch_img_crops) for im in batch['image']]
                
                if self.num_image_with_embedding:
                    features = [f + e for f, e in zip(features, self.img_temperal_embedding)]
                if self.pooling_images is None:
                    visual_features = torch.cat(features, dim=1)
                elif self.pooling_images == 'avg':
                    visual_features = torch.stack(features, dim=1).mean(dim=1)
                else:
                    raise NotImplementedError
            else:
                visual_features = self.image_encoder(batch['image'], batch_img_crops)
        else:
            visual_features = None
        visual_features_valid = None
        if 'context' in batch:
            context_embedding = self.context_embedding if self.context_not_share_embedding else self.textual.embedding
            all_context = [visual_features]
            all_valid = [convert2valid(visual_features.shape[:2])]
            for info in batch['context']:
                context = context_embedding(info['tokens'])
                valid = convert2valid(info['tokens'].shape, info['length'])
                all_context.append(context)
                all_valid.append(valid)
            visual_features = torch.cat(all_context, dim=1)
            visual_features_valid = torch.cat(all_valid, dim=1)
            
        if not self.training or (not self.scst):
            return self.forward_one_ce(batch, visual_features, visual_features_valid, return_info)
        else:
            assert self.training and self.scst
            return self.forward_one_scst(batch, visual_features, visual_features_valid)

    def forward_one_scst(self, batch, visual_features, visual_features_valid):
        self.eval()
        def _ids_to_captions(all_ids):
            captions = []
            for ids in all_ids:
                c = self.tokenizer.decode(ids.tolist(), skip_special_tokens=True)
                captions.append(c)
            return captions
        with torch.no_grad():
            greedy_res = self.infer(batch, visual_features, visual_features_valid)
            greedy_res_raw = greedy_res['predictions']
            greedy_res_raw.squeeze_(1)  # batch_size * max_len
            greedy_res = _ids_to_captions(greedy_res_raw)

        self.train()
        search_param = {
            'do_sample': True,
            #'top_k': 5,
            'top_p': 1,
            'num_return_sequences': 5,
            'temperature': self.scst_temperature,
        }
        infer_res = self.infer(
            batch,
            visual_features,
            visual_features_valid,
            search_param,
        )
        sample_res = _ids_to_captions(infer_res['predictions'])
        gt_res = list(zip(*[[j_th_image_cap for j_th_image_cap in i_th_caption['caption']] for i_th_caption in batch['all_caption']]))
        loss = self.scst_criterion(gt_res, greedy_res, sample_res, infer_res['logprobs'])
        if (self.scst_fwd_times % 100) == 0:
            info = self.scst_criterion.get_info()
            logging.info(pformat(info))
        self.scst_fwd_times += 1
        return {'decoder_loss': loss}

    def forward_one_ce(self, batch, visual_features, visual_features_valid, return_info):
        has_image = (visual_features is not None)
        assert has_image == ('image' in batch)
        if self.training:
            #if self.use_masked_as_input_for_train:
                #caption_token_input = batch["masked_caption_tokens"]
            #else:
            caption_token_input = batch["caption_tokens"]
            #caption_lengths = batch["caption_lengths"]

            output_logits = self.textual(
                visual_features,
                caption_token_input,
                #caption_lengths=caption_lengths,
                hidden_valid_mask=visual_features_valid,
                bi_valid_mask_caption=batch.get('bi_valid_mask_caption'),
            )
            output_dict = {}
            #output_logits = x['output_logits']
            #ipdb> output_logits.shape
            #torch.Size([2, 13, 30522])
            #ipdb> batch['caption_tokens'].shape
            #torch.Size([2, 13])
            if 'need_predict' in batch:
                target = batch["caption_tokens"].clone()
                if self.padding_idx is not None:
                    target[batch['need_predict'] == 0] = self.padding_idx
            else:
                assert ValueError()
                #target = batch["caption_tokens"]
            need_predict = batch['need_predict']
            feat = output_logits[:, :-1].contiguous()
            target = target[:, 1:].contiguous()
            need_predict = need_predict[:, 1:].contiguous()
            feat = feat.view(-1, self.textual.vocab_size)
            target = target.view(-1)
            need_predict = need_predict.view(-1)

            valid_mask = need_predict == 1
            #valid_mask2 = target != self.padding_idx
            #assert (valid_mask.long() - valid_mask2.long()).abs().sum().cpu() == 0

            target = target[valid_mask]
            feat = feat[valid_mask]
            loss = self.loss(feat, target)
            if (self.verbose['num_has_image'] + self.verbose['num_no_image']) % 200 == 0:
                logging.info(self.verbose)
            hint = 'l' if 'context_target_type' not in batch else batch['context_target_type'][0]
            if has_image:
                output_dict.update({'vl_{}_loss'.format(hint): loss})
                self.verbose['num_has_image'] += 1
            else:
                output_dict.update({'l_{}_loss'.format(hint): loss})
                self.verbose['num_no_image'] += 1

            if return_info:
                output_dict['feat'] = feat
        else:
            output_dict = self.infer(batch, visual_features, visual_features_valid)
        return output_dict

    def infer(self, batch, visual_features, visual_features_valid,
              search_param=None):
        batch_size = visual_features.size(0)
        if 'prefix' not in batch:
            start_predictions = visual_features.new_full(
                (batch_size,1), self.sos_index
            ).long()
        else:
            # if batch size is larger than 1, the prefix length could be
            # different, and we have to padding non-valid data, which
            # is not supported
            assert len(batch['prefix']) == 1, 'not supported'
            start_predictions = batch['prefix'].long()

        self.prev_encoded_layers = None
        # Add image features as a default argument to match callable
        # signature accepted by beam search class (partial captions only).
        decoding_step = functools.partial(
            self.decoding_step, visual_features, visual_features_valid,
            batch.get('bi_valid_mask_caption')
        )

        search_param = search_param or {}
        # the start_predictions are not in predicted_caption
        predicted_caption, logprobs = self.decoder.search(
            start_predictions, decoding_step, **search_param
        )
        if 'prefix' in batch:
            # we need to remove prefix from predicted_caption
            predicted_caption = predicted_caption[:, start_predictions.shape[1]:]
        output_dict = {
            'predictions': predicted_caption,
            'logprobs': logprobs,
        }
        return output_dict

    def decoding_step(
        self, visual_features, visual_features_valid, bi_valid_mask_caption, partial_captions
    ):
        # Expand and repeat image features while doing beam search.
        batch_size = visual_features.shape[0]
        beam_size = int(partial_captions.size(0) / batch_size)
        if beam_size > 1:
            batch_size, num_token, channels = visual_features.size()
            # shape: (batch_size * beam_size, channels, height, width)
            visual_features = visual_features.unsqueeze(1).repeat(1, beam_size, 1, 1)
            visual_features = visual_features.view(
                batch_size * beam_size, num_token, channels
            )

        # Provide caption lengths as current length (irrespective of predicted
        # EOS/padding tokens). shape: (batch_size, )
        caption_lengths = torch.ones_like(partial_captions)
        if len(caption_lengths.size()) == 2:
            caption_lengths = caption_lengths.sum(1)
        else:
            # Add a timestep. shape: (batch_size, 1)
            partial_captions = partial_captions.unsqueeze(1)

        # shape: (batch_size * beam_size, partial_caption_length, vocab_size)
        logits = self.textual(
            visual_features,
            partial_captions,
            caption_lengths=caption_lengths,
            hidden_valid_mask=visual_features_valid,
            bi_valid_mask_caption=bi_valid_mask_caption,
            encoder_history_states=self.prev_encoded_layers,
        )
        if self.scst or self.use_history_for_infer:
            if isinstance(logits, tuple) and len(logits) == 2:
                if self.prev_encoded_layers is None:
                    self.prev_encoded_layers = logits[1]
                else:
                    self.prev_encoded_layers = [torch.cat((p, c), dim=1) for p, c in
                                                zip(self.prev_encoded_layers, logits[1])]
                #self.prev_encoded_layers = None
                logits = logits[0]
        return logits[:, -1, :].float()