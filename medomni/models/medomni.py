import logging
import random

import torch
from torch.cuda.amp import autocast as autocast
from torchvision import models
import torch.nn as nn

from medomni.common.registry import registry
from medomni.models.blip2 import Blip2Base, disabled_train
from medomni.models.modeling_llama import LlamaForCausalLM
from transformers import LlamaTokenizer
from transformers import SwinModel
import torch.nn.functional as F
import math
from einops import rearrange, repeat
from einops_exts import rearrange_many
import open_clip
import segmentation_models_pytorch as smp
from medomni.models.UNet import UNet3d
from huggingface_hub import PyTorchModelHubMixin
import ipdb
from peft import (
    get_peft_model,
    LoraConfig,
    PrefixTuningConfig,
    PromptEncoderConfig,
    PromptTuningConfig,
    TaskType,
)

class GroupNorm(nn.GroupNorm):
    """Subclass torch's LayerNorm to handle fp16."""

    def forward(self, x: torch.Tensor):
        orig_type = x.dtype
        ret = super().forward(x.type(torch.float32))
        return ret.type(orig_type)

class LayerNorm(nn.LayerNorm):
    """Subclass torch's LayerNorm to handle fp16."""

    def forward(self, x: torch.Tensor):
        orig_type = x.dtype
        ret = super().forward(x.type(torch.float32))
        return ret.type(orig_type)

def replace_batchnorm_2d(model):
    for name, module in reversed(model._modules.items()):
        if len(list(module.children())) > 0:
            model._modules[name] = replace_batchnorm_2d(module)
        
        if isinstance(module, nn.BatchNorm2d):
            model._modules[name] = GroupNorm(num_groups=16, num_channels=module.num_features)
    return model

def dice_loss(input, target):
    input = torch.sigmoid(input)
    smooth = 1.0
    iflat = input.view(-1)
    tflat = target.view(-1)
    intersection = (iflat * tflat).sum()
    return ((2.0 * intersection + smooth) / (iflat.sum() + tflat.sum() + smooth))

class FocalLoss(nn.Module):
    def __init__(self, gamma):
        super().__init__()
        self.gamma = gamma

    def forward(self, input, target):
        if not (target.size() == input.size()):
            raise ValueError("Target size ({}) must be the same as input size ({})"
                             .format(target.size(), input.size()))
        max_val = (-input).clamp(min=0)
        loss = input - input * target + max_val + \
            ((-max_val).exp() + (-input - max_val).exp()).log()
        invprobs = F.logsigmoid(-input * (target * 2.0 - 1.0))
        loss = (invprobs * self.gamma).exp() * loss
        return loss.mean()

class MixedLoss(nn.Module):
    def __init__(self, alpha, gamma):
        super().__init__()
        self.alpha = alpha
        self.focal = FocalLoss(gamma)

    def forward(self, input, target):
        loss = self.alpha*self.focal(input, target) - torch.log(dice_loss(input, target))
        return loss.mean()

def trans_seg(sample_num, bsz):
    labels = torch.zeros((bsz, 10))
    c_bsz = 0
    for num1 in sample_num:
        num2 = num1.split('-')
        for num3 in num2:
            if num3 != 'n/a':
                c4 = 0
                for num in num3.split(','):
                    labels[c_bsz, c4] = float(num)
                    c4 += 1
                c_bsz += 1
    return labels

def trans_det(sample_num, bsz):
    labels = torch.zeros((bsz, 4))
    c_bsz = 0
    for num1 in sample_num:
        num2 = num1.split(';')
        for num3 in num2:
            if num3 != 'n/a':
                c4 = 0
                for num in num3.split(','):
                    labels[c_bsz, c4] = float(num)
                    c4 += 1
                c_bsz += 1
    return labels

def trans_keypoint(sample_num, bsz):
    labels = torch.zeros((bsz, 2))
    c_bsz = 0
    for num1 in sample_num:
        num2 = num1.split(';')
        for num3 in num2:
            if num3 != 'n/a':
                c4 = 0
                for num in num3.split(','):
                    labels[c_bsz, c4] = float(num)
                    c4 += 1
                c_bsz += 1
    return labels

@registry.register_model("medomni")
class MedOmni(Blip2Base, PyTorchModelHubMixin):
    PRETRAINED_MODEL_CONFIG_DICT = {
        "medomni": "configs/models/medomni.yaml",
    }
    def __init__(
        self,
        config,
    ):
        super().__init__()
        freeze_vit=True
        llama_model=config['llama_model']
        max_txt_len=config['max_txt_len']
        low_resource=False  # use 8 bit and put vit in cpu / have not been tested
        end_sym=config['end_sym']
        # self.tokenizer = self.init_tokenizer()
        self.low_resource = low_resource

        print('Loading VIT')
        self.visual_encoder_2d = SwinModel.from_pretrained('microsoft/swin-base-patch4-window7-224')
        self.visual_encoder_3d = UNet3d(in_channels=1, n_classes=1, n_channels=32)
        self.ln_vision_2d = LayerNorm(1024)
        self.ln_vision_3d = LayerNorm(256)

        if freeze_vit:
            for name, param in self.visual_encoder_2d.named_parameters():
                param.requires_grad = False
            self.visual_encoder_2d = self.visual_encoder_2d.eval()
            self.visual_encoder_2d.train = disabled_train
            for name, param in self.ln_vision_2d.named_parameters():
                param.requires_grad = False
            self.ln_vision_2d = self.ln_vision_2d.eval()
            self.ln_vision_2d.train = disabled_train
            for name, param in self.visual_encoder_3d.named_parameters():
                param.requires_grad = False
            self.visual_encoder_3d = self.visual_encoder_3d.eval()
            self.visual_encoder_3d.train = disabled_train
            for name, param in self.ln_vision_3d.named_parameters():
                param.requires_grad = False
            self.ln_vision_3d = self.ln_vision_3d.eval()
            self.ln_vision_3d.train = disabled_train
            logging.info("freeze vision encoder")
        print('Loading VIT Done')

        print('Loading LLAMA')
        self.llama_tokenizer = LlamaTokenizer.from_pretrained(llama_model, legacy=False, use_fast=False)
        special_token = {}
        special_token["additional_special_tokens"] = ['<ImageHere>']
        self.llama_tokenizer.add_special_tokens(
            special_token
        )
        self.llama_tokenizer.add_tokens("<DET>")
        self.llama_tokenizer.add_tokens("<2DSEG>")
        self.llama_tokenizer.add_tokens("<3DSEG>")
        # self.llama_tokenizer.add_tokens("<2DPOINT>")
        self.llama_tokenizer.add_tokens("<N/A>")
        self.det_token_idx = self.llama_tokenizer("<DET>", add_special_tokens=False).input_ids[0]
        self.seg_token_idx_2d = self.llama_tokenizer("<2DSEG>", add_special_tokens=False).input_ids[0]
        self.seg_token_idx_3d = self.llama_tokenizer("<3DSEG>", add_special_tokens=False).input_ids[0]
        # self.point_token_idx_2d = self.llama_tokenizer("<2DPOINT>", add_special_tokens=False).input_ids[0]
        self.na_token_idx = self.llama_tokenizer("<N/A>", add_special_tokens=False).input_ids[0]
        self.llama_tokenizer.pad_token = 0

        if self.low_resource:
            self.llama_model = LlamaForCausalLM.from_pretrained(
                llama_model,
                torch_dtype=torch.bfloat16,
                load_in_8bit=True,
                device_map="auto"
            )
        else:
            self.llama_model = LlamaForCausalLM.from_pretrained(
                llama_model,
                torch_dtype=torch.bfloat16,
            )

        self.llama_model.resize_token_embeddings(len(self.llama_tokenizer))
        self.embed_tokens = self.llama_model.get_input_embeddings()
        self.embed_states = self.llama_model.get_output_embeddings() # cannot remove
        # ---LoRA---
        class CastOutputToFloat(nn.Sequential):
            def forward(self, x): return super().forward(x).to(torch.bfloat16)
        self.llama_model.lm_head = CastOutputToFloat(self.llama_model.lm_head)
        # ---LoRA---

        print("Setup PEFT")
        peft_config = LoraConfig(
            task_type="CAUSAL_LM", inference_mode=False,
            r=16,
            lora_alpha=16, lora_dropout=0.1, 
            target_modules=['q_proj', 'v_proj']
        ) # 8 32 hyz 9.21
        self.llama_model = get_peft_model(self.llama_model, peft_config)
        self.llama_proj_2d = nn.Linear(1024, self.llama_model.config.hidden_size)
        self.llama_proj_3d = nn.Linear(256, self.llama_model.config.hidden_size)

        # # Detection
        text_det = nn.Sequential(
            LayerNorm(self.llama_model.config.hidden_size),
            nn.Linear(self.llama_model.config.hidden_size, 256),
            nn.ReLU(inplace=True),
            LayerNorm(256),
            nn.Linear(256, 4),
        )
        self.text_det = text_det
        self.det_loss = torch.nn.SmoothL1Loss()

        # # Keypoint
        # text_point = nn.Sequential(
        #     LayerNorm(self.llama_model.config.hidden_size),
        #     nn.Linear(self.llama_model.config.hidden_size, 256),
        #     nn.ReLU(inplace=True),
        #     LayerNorm(256),
        #     nn.Linear(256, 2),
        # )
        # self.text_point = text_point
        # self.keypoint_loss = torch.nn.SmoothL1Loss()

        # Segmentation
        self.model_seg_2d = smp.Unet(encoder_name="resnet18", encoder_weights="imagenet", in_channels=3, classes=1)
        self.model_seg_2d = replace_batchnorm_2d(self.model_seg_2d) # GN is much better than BN

        text2seg_2d = nn.Sequential(
            LayerNorm(self.llama_model.config.hidden_size),
            nn.Linear(self.llama_model.config.hidden_size, 512),
        )
        self.text2seg_2d = text2seg_2d
        self.text2seg_2d_ln = LayerNorm(512)
        self.text2seg_2d_gn = GroupNorm(16, 512)
        text2seg_3d = nn.Sequential(
            LayerNorm(self.llama_model.config.hidden_size),
            nn.Linear(self.llama_model.config.hidden_size, 256),
        )
        self.text2seg_3d = text2seg_3d
        self.text2seg_3d_ln = LayerNorm(256)
        self.text2seg_3d_gn = GroupNorm(16, 256)
        self.seg_loss = MixedLoss(10.0, 2.0)

        self.max_txt_len = max_txt_len
        self.end_sym = end_sym
        self.prompt_list = []

    def vit_to_cpu(self):
        self.ln_vision.to("cpu")
        self.ln_vision.float()
        self.visual_encoder.to("cpu")
        self.visual_encoder.float()

    def encode_img(self, image, modals, task_types=[]):
        B,S,_,_,_ = image.shape
        device = image.device
        image_embeds_list = None
        if self.low_resource:
            self.vit_to_cpu()
            image = image.to("cpu")
        with self.maybe_autocast(device):
            if 'ct' in modals:
                image_embeds_list = self.visual_encoder_3d(image, encoder_only=True)
                image_embeds_list = [_.to(device) for _ in image_embeds_list]
                image_embeds = image_embeds_list[-1].detach()
                image_embeds = F.adaptive_avg_pool3d(image_embeds, (1, 3, 3)).view(B, image_embeds.shape[1], -1).permute(0, 2, 1)
                inputs_llama = self.llama_proj_3d(self.ln_vision_3d(image_embeds))
                inputs_llama = rearrange(inputs_llama, "(b s) c d -> b s c d", b=B, s=S).to(torch.bfloat16)
                atts_llama = torch.ones(inputs_llama.size()[:-2], dtype=torch.long).to(image.device)
            else:
                image = rearrange(image, "b s c h w -> (b s) c h w")
                image_embeds = self.visual_encoder_2d(image)['last_hidden_state'].to(device)
                image_embeds_unp = image_embeds.permute(0, 2, 1).view(B*S,-1,7,7)
                image_embeds_unp = F.adaptive_avg_pool2d(image_embeds_unp, (3, 3))
                image_embeds = image_embeds_unp.view(B*S, -1, 9).permute(0, 2, 1)
                inputs_llama = self.llama_proj_2d(self.ln_vision_2d(image_embeds))
                if 'segmentation' not in task_types:
                    inputs_llama = rearrange(inputs_llama, "(b s) c d -> b s c d", b=B, s=S).to(torch.bfloat16)
                    atts_llama = torch.ones(inputs_llama.size()[:-2], dtype=torch.long).to(image.device)
                else:
                    inputs_llama = rearrange(inputs_llama, "(b s) c d -> b s c d", b=B, s=S).to(torch.bfloat16).detach() # add detach() for segmentation tasks
                    atts_llama = torch.ones(inputs_llama.size()[:-2], dtype=torch.long).to(image.device).detach()
         
        return inputs_llama, atts_llama, image_embeds_list

    def prompt_concat(self, img_embeds, atts_img, prompt):
        if prompt:
            batch_size = img_embeds.shape[0]
            p_after_embeds = self.embed_tokens(prompt.input_ids).expand(batch_size, -1, -1)
            wrapped_img_embeds = torch.cat([img_embeds, p_after_embeds], dim=1)
            wrapped_atts_img = atts_img[:, :1].expand(-1, wrapped_img_embeds.shape[1])                
            return wrapped_img_embeds, wrapped_atts_img
        else:
            return img_embeds, atts_img

    def prompt_wrap(self, img_embeds, atts_img, prompt_list, num_imgs, seg=None):
        bsz = img_embeds.shape[0]
        if prompt_list:
            img_idx = ([], [])
            for i in range(len(num_imgs)):
                for j in range(num_imgs[i]):
                    img_idx[0].append(i)
                    img_idx[1].append(j)
            prompt_tokens = self.llama_tokenizer(prompt_list, return_tensors="pt", padding="longest", truncation=True, max_length=256).to(img_embeds.device)
            idx = (prompt_tokens.input_ids == 32000).nonzero(as_tuple=True)
            prompt_tokens.input_ids[idx] = 123 # avoid memory issue
            p_embeds = self.embed_tokens(prompt_tokens.input_ids).expand(bsz, -1, -1)
            if seg is None:
                p_embeds[idx] = rearrange(img_embeds[img_idx], "b c d -> (b c) d").to(torch.bfloat16)
            else:
                p_embeds[idx] = rearrange(img_embeds[img_idx], "b c d -> (b c) d").to(torch.bfloat16).detach()
            return p_embeds, atts_img
        else:
            return img_embeds, atts_img

    def forward(self, samples):
        image = samples["image"]
        bsz = image.shape[0]
        img_embeds, atts_img, img_embeds_list = self.encode_img(image, samples['modal'], samples['task_type'])
        prefix_list = []
        tag_list = [[] for _ in range(bsz)]
        placeholder = ['<ImageHere>'] * 9 # 9 = the number of visual tokens
        for j in range(bsz):
            num = samples['num_imgs'][j]
            prefix = '' # Can add some prompt, such as 'You will be given an image, please describe everything you see' 
            for i in range(num):
                prefix += '<img' + str(i) + '>' + ''.join(x for x in placeholder) + '</img' + str(i) + '>' 
                tag_list[j].append('<img' + str(i) + '>')
            prefix_list.append('###Human:' + prefix)
        img_embeds, atts_img = self.prompt_wrap(img_embeds, atts_img, prefix_list, samples['num_imgs'], seg = None if 'segmentation' not in samples['task_type'] else 'yes')
        self.llama_tokenizer.padding_side = "right"

        prompt = [t for t in samples['question']]
        for i in range(len(prompt)):
            tags = ''
            for tag in tag_list[i]:
                if tag not in prompt[i]:
                    tags += tag
            prompt[i] = prompt[i].replace('_*_', tags)
        
        if 'detection' in samples['task_type'] or 'keypoint' in samples['task_type']:
            sample_ans = [ans.split('|||')[0] for ans in samples['answer']]
            sample_num = [ans.split('|||')[1] for ans in samples['answer']]
        else:
            sample_ans = samples['answer']
        text = ['###Assistant: ' + str(t) + self.end_sym for t in sample_ans]

        prompt_tokens = self.llama_tokenizer(
            prompt,
            return_tensors="pt",
            padding='longest',
            truncation=True,
            max_length=256,
            add_special_tokens=False
        ).to(image.device)

        img_embeds, atts_img = self.prompt_concat(img_embeds, atts_img, prompt_tokens) 

        to_regress_tokens = self.llama_tokenizer(
            text,
            return_tensors="pt",
            padding="longest",
            truncation=True,
            max_length=self.max_txt_len,
            add_special_tokens=False
        ).to(image.device)

        targets = to_regress_tokens.input_ids.masked_fill(
            to_regress_tokens.input_ids == self.llama_tokenizer.pad_token_id, -100
        )

        empty_targets = (
            torch.ones([atts_img.shape[0], atts_img.shape[1]+1],
                       dtype=torch.long).to(image.device).fill_(-100)  # plus one for bos
        )
        targets = torch.cat([empty_targets, targets], dim=1)

        batch_size = img_embeds.shape[0]
        bos = torch.ones([batch_size, 1],
                         dtype=to_regress_tokens.input_ids.dtype,
                         device=to_regress_tokens.input_ids.device) * self.llama_tokenizer.bos_token_id

        bos_embeds = self.embed_tokens(bos)
        atts_bos = atts_img[:, :1]

        to_regress_embeds = self.embed_tokens(to_regress_tokens.input_ids)
        inputs_embeds = torch.cat([bos_embeds, img_embeds, to_regress_embeds], dim=1)
        attention_mask = torch.cat([atts_bos, atts_img, to_regress_tokens.attention_mask], dim=1)
        with self.maybe_autocast(image.device):
            outputs = self.llama_model(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                return_dict=True,
                labels=targets,
                output_hidden_states=True,
            )
        loss = outputs.loss

        if 'detection' in samples['task_type']:
            with self.maybe_autocast(image.device):
                hidden_states = outputs.hidden_states[-1]
                token_mask = targets == self.det_token_idx
                target_states = hidden_states[token_mask]
                with self.maybe_autocast():
                    det_states = self.text_det(target_states)
                labels = trans_det(sample_num, det_states.shape[0])
                labels = labels.to(targets.device)
                det_loss = self.det_loss(det_states, labels)
                loss += det_loss * 1e2

        if 'keypoint' in samples['task_type']:
            with self.maybe_autocast(image.device):
                hidden_states = outputs.hidden_states[-1]
                token_mask = targets == self.point_token_idx_2d
                target_states = hidden_states[token_mask]
                with self.maybe_autocast():
                    point_states = self.text_point(target_states)
                labels = trans_keypoint(sample_num, point_states.shape[0])
                labels = labels.to(targets.device)
                keypoint_loss = self.keypoint_loss(point_states, labels)
                loss += keypoint_loss * 1e2

        if 'segmentation' in samples['task_type']:
            if 'ct' in samples['modal']:
                masks = samples['answer_img']
                with self.maybe_autocast(image.device):
                    img_embeds_list = self.visual_encoder_3d(image, encoder_only=True)
                    img_embeds_list = [_.to(targets.device) for _ in img_embeds_list]
                    hidden_states = outputs.hidden_states[-1]
                    token_mask = targets == self.seg_token_idx_3d
                    target_states = hidden_states[token_mask]   
                    seg_states = self.text2seg_3d(target_states)
                    last_feats = img_embeds_list[-1]
                    last_feats = last_feats + seg_states.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
                    last_feats = self.text2seg_3d_gn(last_feats)
                    img_embeds_list[-1] = last_feats
                    seg_preds = self.visual_encoder_3d(encoder_only=False, x_=img_embeds_list)
                    loss += self.seg_loss(seg_preds, masks.float()) # +
            else:  
                masks = samples['answer_img']
                with self.maybe_autocast(image.device):
                    feats = self.model_seg_2d.encoder(image[:,0])
                    last_feats = feats[-1]
                    hidden_states = outputs.hidden_states[-1]
                    token_mask = targets == self.seg_token_idx_2d
                    target_states = hidden_states[token_mask]
                    seg_states = self.text2seg_2d(target_states)
                    last_feats = last_feats+seg_states.unsqueeze(-1).unsqueeze(-1)
                    last_feats = self.text2seg_2d_gn(last_feats)
                    feats[-1] = last_feats
                    seg_feats = self.model_seg_2d.decoder(*feats)
                    seg_preds = self.model_seg_2d.segmentation_head(seg_feats)
                    loss += self.seg_loss(seg_preds, masks.float())

        return {"loss": loss, "modal": samples['modal'][0], "task_type": samples['task_type'][0]}

    @classmethod
    def from_config(cls, cfg, finetune=False):
        model = cls(cfg)

        # load checkpoint
        ckpt_path = cfg.get("ckpt", "")
        if ckpt_path:
            print("Load Checkpoint: {}".format(ckpt_path))
            ckpt = torch.load(ckpt_path, map_location="cpu")
            if finetune:
                current_model_dict = model.state_dict()
                weights = ckpt['model']
                new_state_dict = {}
                for k in list(current_model_dict.keys()):
                    if k in list(weights.keys()):
                        if weights[k].size() == current_model_dict[k].size():
                            new_state_dict[k] = weights[k]
                        else:
                            new_state_dict[k] = current_model_dict[k]
                    else:
                        print(k)
                        new_state_dict[k] = current_model_dict[k]
                msg = model.load_state_dict(new_state_dict, strict=False)
            else:
                msg = model.load_state_dict(ckpt['model'], strict=False)

        return model