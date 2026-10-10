from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import logging
import math
from collections import OrderedDict
from types import SimpleNamespace
import torch
from torch import nn
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence
import torch.nn.functional as F
from .module_clip import CLIP, convert_weights, _PT_NAME
from .module_cross import Transformer as TransformerClip
from .module_clip_refiner import CLIPSpatialRefiner
from .module_tome_patch import apply_patch as tome_patch
from .pamr import PAMR
from .module_tome_utils import parse_r
from .until_module import LayerNorm, AllGather, AllGather2, CrossEn, MSE, ArcCrossEn, KL
import numpy as np
import copy
allgather = AllGather.apply
allgather2 = AllGather2.apply

logger = logging.getLogger(__name__)


def distributed_barrier_on_current_device():
    backend = torch.distributed.get_backend().lower()
    if backend == "nccl" and torch.cuda.is_available():
        try:
            torch.distributed.barrier(device_ids=[torch.cuda.current_device()])
            return
        except TypeError:
            pass
    torch.distributed.barrier()

class ResidualLinear(nn.Module):
    def __init__(self, d_int: int):
        super(ResidualLinear, self).__init__()

        self.fc_relu = nn.Sequential(nn.Linear(d_int, d_int),
                                     nn.ReLU(inplace=True))

    def forward(self, x):
        x = x + self.fc_relu(x)
        return x


class VTRModel(nn.Module):
    def __init__(self, config):
        super(VTRModel, self).__init__()
        
        self.config = config
        backbone = getattr(config, 'base_encoder', "ViT-B/32")

        self.lora_dim = config.lora_dim
        logger.info("v_LoRA: {} dim".format(self.lora_dim))
        
        assert backbone in _PT_NAME
        model_path = os.path.join(config.pretrained_path, _PT_NAME[backbone])
        if not os.path.exists(model_path):
            raise FileNotFoundError(model_path)

        state_dict_cache = "{}.state_dict.pt".format(model_path)
        cache_is_fresh = (
            os.path.exists(state_dict_cache)
            and os.path.getmtime(state_dict_cache) >= os.path.getmtime(model_path)
        )
        if cache_is_fresh:
            logger.info("Loading CLIP state_dict cache: %s", state_dict_cache)
            state_dict = torch.load(state_dict_cache, map_location="cpu")
        else:
            logger.info("Loading CLIP TorchScript archive: %s", model_path)
            try:
                # loading JIT archive
                model = torch.jit.load(model_path, map_location="cpu").eval()
                state_dict = model.state_dict()
            except RuntimeError:
                state_dict = torch.load(model_path, map_location="cpu")
        logger.info("Loaded CLIP state_dict with %d tensors", len(state_dict))

        vision_width = state_dict["visual.conv1.weight"].shape[0]
        vision_layers = len(
            [k for k in state_dict.keys() if k.startswith("visual.") and k.endswith(".attn.in_proj_weight")])
        vision_patch_size = state_dict["visual.conv1.weight"].shape[-1]
        grid_size = round((state_dict["visual.positional_embedding"].shape[0] - 1) ** 0.5)
        image_resolution = vision_patch_size * grid_size

        embed_dim = state_dict["text_projection"].shape[1]
        context_length = state_dict["positional_embedding"].shape[0]
        vocab_size = state_dict["token_embedding.weight"].shape[0]
        transformer_width = state_dict["ln_final.weight"].shape[0]
        transformer_heads = transformer_width // 64
        transformer_layers = len(set(k.split(".")[2] for k in state_dict if k.startswith(f"transformer.resblocks")))

        self.merge_layer = [int(_l) for _l in config.merge_layer.split('-')]
        self.merge_frame_num = [int(_l) for _l in config.merge_frame_num.split('-')]
        frame_num_list=[]
        frame_num = config.max_frames
        for _l in range(len(self.merge_layer)):
            frame_num_list.append(frame_num)
            frame_num = frame_num // self.merge_frame_num[_l]
        logger.info('Position_embedding: {}'.format(frame_num_list))
        
        self.clip = CLIP(embed_dim, image_resolution, vision_layers, vision_width, vision_patch_size,
                         context_length, vocab_size, transformer_width, transformer_heads, transformer_layers, self.lora_dim, 
                        self.merge_layer, config.frame_pos, frame_num_list)
            
        self.loss_fct = CrossEn(config)

        self.clip.load_state_dict(state_dict, strict=False)

        self.tome_r = config.tome_r
        self.tome_tracesource = config.tome_tracesource
        self.tome_propattn = config.tome_propattn
        logger.info("tome: {} r | {} tracesource | {} propattn".format(self.tome_r, self.tome_tracesource, self.tome_propattn))
        
        logger.info("merge_layer: {}".format(config.merge_layer))
        logger.info("merge_frame_num: {}".format(config.merge_frame_num))
        logger.info("merge_token_proportion: {}".format(config.merge_token_proportion))
        logger.info("frame_pos: {}".format(config.frame_pos))
        self.use_claa = bool(getattr(config, "use_claa", 1))
        self.use_sap = bool(getattr(config, "use_sap", 1))
        self.resclip_window = max(1, int(getattr(config, "resclip_window", 4)))
        self.attn_claa_weights = (
            float(getattr(config, "claa_tau", 2.0)),
            float(getattr(config, "claa_lambda", 0.6)),
        )
        self.attn_sap_weights = (
            float(getattr(config, "sap_tau", 2.1)),
            float(getattr(config, "sap_lambda", 0.6)),
        )
        self.sap_temp_thd = float(getattr(config, "sap_temp_thd", 0.0))
        self.use_pamr = bool(getattr(config, "use_pamr", 0))
        self.pamr_steps = int(getattr(config, "pamr_steps", 10))
        self.pamr_stride = tuple(
            int(item) for item in str(getattr(config, "pamr_stride", "8-16")).split("-") if item != ""
        )
        self.pamr = PAMR(self.pamr_steps, dilations=self.pamr_stride) if self.use_pamr and self.pamr_steps > 0 else None
        self._pamr_window_logged = False
        self.use_clip_refiner = bool(getattr(config, "use_clip_refiner", 0))
        self.clip_refiner_lambda = float(getattr(config, "clip_refiner_lambda", 0.25))
        self.clip_refiner_scd_weight = float(getattr(config, "clip_refiner_scd_weight", 0.05))
        self.clip_refiner = (
            CLIPSpatialRefiner(
                width=vision_width,
                depth=int(getattr(config, "clip_refiner_depth", 2)),
                hidden_ratio=float(getattr(config, "clip_refiner_hidden_ratio", 0.25)),
                residual_scale=float(getattr(config, "clip_refiner_residual_scale", 0.25)),
                temperature=float(getattr(config, "clip_refiner_temperature", 0.07)),
            )
            if self.use_clip_refiner
            else None
        )
        logger.info(
            "resclip: use_claa=%s use_sap=%s use_pamr=%s window=%d claa=%s sap=%s thd=%.3f pamr_steps=%d pamr_stride=%s",
            self.use_claa,
            self.use_sap,
            self.use_pamr,
            self.resclip_window,
            self.attn_claa_weights,
            self.attn_sap_weights,
            self.sap_temp_thd,
            self.pamr_steps,
            self.pamr_stride,
        )
        logger.info(
            "clip_refiner: enabled=%s depth=%d lambda=%.3f scd_weight=%.3f",
            self.clip_refiner is not None,
            int(getattr(config, "clip_refiner_depth", 2)),
            self.clip_refiner_lambda,
            self.clip_refiner_scd_weight,
        )

        self.merge_token_proportion = [int(_l) / 100 for _l in config.merge_token_proportion.split('-')]
        self.frame_pos = config.frame_pos
        
        self.merge_layer = [int(_l) for _l in config.merge_layer.split('-')]
        self.merge_frame_num = [int(_l) for _l in config.merge_frame_num.split('-')]
        self.TVPt_Video_Positional_embedding = []
        if config.base_encoder == "ViT-B/32":
            patch_num = 50
        else:
            patch_num = 197
        cls_num = 1
        frame_num = config.max_frames
        self.patch_list = [patch_num]
        self.frame_list = [frame_num]
        for _l in range(12):
            if _l not in self.merge_layer:
                if _l < self.merge_layer[0]:
                    patch_num = patch_num - self.tome_r
                    self.patch_list.append(patch_num)
                    self.frame_list.append(frame_num)
                else:
                    patch_num = patch_num - int(patch_num * self.merge_token_proportion[1])
                    self.patch_list.append(patch_num)
                    self.frame_list.append(frame_num)
            else:
                M_frame_num = self.merge_frame_num.pop(0)
                M_token_num = int(patch_num * M_frame_num * self.merge_token_proportion[0])

                assert frame_num % M_frame_num == 0
                patch_num = patch_num * M_frame_num - M_token_num
                cls_num = cls_num * M_frame_num
                frame_num = frame_num // M_frame_num
                self.patch_list.append(patch_num)
                self.frame_list.append(frame_num)

                patch_num = patch_num - int(patch_num * self.merge_token_proportion[1])
                self.patch_list.append(patch_num)
                self.frame_list.append(frame_num)
        
        self.merge_layer = [int(_l) for _l in config.merge_layer.split('-')]
        self.merge_frame_num = [int(_l) for _l in config.merge_frame_num.split('-')]
            
        tome_patch(self.clip, trace_source=self.tome_tracesource, prop_attn=self.tome_propattn)
        
    def forward(self, text_ids, text_mask, video, video_mask=None, idx=None, global_step=0):
        text_ids = text_ids.view(-1, text_ids.shape[-1])
        text_mask = text_mask.view(-1, text_mask.shape[-1])
        video_mask = video_mask.view(-1, video_mask.shape[-1])
        video = torch.as_tensor(video).float()
        if len(video.size()) == 5:
            b, n_v, d, h, w = video.shape
            video = video.view(b * n_v, d, h, w)
        else:
            b, pair, bs, ts, channel, h, w = video.shape
            video = video.view(b * pair * bs * ts, channel, h, w)

        cls = self.get_text_feat(text_ids, text_mask)
        video_feat = self.get_video_feat(video, video_mask)
        refiner_loss = None
        if isinstance(video_feat, tuple):
            video_feat, refiner_loss = video_feat
        
        cls = allgather(cls, self.config)
        video_feat = allgather(video_feat, self.config)
        if self.config.world_size > 1:
            distributed_barrier_on_current_device()
        
        logit_scale = self.clip.logit_scale.exp()
        loss = 0.
        
        t_feat = cls / cls.norm(dim=-1, keepdim=True)
        v_feat = video_feat / video_feat.norm(dim=-1, keepdim=True)

        t2v_logits = torch.einsum('td,vd->tv', [t_feat, v_feat])

        loss_t2v = self.loss_fct(t2v_logits * logit_scale)
        loss_v2t = self.loss_fct(t2v_logits.T * logit_scale)
        loss = (loss_t2v + loss_v2t) / 2
        if refiner_loss is not None and self.clip_refiner_scd_weight > 0:
            loss = loss + self.clip_refiner_scd_weight * refiner_loss
        
        return loss

    def stage1_eval(self, text_ids, text_mask, video, video_mask=None, idx=None, global_step=0):
        text_ids = text_ids.view(-1, text_ids.shape[-1])
        text_mask = text_mask.view(-1, text_mask.shape[-1])
        video_mask = video_mask.view(-1, video_mask.shape[-1])
        video = torch.as_tensor(video).float()
        if len(video.size()) == 5:
            b, n_v, d, h, w = video.shape
            video = video.view(b * n_v, d, h, w)
        else:
            b, pair, bs, ts, channel, h, w = video.shape
            video = video.view(b * pair * bs * ts, channel, h, w)

        cls = self.get_text_feat(text_ids, text_mask)
        video = self.get_video_feat(video, video_mask)
        if isinstance(video, tuple):
            video, _ = video

        return cls, video

    def stage2_eval(self, cls, text_mask, video_feat, video_mask):
        logit_scale = self.clip.logit_scale.exp()
        
        t_feat = cls / cls.norm(dim=-1, keepdim=True) 
        v_feat = video_feat / video_feat.norm(dim=-1, keepdim=True) 

        t2v_logits = torch.einsum('td,vd->tv', [t_feat, v_feat])
        
        return t2v_logits * logit_scale

    def _get_resclip_bounds(self):
        if not (self.use_claa or self.use_sap or self.clip_refiner is not None):
            return None, None
        if len(self.merge_layer) == 0:
            return None, None

        target = self.merge_layer[0] - 1
        if target < 0 or target >= len(self.clip.visual.transformer.resblocks):
            return None, None

        start = max(0, target - self.resclip_window + 1)
        if self.pamr is not None and self.use_sap and start > 0:
            if not self._pamr_window_logged:
                logger.info(
                    "PAMR requires an unmerged patch grid, expanding ResCLIP window start from %d to 0.",
                    start,
                )
                self._pamr_window_logged = True
            start = 0
        return start, target

    @staticmethod
    def _get_token_grid_size(token_states):
        patch_count = token_states.size(1) - 1
        if patch_count <= 0:
            return None

        grid_size = int(round(math.sqrt(patch_count)))
        if grid_size * grid_size != patch_count:
            return None

        return grid_size

    def _apply_pamr_to_confidence(self, confidence, frames, token_states):
        if self.pamr is None or frames is None:
            return confidence

        grid_size = self._get_token_grid_size(token_states)
        if grid_size is None:
            return confidence

        confidence_map = confidence.reshape(confidence.size(0), 1, grid_size, grid_size)
        refined = self.pamr(frames.float(), confidence_map.to(frames.dtype)).to(confidence.dtype)
        refined = F.interpolate(refined, size=(grid_size, grid_size), mode='bilinear', align_corners=False)
        refined = refined.flatten(2).squeeze(1).clamp_min(0)
        refined = refined / (refined.amax(dim=-1, keepdim=True) + 1e-6)

        return refined

    def _build_sap_prior(self, token_states, frames=None):
        with torch.no_grad():
            visual_hidden = self.clip.visual.ln_post(token_states.detach()) @ self.clip.visual.proj
            token_hidden = visual_hidden[:, 1:, :].float()
            if token_hidden.numel() == 0 or token_states.size(1) <= 1:
                return None

            token_hidden = F.normalize(token_hidden, dim=-1)
            sim = token_hidden @ token_hidden.transpose(1, 2)
            sim = sim.clamp_min(0)
            cls_hidden = F.normalize(visual_hidden[:, :1, :].float(), dim=-1)
            confidence = (token_hidden * cls_hidden).sum(dim=-1).clamp_min(0)
            confidence = self._apply_pamr_to_confidence(confidence, frames, token_states)
            confidence = confidence / (confidence.amax(dim=-1, keepdim=True) + 1e-6)
            sim = sim * confidence[:, :, None] * confidence[:, None, :]

            eye = torch.eye(sim.size(-1), device=sim.device, dtype=sim.dtype).unsqueeze(0)
            sim = sim + eye

            if self.sap_temp_thd > 0:
                keep_mask = (confidence >= self.sap_temp_thd).to(sim.dtype)
                sim = sim * keep_mask[:, :, None] * keep_mask[:, None, :]
                sim = sim + eye

            sim = sim / (sim.amax(dim=(-1, -2), keepdim=True) + 1e-6)
            sim = sim / (sim.sum(dim=-1, keepdim=True) + 1e-6)

            attn_prior = token_states.new_zeros((token_states.size(0), 1, token_states.size(1), token_states.size(1)))
            attn_prior[:, :, 1:, 1:] = sim.unsqueeze(1).to(token_states.dtype)
            attn_prior[:, :, 0, 0] = 1

        return attn_prior.detach()

    def _build_clip_refiner_prior(self, token_states, return_loss):
        if self.clip_refiner is None:
            return None, None
        return self.clip_refiner(
            token_states,
            cls_num=self.clip._tome_info["cls_num"],
            return_loss=return_loss,
        )

    @staticmethod
    def _blend_attention_priors(base_prior, refiner_prior, refiner_lambda):
        if refiner_prior is None:
            return base_prior
        if base_prior is None:
            return refiner_prior

        refiner_lambda = min(max(float(refiner_lambda), 0.0), 1.0)
        blended = (1.0 - refiner_lambda) * base_prior + refiner_lambda * refiner_prior
        blended = blended / (blended.sum(dim=-1, keepdim=True) + 1e-6)
        return blended

    def get_text_feat(self, text_ids, orig_mask):
        b = text_ids.size(0)
        x = self.clip.token_embedding(text_ids) 
        max_t_len = x.size(1)
        pos_emd = self.clip.positional_embedding[:max_t_len, :]
        x = x + pos_emd

        mask = orig_mask
        text_length = max_t_len
        attn_mask = self.clip.build_attention_mask(text_length).repeat(x.size(0), 1, 1).to(mask.device)
        inf = torch.zeros((text_length, text_length)).fill_(float("-inf")).repeat(x.size(0), 1, 1).to(mask.device)
        mask = mask.unsqueeze(1).expand(-1, mask.size(1), -1)
        attn_mask = torch.where(mask>0, attn_mask, inf)
    
        x = self.clip.transformer(x, attn_mask)

        hidden = self.clip.ln_final(x) @ self.clip.text_projection
        cls = hidden[torch.arange(hidden.shape[0]), text_ids.argmax(dim=-1)]

        cls = cls.float()
        cls = cls.view(b, -1, cls.size(-1)).squeeze(1)
        return cls

    def get_video_feat(self, video, video_mask):
        self.clip._tome_info["size"] = None
        self.clip._tome_info["source"] = None
        self.clip._tome_info["cls_num"] = 1
        self.clip._tome_info["frame_num"] = self.frame_list[0]
        self.clip._tome_info["token_num"] = self.patch_list[0]

        self.merge_frame_num = [int(_l) for _l in self.config.merge_frame_num.split('-')]
        
        b, n_f = video_mask.size()
        org_n_f = n_f
        x = video
            
        x = self.clip.visual.conv1(x)  

        x = x.reshape(x.shape[0], x.shape[1], -1) 
        x = x.permute(0, 2, 1)  
        x = torch.cat(
            [self.clip.visual.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device),
             x], dim=1)  
        
        x = x + self.clip.visual.positional_embedding.to(x.dtype)
        x = self.clip.visual.ln_pre(x)
        
        _, token_len, d_v = x.size()

        resclip_start, resclip_target = self._get_resclip_bounds()
        claa_attn_maps = []
        refiner_attn_prior = None
        refiner_scd_loss = None
        for res_i, res_block in enumerate(self.clip.visual.transformer.resblocks):
            if resclip_target is not None and resclip_start <= res_i <= resclip_target:
                if res_i < resclip_target:
                    if self.use_claa:
                        x, attn_map = res_block(
                            x,
                            M_frame_num=1,
                            M_token_num=[0],
                            return_attn=True,
                        )
                        claa_attn_maps.append(attn_map.detach())
                    else:
                        x = res_block(x, M_frame_num=1, M_token_num=[0])
                else:
                    attn_claa = None
                    if self.use_claa and len(claa_attn_maps) > 0:
                        attn_claa = torch.stack(claa_attn_maps, dim=0).mean(dim=0).detach()
                    attn_sap = self._build_sap_prior(x, video) if self.use_sap else None
                    refiner_attn_prior, refiner_scd_loss = self._build_clip_refiner_prior(
                        x,
                        return_loss=self.training,
                    )
                    attn_sap = self._blend_attention_priors(
                        attn_sap,
                        refiner_attn_prior,
                        self.clip_refiner_lambda,
                    )
                    x = res_block(
                        x,
                        M_frame_num=1,
                        M_token_num=[0],
                        attn_claa=attn_claa,
                        attn_sap=attn_sap,
                        attn_claa_weights=self.attn_claa_weights,
                        attn_sap_weights=self.attn_sap_weights,
                    )
                continue

            if res_i not in self.merge_layer:
                if res_i < self.merge_layer[0]:
                    x = res_block(x, M_frame_num=1, M_token_num=[self.tome_r])
                else:
                    M_token_num = int(self.clip._tome_info["token_num"] * self.merge_token_proportion[1])
                    M_token_num = min((self.clip._tome_info["token_num"] - self.clip._tome_info["cls_num"]) // 2, M_token_num)
                    x = res_block(x, M_frame_num=1, M_token_num=[M_token_num])
            else:
                M_frame_num = self.merge_frame_num.pop(0)
                M_token_num_0 = int(self.clip._tome_info["token_num"] * M_frame_num * self.merge_token_proportion[0])
                M_token_num_0 = min((self.clip._tome_info["token_num"] - self.clip._tome_info["cls_num"]) * M_frame_num // 2, M_token_num_0)
                M_token_num_1 = int((self.clip._tome_info["token_num"] * M_frame_num - M_token_num_0) * self.merge_token_proportion[1])
                M_token_num_1 = min( ( (self.clip._tome_info["token_num"] - self.clip._tome_info["cls_num"]) * M_frame_num - M_token_num_0) // 2, M_token_num_1)
                    
                x = res_block(x, M_frame_num=M_frame_num, M_token_num=[M_token_num_0, M_token_num_1], frame_pos=self.frame_pos)
        
        n_f = self.clip._tome_info["frame_num"]
        token_len = self.clip._tome_info["token_num"]
        cls_num = self.clip._tome_info["cls_num"]
        x = x.reshape(b, n_f, token_len, d_v)[:, :, :cls_num, :].reshape(b, org_n_f, d_v)
        hidden = self.clip.visual.ln_post(x) @ self.clip.visual.proj
        video_feat = hidden.float()
        
        video_feat = video_feat.contiguous()
        
        video_feat = video_feat / video_feat.norm(dim=-1, keepdim=True)
        video_feat = self.get_video_avg_feat(video_feat, video_mask)
        
        if refiner_scd_loss is not None:
            return video_feat, refiner_scd_loss

        return video_feat

    def get_video_avg_feat(self, video_feat, video_mask):
        video_mask_un = video_mask.to(dtype=torch.float).unsqueeze(-1)
        video_feat = video_feat * video_mask_un
        video_mask_un_sum = torch.sum(video_mask_un, dim=1, dtype=torch.float)
        video_mask_un_sum[video_mask_un_sum == 0.] = 1.
        video_feat = torch.sum(video_feat, dim=1) / video_mask_un_sum
        return video_feat

    @property
    def dtype(self):
        """
        :obj:`torch.dtype`: The dtype of the module (assuming that all the module parameters have the same dtype).
        """
        try:
            return next(self.parameters()).dtype
        except StopIteration:
            def find_tensor_attributes(module: nn.Module):
                tuples = [(k, v) for k, v in module.__dict__.items() if torch.is_tensor(v)]
                return tuples

            gen = self._named_members(get_members_fn=find_tensor_attributes)
            first_tuple = next(gen)
            return first_tuple[1].dtype

    def init_weights(self, module):
        """ Initialize the weights.
        """
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
        elif isinstance(module, LayerNorm):
            if 'beta' in dir(module) and 'gamma' in dir(module):
                module.beta.data.zero_()
                module.gamma.data.fill_(1.0)
            else:
                module.bias.data.zero_()
                module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()
