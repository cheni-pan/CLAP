# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# timm: https://github.com/rwightman/pytorch-image-models/tree/master/timm
# --------------------------------------------------------


from typing import Tuple, Union, List

import torch
import torch.nn.functional as F
from .module_clip import Attention, ResidualAttentionBlock, CLIP
from .module_tome_merge import bipartite_soft_matching, merge_source, merge_wavg
import logging

logger = logging.getLogger(__name__)

### modified from ToMe
class ToMeBlock(ResidualAttentionBlock):

    def forward(
        self,
        x: torch.Tensor,
        M_frame_num: int = 1,
        M_token_num: List[int] = [2],
        frame_pos: int = 0,
        return_attn: bool = False,
        attn_claa: torch.Tensor = None,
        attn_sap: torch.Tensor = None,
        attn_claa_weights: Tuple[float, float] = None,
        attn_sap_weights: Tuple[float, float] = None,
    ) -> torch.Tensor:
        r_f = M_frame_num
        if r_f > 1:
            r = M_token_num[0]
            metric = x.detach()
            bsz, token_num, embed_dim = x.shape
            cls_num = self._tome_info["cls_num"]
            
            if self._tome_info["size"] is None:
                self._tome_info["size"] = torch.ones_like(x[..., 0, None])
            
            x = x.reshape(bsz // r_f, r_f, token_num, embed_dim)
            metric = metric.reshape(bsz // r_f, r_f, token_num, embed_dim)
            info_size = self._tome_info["size"].reshape(bsz // r_f, r_f, token_num, 1)
            
            x_cls = x[:, :, :cls_num, :].reshape(bsz // r_f, -1, embed_dim)
            x_patch = x[:, :, cls_num:, :].reshape(bsz // r_f, -1, embed_dim)
            x = torch.cat([x_cls, x_patch], dim=1)
            
            metric_cls = metric[:, :, :cls_num, :].reshape(bsz // r_f, -1, embed_dim)
            metric_patch = metric[:, :, cls_num:, :].reshape(bsz // r_f, -1, embed_dim)
            metric = torch.cat([metric_cls, metric_patch], dim=1)
            
            info_size_cls = info_size[:, :, :cls_num, :].reshape(bsz // r_f, -1, 1)
            info_size_patch = info_size[:, :, cls_num:, :].reshape(bsz // r_f, -1, 1)
            self._tome_info["size"] = torch.cat([info_size_cls, info_size_patch], dim=1)

            if frame_pos == 1:
                Position_Embed = self.TVPt_Video_Positional_embedding.reshape(1, self._tome_info["frame_num"] // r_f, r_f, 1, embed_dim)
                Position_Embed = Position_Embed.expand(bsz // self._tome_info["frame_num"], -1, -1, token_num, -1).reshape(bsz // r_f, r_f, token_num, embed_dim)

                Position_Embed_cls = Position_Embed[:, :, :cls_num, :].reshape(bsz // r_f, -1, embed_dim)
                Position_Embed_patch = Position_Embed[:, :, cls_num:, :].reshape(bsz // r_f, -1, embed_dim)
                Position_Embed = torch.cat([Position_Embed_cls, Position_Embed_patch], dim=1)

            self._tome_info["cls_num"] = cls_num * r_f
            self._tome_info["token_num"] = token_num * r_f
            self._tome_info["frame_num"] = self._tome_info["frame_num"] // r_f
            if r > 0:
                # Apply ToMe here
                merge, _ = bipartite_soft_matching(
                    metric,
                    r,
                    cls_num=self._tome_info["cls_num"],
                    r_f=r_f
                )
                if self._tome_info["trace_source"]:
                    self._tome_info["source"] = merge_source(
                        merge, x, self._tome_info["source"]
                    )
                ### add position embedd
                if frame_pos == 1:
                    x = x + Position_Embed
                
                x, self._tome_info["size"] = merge_wavg(merge, x, self._tome_info["size"])
                self._tome_info["token_num"] = self._tome_info["token_num"] - r
        
        attn_size = self._tome_info["size"] if self._tome_info["prop_attn"] else None
        attn_outputs = self.attn(
            self.ln_1(x),
            attn_size,
            return_attn=return_attn,
            attn_claa=attn_claa,
            attn_sap=attn_sap,
            attn_claa_weights=attn_claa_weights,
            attn_sap_weights=attn_sap_weights,
        )
        if return_attn:
            x_attn, metric, attn_map = attn_outputs
        else:
            x_attn, metric = attn_outputs
        x = x + x_attn

        r = M_token_num[-1]
        if r > 0:
            # Apply ToMe here
            merge, _ = bipartite_soft_matching(
                metric,
                r,
                cls_num=self._tome_info["cls_num"],
                r_f=1
            )
            if self._tome_info["trace_source"]:
                self._tome_info["source"] = merge_source(
                    merge, x, self._tome_info["source"]
                )
            x, self._tome_info["size"] = merge_wavg(merge, x, self._tome_info["size"])
            self._tome_info["token_num"] = self._tome_info["token_num"] - r
        
        x = x + self.mlp(self.ln_2(x))

        if return_attn:
            return x, attn_map

        return x

class ToMeAttention(Attention):

    def forward(
        self,
        x: torch.Tensor,
        size: torch.Tensor = None,
        return_attn: bool = False,
        attn_claa: torch.Tensor = None,
        attn_sap: torch.Tensor = None,
        attn_claa_weights: Tuple[float, float] = None,
        attn_sap_weights: Tuple[float, float] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        bsz, tgt_len, embed_dim = x.size()

        def _prepare_prior(prior: torch.Tensor, name: str, dtype: torch.dtype) -> torch.Tensor:
            prior = prior.to(device=x.device, dtype=dtype)
            if prior.dim() == 3:
                prior = prior.unsqueeze(1)
            if prior.dim() != 4:
                raise ValueError(f"{name} must be a 3D or 4D tensor, but got shape {tuple(prior.shape)}")
            if prior.size(0) != bsz or prior.size(-2) != tgt_len or prior.size(-1) != tgt_len:
                raise ValueError(
                    f"{name} shape mismatch: expected batch={bsz}, tokens={tgt_len}, got {tuple(prior.shape)}"
                )
            if prior.size(1) == 1:
                prior = prior.expand(-1, self.num_heads, -1, -1)
            elif prior.size(1) != self.num_heads:
                raise ValueError(
                    f"{name} head dimension mismatch: expected 1 or {self.num_heads}, got {prior.size(1)}"
                )
            return prior
        
        qkv = F.linear(x, self.in_proj_weight, self.in_proj_bias).reshape(bsz, tgt_len, 3, self.num_heads, embed_dim // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)

        qkv_delta = F.linear(x, self.TVPt_LoRA_a)
        qkv_delta = F.linear(qkv_delta, self.TVPt_LoRA_b).reshape(bsz, tgt_len, 3, self.num_heads, embed_dim // self.num_heads).permute(2, 0, 3, 1, 4)
        q_delta, k_delta, v_delta = qkv_delta.unbind(0)
        q,k,v = q+q_delta,k+k_delta,v+v_delta
        
        q = q * self.scaling
        attn_logits = (q @ k.transpose(-2, -1))
        if size is not None:
            attn_logits = attn_logits + size.log()[:, None, None, :, 0]

        if attn_sap is not None:
            attn_sap = _prepare_prior(attn_sap, "attn_sap", attn_logits.dtype)
            if attn_sap_weights is None:
                attn_sap_weights = (1.0, 0.5)
            tau_sap, lambda_sap = attn_sap_weights
            attn_logits = tau_sap * ((1.0 - lambda_sap) * attn_logits + lambda_sap * attn_sap)

        attn = attn_logits.softmax(dim=-1)

        if attn_claa is not None:
            attn_claa = _prepare_prior(attn_claa, "attn_claa", attn.dtype)
            if attn_claa_weights is None:
                attn_claa_weights = (1.0, 0.5)
            tau_claa, lambda_claa = attn_claa_weights
            attn = tau_claa * ((1.0 - lambda_claa) * attn + lambda_claa * attn_claa)
            attn = attn / (attn.sum(dim=-1, keepdim=True) + 1e-6)

        x = (attn @ v).transpose(1, 2).reshape(bsz, tgt_len, embed_dim)
        x = F.linear(x, self.out_proj.weight, self.out_proj.bias)

        metric = k.mean(1)
        if return_attn:
            return x, metric, attn

        return x, metric

def apply_patch(
    model: CLIP, trace_source: bool = False, prop_attn: bool = True
):
    model._tome_info = {
        "frame_num": 0,
        "token_num": 0,
        "cls_num": 0,
        "size": None,
        "source": None,
        "trace_source": trace_source,
        "prop_attn": prop_attn
    }

    for module in model.visual.transformer.modules():
        if isinstance(module, ResidualAttentionBlock):
            module.__class__ = ToMeBlock
            module._tome_info = model._tome_info
        elif isinstance(module, Attention):
            module.__class__ = ToMeAttention
            module._tome_info = model._tome_info
