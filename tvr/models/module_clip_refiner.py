from typing import Optional, Tuple

import torch
from torch import nn
import torch.nn.functional as F


class RefinerBlock(nn.Module):
    def __init__(self, width: int, hidden_width: int, residual_scale: float):
        super().__init__()
        self.norm = nn.LayerNorm(width)
        self.fc1 = nn.Linear(width, hidden_width)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_width, width)
        self.residual_scale = residual_scale

        nn.init.zeros_(self.fc2.weight)
        nn.init.zeros_(self.fc2.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        delta = self.fc2(self.act(self.fc1(self.norm(x))))
        return x + self.residual_scale * delta


class CLIPSpatialRefiner(nn.Module):
    """
    Lightweight CLIPRefiner-style module for dense ViT tokens.

    The original CLIPRefiner repository included only method notes in this
    workspace, so this module ports the method's usable ingredients to
    CLAP: refine dense CLIP patch tokens, extract spatial correlations,
    and expose them as a visual-centric attention prior.
    """

    def __init__(
        self,
        width: int,
        depth: int = 2,
        hidden_ratio: float = 0.25,
        residual_scale: float = 0.25,
        temperature: float = 0.07,
        use_confidence: bool = True,
    ):
        super().__init__()
        hidden_width = max(1, int(width * hidden_ratio))
        depth = max(1, int(depth))
        self.blocks = nn.ModuleList(
            [RefinerBlock(width, hidden_width, residual_scale) for _ in range(depth)]
        )
        self.temperature = max(float(temperature), 1e-6)
        self.use_confidence = use_confidence

    @staticmethod
    def _normalize_rows(x: torch.Tensor) -> torch.Tensor:
        return x / (x.sum(dim=-1, keepdim=True) + 1e-6)

    def refine_tokens(self, patch_tokens: torch.Tensor) -> torch.Tensor:
        refined = patch_tokens
        for block in self.blocks:
            refined = block(refined)
        return refined

    def _patch_prior(
        self,
        patch_tokens: torch.Tensor,
        cls_tokens: Optional[torch.Tensor],
    ) -> torch.Tensor:
        patch_hidden = F.normalize(patch_tokens.float(), dim=-1)
        corr_logits = patch_hidden @ patch_hidden.transpose(1, 2)

        if self.use_confidence and cls_tokens is not None and cls_tokens.numel() > 0:
            cls_hidden = F.normalize(cls_tokens.float().mean(dim=1, keepdim=True), dim=-1)
            confidence = (patch_hidden * cls_hidden).sum(dim=-1).clamp_min(0)
            confidence = confidence / (confidence.amax(dim=-1, keepdim=True) + 1e-6)
            corr_logits = corr_logits * confidence[:, :, None] * confidence[:, None, :]

        corr_logits = corr_logits / self.temperature
        return F.softmax(corr_logits, dim=-1)

    def forward(
        self,
        token_states: torch.Tensor,
        cls_num: int = 1,
        return_loss: bool = True,
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        if token_states.size(1) <= cls_num:
            return None, None

        cls_tokens = token_states[:, :cls_num, :]
        patch_tokens = token_states[:, cls_num:, :]
        refined_tokens = self.refine_tokens(patch_tokens)

        refiner_prior_patch = self._patch_prior(refined_tokens, cls_tokens)

        prior = token_states.new_zeros(
            (token_states.size(0), 1, token_states.size(1), token_states.size(1))
        )
        prior[:, :, cls_num:, cls_num:] = refiner_prior_patch.unsqueeze(1).to(token_states.dtype)
        cls_eye = torch.eye(cls_num, device=token_states.device, dtype=token_states.dtype)
        prior[:, :, :cls_num, :cls_num] = cls_eye.view(1, 1, cls_num, cls_num)

        if not return_loss:
            return prior, None

        teacher_prior_patch = self._patch_prior(
            patch_tokens.detach(),
            cls_tokens.detach(),
        ).detach()
        scd_loss = F.kl_div(
            (refiner_prior_patch + 1e-6).log(),
            teacher_prior_patch,
            reduction="none",
        ).sum(dim=-1).mean()

        return prior, scd_loss
