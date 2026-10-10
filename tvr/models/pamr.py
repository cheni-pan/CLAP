# Copyright 2020 TU Darmstadt
# License: Apache 2.0
# Adapted from https://github.com/visinf/1-stage-wseg/blob/master/models/mods/pamr.py

import torch
import torch.nn as nn
import torch.nn.functional as F


class LocalAffinity(nn.Module):
    def __init__(self, dilations=None):
        super(LocalAffinity, self).__init__()
        self.dilations = dilations or [1]
        weight = self._init_aff()
        self.register_buffer("kernel", weight)

    def _init_aff(self):
        weight = torch.zeros(8, 1, 3, 3)

        for i in range(weight.size(0)):
            weight[i, 0, 1, 1] = 1

        weight[0, 0, 0, 0] = -1
        weight[1, 0, 0, 1] = -1
        weight[2, 0, 0, 2] = -1

        weight[3, 0, 1, 0] = -1
        weight[4, 0, 1, 2] = -1

        weight[5, 0, 2, 0] = -1
        weight[6, 0, 2, 1] = -1
        weight[7, 0, 2, 2] = -1
        return weight

    def forward(self, x):
        batch, channels, height, width = x.size()
        x = x.view(batch * channels, 1, height, width)

        x_aff = torch.empty(
            (channels * batch, len(self.dilations) * self.kernel.size(0), height, width),
            device=x.device,
            dtype=x.dtype,
        )
        for idx, dilation in enumerate(self.dilations):
            x_pad = F.pad(x, [dilation] * 4, mode="replicate")
            start = self.kernel.size(0) * idx
            end = self.kernel.size(0) * (idx + 1)
            x_aff[:, start:end, :, :] = F.conv2d(x_pad, self.kernel.to(x.dtype), dilation=dilation)

        return x_aff.view(batch, channels, -1, height, width)


class LocalAffinityCopy(LocalAffinity):
    def _init_aff(self):
        weight = torch.zeros(8, 1, 3, 3)

        weight[0, 0, 0, 0] = 1
        weight[1, 0, 0, 1] = 1
        weight[2, 0, 0, 2] = 1

        weight[3, 0, 1, 0] = 1
        weight[4, 0, 1, 2] = 1

        weight[5, 0, 2, 0] = 1
        weight[6, 0, 2, 1] = 1
        weight[7, 0, 2, 2] = 1
        return weight


class LocalStDev(LocalAffinity):
    def _init_aff(self):
        weight = torch.zeros(9, 1, 3, 3)

        weight[0, 0, 0, 0] = 1
        weight[1, 0, 0, 1] = 1
        weight[2, 0, 0, 2] = 1

        weight[3, 0, 1, 0] = 1
        weight[4, 0, 1, 1] = 1
        weight[5, 0, 1, 2] = 1

        weight[6, 0, 2, 0] = 1
        weight[7, 0, 2, 1] = 1
        weight[8, 0, 2, 2] = 1
        return weight

    def forward(self, x):
        x = super(LocalStDev, self).forward(x)
        return x.std(2, keepdim=True)


class LocalAffinityAbs(LocalAffinity):
    def forward(self, x):
        return torch.abs(super(LocalAffinityAbs, self).forward(x))


class PAMR(nn.Module):
    def __init__(self, num_iter=1, dilations=None):
        super(PAMR, self).__init__()
        self.num_iter = num_iter
        self.aff_x = LocalAffinityAbs(dilations or [1])
        self.aff_m = LocalAffinityCopy(dilations or [1])
        self.aff_std = LocalStDev(dilations or [1])

    def forward(self, x, mask):
        mask = F.interpolate(mask, size=x.size()[-2:], mode="bilinear", align_corners=True)

        x_std = self.aff_std(x)
        x = -self.aff_x(x) / (1e-8 + 0.1 * x_std)
        x = x.mean(1, keepdim=True)
        x = F.softmax(x, 2)

        for _ in range(self.num_iter):
            m = self.aff_m(mask)
            mask = (m * x).sum(2)

        return mask
