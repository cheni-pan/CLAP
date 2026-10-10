import numpy as np
import torch
from PIL import Image


_PIL_INTERPOLATION = {
    "nearest": Image.NEAREST,
    "bilinear": Image.BILINEAR,
    "bicubic": Image.BICUBIC,
}


def _size_tuple(size):
    if isinstance(size, int):
        return (size, size)
    return tuple(size)


def resize_clip(clip, size, interpolation="bilinear"):
    size = _size_tuple(size)
    resample = _PIL_INTERPOLATION.get(interpolation, Image.BILINEAR)

    if isinstance(clip[0], Image.Image):
        return [img.resize(size, resample=resample) for img in clip]
    if isinstance(clip[0], np.ndarray):
        resized = []
        for img in clip:
            pil_img = Image.fromarray(img)
            resized.append(np.asarray(pil_img.resize(size, resample=resample)))
        return resized
    raise TypeError("Expected a clip of PIL.Image or numpy.ndarray frames.")


def crop_clip(clip, y1, x1, h, w):
    if isinstance(clip[0], Image.Image):
        return [img.crop((x1, y1, x1 + w, y1 + h)) for img in clip]
    if isinstance(clip[0], np.ndarray):
        return [img[y1:y1 + h, x1:x1 + w, ...] for img in clip]
    raise TypeError("Expected a clip of PIL.Image or numpy.ndarray frames.")


def normalize(clip, mean, std):
    if not torch.is_tensor(clip):
        clip = torch.as_tensor(clip)
    mean = torch.as_tensor(mean, dtype=clip.dtype, device=clip.device).view(1, -1, 1, 1)
    std = torch.as_tensor(std, dtype=clip.dtype, device=clip.device).view(1, -1, 1, 1)
    return (clip - mean) / std
