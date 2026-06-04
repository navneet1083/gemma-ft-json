"""Image preprocessing.

KEY CHOICE — aspect-ratio-preserving "letterbox" resize:
    Tables are wide, grid-aligned. Squashing a page to a square shears columns
    and smears cell boundaries (a big avoidable source of error). So we scale the
    longest side to `image_size` and pad the rest, preserving true geometry
    (mirrors why SigLIP-2 NaFlex keeps native aspect ratio for documents).

Plain PIL + torch only (no torchvision pretrained transforms -> no download).
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
from PIL import Image

# ImageNet-style normalization constants (fixed numbers, no download). Switch to
# (0.5,0.5,0.5) if you load a local SigLIP encoder.
_MEAN = (0.485, 0.456, 0.406)
_STD = (0.229, 0.224, 0.225)


@dataclass
class ImageTransform:
    image_size: int = 384
    mean: tuple = _MEAN
    std: tuple = _STD

    def __call__(self, img: Image.Image) -> torch.Tensor:
        """Return a normalized CHW float tensor [3, image_size, image_size]."""
        img = img.convert("RGB")
        w, h = img.size
        scale = self.image_size / max(w, h)
        new_w, new_h = max(1, round(w * scale)), max(1, round(h * scale))
        img = img.resize((new_w, new_h), Image.BILINEAR)

        canvas = Image.new("RGB", (self.image_size, self.image_size), (128, 128, 128))
        canvas.paste(img, ((self.image_size - new_w) // 2, (self.image_size - new_h) // 2))

        # PIL -> tensor (HWC uint8 -> CHW float in [0,1]).
        t = torch.frombuffer(bytearray(canvas.tobytes()), dtype=torch.uint8)
        t = t.view(self.image_size, self.image_size, 3).permute(2, 0, 1).contiguous()
        t = t.float().div_(255.0)

        # Normalize per channel: (x - mean) / std; shapes [3,1,1] broadcast over HxW.
        mean = torch.tensor(self.mean).view(3, 1, 1)
        std = torch.tensor(self.std).view(3, 1, 1)
        return (t - mean) / std


def build_image_transform(vision_cfg) -> ImageTransform:
    return ImageTransform(image_size=vision_cfg.image_size)
