"""Task 12: the actual frozen embedding model. A pretrained (ImageNet)
MobileNetV3-Small, used ONLY in eval mode with gradients disabled -- never
fine-tuned, never updated from this project's own data. Isolated in its own
module so the appearance-anchor update/freeze RULES (`oatm.memory.appearance`)
can be unit-tested without ever importing torch or touching a real image.
"""
from __future__ import annotations

import numpy as np
import torch
from PIL import Image
from torchvision import transforms
from torchvision.models import MobileNet_V3_Small_Weights, mobilenet_v3_small

_PREPROCESS = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


class AppearanceEmbedder:
    """Wraps a frozen, pretrained MobileNetV3-Small as a generic appearance
    feature extractor. `embed_crop` returns an L2-normalized 576-d vector
    from the network's final pooled features, before its classifier head
    (the classifier was trained for ImageNet's 1000 classes, not re-id --
    the pooled backbone features are the reusable, general-purpose part)."""

    def __init__(self) -> None:
        model = mobilenet_v3_small(weights=MobileNet_V3_Small_Weights.IMAGENET1K_V1)
        model.eval()
        for p in model.parameters():
            p.requires_grad_(False)
        self._features = model.features
        self._avgpool = model.avgpool

    def embed_crop(self, image: Image.Image) -> np.ndarray:
        """`image` must already be the cropped detection box (RGB)."""
        tensor = _PREPROCESS(image.convert("RGB")).unsqueeze(0)
        with torch.no_grad():
            features = self._features(tensor)
            pooled = self._avgpool(features).flatten(1)
        embedding = pooled.squeeze(0).numpy()
        norm = np.linalg.norm(embedding)
        return embedding / norm if norm > 0 else embedding
