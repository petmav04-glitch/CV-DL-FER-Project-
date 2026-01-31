# src/explainability/gradcam.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image


@dataclass
class GradCAMResult:
    cam: np.ndarray        
    class_idx: int
    score: float           


class GradCAM:
  

    def __init__(self, model: torch.nn.Module, target_layer: torch.nn.Module):
        self.model = model
        self.target_layer = target_layer

        self._acts: Optional[torch.Tensor] = None
        self._grads: Optional[torch.Tensor] = None

        self._fwd = target_layer.register_forward_hook(self._forward_hook)
        self._bwd = target_layer.register_full_backward_hook(self._backward_hook)

    def close(self) -> None:
        self._fwd.remove()
        self._bwd.remove()

    def _forward_hook(self, module, inp, out):
        self._acts = out

    def _backward_hook(self, module, grad_in, grad_out):
        self._grads = grad_out[0]

    def __call__(self, x: torch.Tensor, class_idx: Optional[int] = None) -> GradCAMResult:
        if x.ndim != 4 or x.shape[0] != 1:
            raise ValueError(f"Expected x shape (1,C,H,W), got {tuple(x.shape)}")

        self.model.eval()
        x = x.requires_grad_(True)

        logits = self.model(x)  # (1, num_classes)
        if logits.ndim != 2 or logits.shape[0] != 1:
            raise ValueError(f"Expected logits shape (1,num_classes), got {tuple(logits.shape)}")

        if class_idx is None:
            class_idx = int(torch.argmax(logits, dim=1).item())

        score = logits[0, class_idx]
        self.model.zero_grad(set_to_none=True)
        score.backward()

        if self._acts is None or self._grads is None:
            raise RuntimeError("No activations/grads captured. Check target_layer.")

     
        w = self._grads.mean(dim=(2, 3), keepdim=True)          # (1,C,1,1)
        cam = (w * self._acts).sum(dim=1, keepdim=True)         # (1,1,h,w)
        cam = F.relu(cam)

 
        cam = F.interpolate(cam, size=x.shape[-2:], mode="bilinear", align_corners=False)
        cam = cam[0, 0].detach().cpu().numpy()

        # normalize 0,1

        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-8)

        return GradCAMResult(cam=cam.astype(np.float32), class_idx=class_idx, score=float(score.item()))


def overlay_red(image: Image.Image, cam: np.ndarray, alpha: float = 0.45) -> Image.Image:
    
    img = image.convert("RGB") #only RGB to visualize
    arr = np.asarray(img).astype(np.float32) / 255.0
    h, w = arr.shape[:2]

    if cam.shape != (h, w):
        raise ValueError(f"cam shape {cam.shape} != image shape {(h, w)}")

    heat = np.zeros_like(arr)
    heat[..., 0] = np.clip(cam, 0.0, 1.0)

    out = (1.0 - alpha) * arr + alpha * heat
    out = np.clip(out, 0.0, 1.0)
    return Image.fromarray((out * 255).astype(np.uint8))


def load_checkpoint_flexible(model: torch.nn.Module, ckpt_path: str, device: torch.device) -> None:
  
    ckpt = torch.load(ckpt_path, map_location=device)

    if isinstance(ckpt, dict):
        for k in ("state_dict", "model_state_dict", "model", "net", "weights"):
            if k in ckpt and isinstance(ckpt[k], dict):
                ckpt = ckpt[k]
                break

    if isinstance(ckpt, dict) and any(key.startswith("module.") for key in ckpt.keys()):
        ckpt = {k.replace("module.", "", 1): v for k, v in ckpt.items()}

    model.load_state_dict(ckpt, strict=True)
