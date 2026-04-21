from __future__ import annotations

from typing import Tuple, List

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18, ResNet18_Weights

try:
    import timm
    TIMM_AVAILABLE = True
except Exception:
    timm = None
    TIMM_AVAILABLE = False


class ImageNetBackbone(nn.Module):
    def __init__(self, backbone: str = "resnet18", pretrained: bool = True):
        super().__init__()
        self.backbone_name = backbone

        if backbone == "resnet18":
            weights = ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
            model = resnet18(weights=weights)
            feat_dim = model.fc.in_features
            model.fc = nn.Identity()
            self.model = model
            self.feature_dim = feat_dim
            return

        if backbone == "vit_tiny":
            if not TIMM_AVAILABLE:
                raise RuntimeError("vit_tiny requiere timm instalado")
            model = timm.create_model(
                "vit_tiny_patch16_224.augreg_in21k_ft_in1k",
                pretrained=pretrained,
                num_classes=0,
            )
            self.model = model
            self.feature_dim = int(model.num_features)
            return

        raise ValueError(f"Backbone no soportado: {backbone}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


class SpeechImageModel(nn.Module):
    def __init__(
        self,
        backbone: str,
        projection_type: str,
        n_meg_channels: int = 306,
        img_size: int = 224,
        pretrained: bool = True,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.backbone_name = backbone
        self.projection_type = projection_type
        self.img_size = img_size

        if projection_type == "learnable_1x1_projection":
            self.sensor_projection = nn.Conv2d(
                in_channels=n_meg_channels,
                out_channels=3,
                kernel_size=1,
                bias=False,
            )
            nn.init.xavier_uniform_(self.sensor_projection.weight)
        else:
            self.sensor_projection = None

        self.backbone = ImageNetBackbone(backbone=backbone, pretrained=pretrained)
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(self.backbone.feature_dim, 2),
        )

    def set_finetune_mode(self, mode: str) -> None:
        for p in self.backbone.parameters():
            p.requires_grad = False

        if mode == "frozen":
            return

        if mode == "partial_ft":
            if self.backbone_name == "resnet18":
                for p in self.backbone.model.layer4.parameters():
                    p.requires_grad = True
            elif self.backbone_name == "vit_tiny":
                blocks = self.backbone.model.blocks
                for p in blocks[-1].parameters():
                    p.requires_grad = True
                for p in self.backbone.model.norm.parameters():
                    p.requires_grad = True
            return

        if mode == "full_ft":
            for p in self.backbone.parameters():
                p.requires_grad = True
            return

        raise ValueError(f"Modo de fine-tuning desconocido: {mode}")

    def get_optimizer_groups(self, lr_head: float, lr_backbone: float) -> List[dict]:
        head_params = list(self.classifier.parameters())
        if self.sensor_projection is not None:
            head_params += list(self.sensor_projection.parameters())

        backbone_params = [p for p in self.backbone.parameters() if p.requires_grad]
        return [
            {"params": head_params, "lr": lr_head},
            {"params": backbone_params, "lr": lr_backbone},
        ]

    def _maybe_project(self, x: torch.Tensor) -> torch.Tensor:
        if self.sensor_projection is None:
            return x
        x = self.sensor_projection(x)
        x = F.interpolate(x, size=(self.img_size, self.img_size), mode="bilinear", align_corners=False)
        x_min = x.amin(dim=(2, 3), keepdim=True)
        x_max = x.amax(dim=(2, 3), keepdim=True)
        x = (x - x_min) / (x_max - x_min + 1e-8)
        mean = torch.tensor([0.485, 0.456, 0.406], device=x.device, dtype=x.dtype).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], device=x.device, dtype=x.dtype).view(1, 3, 1, 1)
        x = (x - mean) / std
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self._maybe_project(x)
        feat = self.backbone(x)
        return self.classifier(feat)
