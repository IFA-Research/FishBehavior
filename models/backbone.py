import torch
import torch.nn as nn


# Short names used in configs. The value is what the loader actually builds.
BACKBONE_ALIASES = {
    "mobilevit_small": "apple/mobilevit-small",
    "swin_t": "swin_tiny_patch4_window7_224",
    "swin": "swin_tiny_patch4_window7_224",
    "mvitv2": "mvitv2_small",
    "mvitv2_s": "mvitv2_small",
    "mobilevitv3": "mobilevitv3_xxs",
    "mobilevitv3_xxs": "mobilevitv3_xxs",
    "mobilevitv3_small": "mobilevitv3_small",
}

HF_FEATURE_DIM = {
    "apple/mobilevit-small": 640,
    "apple/mobilevit-x-small": 384,
    "apple/mobilevit-xx-small": 320,
}


class FrameBackbone(nn.Module):
    """Image backbone. forward(x) with x shaped (B, 3, H, W) returns (B, out_dim)."""

    def __init__(self, name, pretrained=True):
        super().__init__()
        resolved = BACKBONE_ALIASES.get(name, name)
        self.name = resolved

        if resolved.startswith("mobilevitv3"):
            if resolved != "mobilevitv3_xxs":
                raise ValueError(
                    "Only MobileViTv3-XXS has the official ImageNet checkpoint in this package. "
                    "Use backbone mobilevitv3_xxs."
                )
            from .mobilevitv3 import MobileViTv3_v1, load_mobilevitv3_xxs_imagenet
            net = MobileViTv3_v1(image_size=(224, 224), mode="xx_small", num_classes=1)
            if pretrained:
                load_mobilevitv3_xxs_imagenet(net)
            self.out_dim = net.out.in_features
            net.out = nn.Identity()
            self.net = net
            self.kind = "mobilevitv3"
        elif resolved.startswith("apple/") or "mobilevit-" in resolved:
            from transformers import MobileViTForImageClassification
            net = MobileViTForImageClassification.from_pretrained(resolved)
            net.classifier = nn.Identity()
            self.net = net
            self.kind = "hf"
            self.out_dim = HF_FEATURE_DIM.get(resolved, 640)
        else:
            try:
                import timm
            except ImportError as exc:
                raise ImportError(
                    "Comparison backbones need timm. Install it with: pip install timm"
                ) from exc
            self.net = timm.create_model(resolved, pretrained=pretrained, num_classes=0)
            self.kind = "timm"
            self.out_dim = int(self.net.num_features)

        for param in self.parameters():
            param.requires_grad = True

    def forward(self, x):
        if self.kind == "hf":
            features = self.net(x).logits
        else:
            features = self.net(x)
        if features.ndim == 4:
            features = features.mean(dim=(2, 3))
        elif features.ndim == 3:
            features = features.mean(dim=1)
        return features


def build_backbone(name, pretrained=True):
    backbone = FrameBackbone(name, pretrained=pretrained)
    return backbone, backbone.out_dim
