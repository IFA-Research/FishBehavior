import torch
import torch.nn as nn
from transformers import MobileViTForImageClassification
from .backbone import BACKBONE_ALIASES, HF_FEATURE_DIM, build_backbone
from .temporal import get_temporal_module

class FeatureStabilizationBlock(nn.Module):
    def __init__(self, in_dim=640, out_dim=512, dropout=0.1):
        super().__init__()
        self.proj1 = nn.Linear(in_dim, out_dim)
        self.proj2 = nn.Linear(out_dim, out_dim)
        self.norm = nn.LayerNorm(out_dim)
        self.dropout = nn.Dropout(dropout)
        self.residual = nn.Linear(in_dim, out_dim) if in_dim != out_dim else nn.Identity()

    def forward(self, x):
        # x: (B, T, in_dim)
        residual = self.residual(x)
        x = self.proj1(x)
        x = torch.relu(x)
        x = self.dropout(x)
        x = self.proj2(x)
        x = self.norm(x)
        return x + residual

class FishBehaviorModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        model_cfg = config["model"]
        backbone_name = model_cfg.get("backbone", "apple/mobilevit-small")
        pretrained = model_cfg.get("pretrained", True)
        resolved = BACKBONE_ALIASES.get(backbone_name, backbone_name)
        self.uses_hf_mobilevit = resolved.startswith("apple/") or "mobilevit-" in resolved
        if self.uses_hf_mobilevit:
            self.backbone = MobileViTForImageClassification.from_pretrained(resolved)
            self.backbone.classifier = nn.Identity()
            for param in self.backbone.parameters():
                param.requires_grad = True
            feature_dim = HF_FEATURE_DIM.get(resolved, 640)
        else:
            self.backbone, feature_dim = build_backbone(backbone_name, pretrained=pretrained)

        self.use_stabilization = model_cfg.get("use_feature_stabilization", False)
        stab_dim = model_cfg.get("stabilization_dim", 512)
        if self.use_stabilization:
            stab_dropout = model_cfg.get("stabilization_dropout", 0.1)
            self.stabilization = FeatureStabilizationBlock(feature_dim, stab_dim, dropout=stab_dropout)
            temporal_input_size = stab_dim
        else:
            temporal_input_size = feature_dim

        temporal_type = model_cfg.get("temporal_type", "bilstm")
        use_attn = model_cfg.get("use_temporal_attention", False)
        self.temporal = get_temporal_module(
            temporal_type,
            use_attention=use_attn,
            input_size=temporal_input_size,
            hidden_size=model_cfg.get("hidden_size", 256),
            num_layers=model_cfg.get("num_transformer_layers", 1),
            num_heads=model_cfg.get("num_heads", 8),
            dropout=model_cfg.get("dropout", 0.1)
        )

        # Determine feature dimension after temporal
        if temporal_type in ["none", "mean", "transformer"]:
            feat_dim = temporal_input_size
        elif temporal_type == "bilstm":
            feat_dim = model_cfg.get("hidden_size", 256) * 2
        elif temporal_type == "bilstm_transformer":
            feat_dim = temporal_input_size
        else:
            feat_dim = temporal_input_size

        self.fc1 = nn.Linear(feat_dim, 128)
        self.bn = nn.BatchNorm1d(128)
        self.dropout = nn.Dropout(model_cfg.get("dropout", 0.3))
        self.fc2 = nn.Linear(128, model_cfg.get("num_classes", 4))

    def forward(self, x):
        B, T, C, H, W = x.shape
        x = x.view(B * T, C, H, W)
        if self.uses_hf_mobilevit:
            features = self.backbone(x).logits
        else:
            features = self.backbone(x)
        features = features.view(B, T, -1)
        if self.use_stabilization:
            features = self.stabilization(features)
        x = self.temporal(features)
        x = self.fc1(x)
        x = self.bn(x)
        x = torch.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x
