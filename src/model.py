"""
model.py — EfficientNetV2-S fine-tuned for binary chest X-ray classification.

Architecture choices:
  - Backbone : EfficientNetV2-S pretrained on ImageNet1k
  - Head     : Dropout → Linear(num_classes)
  - Loss     : BCEWithLogitsLoss with class weights (handles imbalance)
  - Precision: torch.amp mixed precision for faster training
"""

import torch
import torch.nn as nn
from torchvision import models
from torchvision.models import EfficientNet_V2_S_Weights


NUM_CLASSES = 2


# ─── Model ────────────────────────────────────────────────────────────────────
class ChestXRayModel(nn.Module):
    def __init__(self, num_classes: int = NUM_CLASSES, dropout: float = 0.4):
        super().__init__()
        # Load pretrained backbone
        self.backbone = models.efficientnet_v2_s(weights=EfficientNet_V2_S_Weights.IMAGENET1K_V1)

        # Replace classifier head
        in_features = self.backbone.classifier[1].in_features
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(p=dropout, inplace=True),
            nn.Linear(in_features, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)

    def freeze_backbone(self):
        """Freeze all layers except the classifier head (for warm-up phase)."""
        for name, param in self.backbone.named_parameters():
            if "classifier" not in name:
                param.requires_grad = False

    def unfreeze_all(self):
        """Unfreeze all parameters for full fine-tuning."""
        for param in self.parameters():
            param.requires_grad = True

    def param_groups(self, lr_backbone: float, lr_head: float) -> list:
        """Return separate param groups for backbone and head (different LRs)."""
        backbone_params = [
            p for n, p in self.named_parameters()
            if "classifier" not in n and p.requires_grad
        ]
        head_params = [
            p for n, p in self.named_parameters()
            if "classifier" in n and p.requires_grad
        ]
        return [
            {"params": backbone_params, "lr": lr_backbone},
            {"params": head_params,     "lr": lr_head},
        ]


# ─── Loss ─────────────────────────────────────────────────────────────────────
def build_criterion(class_weights: torch.Tensor, device: torch.device) -> nn.Module:
    """
    CrossEntropyLoss with class weights to penalise the majority class less.
    """
    return nn.CrossEntropyLoss(weight=class_weights.to(device))


# ─── Optimiser & Scheduler ────────────────────────────────────────────────────
def build_optimizer_scheduler(
    model: ChestXRayModel,
    lr_backbone: float = 1e-4,
    lr_head: float = 1e-3,
    weight_decay: float = 1e-4,
    epochs: int = 20,
    steps_per_epoch: int = 100,
):
    optimizer = torch.optim.AdamW(
        model.param_groups(lr_backbone, lr_head),
        weight_decay=weight_decay,
    )
    # Cosine annealing over total training steps
    total_steps = epochs * steps_per_epoch
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=total_steps, eta_min=1e-6
    )
    return optimizer, scheduler


# ─── Quick sanity check ───────────────────────────────────────────────────────
if __name__ == "__main__":
    model = ChestXRayModel()
    x = torch.randn(2, 3, 224, 224)
    out = model(x)
    print(f"Output shape: {out.shape}")   # expected: [2, 2]
    total = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"Total parameters: {total:.1f}M")
