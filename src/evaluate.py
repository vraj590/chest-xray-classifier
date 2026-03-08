"""
evaluate.py — Full evaluation suite with:
  - Accuracy, AUC-ROC, F1, Precision, Recall, Confusion Matrix
  - Grad-CAM visualisations saved to outputs/gradcam/
  - Summary report printed to console + saved as outputs/eval_report.txt

Usage:
    python src/evaluate.py \
        --data_dir  data/chest_xray \
        --ckpt_path outputs/best_model.pth \
        --output_dir outputs
"""

import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import (
    roc_auc_score, accuracy_score, f1_score,
    precision_score, recall_score, confusion_matrix, roc_curve,
)
from torch.utils.data import DataLoader

from dataset import ChestXRayDataset, get_val_transforms, CLASS_NAMES
from model import ChestXRayModel


# ─── Grad-CAM ─────────────────────────────────────────────────────────────────
class GradCAM:
    """Grad-CAM for EfficientNet (targets the last conv block)."""

    def __init__(self, model: ChestXRayModel):
        self.model = model
        self.gradients = None
        self.activations = None

        # Hook into last conv block of EfficientNetV2-S
        target_layer = model.backbone.features[-1]
        target_layer.register_forward_hook(self._save_activations)
        target_layer.register_full_backward_hook(self._save_gradients)

    def _save_activations(self, module, input, output):
        self.activations = output.detach()

    def _save_gradients(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def generate(self, image_tensor: torch.Tensor, class_idx: int) -> np.ndarray:
        """Return a (H, W) heatmap in [0, 1]."""
        self.model.eval()
        image_tensor = image_tensor.unsqueeze(0)
        image_tensor.requires_grad_(True)

        logits = self.model(image_tensor)
        self.model.zero_grad()
        logits[0, class_idx].backward()

        weights = self.gradients.mean(dim=(2, 3), keepdim=True)   # GAP
        cam    = (weights * self.activations).sum(dim=1).squeeze()
        cam    = F.relu(cam)
        cam    = cam.cpu().numpy()
        cam    = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        return cam


def save_gradcam_grid(
    images: torch.Tensor,
    labels: list,
    preds: list,
    probs: list,
    gradcam: GradCAM,
    device: torch.device,
    output_path: Path,
    n_samples: int = 8,
):
    """Save a grid of original image + Grad-CAM overlay."""
    import cv2

    fig, axes = plt.subplots(n_samples, 2, figsize=(8, n_samples * 3))
    fig.suptitle("Grad-CAM Visualisations", fontsize=14, fontweight="bold")

    mean = np.array([0.485, 0.456, 0.406])
    std  = np.array([0.229, 0.224, 0.225])

    for i in range(min(n_samples, len(images))):
        img_tensor = images[i].to(device)
        pred_cls   = preds[i]
        true_cls   = labels[i]
        prob       = probs[i]

        cam = gradcam.generate(img_tensor, class_idx=pred_cls)

        # Denormalize for display
        img_np = images[i].permute(1, 2, 0).numpy()
        img_np = (img_np * std + mean).clip(0, 1)
        img_uint8 = (img_np * 255).astype(np.uint8)

        cam_resized = cv2.resize(cam, (img_np.shape[1], img_np.shape[0]))
        heatmap = cv2.applyColorMap((cam_resized * 255).astype(np.uint8), cv2.COLORMAP_JET)
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB) / 255.0
        overlay = (0.55 * img_np + 0.45 * heatmap).clip(0, 1)

        color = "green" if pred_cls == true_cls else "red"
        title = (f"True: {CLASS_NAMES[true_cls]}\n"
                 f"Pred: {CLASS_NAMES[pred_cls]} ({prob:.2f})")

        axes[i, 0].imshow(img_np)
        axes[i, 0].set_title("Original", fontsize=8)
        axes[i, 0].axis("off")

        axes[i, 1].imshow(overlay)
        axes[i, 1].set_title(title, fontsize=8, color=color)
        axes[i, 1].axis("off")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Grad-CAM grid saved → {output_path}")


# ─── Plots ────────────────────────────────────────────────────────────────────
def plot_roc_curve(labels, probs, output_path: Path):
    fpr, tpr, _ = roc_curve(labels, probs)
    auc = roc_auc_score(labels, probs)
    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, lw=2, label=f"AUC = {auc:.4f}")
    plt.plot([0, 1], [0, 1], "k--", lw=1)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve — Chest X-Ray Classifier")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"  ROC curve saved → {output_path}")


def plot_confusion_matrix(cm: np.ndarray, output_path: Path):
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(cm, cmap="Blues")
    plt.colorbar(im, ax=ax)
    ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
    ax.set_xticklabels(CLASS_NAMES); ax.set_yticklabels(CLASS_NAMES)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    ax.set_title("Confusion Matrix")
    for r in range(2):
        for c in range(2):
            ax.text(c, r, str(cm[r, c]), ha="center", va="center",
                    color="white" if cm[r, c] > cm.max() / 2 else "black", fontsize=14)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"  Confusion matrix saved → {output_path}")


# ─── Main ─────────────────────────────────────────────────────────────────────
def evaluate(args):
    device = torch.device("cuda" if torch.cuda.is_available() else
                          "mps" if torch.backends.mps.is_available() else "cpu")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load model
    print("\n[1/4] Loading model checkpoint...")
    model = ChestXRayModel().to(device)
    ckpt = torch.load(args.ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    print(f"  Checkpoint from epoch {ckpt['epoch']} (val AUC: {ckpt['val_auc']:.4f})")

    # Load test data
    print("\n[2/4] Loading test set...")
    test_ds = ChestXRayDataset(
        Path(args.data_dir) / "test", transform=get_val_transforms()
    )
    test_loader = DataLoader(
        test_ds, batch_size=32, shuffle=False, num_workers=args.num_workers
    )
    print(f"  {len(test_ds):,} test images")

    # Inference
    print("\n[3/4] Running inference...")
    model.eval()
    all_labels, all_probs, all_images = [], [], []
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            logits = model(images)
            probs  = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
            all_probs.extend(probs)
            all_labels.extend(labels.numpy())
            if len(all_images) < 8:
                all_images.extend(images.cpu()[:8 - len(all_images)])

    all_labels = np.array(all_labels)
    all_probs  = np.array(all_probs)
    all_preds  = (all_probs >= 0.5).astype(int)

    # Metrics
    auc  = roc_auc_score(all_labels, all_probs)
    acc  = accuracy_score(all_labels, all_preds)
    f1   = f1_score(all_labels, all_preds)
    prec = precision_score(all_labels, all_preds)
    rec  = recall_score(all_labels, all_preds)
    cm   = confusion_matrix(all_labels, all_preds)

    report = (
        "\n" + "=" * 50 + "\n"
        "  EVALUATION REPORT — Chest X-Ray Classifier\n"
        + "=" * 50 + "\n"
        f"  AUC-ROC   : {auc:.4f}\n"
        f"  Accuracy  : {acc:.4f}  ({acc*100:.2f}%)\n"
        f"  F1 Score  : {f1:.4f}\n"
        f"  Precision : {prec:.4f}\n"
        f"  Recall    : {rec:.4f}\n"
        + "-" * 50 + "\n"
        f"  Confusion Matrix:\n"
        f"    TN={cm[0,0]}  FP={cm[0,1]}\n"
        f"    FN={cm[1,0]}  TP={cm[1,1]}\n"
        + "=" * 50
    )
    print(report)
    (output_dir / "eval_report.txt").write_text(report)

    # Plots
    print("\n[4/4] Saving plots & Grad-CAM...")
    plot_roc_curve(all_labels, all_probs, output_dir / "roc_curve.png")
    plot_confusion_matrix(cm, output_dir / "confusion_matrix.png")

    # Grad-CAM
    gradcam = GradCAM(model)
    save_gradcam_grid(
        images=all_images,
        labels=all_labels[:len(all_images)].tolist(),
        preds=all_preds[:len(all_images)].tolist(),
        probs=all_probs[:len(all_images)].tolist(),
        gradcam=gradcam,
        device=device,
        output_path=output_dir / "gradcam_grid.png",
    )

    print("\nAll outputs saved to:", output_dir)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir",    type=str, default="data/chest_xray")
    p.add_argument("--ckpt_path",   type=str, default="outputs/best_model.pth")
    p.add_argument("--output_dir",  type=str, default="outputs")
    p.add_argument("--num_workers", type=int, default=4)
    evaluate(p.parse_args())
