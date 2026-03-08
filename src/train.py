"""
train.py — Full training loop with:
  - Two-phase training (frozen backbone warm-up → full fine-tune)
  - Mixed precision (torch.amp)
  - Early stopping
  - Checkpoint saving (best val AUC)
  - CSV + console logging

Usage:
    python src/train.py --data_dir data/chest_xray --epochs 25 --batch_size 32
"""

import argparse
import csv
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
from sklearn.metrics import roc_auc_score
import numpy as np

from dataset import build_dataloaders, ChestXRayDataset, get_train_transforms
from model import ChestXRayModel, build_criterion, build_optimizer_scheduler


# ─── Argument parsing ─────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser(description="Train Chest X-Ray classifier")
    p.add_argument("--data_dir",    type=str,   default="data/chest_xray")
    p.add_argument("--output_dir",  type=str,   default="outputs")
    p.add_argument("--epochs",      type=int,   default=25)
    p.add_argument("--batch_size",  type=int,   default=32)
    p.add_argument("--warmup_epochs", type=int, default=3,
                   help="Epochs with frozen backbone before full fine-tuning")
    p.add_argument("--lr_head",     type=float, default=1e-3)
    p.add_argument("--lr_backbone", type=float, default=1e-4)
    p.add_argument("--patience",    type=int,   default=7,
                   help="Early stopping patience (epochs without val AUC improvement)")
    p.add_argument("--num_workers", type=int,   default=4)
    p.add_argument("--seed",        type=int,   default=42)
    return p.parse_args()


# ─── Helpers ──────────────────────────────────────────────────────────────────
def set_seed(seed: int):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device() -> torch.device:
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using Apple MPS")
    else:
        device = torch.device("cpu")
        print("Using CPU (training will be slow)")
    return device


# ─── One epoch ────────────────────────────────────────────────────────────────
def run_epoch(
    model: nn.Module,
    loader,
    criterion: nn.Module,
    optimizer,
    scheduler,
    scaler: GradScaler,
    device: torch.device,
    is_train: bool,
):
    model.train() if is_train else model.eval()

    total_loss = 0.0
    all_labels, all_probs = [], []

    ctx = torch.enable_grad() if is_train else torch.no_grad()
    with ctx:
        for images, labels in loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            with autocast(enabled=device.type == "cuda"):
                logits = model(images)
                loss   = criterion(logits, labels)

            if is_train:
                optimizer.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()

            total_loss += loss.item() * images.size(0)
            probs = torch.softmax(logits.float(), dim=1)[:, 1]   # P(pneumonia)
            all_probs.extend(probs.detach().cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    n = len(loader.dataset)
    avg_loss = total_loss / n
    auc = roc_auc_score(all_labels, all_probs)
    acc = np.mean(
        (np.array(all_probs) >= 0.5).astype(int) == np.array(all_labels)
    )
    return avg_loss, auc, acc


# ─── Main training loop ───────────────────────────────────────────────────────
def train(args):
    set_seed(args.seed)
    device = get_device()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Data
    print("\n[1/4] Loading data...")
    train_loader, val_loader, _ = build_dataloaders(
        args.data_dir, args.batch_size, args.num_workers
    )
    train_ds: ChestXRayDataset = train_loader.dataset
    class_weights = train_ds.get_class_weights()

    # Model
    print("\n[2/4] Building model...")
    model = ChestXRayModel().to(device)
    criterion = build_criterion(class_weights, device)
    scaler = GradScaler(enabled=device.type == "cuda")

    # CSV logger
    log_path = output_dir / "training_log.csv"
    log_file = open(log_path, "w", newline="")
    writer = csv.writer(log_file)
    writer.writerow(["epoch", "phase", "train_loss", "train_auc", "train_acc",
                     "val_loss", "val_auc", "val_acc", "lr", "elapsed_s"])

    best_val_auc = 0.0
    patience_counter = 0
    best_ckpt_path = output_dir / "best_model.pth"

    print("\n[3/4] Training...\n")
    print(f"{'Ep':>4} {'Phase':<8} {'TrLoss':>8} {'TrAUC':>7} {'TrAcc':>7} "
          f"{'VlLoss':>8} {'VlAUC':>7} {'VlAcc':>7} {'LR':>10}")
    print("-" * 75)

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()

        # ── Phase: warm-up (frozen backbone) ──────────────────────────────────
        if epoch == 1:
            print(f"  >> Warm-up phase ({args.warmup_epochs} epochs): head only")
            model.freeze_backbone()
            optimizer, scheduler = build_optimizer_scheduler(
                model,
                lr_backbone=0.0,
                lr_head=args.lr_head,
                epochs=args.warmup_epochs,
                steps_per_epoch=len(train_loader),
            )

        # ── Phase: full fine-tuning ────────────────────────────────────────────
        if epoch == args.warmup_epochs + 1:
            print(f"\n  >> Full fine-tuning phase")
            model.unfreeze_all()
            remaining = args.epochs - args.warmup_epochs
            optimizer, scheduler = build_optimizer_scheduler(
                model,
                lr_backbone=args.lr_backbone,
                lr_head=args.lr_head,
                epochs=remaining,
                steps_per_epoch=len(train_loader),
            )

        phase = "warm-up" if epoch <= args.warmup_epochs else "finetune"
        current_lr = optimizer.param_groups[-1]["lr"]

        tr_loss, tr_auc, tr_acc = run_epoch(
            model, train_loader, criterion, optimizer, scheduler, scaler, device, is_train=True
        )
        vl_loss, vl_auc, vl_acc = run_epoch(
            model, val_loader, criterion, optimizer, scheduler, scaler, device, is_train=False
        )

        elapsed = time.time() - t0
        print(f"{epoch:>4} {phase:<8} {tr_loss:>8.4f} {tr_auc:>7.4f} {tr_acc:>7.4f} "
              f"{vl_loss:>8.4f} {vl_auc:>7.4f} {vl_acc:>7.4f} {current_lr:>10.2e}")
        writer.writerow([epoch, phase, tr_loss, tr_auc, tr_acc,
                         vl_loss, vl_auc, vl_acc, current_lr, f"{elapsed:.1f}"])
        log_file.flush()

        # ── Checkpoint & early stopping ───────────────────────────────────────
        if vl_auc > best_val_auc:
            best_val_auc = vl_auc
            patience_counter = 0
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "val_auc": vl_auc,
                "val_acc": vl_acc,
                "args": vars(args),
            }, best_ckpt_path)
            print(f"       ✓ Saved best model (val AUC: {vl_auc:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= args.patience and epoch > args.warmup_epochs:
                print(f"\n  Early stopping at epoch {epoch} (no improvement for {args.patience} epochs)")
                break

    log_file.close()
    print(f"\n[4/4] Done! Best val AUC: {best_val_auc:.4f}")
    print(f"      Checkpoint : {best_ckpt_path}")
    print(f"      Training log: {log_path}")


if __name__ == "__main__":
    train(parse_args())
