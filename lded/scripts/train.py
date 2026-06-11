"""
Trainings-Skript für LDED.

Implementiert 3 Phasen:
1. Backbone Freeze (Epochs 1–10)
2. Full Fine-Tuning mit differentiellem LR (Epochs 11–80)
3. Quantization-Aware Training (Epochs 81–100)
"""

import argparse
import math
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import yaml

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from model.model import LDED
from model.loss.awing import LDEDLoss
from model.data.dataset import LDEDDataset
from model.utils.metrics import compute_all_metrics
import numpy as np


def load_config(config_path: str) -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def create_dataloaders(
    cfg: dict, max_samples: int | None = None, overfit_test: bool = False
) -> tuple[DataLoader, DataLoader]:
    """Erstellt Training- und Validierungs-DataLoader.

    Args:
        cfg: Trainings-Konfiguration.
        max_samples: Begrenzt Train/Val auf die ersten N Samples.
        overfit_test: Wenn True, werden die gleichen Samples ohne Augmentation
            für Train UND Val verwendet (Bug-Check / Overfit-Test).
    """
    train_ds = LDEDDataset(
        data_dir=cfg["data"]["train_dir"],
        input_size=tuple(cfg["input_size"]),
        heatmap_size=(256, 256),
        is_train=not overfit_test,  # Keine Augmentation im Overfit-Test
        padding_ratio=cfg["data"]["padding_ratio"],
    )

    # Dataset auf max_samples begrenzen (Debug-Modus)
    if max_samples is not None:
        train_ds.image_paths = train_ds.image_paths[:max_samples]

    if overfit_test:
        # Overfit-Test: Val = Train (gleiche Samples, keine Augmentation)
        val_ds = train_ds
        print(f"\n{'='*60}")
        print(f"🧪 OVERFIT-TEST MODUS")
        print(f"   Samples: {len(train_ds)} (Train = Val, keine Augmentation)")
        print(f"   Erwartung: Loss → ~0 nach 100–200 Epochen")
        print(f"   Wenn nicht: Bug im Trainingsloop!")
        print(f"{'='*60}\n")
    else:
        val_ds = LDEDDataset(
            data_dir=cfg["data"]["val_dir"],
            input_size=tuple(cfg["input_size"]),
            heatmap_size=(256, 256),
            is_train=False,
            padding_ratio=cfg["data"]["padding_ratio"],
        )
        if max_samples is not None:
            val_ds.image_paths = val_ds.image_paths[:max_samples]
            print(f"⚠ Debug-Modus: max_samples={max_samples} (Train: {len(train_ds)}, Val: {len(val_ds)})")

    pin_memory = not torch.backends.mps.is_available()

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg["batch_size"],
        shuffle=True,
        num_workers=cfg["num_workers"],
        pin_memory=pin_memory,
        drop_last=False if overfit_test else True,
        persistent_workers=cfg["num_workers"] > 0,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg["batch_size"],
        shuffle=False,
        num_workers=cfg["num_workers"],
        pin_memory=pin_memory,
        persistent_workers=cfg["num_workers"] > 0,
    )
    return train_loader, val_loader


def train_one_epoch(
    model: LDED,
    loader: DataLoader,
    criterion: LDEDLoss,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    log_interval: int = 50,
) -> dict[str, float]:
    """Trainiert eine Epoche."""
    model.train()
    total_loss = 0.0
    total_awing = 0.0
    total_bce = 0.0
    num_batches = 0

    pbar = tqdm(loader, desc=f"Epoch {epoch} [Train]")
    for step, batch in enumerate(pbar):
        images = batch["image"].to(device)
        target_heatmaps = batch["heatmaps"].to(device)
        target_confidence = batch["confidence"].to(device)

        target_corners = batch["corners"].to(device)

        # Forward
        pred_heatmaps, pred_coords, pred_confidence = model(images)
        losses = criterion(
            pred_heatmaps, target_heatmaps, pred_confidence, target_confidence,
            pred_coords=pred_coords, target_coords=target_corners,
        )

        # Backward
        optimizer.zero_grad()
        losses["total"].backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()

        total_loss += losses["total"].item()
        total_awing += losses["awing"].item()
        total_bce += losses["bce"].item()
        num_batches += 1

        if (step + 1) % log_interval == 0:
            pbar.set_postfix(
                loss=f"{total_loss / num_batches:.4f}",
                awing=f"{total_awing / num_batches:.4f}",
                bce=f"{total_bce / num_batches:.4f}",
            )

    return {
        "train_loss": total_loss / max(num_batches, 1),
        "train_awing": total_awing / max(num_batches, 1),
        "train_bce": total_bce / max(num_batches, 1),
    }


@torch.no_grad()
def validate(
    model: LDED,
    loader: DataLoader,
    criterion: LDEDLoss,
    device: torch.device,
    epoch: int,
) -> dict[str, float]:
    """Validierung."""
    model.eval()
    total_loss = 0.0
    all_pred_corners = []
    all_target_corners = []
    num_batches = 0

    for batch in tqdm(loader, desc=f"Epoch {epoch} [Val]"):
        images = batch["image"].to(device)
        target_heatmaps = batch["heatmaps"].to(device)
        target_confidence = batch["confidence"].to(device)
        target_corners = batch["corners"].to(device)

        pred_heatmaps, pred_coords, pred_confidence = model(images)
        losses = criterion(
            pred_heatmaps, target_heatmaps, pred_confidence, target_confidence,
            pred_coords=pred_coords, target_coords=target_corners,
        )

        total_loss += losses["total"].item()
        num_batches += 1

        # Koordinaten für Metriken sammeln (in Pixel-Raum)
        pred_px = pred_coords.cpu().numpy() * 512  # Skalierung auf Bildgröße
        target_px = target_corners.cpu().numpy() * 512
        all_pred_corners.append(pred_px)
        all_target_corners.append(target_px)

    all_pred = np.concatenate(all_pred_corners, axis=0)
    all_target = np.concatenate(all_target_corners, axis=0)
    metrics = compute_all_metrics(all_pred, all_target, image_size=(512, 512))

    metrics["val_loss"] = total_loss / max(num_batches, 1)
    return metrics


def generate_metrics_chart(metrics: dict[str, float], save_dir: Path) -> None:
    """Erzeugt ein Balkendiagramm der Evaluationsmetriken und speichert es als PNG.

    Zeigt PCK@5, PCK@10, IoU (als Prozent, höher = besser) und NME (niedriger = besser)
    in einem übersichtlichen Balkendiagramm.
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Linkes Diagramm: PCK und IoU (höher = besser, Skala 0–100%)
    accuracy_names = ["PCK@5", "PCK@10", "IoU"]
    accuracy_values = [
        metrics.get("pck5", 0) * 100,
        metrics.get("pck10", 0) * 100,
        metrics.get("iou", 0) * 100,
    ]
    colors = ["#2196F3", "#4CAF50", "#FF9800"]

    bars = axes[0].bar(accuracy_names, accuracy_values, color=colors, edgecolor="black", width=0.6)
    axes[0].set_ylim(0, 105)
    axes[0].set_ylabel("Prozent (%)")
    axes[0].set_title("Genauigkeitsmetriken (höher = besser)")
    axes[0].axhline(y=90, color="gray", linestyle="--", alpha=0.5, label="Ziel: 90%")
    axes[0].legend()
    for bar, val in zip(bars, accuracy_values):
        axes[0].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                     f"{val:.1f}%", ha="center", va="bottom", fontweight="bold")

    # Rechtes Diagramm: NME und Loss (niedriger = besser)
    error_names = ["NME", "Val Loss"]
    error_values = [
        metrics.get("nme", 0),
        metrics.get("val_loss", 0),
    ]
    colors_err = ["#F44336", "#9C27B0"]

    bars2 = axes[1].bar(error_names, error_values, color=colors_err, edgecolor="black", width=0.6)
    axes[1].set_ylabel("Wert")
    axes[1].set_title("Fehlermetriken (niedriger = besser)")
    for bar, val in zip(bars2, error_values):
        axes[1].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.001,
                     f"{val:.4f}", ha="center", va="bottom", fontweight="bold")

    plt.tight_layout()
    chart_path = save_dir / "metrics_chart.png"
    plt.savefig(chart_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  → Metriken-Diagramm gespeichert: {chart_path}")


def main():
    parser = argparse.ArgumentParser(description="LDED Training")
    parser.add_argument("--config", type=str, default="configs/train/default.yaml")
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--max_samples", type=int, default=None,
                        help="Limit training/val to first N samples (for debugging)")
    parser.add_argument("--overfit_test", type=int, nargs="?", const=5, default=None,
                        help="Overfit-Test: Train+Val auf N gleichen Samples ohne Augmentation "
                             "(default: 5 Samples, 200 Epochen). Loss muss → ~0 gehen.")
    args = parser.parse_args()

    cfg = load_config(args.config)

    # Overfit-Test: Epochen und Samples überschreiben
    overfit_mode = args.overfit_test is not None
    if overfit_mode:
        cfg["epochs"] = 200
        cfg["batch_size"] = min(cfg["batch_size"], args.overfit_test)
        args.max_samples = args.overfit_test
        # Warmup praktisch deaktivieren und LR etwas anheben, damit der
        # Overfit-Test schnell konvergiert.
        cfg["warmup"]["epochs"] = 1
        cfg["optimizer"]["lr"] = 1.0e-3
        cfg["lr_schedule"]["T_max"] = cfg["epochs"]

    # Device
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    print(f"Device: {device}")

    # Modell
    model_cfg_path = cfg.get("model_config", "configs/model/mobilenetv3.yaml")
    model = LDED.from_config(model_cfg_path).to(device)

    # Loss
    criterion = LDEDLoss(
        awing_weight=cfg["loss"]["awing_weight"],
        bce_weight=cfg["loss"]["bce_weight"],
        coord_weight=cfg["loss"].get("coord_weight", 5.0),
    )

    # Dataloaders
    train_loader, val_loader = create_dataloaders(
        cfg, max_samples=args.max_samples, overfit_test=overfit_mode
    )

    # Checkpoint-Verzeichnis
    save_dir = Path(cfg["checkpoint"]["save_dir"])
    save_dir.mkdir(parents=True, exist_ok=True)

    # TensorBoard-Writer für Live-Visualisierung des Trainingsverlaufs
    writer = SummaryWriter(log_dir=cfg["logging"]["log_dir"])

    best_pck5 = 0.0
    phases = cfg["phases"]

    for epoch in range(1, cfg["epochs"] + 1):
        t0 = time.time()

        # Phase Management
        if epoch <= phases["freeze_backbone_until_epoch"]:
            # Phase 1: Backbone einfrieren
            model.freeze_backbone()
            lr = cfg["optimizer"]["lr"]
            optimizer = AdamW(
                filter(lambda p: p.requires_grad, model.parameters()),
                lr=lr,
                weight_decay=cfg["optimizer"]["weight_decay"],
            )
        elif epoch == phases["freeze_backbone_until_epoch"] + 1:
            # Phase 2 Start: Backbone auftauen, differentielles LR
            model.unfreeze_backbone()
            head_lr = cfg["optimizer"]["lr"]
            backbone_lr = head_lr * phases["backbone_lr_factor"]
            optimizer = AdamW(
                model.get_param_groups(backbone_lr, head_lr),
                weight_decay=cfg["optimizer"]["weight_decay"],
            )
        # Phase 3: QAT (Placeholder — wird in späterem Sprint implementiert)

        # Base-LR pro Param-Gruppe festhalten, damit Warmup unabhängig von
        # Optimizer-Neuerstellung immer auf der ursprünglichen LR aufsetzt.
        for pg in optimizer.param_groups:
            pg.setdefault("base_lr", pg["lr"])

        # LR-Schedule: Linearer Warmup gefolgt von CosineAnnealingLR.
        # Wird pro Epoche neu aus base_lr berechnet, damit es robust gegen die
        # Optimizer-Neuerstellung in den Phasen-Wechseln ist.
        warmup_epochs = cfg["warmup"]["epochs"]
        t_max = cfg["lr_schedule"].get("T_max", cfg["epochs"])
        eta_min = cfg["lr_schedule"].get("eta_min", 0.0)
        if warmup_epochs > 0 and epoch <= warmup_epochs:
            lr_factor = epoch / warmup_epochs
            for pg in optimizer.param_groups:
                pg["lr"] = pg["base_lr"] * lr_factor
        else:
            # Cosine-Annealing über die Epochen nach dem Warmup.
            progress = (epoch - warmup_epochs) / max(t_max - warmup_epochs, 1)
            progress = min(max(progress, 0.0), 1.0)
            cosine = 0.5 * (1 + math.cos(math.pi * progress))
            for pg in optimizer.param_groups:
                pg["lr"] = eta_min + (pg["base_lr"] - eta_min) * cosine

        # Training
        train_metrics = train_one_epoch(
            model, train_loader, criterion, optimizer, device, epoch,
            log_interval=cfg["logging"]["log_interval"],
        )

        # Validierung
        val_metrics = validate(model, val_loader, criterion, device, epoch)

        # MPS-Speicher freigeben um Fragmentierung zu vermeiden
        if device.type == "mps":
            torch.mps.empty_cache()

        elapsed = time.time() - t0
        print(
            f"Epoch {epoch}/{cfg['epochs']} "
            f"| train_loss={train_metrics['train_loss']:.4f} "
            f"| val_loss={val_metrics['val_loss']:.4f} "
            f"| PCK@5={val_metrics['pck5']:.4f} "
            f"| PCK@10={val_metrics['pck10']:.4f} "
            f"| NME={val_metrics['nme']:.4f} "
            f"| IoU={val_metrics['iou']:.4f} "
            f"| {elapsed:.1f}s"
        )

        # TensorBoard: Metriken loggen
        writer.add_scalar("Loss/train", train_metrics["train_loss"], epoch)
        writer.add_scalar("Loss/train_awing", train_metrics["train_awing"], epoch)
        writer.add_scalar("Loss/train_bce", train_metrics["train_bce"], epoch)
        writer.add_scalar("Loss/val", val_metrics["val_loss"], epoch)
        writer.add_scalar("Metrics/PCK@5", val_metrics["pck5"], epoch)
        writer.add_scalar("Metrics/PCK@10", val_metrics["pck10"], epoch)
        writer.add_scalar("Metrics/NME", val_metrics["nme"], epoch)
        writer.add_scalar("Metrics/IoU", val_metrics["iou"], epoch)
        writer.add_scalar("LR", optimizer.param_groups[0]["lr"], epoch)

        # Checkpoint speichern
        if val_metrics["pck5"] > best_pck5:
            best_pck5 = val_metrics["pck5"]
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "best_pck5": best_pck5,
                    "metrics": val_metrics,
                },
                save_dir / "best.pt",
            )
            print(f"  → Best model saved (PCK@5={best_pck5:.4f})")

    writer.close()
    print(f"\nTraining abgeschlossen. Bestes PCK@5: {best_pck5:.4f}")

    # Balkendiagramm der finalen Metriken erzeugen
    generate_metrics_chart(val_metrics, save_dir)


if __name__ == "__main__":
    main()
