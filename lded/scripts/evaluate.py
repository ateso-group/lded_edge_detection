"""
Evaluations-Skript für LDED.

Berechnet PCK, NME, IoU und Latenz auf dem Testset.
"""

import argparse
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort
import cv2
from tqdm import tqdm

from model.inference.preprocess import Preprocessor
from model.utils.metrics import compute_all_metrics


def evaluate_onnx(
    model_path: str,
    data_dir: str,
    input_size: tuple[int, int] = (512, 512),
    padding_ratio: float = 0.05,
) -> dict[str, float]:
    """Evaluiert ein ONNX-Modell auf dem Testset.

    Args:
        model_path: Pfad zum ONNX-Modell.
        data_dir: Verzeichnis mit images/ und annotations/.
        input_size: Modell-Input-Größe.
        padding_ratio: Padding-Anteil.

    Returns:
        Dict mit Metriken und Latenz.
    """
    import json

    session = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
    preprocessor = Preprocessor(input_size=input_size, padding_ratio=padding_ratio)

    data_path = Path(data_dir)
    image_paths = sorted(data_path.glob("images/*.jpg"))
    image_paths += sorted(data_path.glob("images/*.png"))

    all_pred_corners = []
    all_target_corners = []
    latencies = []

    for img_path in tqdm(image_paths, desc="Evaluating"):
        # Annotation laden
        ann_path = data_path / "annotations" / f"{img_path.stem}.json"
        if not ann_path.exists():
            continue

        with open(ann_path, "r") as f:
            annotation = json.load(f)

        target_corners = np.array(annotation["corners"], dtype=np.float32)

        # Bild laden & vorverarbeiten
        image = cv2.imread(str(img_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        orig_h, orig_w = image.shape[:2]

        tensor, meta = preprocessor.preprocess(image)

        # Inferenz + Latenz messen
        input_name = session.get_inputs()[0].name
        t0 = time.perf_counter()
        outputs = session.run(None, {input_name: tensor})
        latency = (time.perf_counter() - t0) * 1000  # ms
        latencies.append(latency)

        # Koordinaten extrahieren
        coords = outputs[0][0]  # [4, 2] normalisiert (im gepaddeten Raum)

        # Predictions: normalisiert → 512×512 Pixel (gleicher Raum wie Training)
        pred_corners = coords.copy()
        pred_corners[:, 0] *= input_size[1]  # * 512
        pred_corners[:, 1] *= input_size[0]  # * 512

        # Targets: Original-normalisiert → gepaddeten Raum → 512×512 Pixel
        # (gleiche Transformation wie im Dataset/Training)
        pad = meta["pad"]
        padded_w = orig_w + 2 * pad
        padded_h = orig_h + 2 * pad

        target_px = target_corners.copy()
        target_px[:, 0] = (target_corners[:, 0] * orig_w + pad) / padded_w * input_size[1]
        target_px[:, 1] = (target_corners[:, 1] * orig_h + pad) / padded_h * input_size[0]

        all_pred_corners.append(pred_corners)
        all_target_corners.append(target_px)

    all_pred = np.array(all_pred_corners)
    all_target = np.array(all_target_corners)

    metrics = compute_all_metrics(all_pred, all_target, image_size=input_size)
    metrics["latency_mean_ms"] = float(np.mean(latencies)) if latencies else 0.0
    metrics["latency_p95_ms"] = float(np.percentile(latencies, 95)) if latencies else 0.0
    metrics["num_samples"] = len(all_pred_corners)

    return metrics


def main():
    parser = argparse.ArgumentParser(description="LDED Evaluation")
    parser.add_argument("--model", type=str, required=True, help="Pfad zum ONNX-Modell")
    parser.add_argument("--data", type=str, required=True, help="Testdaten-Verzeichnis")
    parser.add_argument("--input-size", type=int, nargs=2, default=[512, 512])
    parser.add_argument(
        "--metrics", nargs="+", default=["pck", "nme", "iou", "latency"],
        help="Zu berechnende Metriken",
    )
    args = parser.parse_args()

    metrics = evaluate_onnx(
        model_path=args.model,
        data_dir=args.data,
        input_size=tuple(args.input_size),
    )

    print("\n=== Evaluationsergebnisse ===")
    print(f"  Samples:      {metrics['num_samples']}")
    if "pck" in args.metrics:
        print(f"  PCK@5:        {metrics['pck5']:.4f}")
        print(f"  PCK@10:       {metrics['pck10']:.4f}")
    if "nme" in args.metrics:
        print(f"  NME:          {metrics['nme']:.4f}")
    if "iou" in args.metrics:
        print(f"  IoU:          {metrics['iou']:.4f}")
    if "latency" in args.metrics:
        print(f"  Latenz (avg): {metrics['latency_mean_ms']:.1f} ms")
        print(f"  Latenz (p95): {metrics['latency_p95_ms']:.1f} ms")


if __name__ == "__main__":
    main()
