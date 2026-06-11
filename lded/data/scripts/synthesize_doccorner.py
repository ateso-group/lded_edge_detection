"""
Synthetische Datenaugmentation für das LDED-Training aus dem DocCornerDataset.

Liest Bilder + normalisierte Eckpunkte aus Parquet-Dateien und schreibt
direkt nach data/processed/{train,val,test}/. Die DocCorner-Splits werden
1:1 übernommen. Augmentation wird nur auf den train-Split angewandt.

Falls im Zielverzeichnis bereits Daten liegen (z.B. von MIDV-2020),
wird die Nummerierung automatisch fortgesetzt.
"""

import argparse
import io
import json
import random
from pathlib import Path

import albumentations as A
import cv2
import numpy as np
import pyarrow.parquet as pq
from PIL import Image
from tqdm import tqdm


def get_next_sample_id(output_dir: Path) -> int:
    """Ermittelt die nächste freie Sample-ID anhand existierender Dateien."""
    images_dir = output_dir / "images"
    if not images_dir.exists():
        return 0
    existing = sorted(images_dir.glob("*.jpg"))
    if not existing:
        return 0
    return int(existing[-1].stem) + 1


def build_augmentation_pipeline() -> A.Compose:
    """Milde Offline-Augmentation für synthetische Daten.

    Die Hauptdiversität (Blur, JPEG, starke Perspektive, Flip) wird bewusst
    NICHT hier, sondern on-the-fly im Training erzeugt. Offline werden nur
    leichte Varianten ergänzt, damit der saubere Standard erhalten bleibt.
    """
    return A.Compose(
        [
            A.Perspective(scale=(0.02, 0.06), p=0.5),
            A.RandomBrightnessContrast(
                brightness_limit=0.2,
                contrast_limit=0.2,
                p=0.4,
            ),
        ],
        keypoint_params=A.KeypointParams(format="xy", remove_invisible=False),
    )


def synthesize_sample(
    image: np.ndarray,
    corners: list[tuple[float, float]],
    transform: A.Compose,
) -> tuple[np.ndarray, list[tuple[float, float]]]:
    """Wendet Augmentation auf ein Bild + Eckpunkte an."""
    result = transform(image=image, keypoints=corners)
    return result["image"], result["keypoints"]


def decode_image(image_bytes: bytes) -> np.ndarray | None:
    """Dekodiert ein Bild aus Bytes (JPEG/PNG) zu einem numpy Array (RGB)."""
    try:
        pil_img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        return np.array(pil_img)
    except Exception:
        return None


def process_parquet_split(
    parquet_dir: Path,
    output_dir: Path,
    num_augmentations: int,
    transform: A.Compose | None,
    skip_negatives: bool = True,
) -> tuple[int, int]:
    """Verarbeitet Parquet-Dateien eines einzelnen Splits.

    Args:
        parquet_dir: Verzeichnis mit Parquet-Dateien.
        output_dir: Ausgabeverzeichnis für diesen Split.
        num_augmentations: Augmentationen pro Bild (0 = nur Original).
        transform: Augmentations-Pipeline (None wenn num_augmentations == 0).
        skip_negatives: Negative Samples überspringen.

    Returns:
        Tuple (erzeugte Samples, übersprungene Negatives).
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "images").mkdir(exist_ok=True)
    (output_dir / "annotations").mkdir(exist_ok=True)

    sample_id = get_next_sample_id(output_dir)
    start_id = sample_id
    print(f"Starte bei Sample-ID {sample_id} in {output_dir}")

    parquet_files = sorted(parquet_dir.glob("*.parquet"))
    print(f"Gefunden: {len(parquet_files)} Parquet-Dateien in {parquet_dir}")

    skipped_negatives = 0

    for pq_file in tqdm(parquet_files, desc="Processing parquet files"):
        table = pq.read_table(pq_file)
        data = table.to_pydict()
        num_rows = len(data["filename"])

        for row_idx in tqdm(range(num_rows), desc=f"  {pq_file.name}", leave=False):
            # Negative Samples überspringen (keine Dokumente im Bild)
            if skip_negatives and data["is_negative"][row_idx]:
                skipped_negatives += 1
                continue

            # Eckpunkte lesen (normalisiert 0-1)
            tl_x = data["corner_tl_x"][row_idx]
            tl_y = data["corner_tl_y"][row_idx]
            tr_x = data["corner_tr_x"][row_idx]
            tr_y = data["corner_tr_y"][row_idx]
            br_x = data["corner_br_x"][row_idx]
            br_y = data["corner_br_y"][row_idx]
            bl_x = data["corner_bl_x"][row_idx]
            bl_y = data["corner_bl_y"][row_idx]

            # Ungültige Annotationen überspringen
            if any(v is None for v in [tl_x, tl_y, tr_x, tr_y, br_x, br_y, bl_x, bl_y]):
                continue

            # Bild dekodieren
            image_dict = data["image"][row_idx]
            image_bytes = image_dict["bytes"]
            if image_bytes is None:
                continue

            image = decode_image(image_bytes)
            if image is None:
                continue

            h, w = image.shape[:2]

            # Normalisierte Koordinaten in Pixel umrechnen
            corners_px = [
                (tl_x * w, tl_y * h),
                (tr_x * w, tr_y * h),
                (br_x * w, br_y * h),
                (bl_x * w, bl_y * h),
            ]

            # aug_idx == 0: unverändertes Original mitspeichern (sauberer Standard),
            # aug_idx > 0: milde augmentierte Varianten.
            for aug_idx in range(num_augmentations + 1):
                if aug_idx == 0:
                    aug_image, aug_corners = image, corners_px
                else:
                    try:
                        aug_image, aug_corners = synthesize_sample(
                            image, corners_px, transform
                        )
                    except Exception:
                        continue

                # Augmentierte Ecken zurück zu normalisierten Koordinaten
                aug_h, aug_w = aug_image.shape[:2]
                corners_norm = [
                    [float(x) / aug_w, float(y) / aug_h]
                    for x, y in aug_corners
                ]

                # Speichern
                out_name = f"{sample_id:06d}"
                out_img_path = output_dir / "images" / f"{out_name}.jpg"
                out_ann_path = output_dir / "annotations" / f"{out_name}.json"

                cv2.imwrite(
                    str(out_img_path),
                    cv2.cvtColor(aug_image, cv2.COLOR_RGB2BGR),
                )
                annotation_out = {
                    "corners": corners_norm,
                    "corners_px": [[float(x), float(y)] for x, y in aug_corners],
                    "image_size": [aug_w, aug_h],
                    "source": data["filename"][row_idx],
                    "augmentation_index": aug_idx,
                }
                with open(out_ann_path, "w") as f:
                    json.dump(annotation_out, f, indent=2)

                sample_id += 1

    n_new = sample_id - start_id
    return n_new, skipped_negatives


def process_all_splits(
    dataset_dir: Path,
    output_dir: Path,
    num_augmentations: int = 2,
    seed: int = 42,
    skip_negatives: bool = True,
) -> None:
    """Verarbeitet alle DocCorner-Splits (train/val/test).

    Schreibt direkt nach data/processed/{train,val,test}/.
    - train: Original + augmentierte Varianten
    - val/test: nur Originale (keine Augmentation)
    """
    random.seed(seed)
    np.random.seed(seed)

    transform = build_augmentation_pipeline()

    splits = {
        "train": num_augmentations,
        "val": 0,
        "test": 0,
    }

    for split_name, n_aug in splits.items():
        split_input = dataset_dir / split_name
        if not split_input.exists():
            print(f"WARNUNG: {split_input} existiert nicht, überspringe {split_name}.")
            continue

        split_output = output_dir / split_name
        print(f"\n{'='*60}")
        print(f"Split: {split_name} (Augmentationen: {n_aug})")
        print(f"{'='*60}")

        n_samples, n_skipped = process_parquet_split(
            parquet_dir=split_input,
            output_dir=split_output,
            num_augmentations=n_aug,
            transform=transform if n_aug > 0 else None,
            skip_negatives=skip_negatives,
        )

        print(f"Erzeugt: {n_samples} neue Samples in {split_output}")
        if n_skipped > 0:
            print(f"Übersprungen: {n_skipped} Negative Samples (kein Dokument)")


def main():
    parser = argparse.ArgumentParser(
        description="DocCornerDataset → data/processed/{train,val,test}/"
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("data/raw/DocCornerDataset"),
        help="DocCornerDataset-Wurzelverzeichnis (enthält train/, val/, test/)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/processed"),
        help="Ausgabe-Wurzelverzeichnis (erzeugt train/, val/, test/ Unterordner)",
    )
    parser.add_argument(
        "--num-augmentations",
        type=int,
        default=2,
        help="Anzahl milder Augmentationen pro train-Bild zusätzlich zum Original (default: 2)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )
    parser.add_argument(
        "--include-negatives",
        action="store_true",
        help="Negative Samples (ohne Dokument) einschließen",
    )
    args = parser.parse_args()

    process_all_splits(
        dataset_dir=args.input_dir,
        output_dir=args.output_dir,
        num_augmentations=args.num_augmentations,
        seed=args.seed,
        skip_negatives=not args.include_negatives,
    )


if __name__ == "__main__":
    main()
