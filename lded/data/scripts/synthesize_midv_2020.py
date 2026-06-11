"""
Synthetische Datenaugmentation für das LDED-Training.

Erzeugt augmentierte Trainingsbilder aus MIDV-2020 Rohdaten und schreibt
direkt nach data/processed/{train,val,test}/. Die Quellbilder werden
zufällig auf die Splits aufgeteilt (70/15/15).

Falls im Zielverzeichnis bereits Daten liegen (z.B. von DocCorner),
wird die Nummerierung automatisch fortgesetzt.
"""

import argparse
import json
import random
from pathlib import Path

import albumentations as A
import cv2
import numpy as np
from tqdm import tqdm

SPLIT_RATIOS = {"train": 0.70, "val": 0.15, "test": 0.15}


def build_augmentation_pipeline() -> A.Compose:
    """Erstellt die Augmentations-Pipeline für synthetische Daten."""
    return A.Compose(
        [
            A.Perspective(scale=(0.02, 0.10), p=0.7),
            A.HorizontalFlip(p=0.5),
            A.RandomBrightnessContrast(
                brightness_limit=(-0.5, 0.5),
                contrast_limit=(-0.3, 0.3),
                p=0.6,
            ),
            A.MotionBlur(blur_limit=(3, 9), p=0.3),
            A.ImageCompression(quality_range=(50, 95), p=0.4),
            A.GaussNoise(std_range=(0.01, 0.05), p=0.3),
        ],
        keypoint_params=A.KeypointParams(format="xy", remove_invisible=False),
    )


def get_next_sample_id(output_dir: Path) -> int:
    """Ermittelt die nächste freie Sample-ID anhand existierender Dateien."""
    images_dir = output_dir / "images"
    if not images_dir.exists():
        return 0
    existing = sorted(images_dir.glob("*.jpg"))
    if not existing:
        return 0
    return int(existing[-1].stem) + 1


def load_annotations(annotation_path: Path) -> dict:
    """Lädt Annotationen aus JSON-Datei."""
    with open(annotation_path, "r") as f:
        return json.load(f)


def load_midv2020_annotations(annotations_dir: Path, images_dir: Path) -> dict[str, list[tuple[float, float]]]:
    """Lädt MIDV-2020 VIA-Annotationen und gibt ein Dict {image_path: corners} zurück."""
    image_corners = {}
    for ann_file in sorted(annotations_dir.glob("*.json")):
        doc_type = ann_file.stem  # z.B. "lva_passport"
        with open(ann_file, "r") as f:
            data = json.load(f)

        img_metadata = data.get("_via_img_metadata", {})
        for _key, entry in img_metadata.items():
            filename = entry.get("filename", "")
            if not filename:
                continue
            img_path = images_dir / doc_type / filename
            if not img_path.exists():
                continue

            # Suche das doc_quad Polygon
            for region in entry.get("regions", []):
                attrs = region.get("region_attributes", {})
                if attrs.get("field_name") == "doc_quad":
                    shape = region.get("shape_attributes", {})
                    xs = shape.get("all_points_x", [])
                    ys = shape.get("all_points_y", [])
                    if len(xs) == 4 and len(ys) == 4:
                        corners = [(float(xs[i]), float(ys[i])) for i in range(4)]
                        image_corners[str(img_path)] = corners
                    break
    return image_corners


def synthesize_sample(
    image: np.ndarray,
    corners: list[tuple[float, float]],
    transform: A.Compose,
) -> tuple[np.ndarray, list[tuple[float, float]]]:
    """Wendet Augmentation auf ein Bild + Eckpunkte an."""
    result = transform(image=image, keypoints=corners)
    return result["image"], result["keypoints"]


def assign_splits(
    image_paths: list[str],
    seed: int,
) -> dict[str, list[str]]:
    """Teilt Quellbilder zufällig in train/val/test auf."""
    rng = random.Random(seed)
    paths = list(image_paths)
    rng.shuffle(paths)

    n = len(paths)
    n_train = int(n * SPLIT_RATIOS["train"])
    n_val = int(n * SPLIT_RATIOS["val"])

    return {
        "train": paths[:n_train],
        "val": paths[n_train : n_train + n_val],
        "test": paths[n_train + n_val :],
    }


def save_sample(
    aug_image: np.ndarray,
    aug_corners: list[tuple[float, float]],
    source_name: str,
    aug_idx: int,
    sample_id: int,
    output_dir: Path,
) -> None:
    """Speichert ein einzelnes Sample (Bild + Annotation)."""
    out_name = f"{sample_id:06d}"
    out_img_path = output_dir / "images" / f"{out_name}.jpg"
    out_ann_path = output_dir / "annotations" / f"{out_name}.json"

    cv2.imwrite(str(out_img_path), cv2.cvtColor(aug_image, cv2.COLOR_RGB2BGR))
    annotation_out = {
        "corners": [[float(x), float(y)] for x, y in aug_corners],
        "source": source_name,
        "augmentation_index": aug_idx,
    }
    with open(out_ann_path, "w") as f:
        json.dump(annotation_out, f, indent=2)


def process_dataset(
    raw_dir: Path,
    output_dir: Path,
    num_augmentations: int = 10,
    seed: int = 42,
) -> None:
    """Verarbeitet den gesamten Rohdatensatz und schreibt nach data/processed/{train,val,test}/."""
    random.seed(seed)
    np.random.seed(seed)

    transform = build_augmentation_pipeline()

    # MIDV-2020 Struktur erkennen: annotations/ und images/ Ordner
    annotations_dir = None
    images_dir = None
    for candidate in [raw_dir, raw_dir / "MIDV2020" / "dataset"]:
        if (candidate / "annotations").is_dir() and (candidate / "images").is_dir():
            annotations_dir = candidate / "annotations"
            images_dir = candidate / "images"
            break

    # Quellbilder + Corners laden
    if annotations_dir and images_dir:
        print(f"MIDV-2020 Format erkannt in {annotations_dir.parent}")
        image_corners = load_midv2020_annotations(annotations_dir, images_dir)
    else:
        # Fallback: Einfaches Format (Bild + gleichnamige .json)
        image_corners = {}
        image_paths = sorted(raw_dir.rglob("*.jpg")) + sorted(raw_dir.rglob("*.png"))
        for img_path in image_paths:
            ann_path = img_path.with_suffix(".json")
            if not ann_path.exists():
                continue
            annotations = load_annotations(ann_path)
            corners = annotations.get("corners", [])
            if len(corners) != 4:
                continue
            image_corners[str(img_path)] = [(c[0], c[1]) for c in corners]

    print(f"Gefunden: {len(image_corners)} Bilder mit Annotationen")
    if not image_corners:
        print("FEHLER: Keine Bilder gefunden. Abbruch.")
        return

    # Quellbilder auf Splits aufteilen
    split_assignments = assign_splits(list(image_corners.keys()), seed)
    for split_name, paths in split_assignments.items():
        print(f"  {split_name}: {len(paths)} Quellbilder")

    # Ausgabe-Verzeichnisse vorbereiten + nächste IDs ermitteln
    next_ids: dict[str, int] = {}
    for split_name in ["train", "val", "test"]:
        split_dir = output_dir / split_name
        split_dir.mkdir(parents=True, exist_ok=True)
        (split_dir / "images").mkdir(exist_ok=True)
        (split_dir / "annotations").mkdir(exist_ok=True)
        next_ids[split_name] = get_next_sample_id(split_dir)
        print(f"  {split_name}: starte bei Sample-ID {next_ids[split_name]}")

    # Pro Split augmentieren und speichern
    total_new = {"train": 0, "val": 0, "test": 0}

    for split_name, img_paths in split_assignments.items():
        split_dir = output_dir / split_name
        sample_id = next_ids[split_name]

        for img_path_str in tqdm(img_paths, desc=f"Synthesizing ({split_name})"):
            img_path = Path(img_path_str)
            image = cv2.imread(str(img_path))
            if image is None:
                continue
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            corners_tuples = image_corners[img_path_str]

            for aug_idx in range(num_augmentations):
                try:
                    aug_image, aug_corners = synthesize_sample(
                        image, corners_tuples, transform
                    )
                except Exception:
                    continue

                save_sample(
                    aug_image, aug_corners,
                    str(img_path.name), aug_idx,
                    sample_id, split_dir,
                )
                sample_id += 1

        total_new[split_name] = sample_id - next_ids[split_name]

    print(f"\nErzeugt:")
    for split_name, count in total_new.items():
        print(f"  {split_name}: {count} neue Samples")


def main():
    parser = argparse.ArgumentParser(
        description="MIDV-2020 → data/processed/{train,val,test}/"
    )
    parser.add_argument("--raw-dir", type=Path, default=Path("data/raw/midv-2020"))
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/processed"),
        help="Ausgabe-Wurzelverzeichnis (erzeugt train/, val/, test/ Unterordner)",
    )
    parser.add_argument("--num-augmentations", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    process_dataset(
        raw_dir=args.raw_dir,
        output_dir=args.output_dir,
        num_augmentations=args.num_augmentations,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
