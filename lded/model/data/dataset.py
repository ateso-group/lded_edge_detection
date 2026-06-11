"""
PyTorch Dataset für LDED-Trainings- und Validierungsdaten.

Lädt Bilder und 4-Eckpunkt-Annotationen, erzeugt Heatmap-Targets.

Was ist ein Dataset?
    Ein Dataset ist eine Datenquelle, aus der das Training seine Bilder und
    zugehörigen Informationen (Labels/Annotationen) bezieht. PyTorch ruft bei
    jedem Trainingsschritt __getitem__ auf, um ein einzelnes Datensample
    (Bild + Eckpunkte + Heatmaps) abzurufen. Mehrere Samples werden dann
    automatisch zu einem Batch (Stapel) zusammengefasst.
"""

# json: Python-Standardbibliothek zum Lesen von JSON-Dateien (Annotationen).
import json
# Path: Objektorientierter Umgang mit Dateipfaden.
from pathlib import Path

# cv2 (OpenCV): Bildverarbeitung (Laden, Resize, Farbkonvertierung).
import cv2
# numpy: Effiziente numerische Berechnungen mit mehrdimensionalen Arrays.
import numpy as np
# torch: Framework für neuronale Netze und Tensor-Berechnungen.
import torch
# Dataset: Basisklasse für alle PyTorch-Datasets – definiert die Schnittstelle
# (__len__ und __getitem__), die der DataLoader erwartet.
from torch.utils.data import Dataset

from model.data.augmentation import get_train_transforms, get_val_transforms


def generate_heatmap(
    corners: np.ndarray,
    heatmap_size: tuple[int, int] = (256, 256),
    sigma: float = 10.0,
) -> np.ndarray:
    """Erzeugt Gaussian Heatmaps für 4 Eckpunkte.

    Für jeden Eckpunkt wird eine Gauß-Glocke (Normalverteilung) zentriert auf
    die Position des Eckpunkts erzeugt. Das Ergebnis ist ein „Wärmebild", das
    an der Ecke am hellsten ist und zum Rand hin abfällt. Diese Heatmaps dienen
    als Ziel-Werte (Ground Truth) für das Training.

    Args:
        corners: [4, 2] normalisierte Eckpunkte (x, y) im Bereich [0, 1].
        heatmap_size: (Höhe, Breite) der Heatmap in Pixel.
        sigma: Standardabweichung der Gauß-Glocke – bestimmt, wie breit
            der „heiße" Bereich um den Eckpunkt ist. Größerer Wert = breiterer Fleck.

    Returns:
        heatmaps: [4, H, W] – 4 Heatmaps (eine pro Ecke) als NumPy-Array.
    """
    H, W = heatmap_size
    heatmaps = np.zeros((4, H, W), dtype=np.float32)

    for i, (cx, cy) in enumerate(corners):
        # Normalisierte Koordinaten [0, 1] → Pixel-Koordinaten der Heatmap
        px, py = cx * W, cy * H
        # Koordinaten-Gitter erzeugen: Für jeden Pixel die (x, y)-Position
        y_grid, x_grid = np.mgrid[0:H, 0:W].astype(np.float32)
        # Gauß-Formel: e^(-(Abstand²) / (2σ²)) → höchster Wert am Eckpunkt,
        # fällt mit zunehmendem Abstand ab.
        gaussian = np.exp(-((x_grid - px) ** 2 + (y_grid - py) ** 2) / (2 * sigma ** 2))
        heatmaps[i] = gaussian

    return heatmaps


class LDEDDataset(Dataset):
    """Dataset für Document Edge Detection.

    Lädt Bilder und ihre Eckpunkt-Annotationen aus einem Verzeichnis, wendet
    Augmentation an und erzeugt die für das Training benötigten Tensoren
    (Bild, Heatmaps, Eckpunkt-Koordinaten, Confidence).

    Erwartet folgende Verzeichnisstruktur:
        data_dir/
            images/       – Bilddateien (*.jpg, *.png)
            annotations/  – JSON-Dateien mit Eckpunkten
                            ({"corners": [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]})
                            Koordinaten sind normalisiert auf [0, 1].

    Args:
        data_dir: Pfad zum Datenverzeichnis mit images/ und annotations/.
        input_size: Ziel-Bildgröße (Höhe, Breite), auf die alle Bilder skaliert werden.
        heatmap_size: Größe der erzeugten Heatmaps (Höhe, Breite). Typischerweise
            kleiner als input_size (z.B. 128×128 statt 512×512), um Speicher zu sparen.
        is_train: True = Trainingsmodus mit Augmentation (Datenvervielfältigung).
            False = Validierungsmodus (nur Resize, keine Veränderungen).
        padding_ratio: Anteil des Bildumfangs, der als schwarzer Rand hinzugefügt wird.
    """

    def __init__(
        self,
        data_dir: str | Path,
        input_size: tuple[int, int] = (512, 512),
        heatmap_size: tuple[int, int] = (256, 256),
        is_train: bool = True,
        padding_ratio: float = 0.05,
    ):
        self.data_dir = Path(data_dir)
        self.input_size = input_size
        self.heatmap_size = heatmap_size
        self.padding_ratio = padding_ratio

        # Alle Bilddateien sammeln (alphabetisch sortiert)
        self.image_paths = sorted((self.data_dir / "images").glob("*.jpg"))
        self.image_paths += sorted((self.data_dir / "images").glob("*.png"))

        # Augmentation-Pipeline wählen (Training oder Validierung)
        if is_train:
            self.transforms = get_train_transforms(input_size)
        else:
            self.transforms = get_val_transforms(input_size)

        # ImageNet-Normalisierungskonstanten (Mittelwert und Standardabweichung der
        # RGB-Kanäle) – gleiche Werte wie beim vortrainierten Backbone.
        self.mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        self.std = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    def __len__(self) -> int:
        """Gibt die Anzahl der Bilder im Dataset zurück."""
        return len(self.image_paths)

    def _load_annotation(self, img_path: Path) -> dict:
        """Lädt die zugehörige JSON-Annotation zu einem Bild.

        Die JSON-Datei hat den gleichen Namen wie das Bild (z.B. foto.jpg → foto.json)
        und enthält die 4 Eckpunkte des Dokuments.
        """
        ann_path = self.data_dir / "annotations" / f"{img_path.stem}.json"
        with open(ann_path, "r") as f:
            return json.load(f)

    def _apply_padding(
        self, image: np.ndarray, corners: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Fügt symmetrisches Padding hinzu und passt die Eckpunkt-Koordinaten an.

        Da das Bild durch das Padding größer wird, müssen die Eckpunkt-Koordinaten
        entsprechend verschoben und neu normalisiert werden.
        """
        h, w = image.shape[:2]
        pad = int(self.padding_ratio * (h + w))

        # Schwarzen Rand hinzufügen (oben, unten, links, rechts)
        padded = cv2.copyMakeBorder(
            image, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=(0, 0, 0)
        )
        new_h, new_w = padded.shape[:2]

        # Eckpunkte anpassen: Normalisierte Koordinaten → Pixel → + Padding-Offset
        corners_px = corners.copy()
        corners_px[:, 0] = corners[:, 0] * w + pad  # x: normalisiert → Pixel + Offset
        corners_px[:, 1] = corners[:, 1] * h + pad  # y: normalisiert → Pixel + Offset

        # Zurück zu normalisierten Koordinaten (jetzt bezogen auf das gepaddete Bild)
        corners_px[:, 0] /= new_w
        corners_px[:, 1] /= new_h

        return padded, corners_px

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        """Gibt ein einzelnes Trainings-/Validierungssample zurück.

        Wird automatisch von PyTorchs DataLoader aufgerufen.

        Args:
            idx: Index des gewünschten Bildes.

        Returns:
            Dictionary mit:
              - 'image': Normalisiertes Bild als Tensor [3, H, W].
              - 'heatmaps': Ziel-Heatmaps als Tensor [4, Hm, Wm].
              - 'corners': Normalisierte Eckpunkt-Koordinaten [4, 2].
              - 'confidence': Confidence-Zielwert [1] (immer 1.0, da ein Dokument vorhanden ist).
        """
        img_path = self.image_paths[idx]
        annotation = self._load_annotation(img_path)

        # Bild laden und von BGR (OpenCV-Standard) nach RGB konvertieren
        image = cv2.imread(str(img_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Eckpunkte laden [4, 2] — erwartet normalisiert auf [0, 1].
        corners = np.array(annotation["corners"], dtype=np.float32)

        # Robustheit gegen inkonsistente Annotationen: Einige JSONs speichern die
        # Eckpunkte versehentlich in Pixel-Koordinaten (Werte deutlich > 1) statt
        # normalisiert. In dem Fall anhand der tatsächlichen Bildmaße normalisieren,
        # damit downstream konsistent mit [0, 1] gerechnet wird.
        if np.abs(corners).max() > 1.5:
            img_h, img_w = image.shape[:2]
            corners[:, 0] /= img_w
            corners[:, 1] /= img_h

        # Padding hinzufügen und Koordinaten anpassen
        image, corners = self._apply_padding(image, corners)

        # Augmentation anwenden (Spiegeln, Perspektive, Rauschen etc.)
        # Keypoints werden in Pixel-Koordinaten übergeben und automatisch mittransformiert.
        h, w = image.shape[:2]
        keypoints = [(c[0] * w, c[1] * h) for c in corners]
        transformed = self.transforms(image=image, keypoints=keypoints)
        image = transformed["image"]
        kps = transformed["keypoints"]

        # Hinweis: A.Resize ist bereits Teil der Pipeline – kein zusätzliches cv2.resize.
        # Eckpunkte nach der Augmentation zurück in normalisierte Koordinaten [0, 1] umrechnen
        th, tw = self.input_size
        if len(kps) == 4:
            corners = np.array(
                [[kp[0] / tw, kp[1] / th] for kp in kps], dtype=np.float32
            )
            # Eckpunkte, die durch starke Perspektive aus dem Bild geschoben wurden,
            # auf den gültigen Bereich [0, 1] begrenzen (verhindert verrauschte Targets).
            corners = np.clip(corners, 0.0, 1.0)
        else:
            # Fallback: Falls Keypoints bei der Transformation verloren gingen,
            # die Original-Annotationen verwenden.
            corners = np.array(annotation["corners"], dtype=np.float32)

        # Bild normalisieren: [0, 255] → [0, 1] → ImageNet-zentriert
        image = image.astype(np.float32) / 255.0
        image = (image - self.mean) / self.std
        # Format: HWC → CHW (Kanäle, Höhe, Breite) – das PyTorch-Standardformat
        image = torch.from_numpy(image.transpose(2, 0, 1))  # [3, H, W]

        # Ziel-Heatmaps erzeugen: Gauß-Glocken an den Eckpunkt-Positionen
        heatmaps = generate_heatmap(corners, self.heatmap_size)
        heatmaps = torch.from_numpy(heatmaps)

        # Eckpunkte als Tensor
        corners = torch.from_numpy(corners)

        # Confidence: Immer 1.0, da in diesem Dataset jedes Bild ein Dokument enthält.
        # Bei Bildern ohne Dokument wäre dieser Wert 0.0.
        confidence = torch.tensor([1.0], dtype=torch.float32)

        return {
            "image": image,
            "heatmaps": heatmaps,
            "corners": corners,
            "confidence": confidence,
        }
