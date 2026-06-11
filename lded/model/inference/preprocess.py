"""
Pre-Processing Pipeline für Inferenz.

Schritte:
1. Symmetrisches Padding (5 % des Bildumfangs)
2. Resize → 512×512
3. Normalisierung: μ=[0.485, 0.456, 0.406], σ=[0.229, 0.224, 0.225]

Was ist Pre-Processing?
    Bevor ein Bild dem neuronalen Netz übergeben wird, muss es in ein
    einheitliches Format gebracht werden. Dazu gehören:
    - Padding: Einen schwarzen Rand hinzufügen, damit Ecken am Bildrand nicht
      abgeschnitten werden.
    - Resize: Das Bild auf die feste Eingangsgröße des Modells (512×512) skalieren.
    - Normalisierung: Pixelwerte (0–255) in kleine Zahlen umwandeln, die das Netz
      besser verarbeiten kann. Die Werte μ (Mittelwert) und σ (Standardabweichung)
      stammen aus dem ImageNet-Datensatz und sorgen dafür, dass die Eingabedaten
      zum vortrainierten Backbone passen.
"""

# cv2 (OpenCV): Bibliothek zur Bildverarbeitung (Resize, Padding, Farbkonvertierung etc.).
import cv2
# numpy: Bibliothek für effiziente numerische Berechnungen mit mehrdimensionalen Arrays.
import numpy as np


class Preprocessor:
    """Bildvorverarbeitung für LDED-Inferenz.

    Bereitet ein beliebiges Eingabebild so auf, dass es dem Modell übergeben
    werden kann, und speichert Metadaten, um die Ergebnisse anschließend
    zurück in den Original-Bildraum umrechnen zu können.

    Args:
        input_size: Ziel-Größe (Höhe, Breite) in Pixel, auf die das Bild skaliert wird.
        padding_ratio: Anteil des Bildumfangs (Höhe + Breite), der als schwarzer
            Rand hinzugefügt wird. 0.05 = 5 % → bei einem 1000×1000-Bild sind das
            100 Pixel Padding auf jeder Seite.
        mean: RGB-Mittelwerte für die Normalisierung. Diese Werte stammen aus dem
            ImageNet-Datensatz und passen zum vortrainierten Backbone.
        std: RGB-Standardabweichungen für die Normalisierung.
    """

    def __init__(
        self,
        input_size: tuple[int, int] = (512, 512),
        padding_ratio: float = 0.05,
        mean: tuple[float, ...] = (0.485, 0.456, 0.406),
        std: tuple[float, ...] = (0.229, 0.224, 0.225),
    ):
        self.input_size = input_size
        self.padding_ratio = padding_ratio
        self.mean = np.array(mean, dtype=np.float32)
        self.std = np.array(std, dtype=np.float32)

    def add_padding(self, image: np.ndarray) -> tuple[np.ndarray, int]:
        """Fügt einen symmetrischen schwarzen Rand um das Bild hinzu.

        Der Rand verhindert, dass Dokumenten-Ecken am Bildrand abgeschnitten
        werden und gibt dem Modell Kontext über den Bildrand hinaus.

        Args:
            image: Eingabebild (H, W, 3) in BGR oder RGB.

        Returns:
            padded: Bild mit schwarzem Rand.
            pad: Breite des Rands in Pixel.
        """
        h, w = image.shape[:2]
        # Padding-Größe = 5 % des Bildumfangs (Höhe + Breite)
        pad = int(self.padding_ratio * (h + w))
        # copyMakeBorder fügt oben, unten, links, rechts jeweils 'pad' Pixel
        # schwarzen Rand (value=0,0,0) hinzu.
        padded = cv2.copyMakeBorder(
            image, pad, pad, pad, pad,
            cv2.BORDER_CONSTANT, value=(0, 0, 0),
        )
        return padded, pad

    def preprocess(self, image: np.ndarray) -> tuple[np.ndarray, dict]:
        """Vollständige Vorverarbeitung: Padding → Resize → Normalisierung → Tensor-Format.

        Args:
            image: Eingabebild (H, W, 3) in RGB.

        Returns:
            tensor: Vorverarbeitetes Bild als float32-Array der Form [1, 3, H, W].
                1 = Batch-Dimension (ein einzelnes Bild),
                3 = Farbkanäle (RGB),
                H, W = Ziel-Höhe und -Breite.
            meta: Dictionary mit Metadaten, die für die Rücktransformation der
                Koordinaten benötigt werden (Originalgröße, Padding, Skalierung).
        """
        orig_h, orig_w = image.shape[:2]

        # 1) Padding: Schwarzen Rand hinzufügen
        padded, pad = self.add_padding(image)
        padded_h, padded_w = padded.shape[:2]

        # 2) Resize: Auf die feste Modell-Eingangsgröße skalieren (z.B. 512×512)
        resized = cv2.resize(padded, (self.input_size[1], self.input_size[0]))

        # 3) Normalisierung:
        #    - Pixelwerte von [0, 255] auf [0, 1] skalieren (/ 255)
        #    - Dann mit ImageNet-Statistiken zentrieren: (Wert - Mittelwert) / Standardabweichung
        #    Das sorgt dafür, dass die Eingabewerte im gleichen Bereich liegen wie
        #    die Trainingsdaten des vortrainierten Backbones.
        normalized = resized.astype(np.float32) / 255.0
        normalized = (normalized - self.mean) / self.std

        # 4) Format-Umwandlung:
        #    - transpose(2, 0, 1): Ändert das Format von HWC (Höhe, Breite, Kanäle)
        #      zu CHW (Kanäle, Höhe, Breite) – das Format, das PyTorch/ONNX erwartet.
        #    - [np.newaxis, ...]: Fügt eine Batch-Dimension hinzu → [1, 3, H, W].
        tensor = normalized.transpose(2, 0, 1)[np.newaxis, ...]

        # Metadaten speichern, damit die Ergebnisse später zurückgerechnet werden können.
        meta = {
            "orig_size": (orig_h, orig_w),
            "padded_size": (padded_h, padded_w),
            "pad": pad,
            "scale_x": padded_w / self.input_size[1],
            "scale_y": padded_h / self.input_size[0],
        }

        return tensor, meta
