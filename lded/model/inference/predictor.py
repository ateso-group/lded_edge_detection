"""
ONNX-basierter Predictor für LDED.

Führt Inferenz mit ONNX Runtime durch und wandelt Heatmap-Koordinaten
zurück in den Original-Bildraum.

Was ist Inferenz?
    Inferenz ist der Einsatz eines fertig trainierten Modells, um Vorhersagen
    auf neuen, ungesehenen Bildern zu machen. Im Gegensatz zum Training werden
    dabei keine Gewichte verändert – das Modell wird nur „benutzt".

Was ist ONNX?
    ONNX (Open Neural Network Exchange) ist ein offenes Format, um trainierte
    Modelle zwischen verschiedenen Frameworks (PyTorch, TensorFlow, etc.) und
    Geräten auszutauschen. ONNX Runtime ist eine schnelle Laufzeitumgebung,
    die ONNX-Modelle effizient ausführt – auch ohne GPU.
"""

# numpy: Bibliothek für effiziente Berechnungen mit mehrdimensionalen Zahlenarrays.
import numpy as np
# onnxruntime: Führt ONNX-Modelle effizient aus (schnelle Inferenz).
import onnxruntime as ort

from model.inference.preprocess import Preprocessor


class LDEDPredictor:
    """ONNX-basierte Dokumenteneck-Erkennung.

    Diese Klasse kapselt den gesamten Vorhersage-Ablauf:
    Bild einlesen → vorverarbeiten → Modell ausführen → Ergebnis zurückrechnen.

    Args:
        model_path: Dateipfad zum exportierten ONNX-Modell (z.B. "lded.onnx").
        input_size: Bildgröße (Höhe, Breite), die das Modell erwartet. Das Eingabebild
            wird automatisch auf diese Größe skaliert.
        padding_ratio: Anteil des Bildumfangs, der als Rand hinzugefügt wird (z.B. 0.05 = 5 %).
            Das Padding stellt sicher, dass Eckpunkte am Bildrand nicht abgeschnitten werden.
        confidence_threshold: Mindestwert der Confidence (0–1), ab dem eine Erkennung
            als „Dokument gefunden" gilt. Beispiel: 0.5 = das Modell muss mindestens
            50 % sicher sein.
        providers: Liste der ONNX Runtime Execution Providers – bestimmt, auf welcher
            Hardware das Modell ausgeführt wird (z.B. ["CUDAExecutionProvider"] für GPU,
            ["CPUExecutionProvider"] für CPU).
    """

    def __init__(
        self,
        model_path: str,
        input_size: tuple[int, int] = (512, 512),
        padding_ratio: float = 0.05,
        confidence_threshold: float = 0.5,
        providers: list[str] | None = None,
    ):
        self.confidence_threshold = confidence_threshold

        if providers is None:
            providers = ["CPUExecutionProvider"]

        # InferenceSession lädt das ONNX-Modell und bereitet es zur Ausführung vor.
        self.session = ort.InferenceSession(model_path, providers=providers)
        # Preprocessor bereitet das Eingabebild für das Modell auf (Resize, Normalisierung etc.).
        self.preprocessor = Preprocessor(
            input_size=input_size,
            padding_ratio=padding_ratio,
        )

    def predict(self, image: np.ndarray) -> dict:
        """Führt Dokumenteneck-Erkennung auf einem Bild durch.

        Ablauf:
          1. Bild vorverarbeiten (Padding, Resize, Normalisierung).
          2. Modell ausführen (ONNX Inferenz).
          3. Koordinaten vom Modell-Raum [0, 1] zurück in Pixel des Originalbilds rechnen.

        Args:
            image: Eingabebild als NumPy-Array (H, W, 3) in RGB-Farbformat.

        Returns:
            Dictionary mit:
                - corners: [4, 2] – die vier Eckpunkte in Pixel-Koordinaten
                    des Originalbilds. Reihenfolge: TL, TR, BR, BL.
                - confidence: Zahl zwischen 0 und 1 – wie sicher das Modell ist,
                    dass ein Dokument erkannt wurde.
                - detected: True/False – ob die Confidence über dem Schwellwert liegt.
        """
        # 1) Bild vorverarbeiten: Padding → Resize → Normalisierung
        tensor, meta = self.preprocessor.preprocess(image)

        # 2) ONNX Inferenz: Bild durchs Modell schicken
        input_name = self.session.get_inputs()[0].name
        outputs = self.session.run(None, {input_name: tensor})

        # Ergebnisse auspacken:
        # outputs[0] = Eckpunkt-Koordinaten [B, 4, 2], normalisiert auf [0, 1]
        coords = outputs[0][0]  # [4, 2] – erstes (und einziges) Bild im Batch
        # outputs[1] = Confidence-Logit (Rohwert, noch keine Wahrscheinlichkeit)
        confidence_logit = outputs[1][0, 0]  # Skalar

        # Sigmoid: Wandelt den Logit in eine Wahrscheinlichkeit [0, 1] um.
        # Formel: sigmoid(x) = 1 / (1 + e^(-x))
        confidence = 1.0 / (1.0 + np.exp(-confidence_logit))

        # 3) Koordinaten zurück in den Original-Bildraum umrechnen
        corners = self._rescale_corners(coords, meta)

        return {
            "corners": corners,
            "confidence": float(confidence),
            "detected": bool(confidence >= self.confidence_threshold),
        }

    def _rescale_corners(
        self, coords: np.ndarray, meta: dict
    ) -> np.ndarray:
        """Skaliert normalisierte Koordinaten zurück in den Original-Bildraum.

        Da das Bild vor der Inferenz gepaddet und resized wurde, müssen die
        vom Modell vorhergesagten Koordinaten [0, 1] wieder in die tatsächlichen
        Pixel-Positionen des Originalbilds umgerechnet werden.

        Args:
            coords: [4, 2] normalisierte Koordinaten (x, y) im Bereich [0, 1].
            meta: Metadaten aus dem Preprocessing (Originalgrößen, Padding etc.).

        Returns:
            corners: [4, 2] Pixel-Koordinaten im Originalbild.
        """
        padded_h, padded_w = meta["padded_size"]
        pad = meta["pad"]
        orig_h, orig_w = meta["orig_size"]

        # Schritt 1: [0, 1] → Pixel-Koordinaten im gepaddeten Bild
        corners_px = coords.copy()
        corners_px[:, 0] *= padded_w  # x-Koordinaten skalieren
        corners_px[:, 1] *= padded_h  # y-Koordinaten skalieren

        # Schritt 2: Padding-Offset abziehen → Koordinaten im ungepaddeten Bild
        corners_px[:, 0] -= pad
        corners_px[:, 1] -= pad

        # Schritt 3: Clipping – sicherstellen, dass Koordinaten innerhalb
        # der Originalbildgrenzen liegen (nicht negativ und nicht zu groß).
        corners_px[:, 0] = np.clip(corners_px[:, 0], 0, orig_w - 1)
        corners_px[:, 1] = np.clip(corners_px[:, 1], 0, orig_h - 1)

        return corners_px
