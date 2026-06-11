"""
Evaluationsmetriken für LDED.

Metriken messen, wie gut das Modell arbeitet. Hier werden drei verschiedene
Bewertungsmaße verwendet, die jeweils einen anderen Aspekt der Genauigkeit erfassen:

- PCK (Percentage of Correct Keypoints):
    „Wie viel Prozent der vorhergesagten Eckpunkte liegen nahe genug an der
    tatsächlichen Position?" Nahe genug = innerhalb einer bestimmten Pixel-Toleranz.
    Beispiel: PCK@5 = 0.95 bedeutet: 95 % der Eckpunkte weichen max. 5 Pixel ab.

- NME (Normalized Mean Error):
    „Wie groß ist der durchschnittliche Fehler relativ zur Bildgröße?"
    Wird durch die Bilddiagonale normalisiert, damit der Wert unabhängig von
    der Bildauflösung vergleichbar ist. Kleiner = besser.

- IoU (Intersection over Union):
    „Wie stark überlappen sich das vorhergesagte und das tatsächliche Dokument-Viereck?"
    IoU = 1.0 = perfekte Übereinstimmung, IoU = 0.0 = keine Überlappung.
"""

# numpy: Bibliothek für effiziente numerische Berechnungen mit Arrays.
import numpy as np
# Polygon aus shapely: Ermöglicht geometrische Berechnungen mit Vielecken
# (Fläche, Überlappung, Vereinigung etc.).
from shapely.geometry import Polygon


def pck(
    pred: np.ndarray,
    target: np.ndarray,
    threshold: float = 5.0,
    image_size: tuple[int, int] = (512, 512),
) -> float:
    """Percentage of Correct Keypoints – Anteil korrekt erkannter Eckpunkte.

    Ein Eckpunkt gilt als „korrekt", wenn sein Abstand zur tatsächlichen
    Position kleiner als die Toleranzschwelle (threshold) ist.

    Args:
        pred: [N, 4, 2] vorhergesagte Eckpunkte in Pixel.
            N = Anzahl Bilder, 4 = Ecken, 2 = (x, y).
        target: [N, 4, 2] tatsächliche Eckpunkte (Ground Truth) in Pixel.
        threshold: Toleranz in Pixel – maximaler Abstand, bei dem ein Punkt
            noch als „korrekt" gilt (z.B. 5.0 = max. 5 Pixel Abweichung).
        image_size: Bildgröße (wird hier nicht verwendet, aber für Kompatibilität).

    Returns:
        PCK-Wert zwischen 0 und 1 (0 = alle falsch, 1 = alle korrekt).
    """
    # Euklidischer Abstand (Luftlinie) zwischen Vorhersage und Ground Truth pro Eckpunkt
    distances = np.linalg.norm(pred - target, axis=-1)  # [N, 4]
    # Prüfen, ob jeder Abstand unter der Toleranz liegt
    correct = (distances < threshold).astype(np.float32)
    # Anteil der korrekten Punkte (Durchschnitt über alle Bilder und Ecken)
    return float(correct.mean())


def nme(
    pred: np.ndarray,
    target: np.ndarray,
    image_size: tuple[int, int] = (512, 512),
) -> float:
    """Normalized Mean Error – durchschnittlicher Fehler relativ zur Bildgröße.

    Der Fehler wird durch die Bilddiagonale geteilt, damit der Wert unabhängig
    von der Bildauflösung ist (ein 5-Pixel-Fehler ist bei einem 100×100-Bild
    schlimmer als bei einem 4000×3000-Bild).

    Args:
        pred: [N, 4, 2] vorhergesagte Eckpunkte in Pixel.
        target: [N, 4, 2] tatsächliche Eckpunkte (Ground Truth) in Pixel.
        image_size: (Höhe, Breite) des Bildes – wird für die Diagonale benötigt.

    Returns:
        NME-Wert zwischen 0 und 1 (kleiner = genauer).
    """
    # Bilddiagonale berechnen: √(H² + W²) – z.B. bei 512×512 ≈ 724 Pixel
    diag = np.sqrt(image_size[0] ** 2 + image_size[1] ** 2)
    # Abstände berechnen und durch Diagonale normalisieren
    distances = np.linalg.norm(pred - target, axis=-1)  # [N, 4]
    return float((distances / diag).mean())


def polygon_iou(pred_corners: np.ndarray, target_corners: np.ndarray) -> float:
    """IoU (Intersection over Union) zwischen zwei 4-Eck-Polygonen.

    Misst die Überlappung der beiden Vierecke:
    IoU = Schnittfläche / Vereinigungsfläche.
    IoU = 1.0 → perfekte Übereinstimmung, IoU = 0.0 → keine Überlappung.

    Args:
        pred_corners: [4, 2] vorhergesagte Eckpunkte (x, y).
        target_corners: [4, 2] tatsächliche Eckpunkte (Ground Truth).

    Returns:
        IoU-Wert zwischen 0 und 1.
    """
    try:
        # Polygone (Vielecke) aus den Eckpunkten erstellen
        poly_pred = Polygon(pred_corners)
        poly_target = Polygon(target_corners)

        # Prüfen, ob die Polygone geometrisch gültig sind
        # (z.B. keine sich selbst kreuzenden Kanten)
        if not poly_pred.is_valid or not poly_target.is_valid:
            return 0.0

        # Schnittfläche (Bereich, der in BEIDEN Polygonen liegt)
        intersection = poly_pred.intersection(poly_target).area
        # Vereinigungsfläche (Bereich, der in MINDESTENS EINEM Polygon liegt)
        union = poly_pred.union(poly_target).area

        if union == 0:
            return 0.0
        return float(intersection / union)
    except Exception:
        return 0.0


def batch_polygon_iou(pred: np.ndarray, target: np.ndarray) -> float:
    """Durchschnittliche Polygon-IoU über einen Batch (Stapel von Bildern).

    Args:
        pred: [N, 4, 2] vorhergesagte Eckpunkte für N Bilder.
        target: [N, 4, 2] tatsächliche Eckpunkte für N Bilder.

    Returns:
        Mittlere IoU über alle N Bilder.
    """
    ious = [polygon_iou(p, t) for p, t in zip(pred, target)]
    return float(np.mean(ious)) if ious else 0.0


def compute_all_metrics(
    pred: np.ndarray,
    target: np.ndarray,
    image_size: tuple[int, int] = (512, 512),
) -> dict[str, float]:
    """Berechnet alle Evaluationsmetriken auf einmal.

    Praktische Hilfsfunktion, die PCK@5, PCK@10, NME und IoU in einem
    Aufruf berechnet und als Dictionary zurückgibt.

    Args:
        pred: [N, 4, 2] vorhergesagte Eckpunkte in Pixel.
        target: [N, 4, 2] tatsächliche Eckpunkte (Ground Truth) in Pixel.
        image_size: (Höhe, Breite) des Bildes.

    Returns:
        Dictionary mit:
          - 'pck5': PCK mit 5 Pixel Toleranz
          - 'pck10': PCK mit 10 Pixel Toleranz
          - 'nme': Normalisierter mittlerer Fehler
          - 'iou': Mittlere Polygon-Überlappung
    """
    return {
        "pck5": pck(pred, target, threshold=5.0),
        "pck10": pck(pred, target, threshold=10.0),
        "nme": nme(pred, target, image_size),
        "iou": batch_polygon_iou(pred, target),
    }
