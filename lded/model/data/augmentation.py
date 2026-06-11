"""
Augmentations-Pipelines für Training und Validierung.

Basierend auf Albumentations mit Keypoint-Support.

Was ist Augmentation (Datenvervielfältigung)?
    Augmentation erzeugt aus jedem Trainingsbild viele verschiedene Varianten durch
    zufällige Veränderungen (Spiegeln, Drehen, Helligkeit ändern, Rauschen etc.).
    Dadurch „sieht" das Modell deutlich mehr verschiedene Bilder und lernt besser,
    mit realen Bedingungen (unterschiedliche Beleuchtung, Kamerawinkel etc.) umzugehen.

    Wichtig: Die Eckpunkte (Keypoints) werden bei geometrischen Transformationen
    (z.B. Spiegeln, Perspektive) automatisch mit verschoben, damit sie weiterhin
    zur richtigen Position im veränderten Bild passen.

    Bei der Validierung werden KEINE Augmentationen angewendet (nur Resize), um die
    echte Leistung des Modells auf unveränderten Bildern zu messen.
"""

# albumentations: Bibliothek für schnelle und flexible Bild-Augmentationen.
# Unterstützt auch die automatische Transformation von Keypoints (Eckpunkten).
import albumentations as A


def get_train_transforms(input_size: tuple[int, int] = (512, 512)) -> A.Compose:
    """Augmentations-Pipeline für das Training.

    Definiert eine Kette von zufälligen Bildveränderungen, die beim Training
    angewendet werden. Jede Transformation hat eine Wahrscheinlichkeit (p),
    mit der sie bei jedem Bild aktiviert wird.

    Args:
        input_size: Ziel-Bildgröße (Höhe, Breite) in Pixel.

    Returns:
        Albumentations Compose-Objekt – eine verkettete Augmentations-Pipeline.
    """
    # Hinweis: KEIN HorizontalFlip. Die 4 Eckpunkte haben eine feste Reihenfolge
    # (TL, TR, BR, BL). Ein Flip spiegelt zwar die Positionen, vertauscht aber nicht
    # die Label-Indizes → inkonsistente Targets. Wenn Flip gewünscht ist, müssen
    # TL↔TR und BL↔BR nach dem Flip explizit getauscht werden.
    return A.Compose(
        [
            # Perspective: Simuliert verschiedene Kamerawinkel durch perspektivische
            # Verzerrung. scale = Stärke der Verzerrung. Hauptquelle der Geometrie-Diversität.
            A.Perspective(scale=(0.02, 0.08), p=0.7),
            # RandomBrightnessContrast: Ändert zufällig Helligkeit und Kontrast,
            # um unterschiedliche Lichtverhältnisse zu simulieren.
            A.RandomBrightnessContrast(
                brightness_limit=(-0.5, 0.5),
                contrast_limit=(-0.3, 0.3),
                p=0.5,
            ),
            # MotionBlur: Simuliert Bewegungsunschärfe (z.B. wenn die Kamera wackelt).
            # blur_limit = Stärke der Unschärfe in Pixeln.
            A.MotionBlur(blur_limit=(3, 7), p=0.3),
            # ImageCompression: Simuliert JPEG-Kompressionsartefakte (niedrige Bildqualität).
            # quality_lower/upper = Bereich der simulierten JPEG-Qualität (0–100).
            A.ImageCompression(quality_lower=50, quality_upper=95, p=0.3),
            # GaussNoise: Fügt zufälliges Bildrauschen hinzu (körniger Look).
            # var_limit = Varianz-Bereich des Rauschens.
            A.GaussNoise(var_limit=(5.0, 25.0), p=0.2),
            # Resize: Skaliert das Bild auf die feste Eingangsgröße des Modells.
            A.Resize(height=input_size[0], width=input_size[1]),
        ],
        # keypoint_params: Stellt sicher, dass die Eckpunkte (Keypoints) bei
        # geometrischen Transformationen korrekt mitbewegt werden.
        # format="xy" = Koordinaten als (x, y)-Paare.
        # remove_invisible=False = Punkte behalten, auch wenn sie nach der
        # Transformation außerhalb des Bildes liegen.
        keypoint_params=A.KeypointParams(format="xy", remove_invisible=False),
    )


def get_val_transforms(input_size: tuple[int, int] = (512, 512)) -> A.Compose:
    """Augmentations-Pipeline für die Validierung – nur Resize, keine Veränderungen.

    Bei der Validierung soll die echte Leistung des Modells gemessen werden,
    daher werden die Bilder nur auf die richtige Größe skaliert, ohne
    weitere Veränderungen.

    Args:
        input_size: Ziel-Bildgröße (Höhe, Breite) in Pixel.

    Returns:
        Albumentations Compose-Objekt.
    """
    return A.Compose(
        [
            A.Resize(height=input_size[0], width=input_size[1]),
        ],
        keypoint_params=A.KeypointParams(format="xy", remove_invisible=False),
    )
