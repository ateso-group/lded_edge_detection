"""
Heatmap Head mit Soft-Argmax für Eckpunkt-Regression.

Erzeugt 4 Heatmap-Kanäle (TL, TR, BR, BL) + 1 Confidence-Kanal.
Soft-Argmax extrahiert differenzierbare (x, y)-Koordinaten aus den Heatmaps.

Was ist eine Heatmap?
    Eine Heatmap ist ein „Wärmebild" – eine Rastergrafik, bei der jeder Pixel einen
    Wert hat: hohe Werte (= „heiß") dort, wo das Netz eine Dokumenten-Ecke vermutet,
    niedrige Werte (= „kalt") überall sonst. Für jede der 4 Ecken (oben-links,
    oben-rechts, unten-rechts, unten-links) wird eine eigene Heatmap erzeugt.

Was ist Soft-Argmax?
    Soft-Argmax ist eine Methode, um aus einer Heatmap eine präzise (x, y)-Koordinate
    abzuleiten. Statt einfach den hellsten Pixel zu nehmen (= Argmax, nicht
    differenzierbar), berechnet Soft-Argmax einen gewichteten Durchschnitt über alle
    Pixel-Positionen. Das macht die Koordinaten-Berechnung differenzierbar, sodass
    das Netz per Gradient lernen kann, wo die Ecke genau liegt.
"""

# torch: Framework für neuronale Netze und Tensor-Berechnungen.
import torch
import torch.nn as nn
# F.softmax wandelt beliebige Werte in Wahrscheinlichkeiten um (Summe = 1).
import torch.nn.functional as F


class SoftArgmax2D(nn.Module):
    """Differenzierbarer Soft-Argmax über 2D-Heatmaps.

    Berechnet den Erwartungswert (gewichteten Durchschnitt) der (x, y)-Koordinaten,
    wobei die Heatmap-Werte als Gewichte dienen. Hohe Werte in der Heatmap „ziehen"
    die berechnete Koordinate zu sich hin.

    Analogie: Stell dir die Heatmap als Landkarte mit Erhebungen vor. Der
    Schwerpunkt (Massezentrum) dieser Erhebungen ist die berechnete Koordinate.
    """

    def __init__(self, temperature: float = 0.05):
        super().__init__()
        # temperature steuert, wie „spitz" die Gewichtsverteilung ist:
        # Klein (z.B. 0.1) → fast alles Gewicht auf dem Maximum → sehr präzise.
        # Groß (z.B. 10) → Gewicht verteilt sich gleichmäßiger → robuster, aber ungenauer.
        self.temperature = temperature

    def forward(self, heatmaps: torch.Tensor) -> torch.Tensor:
        """Soft-Argmax über Heatmaps.

        Args:
            heatmaps: [B, C, H, W] — C Heatmap-Kanäle (einer pro Eckpunkt).
                B = Batch-Größe, H/W = Höhe/Breite der Heatmap.

        Returns:
            coords: [B, C, 2] — (x, y) Koordinaten pro Kanal, normalisiert auf [0, 1].
        """
        B, C, H, W = heatmaps.shape

        # Spatial Softmax: Wandelt die Heatmap-Rohwerte in Wahrscheinlichkeiten um.
        # Jeder Pixel bekommt einen Wert zwischen 0 und 1; die Summe aller Pixel = 1.
        # Division durch temperature kontrolliert die Schärfe der Verteilung.
        flat = heatmaps.view(B, C, -1) / self.temperature
        weights = F.softmax(flat, dim=-1)
        weights = weights.view(B, C, H, W)

        # Koordinaten-Gitter: Erstellt für jede Pixel-Position einen x- und y-Wert
        # im Bereich [0, 1]. (0, 0) = oben-links, (1, 1) = unten-rechts.
        device = heatmaps.device
        y_coords = torch.linspace(0, 1, H, device=device).view(1, 1, H, 1).expand(B, C, H, W)
        x_coords = torch.linspace(0, 1, W, device=device).view(1, 1, 1, W).expand(B, C, H, W)

        # Erwartungswert: Gewichteter Durchschnitt der x- und y-Koordinaten.
        # Pixel mit hoher Wahrscheinlichkeit tragen mehr zum Ergebnis bei.
        x = (weights * x_coords).sum(dim=(2, 3))
        y = (weights * y_coords).sum(dim=(2, 3))

        return torch.stack([x, y], dim=-1)  # [B, C, 2]


class HeatmapHead(nn.Module):
    """Heatmap Head: Erzeugt Corner-Heatmaps + Confidence Score.

    Dieses Modul ist der letzte Teil des Netzwerks. Es nimmt die fusionierte
    Feature Map vom FPN und erzeugt daraus:
      1. Heatmaps – Wärmebilder, die zeigen, wo jede Ecke liegt.
      2. Koordinaten – präzise (x, y)-Positionen der Ecken (via Soft-Argmax).
      3. Confidence – wie sicher das Netz ist, dass überhaupt ein Dokument im Bild ist.

    Args:
        in_channels: Anzahl der Eingangskanäle der Feature Map (z.B. 128).
        num_corners: Anzahl Eckpunkte (Standard: 4 für TL, TR, BR, BL –
            top-left, top-right, bottom-right, bottom-left).
        temperature: Soft-Argmax Temperatur (siehe SoftArgmax2D).
    """

    def __init__(
        self,
        in_channels: int = 128,
        num_corners: int = 4,
        temperature: float = 0.05,
    ):
        super().__init__()
        self.num_corners = num_corners

        # Heatmap-Regression: Reduziert die Feature Map von in_channels Kanälen
        # auf num_corners Kanäle (eine Heatmap pro Ecke).
        # Conv2d = Faltungsschicht: Verarbeitet räumliche Muster im Bild.
        # BatchNorm2d = Normalisierung für stabiles Training.
        # ReLU6 = Aktivierung: Setzt negative Werte auf 0, begrenzt auf max. 6.
        self.heatmap_conv = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // 2, 3, padding=1, bias=False),
            nn.BatchNorm2d(in_channels // 2),
            nn.ReLU6(inplace=True),
            # Letzte 1×1-Faltung: Erzeugt genau num_corners Ausgabe-Kanäle.
            nn.Conv2d(in_channels // 2, num_corners, 1),
            # Sigmoid: Begrenzt die Heatmap-Werte auf [0, 1] – passend zu den
            # Gaussian-Targets und nötig für korrektes Soft-Argmax-Verhalten.
            nn.Sigmoid(),
        )

        # Confidence Head: Bestimmt, ob ein Dokument im Bild vorhanden ist.
        # AdaptiveAvgPool2d(1): Berechnet den Durchschnitt über alle Pixel → ein Wert pro Kanal.
        # Flatten: Formt den Tensor in einen 1D-Vektor um.
        # Linear: Vollverbundene Schicht (jeder Eingang mit jedem Ausgang verknüpft).
        # Am Ende: 1 Ausgabe-Neuron → Confidence-Logit (wird später per Sigmoid
        # in eine Wahrscheinlichkeit [0, 1] umgewandelt).
        self.confidence_head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(in_channels, 64),
            nn.ReLU6(inplace=True),
            nn.Linear(64, 1),
        )

        # Soft-Argmax wandelt die Heatmaps in präzise (x, y)-Koordinaten um.
        self.soft_argmax = SoftArgmax2D(temperature=temperature)

    def forward(
        self, features: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward Pass – erzeugt Heatmaps, Koordinaten und Confidence.

        Args:
            features: [B, C, H, W] Feature Map vom FPN-Neck.

        Returns:
            heatmaps: [B, num_corners, H, W] — Rohwerte der Heatmaps
                (hoher Wert = hier vermutet das Netz eine Ecke).
            coords: [B, num_corners, 2] — (x, y) Koordinaten jeder Ecke,
                normalisiert auf [0, 1].
            confidence: [B, 1] — Logit für Dokument-Confidence
                (positiv = Dokument erkannt, negativ = kein Dokument).
        """
        heatmaps = self.heatmap_conv(features)
        coords = self.soft_argmax(heatmaps)
        confidence = self.confidence_head(features)
        return heatmaps, coords, confidence
