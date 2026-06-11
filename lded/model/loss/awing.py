"""
Adaptive Wing Loss für robuste Heatmap-Regression.

Kombiniert Wing Loss mit einer adaptiven Gewichtung, die kleine Fehler
stärker bestraft — ideal für präzise Keypoint-Lokalisierung.

Was ist ein Loss (Verlustfunktion)?
    Der Loss misst, wie weit die Vorhersage des Modells vom richtigen Ergebnis
    entfernt ist. Je kleiner der Loss, desto besser die Vorhersage. Während des
    Trainings versucht das Netz, den Loss zu minimieren – es „lernt", indem es
    seine Gewichte so anpasst, dass der Loss kleiner wird.

Warum Adaptive Wing Loss?
    Standard-Verlustfunktionen (z.B. MSE) behandeln kleine und große Fehler gleich.
    Adaptive Wing Loss bestraft dagegen kleine Fehler überproportional stark, was
    besonders wichtig ist, wenn es auf präzise Eckpunkt-Lokalisierung ankommt:
    Ein Fehler von 1 Pixel ist bei Dokumenten-Ecken kritischer als ein Fehler von 20 Pixel.

Referenz: Wang et al., "Adaptive Wing Loss for Robust Face Alignment
via Heatmap Regression" (ICCV 2019).
"""

# math: Python-Standardbibliothek für mathematische Funktionen (z.B. Logarithmus).
import math

import torch
import torch.nn as nn


class AdaptiveWingLoss(nn.Module):
    """Adaptive Wing Loss.

    Berechnet den Fehler zwischen vorhergesagter und Ziel-Heatmap mit zwei Bereichen:
      - Kleine Fehler (< theta): Nicht-linearer Bereich → überproportional bestraft,
        damit das Netz bei kleinen Abweichungen besonders genau wird.
      - Große Fehler (≥ theta): Linearer Bereich → moderate Bestrafung,
        damit Ausreißer das Training nicht destabilisieren.

    Args:
        omega: Skalierungsfaktor für den nicht-linearen Bereich – höhere Werte
            verstärken die Bestrafung kleiner Fehler.
        theta: Schwellwert, ab dem vom nicht-linearen in den linearen Bereich
            gewechselt wird. Fehler < theta → nicht-linear, ≥ theta → linear.
        epsilon: Stabilisierungswert, der Division durch null verhindert.
        alpha: Exponent (> 1, typisch 2.1) – steuert die Krümmung der
            nicht-linearen Funktion. Höhere Werte → stärkere Bestrafung kleiner Fehler.
    """

    def __init__(
        self,
        omega: float = 14.0,
        theta: float = 0.5,
        epsilon: float = 1.0,
        alpha: float = 2.1,
    ):
        super().__init__()
        self.omega = omega
        self.theta = theta
        self.epsilon = epsilon
        self.alpha = alpha

        # Vorab berechnete Konstanten (A und C), damit die Funktion am Schwellwert
        # theta stetig und glatt ist (kein „Knick" in der Kurve).
        self._A = self.omega * (
            1 / (1 + (self.theta / self.epsilon) ** (self.alpha - 1))
        ) * (self.alpha - 1) * (
            (self.theta / self.epsilon) ** (self.alpha - 2)
        ) * (1 / self.epsilon)

        self._C = self.theta * self._A - self.omega * math.log(
            1 + (self.theta / self.epsilon) ** (self.alpha - 1)
        )

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Berechnet Adaptive Wing Loss.

        Args:
            pred: Vorhergesagte Heatmaps [B, C, H, W] – was das Modell vorhersagt.
            target: Ziel-Heatmaps [B, C, H, W] – der gewünschte Sollwert (Ground Truth).

        Returns:
            Skalarer Loss-Wert (eine einzelne Zahl) – je kleiner, desto besser.
        """
        # delta = absoluter Fehler pro Pixel (Betrag der Differenz).
        delta = (pred - target).abs()

        # Nicht-linearer Bereich (kleine Fehler < theta):
        # Logarithmische Funktion → kleine Fehler werden überproportional bestraft.
        loss_nonlinear = self.omega * torch.log(
            1 + (delta / self.epsilon) ** (self.alpha - 1)
        )

        # Linearer Bereich (große Fehler ≥ theta):
        # Einfache Gerade → moderate, stabile Bestrafung.
        loss_linear = self._A * delta - self._C

        # torch.where wählt pro Pixel: kleiner Fehler → nicht-linear, sonst → linear.
        loss = torch.where(delta < self.theta, loss_nonlinear, loss_linear)
        # .mean() → Durchschnitt über alle Pixel → ein einziger Loss-Wert.
        return loss.mean()


class LDEDLoss(nn.Module):
    """Kombinierter Loss für LDED: Adaptive Wing + BCE + Koordinaten-L1.

    Das Modell hat drei Aufgaben, die jeweils einen eigenen Loss bekommen:
      1. Heatmap-Loss (Adaptive Wing): Wie genau sind die vorhergesagten Heatmaps?
      2. Confidence-Loss (BCE): Wie gut erkennt das Modell, ob ein Dokument vorhanden ist?
      3. Koordinaten-Loss (L1): Wie weit liegen die Soft-Argmax-Koordinaten von den
         tatsächlichen Eckpunkten entfernt? Gibt dem Netz ein direktes Signal zur
         Positionskorrektur.

    Args:
        awing_weight: Gewichtungsfaktor für den Heatmap-Loss. Höher = Heatmap-Genauigkeit
            wird stärker priorisiert.
        bce_weight: Gewichtungsfaktor für den Confidence-Loss. Niedriger = Confidence
            hat weniger Einfluss auf das Training.
        coord_weight: Gewichtungsfaktor für den Koordinaten-Loss. Höher = direktere
            Optimierung der Eckpunkt-Positionen.
    """

    def __init__(self, awing_weight: float = 1.0, bce_weight: float = 0.1, coord_weight: float = 5.0):
        super().__init__()
        self.awing_weight = awing_weight
        self.bce_weight = bce_weight
        self.coord_weight = coord_weight
        self.awing_loss = AdaptiveWingLoss()
        # BCEWithLogitsLoss: Binary Cross Entropy Loss – Standard-Verlustfunktion
        # für Ja/Nein-Entscheidungen (hier: „Ist ein Dokument im Bild?").
        # „WithLogits" bedeutet: Sigmoid wird intern angewendet, man gibt direkt
        # den Rohwert (Logit) ein.
        self.bce_loss = nn.BCEWithLogitsLoss()
        # SmoothL1Loss (Huber): Robust gegen Ausreißer, stabilere Gradienten als
        # reines L1 – direkt auf die Soft-Argmax-Koordinaten angewandt.
        self.coord_loss = nn.SmoothL1Loss()

    def forward(
        self,
        pred_heatmaps: torch.Tensor,
        target_heatmaps: torch.Tensor,
        pred_confidence: torch.Tensor,
        target_confidence: torch.Tensor,
        pred_coords: torch.Tensor | None = None,
        target_coords: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """Berechnet den kombinierten Loss.

        Args:
            pred_heatmaps: [B, 4, H, W] – vom Modell vorhergesagte Heatmaps.
            target_heatmaps: [B, 4, H, W] – Ziel-Heatmaps (Ground Truth).
            pred_confidence: [B, 1] – Confidence-Logit des Modells.
            target_confidence: [B, 1] – Ziel-Confidence (1.0 = Dokument vorhanden,
                0.0 = kein Dokument).
            pred_coords: [B, 4, 2] – vom Soft-Argmax vorhergesagte Koordinaten (optional).
            target_coords: [B, 4, 2] – Ziel-Koordinaten normalisiert auf [0, 1] (optional).

        Returns:
            Dictionary mit Loss-Werten:
              - 'total': Gewichtete Summe (das wird beim Training optimiert).
              - 'awing': Nur der Heatmap-Loss (zum Monitoring/Überwachung).
              - 'bce': Nur der Confidence-Loss (zum Monitoring/Überwachung).
              - 'coord': Nur der Koordinaten-Loss (zum Monitoring/Überwachung).
        """
        l_awing = self.awing_loss(pred_heatmaps, target_heatmaps)
        l_bce = self.bce_loss(pred_confidence, target_confidence)
        # Gewichtete Summe: total = awing_weight × Heatmap-Loss + bce_weight × Confidence-Loss
        total = self.awing_weight * l_awing + self.bce_weight * l_bce

        # Koordinaten-Loss: Direkte L1-Distanz zwischen vorhergesagten und echten Eckpunkten
        l_coord = torch.tensor(0.0, device=pred_heatmaps.device)
        if pred_coords is not None and target_coords is not None:
            l_coord = self.coord_loss(pred_coords, target_coords)
            total = total + self.coord_weight * l_coord

        return {
            "total": total,
            "awing": l_awing,
            "bce": l_bce,
            "coord": l_coord,
        }
