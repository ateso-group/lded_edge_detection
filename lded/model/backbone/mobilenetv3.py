"""
MobileNetV3-Small Backbone mit Feature-Map Extraction über timm.

Liefert Multi-Scale Feature Maps (C3, C4, C5) für den FPN-Neck.

Was ist ein Backbone?
    Das Backbone ist der erste Teil eines Bilderkennungs-Netzwerks. Es funktioniert
    wie das menschliche Auge: Es schaut auf das Bild und erkennt grundlegende visuelle
    Merkmale wie Kanten, Ecken und Texturen. MobileNetV3-Small ist ein besonders
    leichtgewichtiges Netzwerk, das auch auf Smartphones schnell läuft.

Was sind Feature Maps?
    Feature Maps sind die „Zwischenergebnisse" des Netzwerks – sie zeigen, welche
    visuellen Muster das Netz an welcher Stelle im Bild erkannt hat. Multi-Scale
    bedeutet, dass das Netz Merkmale auf verschiedenen Detailstufen liefert:
    - C3: feine Details (hohe Auflösung, 1/4 des Originalbildes)
    - C4: mittlere Details (1/8)
    - C5: grobe Strukturen (1/32)

Verfügbare Stufen von MobileNetV3-Small (timm):
    MobileNetV3-Small stellt 5 Feature-Stufen (Index 0–4) bereit:

    Index | Stride | Auflösung (512×512) | Kanäle | Beschreibung
    ------|--------|---------------------|--------|---------------------------
      0   |   2    |     256×256          |   16   | Sehr früh, kaum nützliche Merkmale
      1   |   4    |     128×128          |   24   | C3 — feine Details ← gewählt
      2   |   8    |      64×64           |   40   | C4 — mittlere Details ← gewählt
      3   |  16    |      32×32           |   48   | Ähnlich wie Stufe 2, übersprungen
      4   |  32    |      16×16           |   96   | C5 — grobe Strukturen ← gewählt

    Warum out_indices=(1, 2, 4)?
    - Index 0 wird weggelassen: Zu früh im Netz (nur 16 Kanäle), enthält kaum
      semantische Information und wäre mit 256×256 sehr groß → mehr Rechenaufwand
      im FPN ohne echten Mehrwert.
    - Index 3 wird übersprungen: Stufe 3 (48 Kanäle) und Stufe 2 (40 Kanäle) sind
      sich in Kanalanzahl und semantischem Gehalt sehr ähnlich. Stufe 3 bringt wenig
      Zusatzinformation, würde aber das FPN komplizierter machen (FPNLite erwartet
      genau 3 Inputs).
    - Die gewählten Stufen (1, 2, 4) decken das optimale Spektrum ab: hohe räumliche
      Auflösung (Stride 4) für präzise Lokalisierung, mittlere Ebene (Stride 8) für
      Kontext + Detail, und niedrige Auflösung mit hohem semantischem Gehalt (Stride 32)
      für die Erkennung des Dokuments als Ganzes.
"""

# timm: Bibliothek mit hunderten vortrainierten Bilderkennungs-Modellen, die man direkt nutzen kann.
import timm
# torch: Framework für neuronale Netze und Tensor-Berechnungen.
import torch
# nn.Module: Basisklasse für alle neuronalen Netz-Bausteine in PyTorch.
import torch.nn as nn


class MobileNetV3Backbone(nn.Module):
    """MobileNetV3-Small Backbone mit Multi-Scale Feature Extraction.

    Dieses Modul nimmt ein Eingabebild und erzeugt daraus mehrere Feature Maps
    auf verschiedenen Auflösungsstufen. Diese werden anschließend vom FPN-Neck
    zusammengeführt.

    Args:
        pretrained: Ob vortrainierte Gewichte geladen werden sollen. True = das Netz
            hat bereits auf ImageNet (1,2 Mio. Bilder) gelernt und kennt grundlegende
            visuelle Merkmale. False = Training bei null beginnen.
        out_indices: Welche Stufen des Netzwerks als Ausgabe genutzt werden.
            (1, 2, 4) bedeutet: die 2., 3. und 5. Stufe → C3, C4, C5.
            Gewählt werden Stufen mit möglichst weit auseinander liegenden
            Auflösungen (Stride 4, 8, 32), um dem FPN eine maximale Bandbreite
            von fein → grob zu geben. Stufe 0 (zu früh, 16 Kanäle) und Stufe 3
            (zu ähnlich zu Stufe 2, 48 vs. 40 Kanäle) werden übersprungen.
    """

    def __init__(
        self,
        pretrained: bool = True,
        out_indices: tuple[int, ...] = (1, 2, 4),
    ):
        super().__init__()
        # timm.create_model erstellt ein fertiges MobileNetV3-Small-Netz.
        # features_only=True: Nur die Zwischen-Feature-Maps ausgeben, nicht die
        # finale Klassifikation (wir wollen ja keine Bildklasse, sondern Eckpunkte).
        self.backbone = timm.create_model(
            "mobilenetv3_small_100",
            pretrained=pretrained,
            features_only=True,
            out_indices=out_indices,
        )
        # Kanal-Anzahl jeder Feature-Map-Stufe auslesen (z.B. [24, 40, 96]).
        # „Kanäle" sind vergleichbar mit Farbkanälen (R, G, B = 3 Kanäle),
        # aber hier gibt es viel mehr Kanäle, die jeweils ein bestimmtes Merkmal kodieren.
        self._out_channels = self.backbone.feature_info.channels()

    @property
    def out_channels(self) -> list[int]:
        """Kanal-Anzahl pro Feature-Map Stage (z.B. [24, 40, 96])."""
        return self._out_channels

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        """Extrahiert Multi-Scale Feature Maps aus dem Eingabebild.

        Args:
            x: Eingabebild als Tensor [B, 3, H, W].
                B = Batch-Größe (Anzahl Bilder im Stapel),
                3 = Farbkanäle (RGB),
                H, W = Höhe und Breite in Pixeln.

        Returns:
            Liste von Feature Maps [C3, C4, C5] – jeweils ein Tensor mit
            abnehmender räumlicher Auflösung, aber zunehmender Merkmalstiefe.
        """
        return self.backbone(x)
