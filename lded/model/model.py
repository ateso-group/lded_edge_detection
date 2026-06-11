"""
LDED — Lightweight Document Edge Detector.

Hauptmodell: MobileNetV3-Small Backbone + FPN-Lite Neck + Heatmap Head.

Das Modell besteht aus drei Teilen (wie eine Fertigungsstraße):
  1. Backbone – Extrahiert visuelle Merkmale aus dem Eingangsbild (z.B. Kanten, Texturen).
  2. Neck (FPN) – Kombiniert Merkmale verschiedener Detailstufen zu einer einzigen Darstellung.
  3. Head – Erzeugt aus den Merkmalen Heatmaps (Wärmebilder), aus denen die vier
     Eckpunkte des Dokuments abgelesen werden.

Warum Neck und Head eigene Module sind (und nicht Teil von MobileNetV3):
  MobileNetV3 ist ursprünglich ein Klassifikationsnetzwerk (z.B. „Ist das ein Hund oder
  eine Katze?"). Es bringt von Haus aus nur einen eigenen Classifier-Head mit, der
  ImageNet-Klassen vorhersagt. In diesem Projekt wird MobileNetV3 jedoch mit
  ``features_only=True`` geladen – dadurch wird der originale Classifier-Head entfernt
  und MobileNetV3 dient ausschließlich als Feature-Extractor (Backbone).

  Da unsere Aufgabe (Dokumenten-Eckpunkt-Erkennung) eine völlig andere ist als
  Bildklassifikation, definieren wir eigene, aufgabenspezifische Module:

  - **Neck (FPN-Lite):** MobileNetV3 liefert Feature Maps auf verschiedenen Auflösungen
    (C3, C4, C5), führt diese aber nicht selbst zusammen. Das FPN übernimmt diese
    Multi-Scale-Feature-Fusion – ein Schritt, den MobileNetV3 nicht beherrscht.
  - **Head (HeatmapHead):** Erzeugt Heatmaps, extrahiert (x, y)-Koordinaten per
    Soft-Argmax und berechnet einen Confidence-Score. All das ist komplett
    aufgabenspezifisch und hat kein Äquivalent in MobileNetV3.

  Das Architektur-Pattern „Backbone → Neck → Head" ist ein weit verbreiteter Standard
  in der Computer Vision (z.B. auch bei YOLO, RetinaNet, FCOS). Dabei ist das Backbone
  immer ein vortrainiertes Netzwerk, während Neck und Head je nach Aufgabe frei
  definiert werden.
"""

# torch: Framework für neuronale Netze und Tensor-Berechnungen (ähnlich NumPy, aber mit GPU-Unterstützung).
import torch
# nn.Module: Basisklasse für alle neuronalen Netz-Schichten in PyTorch.
import torch.nn as nn
# yaml: Liest YAML-Konfigurationsdateien (einfaches, lesbares Textformat für Einstellungen).
import yaml

from model.backbone.mobilenetv3 import MobileNetV3Backbone
from model.neck.fpn_lite import FPNLite
from model.head.heatmap_head import HeatmapHead


class LDED(nn.Module):
    """Lightweight Document Edge Detector.

    Dieses Modell erkennt die vier Eckpunkte eines Dokuments in einem Foto.
    Es gibt Heatmaps (Wärmebilder, die zeigen, wo eine Ecke liegt),
    Koordinaten (x, y) der Ecken und einen Confidence-Wert
    (wie sicher sich das Modell ist, dass ein Dokument im Bild ist) aus.

    Args:
        backbone_pretrained: Ob vortrainierte Gewichte (aus ImageNet, einem großen
            Bilddatensatz) geladen werden sollen, damit das Netz nicht bei null anfängt.
        out_indices: Welche Zwischenstufen des Backbones genutzt werden sollen.
            Jede Stufe liefert Merkmale auf einer anderen Detailstufe (grob → fein).
        fpn_out_channels: Anzahl der Kanäle (≈ „Informationstiefe") der vom Neck
            erzeugten fusionierten Feature Map.
        num_corners: Anzahl der zu erkennenden Eckpunkte (Standard: 4 für ein Rechteck).
        temperature: Steuert die „Schärfe" der Soft-Argmax-Operation –
            kleinere Werte → spitzere Verteilung → präzisere Koordinaten,
            größere Werte → weichere Verteilung → robustere, aber ungenauere Koordinaten.
    """

    def __init__(
        self,
        backbone_pretrained: bool = True,
        out_indices: tuple[int, ...] = (1, 2, 4),
        fpn_out_channels: int = 128,
        num_corners: int = 4,
        temperature: float = 0.05,
    ):
        # super().__init__() ruft den Konstruktor der Elternklasse nn.Module auf –
        # das ist nötig, damit PyTorch das Modell korrekt verwalten kann.
        super().__init__()

        # Backbone: Extrahiert visuelle Merkmale aus dem Bild auf mehreren Detailstufen.
        self.backbone = MobileNetV3Backbone(
            pretrained=backbone_pretrained,
            out_indices=out_indices,
        )
        # Neck (FPN): Verschmilzt die verschiedenen Detailstufen zu einer einheitlichen
        # Feature Map. use_depthwise=True nutzt recheneffiziente Faltungen.
        self.neck = FPNLite(
            in_channels=self.backbone.out_channels,
            out_channels=fpn_out_channels,
            use_depthwise=True,
        )
        # Head: Erzeugt aus der Feature Map die Heatmaps, Eckpunkt-Koordinaten
        # und den Confidence-Score.
        self.head = HeatmapHead(
            in_channels=fpn_out_channels,
            num_corners=num_corners,
            temperature=temperature,
        )

    def forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward Pass – verarbeitet ein Bild durch das gesamte Netzwerk.

        Ablauf: Bild → Backbone → Neck → Head → Ergebnisse.

        Args:
            x: Eingabebild als Tensor der Form [B, 3, 512, 512].
                B = Batch-Größe (Anzahl Bilder gleichzeitig),
                3 = Farbkanäle (Rot, Grün, Blau),
                512×512 = Bildauflösung in Pixeln.

        Returns:
            heatmaps: [B, 4, H, W] – 4 Wärmebilder (eines pro Ecke), die zeigen,
                wo das Modell die jeweilige Ecke vermutet.
            coords: [B, 4, 2] – die (x, y)-Koordinaten der 4 Eckpunkte,
                normalisiert auf den Bereich [0, 1] (0 = links/oben, 1 = rechts/unten).
            confidence: [B, 1] – ein Logit-Wert, der angibt, wie sicher das Modell ist,
                dass im Bild ein Dokument vorhanden ist (wird später per Sigmoid in
                eine Wahrscheinlichkeit umgewandelt).
        """
        # 1) Backbone extrahiert Merkmale auf verschiedenen Auflösungsstufen
        features = self.backbone(x)
        # 2) Neck fusioniert die Merkmale zu einer einheitlichen Feature Map
        fused = self.neck(features)
        # 3) Head erzeugt Heatmaps, Koordinaten und Confidence
        heatmaps, coords, confidence = self.head(fused)
        return heatmaps, coords, confidence

    def freeze_backbone(self) -> None:
        """Friert alle Backbone-Parameter ein (Phase 1 des Trainings).

        „Einfrieren" bedeutet, dass die Gewichte des Backbones beim Training
        nicht verändert werden. Das ist nützlich, wenn man nur den Neck und
        Head trainieren möchte, während das Backbone seine vortrainierten
        Merkmale behält.
        """
        # requires_grad = False → Gradient wird nicht berechnet → Gewicht ändert sich nicht.
        for param in self.backbone.parameters():
            param.requires_grad = False

    def unfreeze_backbone(self) -> None:
        """Taut alle Backbone-Parameter auf (Phase 2 des Trainings).

        Ab Phase 2 dürfen auch die Backbone-Gewichte mittrainiert (feinabgestimmt)
        werden, damit sich das gesamte Netz an die spezifische Aufgabe anpasst.
        """
        for param in self.backbone.parameters():
            param.requires_grad = True

    def get_param_groups(self, backbone_lr: float, head_lr: float) -> list[dict]:
        """Erstellt Parameter-Gruppen mit unterschiedlichen Lernraten.

        Da das Backbone bereits vortrainiert ist, wird es typischerweise mit
        einer kleineren Lernrate (= vorsichtigere Anpassung) trainiert als
        der Neck und Head, die von Grund auf lernen müssen.

        Args:
            backbone_lr: Lernrate für den Backbone – wie stark seine Gewichte
                pro Trainingsschritt angepasst werden.
            head_lr: Lernrate für Neck + Head.

        Returns:
            Liste von Parameter-Gruppen für den Optimizer (das Objekt, das die
            Gewichte während des Trainings aktualisiert).
        """
        return [
            {"params": self.backbone.parameters(), "lr": backbone_lr},
            {"params": self.neck.parameters(), "lr": head_lr},
            {"params": self.head.parameters(), "lr": head_lr},
        ]

    @classmethod
    def from_config(cls, config_path: str) -> "LDED":
        """Erstellt ein LDED-Modell aus einer YAML-Konfigurationsdatei.

        Damit kann man die Modell-Einstellungen bequem in einer Textdatei
        definieren, anstatt sie direkt im Code anzugeben.

        Args:
            config_path: Pfad zur YAML-Konfigurationsdatei.

        Returns:
            Eine fertig konfigurierte LDED-Instanz.
        """
        with open(config_path, "r") as f:
            # yaml.safe_load liest die YAML-Datei und wandelt sie in ein
            # Python-Dictionary (Schlüssel-Wert-Paare) um.
            cfg = yaml.safe_load(f)

        return cls(
            backbone_pretrained=cfg["backbone"].get("pretrained", True),
            out_indices=tuple(cfg["backbone"].get("out_indices", [1, 2, 4])),
            fpn_out_channels=cfg["neck"].get("out_channels", 128),
            num_corners=cfg["head"].get("num_corners", 4),
            temperature=cfg["head"].get("temperature", 0.05),
        )
