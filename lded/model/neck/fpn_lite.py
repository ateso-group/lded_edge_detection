"""
Lightweight Feature Pyramid Network (FPN) mit Depthwise Separable Convolutions.

Fusioniert Multi-Scale Feature Maps zu einer einzelnen Feature Map.

Was ist ein FPN (Feature Pyramid Network)?
    Das FPN ist der „Neck" (Hals) des Netzwerks – es sitzt zwischen Backbone und Head.
    Seine Aufgabe: Die Feature Maps verschiedener Detailstufen (grob, mittel, fein)
    zu einer einzigen, reichhaltigen Darstellung verschmelzen. So kann das Netz
    sowohl kleine Details als auch große Strukturen gleichzeitig nutzen.

    Analogie: Stell dir vor, du betrachtest ein Dokument einmal aus der Ferne
    (grobe Form), einmal aus mittlerer Entfernung (Textblöcke) und einmal aus
    der Nähe (einzelne Buchstaben). Das FPN kombiniert alle drei Sichtweisen.
"""

# torch: Framework für neuronale Netze und GPU-beschleunigte Berechnungen.
import torch
import torch.nn as nn
# F (functional): Enthält Hilfsfunktionen wie interpolate (Bilder vergrößern/verkleinern)
# und softmax (Wahrscheinlichkeiten berechnen).
import torch.nn.functional as F


class DepthwiseSeparableConv(nn.Module):
    """Depthwise Separable Convolution Block.

    Eine besonders recheneffiziente Variante der Faltung (Convolution).
    Statt alle Eingangskanäle gleichzeitig zu verrechnen, wird in zwei Schritten gearbeitet:
      1. Depthwise: Jeder Kanal wird einzeln mit einem eigenen Filter gefaltet.
      2. Pointwise: Die Ergebnisse werden mit einer 1×1-Faltung kanalübergreifend kombiniert.

    Vorteil: Deutlich weniger Rechenoperationen als eine normale Faltung –
    ideal für mobile/eingebettete Geräte.
    """

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3):
        super().__init__()
        # padding sorgt dafür, dass die räumliche Größe (Höhe × Breite) erhalten bleibt.
        padding = kernel_size // 2
        # Depthwise: groups=in_channels → jeder Kanal bekommt seinen eigenen Filter.
        self.depthwise = nn.Conv2d(
            in_channels, in_channels, kernel_size,
            padding=padding, groups=in_channels, bias=False,
        )
        # Pointwise: 1×1-Faltung kombiniert die Kanäle und ändert ggf. ihre Anzahl.
        self.pointwise = nn.Conv2d(in_channels, out_channels, 1, bias=False)
        # BatchNorm: Normalisiert die Werte, damit das Training stabiler und schneller läuft.
        self.bn = nn.BatchNorm2d(out_channels)
        # ReLU6: Aktivierungsfunktion – setzt negative Werte auf 0 und begrenzt positive
        # Werte auf maximal 6. Das stabilisiert die Berechnungen auf mobilen Geräten.
        self.act = nn.ReLU6(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.depthwise(x)
        x = self.pointwise(x)
        x = self.bn(x)
        return self.act(x)


class FPNLite(nn.Module):
    """Lightweight FPN mit 3 Input-Stufen und einer fusionierten Ausgabe.

    Nimmt drei Feature Maps unterschiedlicher Auflösung (C3, C4, C5) vom Backbone
    und verschmilzt sie zu einer einzigen Feature Map, die sowohl feine als auch
    grobe Merkmale enthält.

    Args:
        in_channels: Kanal-Anzahl pro Input-Stage [C3, C4, C5] – z.B. [24, 40, 96].
            Jede Stufe hat eine andere Anzahl von Kanälen.
        out_channels: Gewünschte Kanal-Anzahl der fusionierten Ausgabe-Feature-Map.
        use_depthwise: Ob die effizientere Depthwise Separable Convolution (True)
            oder eine normale Faltung (False) genutzt werden soll.
    """

    def __init__(
        self,
        in_channels: list[int],
        out_channels: int = 128,
        use_depthwise: bool = True,
    ):
        super().__init__()
        assert len(in_channels) == 3, "FPNLite erwartet genau 3 Input-Stages"

        # Lateral Connections (1×1-Faltungen):
        # Bringen alle Feature Maps auf die gleiche Kanal-Anzahl (out_channels),
        # damit sie anschließend miteinander verrechnet werden können.
        self.lateral_convs = nn.ModuleList([
            nn.Sequential(
                # 1×1-Faltung: Ändert nur die Kanal-Anzahl, nicht die räumliche Größe.
                nn.Conv2d(ch, out_channels, 1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU6(inplace=True),
            )
            for ch in in_channels
        ])

        # Fusion Convolutions: Glätten die kombinierten Feature Maps nach dem Addieren.
        conv_cls = DepthwiseSeparableConv if use_depthwise else None
        if use_depthwise:
            self.fusion_convs = nn.ModuleList([
                DepthwiseSeparableConv(out_channels, out_channels) for _ in range(3)
            ])
        else:
            self.fusion_convs = nn.ModuleList([
                nn.Sequential(
                    nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
                    nn.BatchNorm2d(out_channels),
                    nn.ReLU6(inplace=True),
                )
                for _ in range(3)
            ])

        # Final Fusion: Alle 3 Stufen werden per Concatenation (Aneinanderhängen)
        # zusammengeführt und durch eine letzte Faltung auf out_channels reduziert.
        self.final_conv = DepthwiseSeparableConv(out_channels * 3, out_channels)

    def forward(self, features: list[torch.Tensor]) -> torch.Tensor:
        """Fusioniert Multi-Scale Features zu einer einzelnen Feature Map.

        Ablauf:
          1. Lateral: Kanal-Anzahl angleichen.
          2. Top-Down: Grobe Features hochskalieren und mit feinen addieren.
          3. Fusion: Ergebnisse glätten.
          4. Concatenation: Alle Stufen zusammenfügen → eine Ausgabe.

        Args:
            features: Liste von 3 Feature Maps vom Backbone:
                - C3 (1/4 Auflösung) – feine Details
                - C4 (1/8 Auflösung) – mittlere Details
                - C5 (1/32 Auflösung) – grobe Strukturen

        Returns:
            Fusionierte Feature Map [B, out_channels, H/4, W/4] – eine einzige
            Feature Map, die Informationen aller Detailstufen enthält.
        """
        c3, c4, c5 = features

        # 1) Lateral Connections: Alle auf gleiche Kanal-Anzahl bringen
        p3 = self.lateral_convs[0](c3)
        p4 = self.lateral_convs[1](c4)
        p5 = self.lateral_convs[2](c5)

        # 2) Top-Down Pathway: Grobe Feature Maps (p4, p5) auf die Größe von p3
        #    hochskalieren (interpolate = „vergrößern"), damit sie addiert werden können.
        #    bilinear = glatte Vergrößerung durch gewichtete Mittelwerte benachbarter Pixel.
        target_size = p3.shape[2:]

        p4_up = F.interpolate(p4, size=target_size, mode="bilinear", align_corners=False)
        p5_up = F.interpolate(p5, size=target_size, mode="bilinear", align_corners=False)

        # 3) Fusion: Feature Maps addieren und durch eine Faltung glätten.
        #    p3 + p4_up = feine + mittlere Details kombiniert usw.
        f3 = self.fusion_convs[0](p3 + p4_up)
        f4 = self.fusion_convs[1](p4_up + p5_up)
        f5 = self.fusion_convs[2](p5_up)

        # 4) Alle auf 1/4-Auflösung hochskalieren (2× die C3-Auflösung von 1/8)
        out_size = (target_size[0] * 2, target_size[1] * 2)
        f3_up = F.interpolate(f3, size=out_size, mode="bilinear", align_corners=False)
        f4_up = F.interpolate(f4, size=out_size, mode="bilinear", align_corners=False)
        f5_up = F.interpolate(f5, size=out_size, mode="bilinear", align_corners=False)

        # 5) Concatenate (Kanäle aneinanderhängen) + finale Faltung → eine Ausgabe
        fused = torch.cat([f3_up, f4_up, f5_up], dim=1)
        return self.final_conv(fused)
