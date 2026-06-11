# LDED Architektur

## Übersicht

Das Lightweight Document Edge Detector (LDED) Modell besteht aus drei Hauptkomponenten:

1. **Backbone**: MobileNetV3-Small (vortrainiert auf ImageNet)
2. **Neck**: Lightweight Feature Pyramid Network (FPN-Lite)
3. **Head**: Heatmap-Regression + Soft-Argmax + Confidence

## Wahl des Backbones

MobileNetV3-Small wurde als Backbone gewählt, weil es optimal zu den Projektanforderungen
passt: **Lightweight** Document Edge Detection auf mobilen/eingebetteten Geräten.

### Warum MobileNetV3-Small?

| Kriterium               | MobileNetV3-Small | Alternativen (z.B. ResNet-50) |
|--------------------------|-------------------|-------------------------------|
| Parameterzahl            | ~2.5M → sehr klein | ~25M → 10× mehr              |
| Mobile/Edge-tauglich     | ✅ Dafür designed  | ❌ Zu schwer für Smartphones   |
| ImageNet-Pretrained      | ✅ Via `timm`      | ✅ Ja                          |
| Multi-Scale Feature Maps | ✅ 5 Stufen        | ✅ Ja                          |
| Inferenzgeschwindigkeit  | ✅ Sehr schnell    | ❌ Deutlich langsamer           |

### Warum nicht ein anderes Backbone, das Neck/Head überflüssig macht?

Neck und Head sind **keine Notlösung**, weil MobileNetV3 „nicht passt". Das Architektur-Pattern
**Backbone → Neck → Head** ist ein Standard in der Computer Vision (YOLO, RetinaNet, FCOS, …).
**Jedes** Backbone — egal ob ResNet, EfficientNet, ConvNeXt oder Vision Transformer — liefert
nur generische Feature Maps. Keines davon kann von sich aus:

- Multi-Scale-Features zu einer einzigen Map fusionieren → dafür braucht man immer einen **Neck**
- Aufgabenspezifische Vorhersagen erzeugen (Heatmaps, Koordinaten, Confidence) → dafür braucht
  man immer einen **Head**

Ein anderes Backbone zu wählen würde also **nicht** Neck oder Head überflüssig machen, sondern
nur die Art der Feature Maps ändern.

### Mögliche Alternativen (falls andere Anforderungen gelten)

- **MobileNetV2** — ähnlich leicht, aber etwas älter und weniger effizient
- **EfficientNet-B0** — etwas schwerer (~5M Parameter), leicht bessere Accuracy
- **MobileNetV3-Large** — mehr Kapazität, aber doppelt so viele Parameter
- **ResNet-18/34** — klassisch und gut erforscht, aber nicht für Mobile optimiert

Für die Aufgabe „4 Ecken eines Dokuments finden" (eine relativ einfache geometrische Aufgabe)
wäre ein schwereres Backbone Overkill — MobileNetV3-Small bietet das beste Verhältnis von
Geschwindigkeit, Größe und Genauigkeit.

## Pipeline

```
Input (beliebige Auflösung)
    │
    ▼
Pre-Processing
  ├── Symmetrisches Padding (5% des Bildumfangs)
  ├── Resize → 512×512
  └── ImageNet-Normalisierung: μ=[0.485, 0.456, 0.406], σ=[0.229, 0.224, 0.225]
    │
    ▼
Backbone: MobileNetV3-Small
  Feature Maps: C3 [24ch, 1/4], C4 [40ch, 1/8], C5 [96ch, 1/32]
    │
    ▼
Neck: FPN-Lite (Depthwise Separable Conv)
  Fusionierte Feature Map @ 128ch, 1/4 Auflösung (128×128 bei 512 Input)
    │
    ▼
Head: 4× Heatmap-Kanäle (TL, TR, BR, BL) + 1× Confidence
    │
    ▼
Post-Processing
  ├── Soft-Argmax → (x, y) im Heatmap-Raum
  ├── Koordinaten-Skalierung → Original-Bildraum
  └── Padding-Offset-Korrektur
    │
    ▼
Output: Polygon [4 × (x, y)] + Confidence Score
```

## Layer-Shapes (512×512 Input)

| Stage | Shape | Channels | Stride |
|-------|-------|----------|--------|
| Input | 512×512×3 | 3 | — |
| C3 | 128×128 | 24 | 1/4 |
| C4 | 64×64 | 40 | 1/8 |
| C5 | 16×16 | 96 | 1/32 |
| FPN-Out | 128×128 | 128 | 1/4 |
| Heatmap | 128×128 | 5 | 1/4 |
| Output | 4 × (x,y) | — | — |

## Backbone: MobileNetV3-Small

- Vortrainiert auf ImageNet (timm)
- Feature-Extraction an 3 Stages (C3, C4, C5)
- Leichtgewichtig: ~2.5M Parameter

## Neck: FPN-Lite

- 3 laterale 1×1 Convolutions (Kanal-Angleichung)
- Top-Down Pathway mit Bilinear Upsampling
- Depthwise Separable Convolutions für Fusion
- Ausgabe: Einzelne Feature Map (128ch, 128×128)

## Head: Heatmap + Soft-Argmax

- 2-Layer Conv für Heatmap-Regression (4 Kanäle)
- Soft-Argmax: Differenzierbare Koordinaten-Extraktion
- Global Average Pooling + MLP für Confidence

## Loss

```
L_total = 1.0 × L_awing + 0.1 × L_bce
```

- **Adaptive Wing Loss**: Robuste Heatmap-Regression
- **BCE Loss**: Confidence-Klassifikation (Dokument ja/nein)
