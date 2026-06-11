# LDED — Lightweight Document Edge Detector

Leichtes, browser-taugliches Dokumenteneck-Erkennungsmodell auf Basis von
**MobileNetV3-Small + Heatmap-Head + Soft-Argmax**. Deployment via ONNX Runtime Web.

## Architektur

```
Input (beliebige Auflösung)
  → Pre-Processing (Padding, Resize 512×512, Normalisierung)
  → Backbone: MobileNetV3-Small (C3, C4, C5)
  → Neck: Lightweight FPN (128ch @ 1/4)
  → Head: 4× Heatmap + 1× Confidence
  → Soft-Argmax → 4 Eckpunkte (x, y) + Confidence
```

## Quickstart

```bash
# Verzeichnis wechseln
cd lded

# VENV erstellen oder einfach in bestehendes venv wechseln
python3 -m venv .venv
source .venv/bin/activate

# Installation
pip install -e ".[dev]"

# Daten herunterladen
bash data/scripts/download_training_data.sh

# Training
python scripts/train.py --config configs/train/default.yaml --max_samples 50
tensorboard --logdir runs/

# Prozess finden und killen
pkill -9 -f train.py

# Evaluation
python scripts/evaluate.py --model models/docaligner_lite.onnx --data data/processed/test/

# ONNX Export
python scripts/export_onnx.py --checkpoint models/best.pt --output models/docaligner_lite.onnx

# Tests
pytest tests/
```

## Projektstruktur

```
lded/
├── configs/          # YAML-Konfigurationen
├── data/             # Daten & Synthese-Skripte
├── model/            # PyTorch-Modell (Backbone, Neck, Head, Loss)
├── scripts/          # Training, Evaluation, ONNX-Export
├── tests/            # Unit- & Integrationstests
├── web/              # Browser-Demo (ONNX Runtime Web)
└── docs/             # Dokumentation
```

## Metriken (Zielwerte)

| Metrik | Zielwert |
|--------|----------|
| PCK@5 | > 92 % |
| PCK@10 | > 97 % |
| NME | < 3.0 % |
| IoU | > 0.90 |
| Latenz CPU (WASM) | < 100 ms |
| Modellgrösse FP32 | < 10 MB |
| Modellgrösse INT8 | < 5 MB |
