# LDED Training Guide

## Voraussetzungen

- Python ≥ 3.10
- CUDA-fähige GPU (empfohlen)
- ~50 GB Festplatte für Daten

## Installation

```bash
cd lded/
pip install -e ".[dev]"
```

## Daten vorbereiten

### 1. Rohdaten herunterladen

```bash
bash data/scripts/download_training_data.sh [data/raw]
```

Lädt MIDV-2020 und DocCorner-Datasets nach `data/raw/`.

### 2. Synthetische Augmentation

Beide Skripte schreiben direkt nach `data/processed/{train,val,test}/`.
Falls dort bereits Daten liegen, wird die Nummerierung automatisch fortgesetzt.

**DocCorner** (1:1 Splits aus dem Dataset, Augmentation nur auf train):

```bash
python data/scripts/synthesize_doccorner.py \
    --input-dir data/raw/DocCornerDataset \
    --num-augmentations 3 \
    --seed 42
```

**MIDV-2020** (Random-Split 70/15/15):

```bash
python data/scripts/synthesize_midv_2020.py \
    --raw-dir data/raw/midv-2020 \
    --num-augmentations 3 \
    --seed 42
```

Erzeugt folgende Verzeichnisstruktur:
```
data/processed/train/images/*.jpg       data/processed/train/annotations/*.json
data/processed/val/images/*.jpg         data/processed/val/annotations/*.json
data/processed/test/images/*.jpg        data/processed/test/annotations/*.json
```

Annotationsformat: `{"corners": [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]}` (normalisiert auf [0, 1]).

## Training starten

```bash
python scripts/train.py --config configs/train/default.yaml
```

### Parameter

| Parameter | Default | Beschreibung |
|-----------|---------|-------------|
| `--config` | `configs/train/default.yaml` | Pfad zur Trainings-Konfiguration |
| `--device` | `auto` | Device (`auto`, `cuda`, `cpu`, `mps`) |
| `--max_samples` | `None` (alle) | Begrenzt Train/Val auf die ersten N Bilder (Debug-Modus) |

**Beispiel: Schneller Debug-Lauf mit 50 Bildern:**

```bash
python scripts/train.py --config configs/train/default.yaml --max_samples 50 --device mps

# Overfit-Test mit 5 Samples (default)
python scripts/train.py --overfit_test --device mps

# Overfit-Test mit 1 Sample
python scripts/train.py --overfit_test 1 --device mps

# Overfit-Test mit 10 Samples auf MPS
python scripts/train.py --overfit_test 10 --device mps

```



Nach Abschluss des Trainings wird automatisch ein Balkendiagramm (`models/checkpoints/metrics_chart.png`) mit den finalen PCK@5, PCK@10, IoU, NME und Val-Loss Werten erzeugt.

### Trainings-Phasen

| Phase | Epochs | Beschreibung |
|-------|--------|-------------|
| Phase 1 | 1–10 | Backbone eingefroren, nur Neck + Head |
| Phase 2 | 11–80 | Full Fine-Tuning, diff. LR (Backbone: 1e-4, Rest: 3e-4) |
| Phase 3 | 81–100 | Quantization-Aware Training (INT8) |

### Trainings-Output

Nach dem Training werden folgende Artefakte erzeugt:

| Datei | Beschreibung |
|-------|-------------|
| `models/checkpoints/best.pt` | Bestes Modell (nach val PCK@5) |
| `models/checkpoints/metrics_chart.png` | Balkendiagramm der finalen Metriken |

Das **Metriken-Diagramm** enthält zwei Teilgrafiken:
- **Links**: PCK@5, PCK@10, IoU (in %, höher = besser) mit 90%-Ziellinie
- **Rechts**: NME und Val Loss (niedriger = besser)

### GPU-Tipps

- **Batch Size**: 32 auf einer RTX 3090 (24 GB). Halbieren bei weniger VRAM.
- **Mixed Precision**: Kann bei Bedarf ergänzt werden (torch.amp).

## Evaluation

```bash
python scripts/evaluate.py \
    --model models/docaligner_lite.onnx \
    --data data/processed/test/ \
    --metrics pck nme iou latency
```

### Zielwerte

| Metrik | Zielwert |
|--------|----------|
| PCK@5 | > 92% |
| PCK@10 | > 97% |
| NME | < 3.0% |
| IoU | > 0.90 |
| Latenz (WASM) | < 100 ms |

## ONNX Export

```bash
python scripts/export_onnx.py \
    --checkpoint models/checkpoints/best.pt \
    --output models/docaligner_lite.onnx \
    --opset 17
```

## Tests

```bash
# Alle Tests
pytest tests/

# Nur Unit Tests
pytest tests/unit/

# Nur Integration Tests
pytest tests/integration/

# Mit Coverage
pytest --cov=model tests/
```
