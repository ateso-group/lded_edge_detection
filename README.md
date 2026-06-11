<div align="center">
<br/>
<br/>

# LDED

Lightweight Document Edge Detection — real-time corner detection in the browser.

<br/>

<!-- Version Chips -->
<img alt="version" src="https://img.shields.io/badge/lded-0.1.0-primary" />
<img alt="model-size" src="https://img.shields.io/badge/model-<5MB_(INT8)-0A7CFF" />
<img alt="latency" src="https://img.shields.io/badge/latency-<100ms_(WASM)-FF6B00" />

<br/>
<br/>

<!-- Quick Status -->
<img alt="licence" src="https://img.shields.io/badge/license-academic-6C757D" />
<img alt="status" src="https://img.shields.io/badge/status-in_development-6C757D" />

<br/>
<br/>

<!-- Tech Tags -->
<img alt="pytorch" src="https://img.shields.io/badge/PyTorch-2.x-EE4C2C?logo=pytorch&logoColor=white" />
<img alt="onnx" src="https://img.shields.io/badge/ONNX_Runtime-Web-005CED?logo=onnx&logoColor=white" />
<img alt="mobilenet" src="https://img.shields.io/badge/MobileNetV3-Small-34A853?logo=google&logoColor=white" />
<img alt="python" src="https://img.shields.io/badge/Python-3.10+-3776AB?logo=python&logoColor=white" />
<img alt="cuda" src="https://img.shields.io/badge/CUDA-AMP-76B900?logo=nvidia&logoColor=white" />

<br/>
<br/>

<!-- Positioning Tags -->
<img alt="tag-cv" src="https://img.shields.io/badge/Tag-Computer_Vision-111827" />
<img alt="tag-edge" src="https://img.shields.io/badge/Tag-Edge_AI-111827" />
<img alt="tag-browser" src="https://img.shields.io/badge/Tag-Browser_Inference-111827" />
<img alt="tag-mobile" src="https://img.shields.io/badge/Tag-Mobile--first-111827" />
<img alt="tag-privacy" src="https://img.shields.io/badge/Tag-Privacy--preserving-111827" />

</div>

---

## Why LDED

Mobile document scanning requires fast and accurate detection of document boundaries. Classical computer vision approaches are fragile under varying lighting and backgrounds. Heavyweight deep learning models cannot run in real-time on end-user devices. LDED solves this by providing a neural network small enough to run entirely client-side via WebAssembly — no server roundtrip, no privacy concerns, no infrastructure cost.

## Highlights

- Real-time document corner detection (< 100 ms) directly in the browser
- Privacy-preserving — all inference happens on-device, no data leaves the client
- Trained on diverse datasets (DocCorner, MIDV-500/2020, SmartDoc-2015) with heavy augmentation
- 3-phase training curriculum: Backbone Freeze → Fine-Tuning → Quantization-Aware Training
- Targets > 92 % corner precision (PCK@5) with < 5 MB model size

## Use Cases

- Browser-based document scanners
- Mobile receipt and invoice capture
- ID card and passport boundary detection
- Drop-in integration into existing web applications

## Results

<div align="center">
<img src="lded/models/checkpoints/metrics_chart.png" alt="Model Metrics" width="700" />
</div>

<br/>

### Benchmark (placeholder — will be updated with final model)

| Metric | LDED (ours) | Target | SmartDoc Baseline |
|--------|:-----------:|:------:|:-----------------:|
| PCK@5 ↑ | 100.0 % | > 92 % | — |
| PCK@10 ↑ | 100.0 % | > 97 % | — |
| IoU ↑ | 98.3 % | > 90 % | — |
| NME ↓ | 0.19 % | < 3.0 % | — |
| Val Loss ↓ | 0.1157 | — | — |

> *↑ = higher is better, ↓ = lower is better. Results on validation split. Baseline columns to be filled after final evaluation.*

### Model Card

| Property | Value |
|----------|-------|
| Architecture | MobileNetV3-Small + FPN-Lite + Heatmap Head |
| Input size | 512 × 512 px |
| Parameters | ~2.5 M |
| Model size (FP32) | ~10 MB |
| Model size (INT8) | < 5 MB |
| Output | 4 corner points (x, y) + confidence |
| Framework | PyTorch 2.x → ONNX |

### Training Configuration

| Setting | Value |
|---------|-------|
| Optimizer | AdamW |
| Learning rate | 1e-3 (phase 1), 1e-4 (phase 2) |
| Scheduler | CosineAnnealingLR |
| Batch size | 16 |
| Epochs | 50 (phase 1) + 30 (phase 2) |
| Loss | Adaptive Wing + BCE + Coordinate |
| AMP | Enabled (CUDA) |
| Augmentation | Perspective, Brightness/Contrast, MotionBlur, JPEG, GaussNoise |

### Datasets

| Dataset | Raw Samples | × Augmentation | Effective Samples | Split |
|---------|------------:|:--------------:|------------------:|-------|
| DocCorner (HuggingFace) | ~4,000 | ×3 | ~12,000 | train / val / test |
| MIDV-500/2020 (synthesized) | ~3,000 | ×3 | ~9,000 | train / val |
| Mendeley Corner | ~1,100 | ×3 | ~3,300 | train / val |
| **Total** | **~8,100** | | **~24,300** | — |

### Inference

| Runtime | Device | Latency |
|---------|--------|---------|
| ONNX Runtime Web (WASM) | Desktop browser | < 100 ms |
| ONNX Runtime Web (WASM) | Mobile browser | < 150 ms |
| PyTorch (CPU) | MacBook M1 | < 50 ms |
| PyTorch (CUDA) | NVIDIA GPU | < 10 ms |

> *Latency values are estimates — will be updated after final benchmarking.*

## Contents

- [Architecture](lded/architecture.md)
- [Training Guide](lded/training_guide.md)
- [Technical README](lded/README.md)

---

**CAS project work** by Simon Fuchs & Raphael Fuchs, 2026
