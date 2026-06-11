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

<br/>
<br/>

<!-- Demo Link -->
<a href="https://lded.demo.ateso.ch/" target="_blank">
  <img src="https://img.shields.io/badge/🚀_Try_Demo-lded.demo.ateso.ch-8B5CF6?style=for-the-badge&logo=data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHZpZXdCb3g9IjAgMCAyNCAyNCIgZmlsbD0ibm9uZSIgc3Ryb2tlPSJ3aGl0ZSIgc3Ryb2tlLXdpZHRoPSIyIiBzdHJva2UtbGluZWNhcD0icm91bmQiIHN0cm9rZS1saW5lam9pbj0icm91bmQiPjxjaXJjbGUgY3g9IjEyIiBjeT0iMTIiIHI9IjEwIi8+PHBvbHlnb24gcG9pbnRzPSIxMCA4IDE2IDEyIDEwIDE2IDEwIDgiLz48L3N2Zz4=" alt="Demo" />
</a>

</div>

---

## Why LDED

Mobile document scanning requires fast and accurate detection of document boundaries. Classical computer vision approaches are fragile under varying lighting conditions, cluttered backgrounds, and perspective distortion. At the same time, while both \(iOS\) and \(Android\) already ship with operating-system-level computer vision and machine learning capabilities for tasks such as edge or document detection, these native capabilities are generally not accessible from within the browser. Web applications therefore cannot rely on the optimized on-device models that native apps can use.

This creates a clear gap: existing deep learning models for document boundary detection are typically too large or too computationally expensive to run in real time in a browser environment, especially on end-user mobile devices. In practice, there are currently no broadly available browser-first models that combine the necessary accuracy, speed, and device-level efficiency for this task.

LDED addresses this gap by providing a neural network compact enough to run entirely client-side via WebAssembly. This enables real-time document boundary detection directly in the browser, without server roundtrips, without exposing sensitive images to external infrastructure, and without the operational cost of backend processing.

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

| Training Curves (TensorBoard) | Evaluation Metrics |
|:-----------------------------:|:------------------:|
| <img src="lded/models/checkpoints/img.png" alt="Training Curves" width="480" /> | <img src="lded/models/checkpoints/metrics_chart.png" alt="Model Metrics" width="480" /> |

</div>

<br/>

### Benchmark

| Metric | LDED (ours) | Target | SmartDoc Baseline |
|--------|:-----------:|:------:|:-----------------:|
| PCK@5 ↑ | 93.0 % | > 92 % | — |
| PCK@10 ↑ | 98.2 % | > 97 % | — |
| IoU ↑ | 97.4 % | > 90 % | — |
| NME ↓ | 0.39 % | < 3.0 % | — |
| Val Loss ↓ | 0.0026 | — | — |

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
| ONNX Runtime Web (WebGPU) | Desktop browser | ~60 ms |
| ONNX Runtime Web (WASM) | Desktop browser | ~500 ms |
| ONNX Runtime Web (WebGPU) | iPhone (mobile) | ~237 ms |
| ONNX Runtime Web (WASM) | iPhone (mobile) | ~500 ms |

## Contents

- [Architecture](lded/architecture.md)
- [Training Guide](lded/training_guide.md)
- [Technical README](lded/README.md)

---

**CAS project work** by Simon Fuchs & Raphael Fuchs, 2026
