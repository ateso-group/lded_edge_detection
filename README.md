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

## Contents

- [Architecture](lded/architecture.md)
- [Training Guide](lded/training_guide.md)
- [Technical README](lded/README.md)

---

**CAS project work** by Simon Fuchs & Raphael Fuchs, 2026
