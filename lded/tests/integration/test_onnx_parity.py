"""Integration Test: PyTorch vs. ONNX Ausgabe ≤ 0.5px Abweichung."""

import tempfile
from pathlib import Path

import numpy as np
import onnxruntime as ort
import pytest
import torch

from model.model import LDED


class LDEDExportWrapper(torch.nn.Module):
    def __init__(self, base_model: LDED):
        super().__init__()
        self.model = base_model

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        _, coords, confidence = self.model(x)
        return coords, confidence


@pytest.fixture
def model_and_onnx():
    """Erstellt PyTorch-Modell und exportiert es als ONNX."""
    model = LDED(backbone_pretrained=False)
    model.eval()

    export_model = LDEDExportWrapper(model)
    dummy_input = torch.randn(1, 3, 512, 512)

    with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as f:
        onnx_path = f.name

    torch.onnx.export(
        export_model, dummy_input, onnx_path,
        opset_version=17,
        input_names=["image"],
        output_names=["corners", "confidence"],
    )

    session = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    yield model, export_model, session, onnx_path

    Path(onnx_path).unlink()


def test_pytorch_onnx_parity(model_and_onnx):
    """PyTorch und ONNX Ausgaben weichen maximal 0.5px ab (bei 512px)."""
    model, export_model, session, _ = model_and_onnx

    # Gleicher Input
    input_np = np.random.randn(1, 3, 512, 512).astype(np.float32)
    input_torch = torch.from_numpy(input_np)

    # PyTorch Inferenz
    with torch.no_grad():
        pt_coords, pt_confidence = export_model(input_torch)

    # ONNX Inferenz
    onnx_outputs = session.run(None, {"image": input_np})
    onnx_coords = onnx_outputs[0]
    onnx_confidence = onnx_outputs[1]

    # Vergleich: Koordinaten (in Pixel-Raum bei 512×512)
    pt_coords_px = pt_coords.numpy() * 512
    onnx_coords_px = onnx_coords * 512

    max_diff_px = np.abs(pt_coords_px - onnx_coords_px).max()
    assert max_diff_px < 0.5, f"Max Abweichung: {max_diff_px:.3f}px (Limit: 0.5px)"

    # Vergleich: Confidence
    pt_conf = pt_confidence.numpy()
    max_conf_diff = np.abs(pt_conf - onnx_confidence).max()
    assert max_conf_diff < 0.01, f"Confidence Abweichung: {max_conf_diff:.6f}"
