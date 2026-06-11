"""Integration Test: ONNX Export + Shape-Check."""

import tempfile
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import pytest
import torch

from model.model import LDED


@pytest.fixture
def model():
    model = LDED(backbone_pretrained=False)
    model.eval()
    return model


class LDEDExportWrapper(torch.nn.Module):
    """Wrapper für ONNX Export — nur Koordinaten + Confidence."""

    def __init__(self, base_model: LDED):
        super().__init__()
        self.model = base_model

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        _, coords, confidence = self.model(x)
        return coords, confidence


def test_onnx_export_and_load(model):
    """ONNX Export erzeugt valides Modell."""
    export_model = LDEDExportWrapper(model)
    dummy_input = torch.randn(1, 3, 512, 512)

    with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as f:
        onnx_path = f.name

    torch.onnx.export(
        export_model,
        dummy_input,
        onnx_path,
        opset_version=17,
        input_names=["image"],
        output_names=["corners", "confidence"],
    )

    # Validierung
    onnx_model = onnx.load(onnx_path)
    onnx.checker.check_model(onnx_model)

    # Shape Check
    session = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    input_np = np.random.randn(1, 3, 512, 512).astype(np.float32)
    outputs = session.run(None, {"image": input_np})

    assert outputs[0].shape == (1, 4, 2)   # corners
    assert outputs[1].shape == (1, 1)       # confidence

    Path(onnx_path).unlink()


def test_onnx_model_size(model):
    """ONNX-Modell ist kleiner als 10 MB (Zielwert)."""
    export_model = LDEDExportWrapper(model)
    dummy_input = torch.randn(1, 3, 512, 512)

    with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as f:
        onnx_path = f.name

    torch.onnx.export(
        export_model, dummy_input, onnx_path,
        opset_version=17, input_names=["image"],
        output_names=["corners", "confidence"],
    )

    size_mb = Path(onnx_path).stat().st_size / (1024 * 1024)
    Path(onnx_path).unlink()

    assert size_mb < 10.0, f"ONNX Modell zu groß: {size_mb:.2f} MB"
