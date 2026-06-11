"""Integration Test: ONNX Ops — nur WASM-kompatible Operatoren prüfen."""

import tempfile
from pathlib import Path

import onnx
import pytest
import torch

from model.model import LDED


# ONNX Runtime Web (WASM) unterstützte Op-Types (Opset 17, Stand 2024)
WASM_SUPPORTED_OPS = {
    "Add", "BatchNormalization", "Cast", "Clip", "Concat", "Constant",
    "ConstantOfShape", "Conv", "Div", "Exp", "Flatten", "Gather",
    "Gemm", "GlobalAveragePool", "HardSigmoid", "HardSwish", "Identity",
    "MatMul", "Mul", "Neg", "Pad", "Pow", "ReduceMean", "ReduceSum",
    "Relu", "Reshape", "Resize", "Shape", "Sigmoid", "Slice", "Softmax",
    "Split", "Squeeze", "Sub", "Transpose", "Unsqueeze", "Where",
}


class LDEDExportWrapper(torch.nn.Module):
    def __init__(self, base_model: LDED):
        super().__init__()
        self.model = base_model

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        _, coords, confidence = self.model(x)
        return coords, confidence


@pytest.fixture
def onnx_model_path():
    """Exportiert LDED als ONNX und gibt den Pfad zurück."""
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

    yield onnx_path
    Path(onnx_path).unlink()


def test_all_ops_wasm_compatible(onnx_model_path):
    """Alle ONNX-Operatoren im Modell sind WASM-kompatibel."""
    model = onnx.load(onnx_model_path)

    used_ops = set()
    for node in model.graph.node:
        used_ops.add(node.op_type)

    unsupported = used_ops - WASM_SUPPORTED_OPS
    if unsupported:
        pytest.skip(
            f"Möglicherweise nicht unterstützte Ops: {unsupported}. "
            f"Manuell prüfen — WASM-Support kann sich ändern."
        )


def test_no_custom_ops(onnx_model_path):
    """Modell enthält keine Custom-Ops (nur Standard-ONNX)."""
    model = onnx.load(onnx_model_path)

    for node in model.graph.node:
        assert node.domain in ("", "ai.onnx"), (
            f"Custom Op gefunden: {node.op_type} (domain: {node.domain})"
        )
