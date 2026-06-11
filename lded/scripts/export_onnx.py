"""
ONNX Export für LDED.

Exportiert das PyTorch-Modell als ONNX für Browser-Deployment via ONNX Runtime Web.
"""

import argparse
from pathlib import Path

import torch
import onnx

from model.model import LDED


def export_to_onnx(
    checkpoint_path: str,
    output_path: str,
    model_config: str = "configs/model/mobilenetv3.yaml",
    input_size: tuple[int, int] = (512, 512),
    opset_version: int = 17,
) -> None:
    """Exportiert LDED als ONNX-Modell.

    Args:
        checkpoint_path: Pfad zum PyTorch-Checkpoint (.pt).
        output_path: Ausgabepfad für das ONNX-Modell.
        model_config: Pfad zur Modell-Konfiguration.
        input_size: Input-Größe (H, W).
        opset_version: ONNX Opset Version.
    """
    # Modell laden
    model = LDED.from_config(model_config)
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    # Dummy Input
    dummy_input = torch.randn(1, 3, input_size[0], input_size[1])

    # Wrapper für saubere ONNX-Outputs (nur Koordinaten + Confidence)
    class LDEDExport(torch.nn.Module):
        def __init__(self, base_model: LDED):
            super().__init__()
            self.model = base_model

        def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
            heatmaps, coords, confidence = self.model(x)
            return coords, confidence

    export_model = LDEDExport(model)

    # ONNX Export
    output_dir = Path(output_path).parent
    output_dir.mkdir(parents=True, exist_ok=True)

    torch.onnx.export(
        export_model,
        dummy_input,
        output_path,
        opset_version=opset_version,
        input_names=["image"],
        output_names=["corners", "confidence"],
        dynamic_axes={"image": {0: "batch"}},
        do_constant_folding=True,
    )

    # Validierung
    onnx_model = onnx.load(output_path)
    onnx.checker.check_model(onnx_model)

    # Modellgröße
    file_size_mb = Path(output_path).stat().st_size / (1024 * 1024)
    print(f"ONNX-Modell exportiert: {output_path}")
    print(f"  Opset:  {opset_version}")
    print(f"  Größe:  {file_size_mb:.2f} MB")
    print(f"  Inputs: {[inp.name for inp in onnx_model.graph.input]}")
    print(f"  Outputs: {[out.name for out in onnx_model.graph.output]}")

    # Shape-Check
    with torch.no_grad():
        coords, confidence = export_model(dummy_input)
        print(f"  Corners Shape:    {coords.shape}")
        print(f"  Confidence Shape: {confidence.shape}")


def main():
    parser = argparse.ArgumentParser(description="LDED ONNX Export")
    parser.add_argument(
        "--checkpoint", type=str, required=True, help="Pfad zum PyTorch-Checkpoint"
    )
    parser.add_argument(
        "--output", type=str, default="models/docaligner_lite.onnx",
        help="Ausgabepfad für ONNX-Modell",
    )
    parser.add_argument("--model-config", type=str, default="configs/model/mobilenetv3.yaml")
    parser.add_argument("--input-size", type=int, nargs=2, default=[512, 512])
    parser.add_argument("--opset", type=int, default=17)
    args = parser.parse_args()

    export_to_onnx(
        checkpoint_path=args.checkpoint,
        output_path=args.output,
        model_config=args.model_config,
        input_size=tuple(args.input_size),
        opset_version=args.opset,
    )


if __name__ == "__main__":
    main()
