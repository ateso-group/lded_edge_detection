"""Tests für das LDED-Modell: Forward Pass, Output Shapes."""

import pytest
import torch

from model.model import LDED


@pytest.fixture
def model():
    return LDED(backbone_pretrained=False, fpn_out_channels=128, num_corners=4)


def test_forward_pass_output_types(model):
    """Forward Pass gibt 3 Tensoren zurück."""
    x = torch.randn(2, 3, 512, 512)
    heatmaps, coords, confidence = model(x)

    assert isinstance(heatmaps, torch.Tensor)
    assert isinstance(coords, torch.Tensor)
    assert isinstance(confidence, torch.Tensor)


def test_heatmap_shape(model):
    """Heatmaps haben Shape [B, 4, H, W]."""
    x = torch.randn(2, 3, 512, 512)
    heatmaps, _, _ = model(x)

    assert heatmaps.shape[0] == 2
    assert heatmaps.shape[1] == 4
    assert heatmaps.shape[2] > 0
    assert heatmaps.shape[3] > 0


def test_coords_shape(model):
    """Koordinaten haben Shape [B, 4, 2]."""
    x = torch.randn(2, 3, 512, 512)
    _, coords, _ = model(x)

    assert coords.shape == (2, 4, 2)


def test_coords_range(model):
    """Koordinaten liegen in [0, 1]."""
    x = torch.randn(1, 3, 512, 512)
    _, coords, _ = model(x)

    assert coords.min() >= 0.0
    assert coords.max() <= 1.0


def test_confidence_shape(model):
    """Confidence hat Shape [B, 1]."""
    x = torch.randn(2, 3, 512, 512)
    _, _, confidence = model(x)

    assert confidence.shape == (2, 1)


def test_freeze_backbone(model):
    """Backbone einfrieren setzt requires_grad=False."""
    model.freeze_backbone()
    for param in model.backbone.parameters():
        assert not param.requires_grad


def test_unfreeze_backbone(model):
    """Backbone auftauen setzt requires_grad=True."""
    model.freeze_backbone()
    model.unfreeze_backbone()
    for param in model.backbone.parameters():
        assert param.requires_grad


def test_param_groups(model):
    """get_param_groups gibt 3 Gruppen mit korrektem LR."""
    groups = model.get_param_groups(backbone_lr=1e-4, head_lr=3e-4)
    assert len(groups) == 3
    assert groups[0]["lr"] == 1e-4
    assert groups[1]["lr"] == 3e-4
    assert groups[2]["lr"] == 3e-4
