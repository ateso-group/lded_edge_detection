"""Tests für Heatmap Head: Output Range, Soft-Argmax Genauigkeit."""

import pytest
import torch

from model.head.heatmap_head import HeatmapHead, SoftArgmax2D


@pytest.fixture
def head():
    return HeatmapHead(in_channels=128, num_corners=4, temperature=1.0)


def test_heatmap_head_output_shapes(head):
    """Head liefert korrekte Output Shapes."""
    features = torch.randn(2, 128, 128, 128)
    heatmaps, coords, confidence = head(features)

    assert heatmaps.shape == (2, 4, 128, 128)
    assert coords.shape == (2, 4, 2)
    assert confidence.shape == (2, 1)


def test_soft_argmax_center():
    """Soft-Argmax erkennt Peak in der Mitte."""
    soft_argmax = SoftArgmax2D(temperature=0.1)

    heatmap = torch.zeros(1, 1, 64, 64)
    heatmap[0, 0, 32, 32] = 10.0  # Peak in der Mitte

    coords = soft_argmax(heatmap)
    assert coords.shape == (1, 1, 2)

    # Sollte nahe (0.5, 0.5) sein
    assert coords[0, 0, 0].item() == pytest.approx(0.5, abs=0.05)
    assert coords[0, 0, 1].item() == pytest.approx(0.5, abs=0.05)


def test_soft_argmax_corner():
    """Soft-Argmax erkennt Peak in der Ecke."""
    soft_argmax = SoftArgmax2D(temperature=0.1)

    heatmap = torch.zeros(1, 1, 64, 64)
    heatmap[0, 0, 0, 0] = 10.0  # Peak oben links

    coords = soft_argmax(heatmap)
    assert coords[0, 0, 0].item() == pytest.approx(0.0, abs=0.1)
    assert coords[0, 0, 1].item() == pytest.approx(0.0, abs=0.1)


def test_soft_argmax_differentiable():
    """Soft-Argmax ist differenzierbar."""
    soft_argmax = SoftArgmax2D(temperature=1.0)
    heatmap = torch.randn(1, 4, 64, 64, requires_grad=True)

    coords = soft_argmax(heatmap)
    loss = coords.sum()
    loss.backward()

    assert heatmap.grad is not None
    assert heatmap.grad.shape == heatmap.shape


def test_soft_argmax_output_range():
    """Soft-Argmax Koordinaten liegen in [0, 1]."""
    soft_argmax = SoftArgmax2D(temperature=1.0)
    heatmap = torch.randn(4, 4, 64, 64)

    coords = soft_argmax(heatmap)
    assert coords.min() >= 0.0
    assert coords.max() <= 1.0
