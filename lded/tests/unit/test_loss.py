"""Tests für AWing Loss: Werte, Gradientenfluss."""

import pytest
import torch

from model.loss.awing import AdaptiveWingLoss, LDEDLoss


@pytest.fixture
def awing_loss():
    return AdaptiveWingLoss()


@pytest.fixture
def lded_loss():
    return LDEDLoss(awing_weight=1.0, bce_weight=0.1)


def test_awing_zero_loss(awing_loss):
    """AWing Loss ist 0 bei identischen Inputs."""
    pred = torch.ones(2, 4, 64, 64)
    target = torch.ones(2, 4, 64, 64)
    loss = awing_loss(pred, target)
    assert loss.item() == pytest.approx(0.0, abs=1e-6)


def test_awing_positive_loss(awing_loss):
    """AWing Loss ist positiv bei unterschiedlichen Inputs."""
    pred = torch.randn(2, 4, 64, 64)
    target = torch.randn(2, 4, 64, 64)
    loss = awing_loss(pred, target)
    assert loss.item() > 0.0


def test_awing_gradient_flow(awing_loss):
    """AWing Loss ermöglicht Gradientenfluss."""
    pred = torch.randn(2, 4, 64, 64, requires_grad=True)
    target = torch.randn(2, 4, 64, 64)
    loss = awing_loss(pred, target)
    loss.backward()

    assert pred.grad is not None
    assert not torch.all(pred.grad == 0)


def test_awing_small_error_more_penalized(awing_loss):
    """Kleine Fehler werden durch AWing relativ stärker bestraft als linear."""
    small_error = torch.tensor([[[[0.01]]]])
    large_error = torch.tensor([[[[1.0]]]])
    target = torch.zeros_like(small_error)

    loss_small = awing_loss(small_error, target)
    loss_large = awing_loss(large_error, target)

    # AWing Loss sollte für große Fehler größer sein
    assert loss_large.item() > loss_small.item()


def test_lded_loss_keys(lded_loss):
    """LDEDLoss gibt Dict mit total, awing, bce zurück."""
    pred_hm = torch.randn(2, 4, 64, 64)
    target_hm = torch.randn(2, 4, 64, 64)
    pred_conf = torch.randn(2, 1)
    target_conf = torch.ones(2, 1)

    losses = lded_loss(pred_hm, target_hm, pred_conf, target_conf)
    assert "total" in losses
    assert "awing" in losses
    assert "bce" in losses


def test_lded_loss_total_composition(lded_loss):
    """Total Loss = awing_weight * awing + bce_weight * bce."""
    pred_hm = torch.randn(2, 4, 64, 64)
    target_hm = torch.randn(2, 4, 64, 64)
    pred_conf = torch.randn(2, 1)
    target_conf = torch.ones(2, 1)

    losses = lded_loss(pred_hm, target_hm, pred_conf, target_conf)
    expected_total = 1.0 * losses["awing"] + 0.1 * losses["bce"]
    assert losses["total"].item() == pytest.approx(expected_total.item(), abs=1e-5)
