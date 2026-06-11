"""Tests für Pre-Processing: Padding, Resize, Normalisierung."""

import numpy as np
import pytest

from model.inference.preprocess import Preprocessor


@pytest.fixture
def preprocessor():
    return Preprocessor(input_size=(512, 512), padding_ratio=0.05)


def test_add_padding_shape(preprocessor):
    """Padding vergrößert das Bild symmetrisch."""
    image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    padded, pad = preprocessor.add_padding(image)

    expected_h = 480 + 2 * pad
    expected_w = 640 + 2 * pad
    assert padded.shape == (expected_h, expected_w, 3)


def test_add_padding_value(preprocessor):
    """Padding berechnet korrekte Pixel-Anzahl (5% des Umfangs)."""
    image = np.zeros((480, 640, 3), dtype=np.uint8)
    _, pad = preprocessor.add_padding(image)

    expected_pad = int(0.05 * (480 + 640))
    assert pad == expected_pad


def test_preprocess_output_shape(preprocessor):
    """Vorverarbeitung liefert [1, 3, 512, 512] Tensor."""
    image = np.random.randint(0, 255, (1080, 1920, 3), dtype=np.uint8)
    tensor, meta = preprocessor.preprocess(image)

    assert tensor.shape == (1, 3, 512, 512)
    assert tensor.dtype == np.float32


def test_preprocess_normalization_range(preprocessor):
    """Normalisierte Werte liegen im erwarteten Bereich."""
    image = np.ones((512, 512, 3), dtype=np.uint8) * 128
    tensor, _ = preprocessor.preprocess(image)

    # Werte sollten nach ImageNet-Normalisierung ungefähr bei 0 liegen
    assert tensor.mean() == pytest.approx(0.0, abs=1.0)


def test_preprocess_meta_keys(preprocessor):
    """Meta-Dict enthält alle notwendigen Schlüssel."""
    image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    _, meta = preprocessor.preprocess(image)

    assert "orig_size" in meta
    assert "padded_size" in meta
    assert "pad" in meta
    assert "scale_x" in meta
    assert "scale_y" in meta
    assert meta["orig_size"] == (480, 640)
