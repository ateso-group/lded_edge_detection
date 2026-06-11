"""Integration Test: End-to-end Bild → Polygon."""

import numpy as np
import pytest
import torch

from model.model import LDED
from model.inference.preprocess import Preprocessor


@pytest.fixture
def model():
    model = LDED(backbone_pretrained=False)
    model.eval()
    return model


@pytest.fixture
def preprocessor():
    return Preprocessor(input_size=(512, 512), padding_ratio=0.05)


def test_full_pipeline_rgb_image(model, preprocessor):
    """End-to-end: RGB-Bild → Preprocessing → Modell → 4 Eckpunkte."""
    # Simuliertes RGB-Bild
    image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)

    # Preprocessing
    tensor, meta = preprocessor.preprocess(image)
    assert tensor.shape == (1, 3, 512, 512)

    # Modell-Inferenz
    with torch.no_grad():
        input_tensor = torch.from_numpy(tensor)
        heatmaps, coords, confidence = model(input_tensor)

    # Outputs prüfen
    assert coords.shape == (1, 4, 2)
    assert confidence.shape == (1, 1)

    # Koordinaten im gültigen Bereich
    coords_np = coords.numpy()[0]
    assert np.all(coords_np >= 0.0)
    assert np.all(coords_np <= 1.0)


def test_full_pipeline_batch(model, preprocessor):
    """End-to-end mit Batch von 4 Bildern."""
    batch = []
    for _ in range(4):
        image = np.random.randint(0, 255, (720, 1280, 3), dtype=np.uint8)
        tensor, _ = preprocessor.preprocess(image)
        batch.append(tensor)

    batch_tensor = torch.from_numpy(np.concatenate(batch, axis=0))
    assert batch_tensor.shape == (4, 3, 512, 512)

    with torch.no_grad():
        heatmaps, coords, confidence = model(batch_tensor)

    assert coords.shape == (4, 4, 2)
    assert confidence.shape == (4, 1)


def test_full_pipeline_deterministic(model, preprocessor):
    """Gleiches Bild liefert gleiche Koordinaten (deterministic)."""
    image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    tensor, _ = preprocessor.preprocess(image)
    input_tensor = torch.from_numpy(tensor)

    with torch.no_grad():
        _, coords1, _ = model(input_tensor)
        _, coords2, _ = model(input_tensor)

    np.testing.assert_array_almost_equal(
        coords1.numpy(), coords2.numpy(), decimal=5
    )
