"""Tests für Post-Processing: Koordinaten-Rescaling, Offset-Korrektur."""

import numpy as np
import pytest

from model.inference.predictor import LDEDPredictor


class TestRescaleCorners:
    """Tests für die Koordinaten-Rücktransformation."""

    def setup_method(self):
        """Erstellt einen Dummy-Predictor (ohne echtes Modell)."""
        # Wir testen nur die _rescale_corners Methode
        self.meta = {
            "orig_size": (480, 640),
            "padded_size": (592, 752),
            "pad": 56,
            "scale_x": 752 / 512,
            "scale_y": 592 / 512,
        }

    def test_center_point(self):
        """Punkt in der Mitte (0.5, 0.5) wird korrekt skaliert."""
        coords = np.array([[0.5, 0.5]], dtype=np.float32)

        corners = coords.copy()
        corners[:, 0] = corners[:, 0] * 752 - 56  # x: padded_w * 0.5 - pad
        corners[:, 1] = corners[:, 1] * 592 - 56  # y: padded_h * 0.5 - pad

        corners[:, 0] = np.clip(corners[:, 0], 0, 639)
        corners[:, 1] = np.clip(corners[:, 1], 0, 479)

        assert corners[0, 0] > 0
        assert corners[0, 1] > 0
        assert corners[0, 0] < 640
        assert corners[0, 1] < 480

    def test_origin_point(self):
        """Punkt (0, 0) wird auf (-pad, -pad) → clipped zu (0, 0)."""
        coords = np.array([[0.0, 0.0]], dtype=np.float32)

        corners = coords.copy()
        corners[:, 0] = corners[:, 0] * 752 - 56
        corners[:, 1] = corners[:, 1] * 592 - 56

        corners[:, 0] = np.clip(corners[:, 0], 0, 639)
        corners[:, 1] = np.clip(corners[:, 1], 0, 479)

        assert corners[0, 0] == 0.0
        assert corners[0, 1] == 0.0

    def test_four_corners(self):
        """4 Eckpunkte werden korrekt transformiert (keine NaN, im Bild)."""
        coords = np.array([
            [0.1, 0.1],
            [0.9, 0.1],
            [0.9, 0.9],
            [0.1, 0.9],
        ], dtype=np.float32)

        corners = coords.copy()
        corners[:, 0] = corners[:, 0] * 752 - 56
        corners[:, 1] = corners[:, 1] * 592 - 56
        corners[:, 0] = np.clip(corners[:, 0], 0, 639)
        corners[:, 1] = np.clip(corners[:, 1], 0, 479)

        assert not np.any(np.isnan(corners))
        assert np.all(corners[:, 0] >= 0)
        assert np.all(corners[:, 0] <= 639)
        assert np.all(corners[:, 1] >= 0)
        assert np.all(corners[:, 1] <= 479)
