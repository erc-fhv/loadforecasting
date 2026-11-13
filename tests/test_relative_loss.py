"""Unit tests for the reference of the relative test loss (loss_relative_to)."""

import unittest
import torch

from loadforecasting_models import Normalizer, Persistence, Transformer
from loadforecasting_models.helpers import relative_loss_reference


class TestRelativeLossReference(unittest.TestCase):
    """Tests for relative_loss_reference and its use in the models' evaluate()."""

    def setUp(self):
        self.y = torch.tensor([[[-2.0], [1.0]], [[3.0], [-4.0]]])  # net load can be negative

    def test_references(self):
        self.assertAlmostEqual(relative_loss_reference(self.y, "mean"), 0.5)
        self.assertAlmostEqual(relative_loss_reference(self.y, ""), 0.5)  # default
        self.assertAlmostEqual(relative_loss_reference(self.y, "mean_abs"), 2.5)
        self.assertAlmostEqual(relative_loss_reference(self.y, "max"), 3.0)
        self.assertAlmostEqual(relative_loss_reference(self.y, "range"), 7.0)
        with self.assertRaises(ValueError):
            relative_loss_reference(self.y, "median")

    def test_sklearn_style_model(self):
        x = torch.zeros(2, 2, 1)  # persistence predicts the zero lag
        model = Persistence(lagged_load_feature=0, normalizer=Normalizer(), loss_relative_to="mean_abs")
        results = model.evaluate(x, self.y)
        self.assertAlmostEqual(results["test_loss"][0], 2.5, places=5)
        self.assertAlmostEqual(results["test_loss_relative"][0], 100.0, places=4)
        # an explicit argument overrides the attribute
        results = model.evaluate(x, self.y, loss_relative_to="range")
        self.assertAlmostEqual(results["test_loss_relative"][0], 100.0 * 2.5 / 7.0, places=4)

    def test_pytorch_model(self):
        x = torch.zeros(2, 2, 1)
        model = Transformer('0.1k', normalizer=Normalizer(), loss_relative_to="mean_abs")
        results = model.evaluate(x, self.y)
        mae = results["test_loss"][0]
        self.assertAlmostEqual(results["test_loss_relative"][0], 100.0 * mae / 2.5, places=4)
        # an explicit argument overrides the attribute
        results = model.evaluate(x, self.y, loss_relative_to="range")
        self.assertAlmostEqual(results["test_loss_relative"][0], 100.0 * mae / 7.0, places=4)


if __name__ == '__main__':
    unittest.main()
