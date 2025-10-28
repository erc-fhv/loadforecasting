import numpy as np
import torch

class Normalizer():
    """
    Simple normalizer class for input and output normalization and denormalization.
    Works with ndarrays and torch tensors.
    """

    def __init__(self):
        self.mean_x = 0.
        self.std_x = 1.
        self.mean_y = 0.
        self.std_y = 1.

    def _is_tensor(self, x):
        return torch.is_tensor(x)

    def _mean(self, x, axis=None):
        return torch.mean(x, dim=axis) if self._is_tensor(x) else np.mean(x, axis=axis)

    def _std(self, x, axis=None):
        return torch.std(x, dim=axis, unbiased=False) if self._is_tensor(x) else np.std(x, axis=axis)

    def _isclose(self, a, b=0):
        return torch.isclose(a, torch.tensor(b, dtype=a.dtype, device=a.device)) if self._is_tensor(a) else np.isclose(a, b)

    def _where(self, cond, a, b):
        return torch.where(cond, a, b) if self._is_tensor(cond) else np.where(cond, a, b)

    def normalize(self, x, y, training=True):
        """Normalize both input and output data of the model."""

        x_normalized = self.normalize_x(x, training)
        y_normalized = self.normalize_y(y, training)

        return x_normalized, y_normalized

    def de_normalize(self, x, y):
        """De-Normalize both input and output data of the model."""

        x_de_normalized = self.de_normalize_x(x)
        y_de_normalized = self.de_normalize_y(y)

        return x_de_normalized, y_de_normalized

    def normalize_x(self, x, training=True):
        """Z-Normalize the input data of the model."""

        if training:
            self.mean_x = self._mean(x, axis=(0, 1))
            self.std_x = self._std(x, axis=(0, 1))

            if self._isclose(self.std_x, 0).any():
                self.std_x = self._where(self._isclose(self.std_x, 0), 
                                         torch.tensor(1e-8, dtype=self.std_x.dtype, device=self.std_x.device) if self._is_tensor(x) else 1e-8, 
                                         self.std_x)

        x_normalized = (x - self.mean_x) / self.std_x
        return x_normalized

    def normalize_y(self, y, training=True):
        """Z-Normalize the output data of the model."""

        if training:
            self.mean_y = self._mean(y, axis=(0, 1))
            self.std_y = self._std(y, axis=(0, 1))

        if self._is_tensor(y):
            if torch.any(self._isclose(self.std_y, 0)):
                raise ValueError("Normalization leads to division by zero.")
        else:
            if np.isclose(self.std_y, 0).any():
                raise ValueError("Normalization leads to division by zero.")

        y_normalized = (y - self.mean_y) / self.std_y
        return y_normalized

    def de_normalize_y(self, y):
        """Undo normalization"""
        return (y * self.std_y) + self.mean_y

    def de_normalize_x(self, x):
        """Undo z-normalization."""
        return (x * self.std_x) + self.mean_x
