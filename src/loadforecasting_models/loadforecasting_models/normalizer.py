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

        # Optionally convert to ndarray
        is_torch_tensor = self.is_tensor(x)
        if is_torch_tensor:
            x = self.convert_to_ndarray(x)

        if training:
            # Estimate the mean and standard deviation of the data during training
            self.mean_x = np.mean(x, axis=(0, 1))
            self.std_x = np.std(x, axis=(0, 1))

            if np.isclose(self.std_x, 0).any():
                # Avoid a division by zero (which can occur for constant features)
                self.std_x = np.where(np.isclose(self.std_x, 0), 1e-8, self.std_x)

        x_normalized = (x - self.mean_x) / self.std_x

        # Optionally convert back to torch tensor
        if is_torch_tensor:
            x_normalized = self.convert_to_torch_tensor(x_normalized)

        return x_normalized

    def normalize_y(self, y, training=True):
        """Z-Normalize the output data of the model."""

        # Optionally convert to ndarray
        is_torch_tensor = self.is_tensor(y)
        if is_torch_tensor:
            y = self.convert_to_ndarray(y)

        if training:
            # Estimate the mean and standard deviation of the data during training
            self.mean_y = np.mean(y, axis=(0, 1))
            self.std_y = np.std(y)

        if np.isclose(self.std_y, 0):
            assert False, "Normalization leads to division by zero."

        y_normalized = (y - self.mean_y) / self.std_y

        # Optionally convert back to torch tensor
        if is_torch_tensor:
            y_normalized = self.convert_to_torch_tensor(y_normalized)

        return y_normalized

    def de_normalize_y(self, y):
        """Undo normalization"""

        # Optionally convert to ndarray
        is_torch_tensor = self.is_tensor(y)
        if is_torch_tensor:
            y = self.convert_to_ndarray(y)

        y_denormalized = (y * self.std_y) + self.mean_y

        # Optionally convert back to torch tensor
        if is_torch_tensor:
            y_denormalized = self.convert_to_torch_tensor(y_denormalized)

        return y_denormalized

    def de_normalize_x(self, x):
        """Undo z-normalization."""

        # Optionally convert to ndarray
        is_torch_tensor = self.is_tensor(x)
        if is_torch_tensor:
            x = self.convert_to_ndarray(x)

        x_denormalized = (x * self.std_x) + self.mean_x

        # Optionally convert back to torch tensor
        if is_torch_tensor:
            x_denormalized = self.convert_to_torch_tensor(x_denormalized)

        return x_denormalized

    def is_tensor(self, x):
        return torch.is_tensor(x)

    def convert_to_ndarray(self, x):
        return np.array(x)

    def convert_to_torch_tensor(self, x):
        return torch.tensor(x)
