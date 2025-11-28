from typing import Union, Tuple, Any
import numpy as np
import torch

# Define a type that can be either a torch Tensor or a numpy ndarray
ArrayLike = Union[torch.Tensor, np.ndarray]

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

    def normalize(
        self,
        x: ArrayLike,
        y: ArrayLike,
        training: bool =True
        ) -> Tuple[ArrayLike, ArrayLike]:
        """Normalize both input and output data of the model."""

        x_normalized = self.normalize_x(x, training)
        y_normalized = self.normalize_y(y, training)

        return x_normalized, y_normalized

    def de_normalize(
        self,
        x: ArrayLike,
        y: ArrayLike,
        ) -> Tuple[ArrayLike, ArrayLike]:
        """De-Normalize both input and output data of the model."""

        x_de_normalized = self.de_normalize_x(x)
        y_de_normalized = self.de_normalize_y(y)

        return x_de_normalized, y_de_normalized

    def normalize_x(
        self,
        x: ArrayLike,
        training: bool =True,
        ) -> ArrayLike:
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
                # Avoid a division by zero (which can occur for constant features).
                # For zero-features (i.e. features that are always 0), we set the std to 1.
                # For other constant features, we set the std to the max value of that feature.
                max_vals = np.max(x, axis=(0, 1))
                is_null_feature = np.isclose(max_vals, 0)
                fallback_vals = np.where(is_null_feature, 1.0, max_vals)
                self.std_x = np.where(np.isclose(self.std_x, 0), fallback_vals, self.std_x)

        x_normalized = (x - self.mean_x) / self.std_x

        # Optionally convert back to torch tensor
        if is_torch_tensor:
            x_normalized = self.convert_to_torch_tensor(x_normalized)

        return x_normalized

    def normalize_y(
        self,
        y: ArrayLike,
        training: bool =True,
        ) -> ArrayLike:
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

    def de_normalize_y(
        self,
        y: ArrayLike
        ) -> ArrayLike:
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

    def de_normalize_x(
        self,
        x: ArrayLike
        ) -> ArrayLike:
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

    def is_tensor(self, x: ArrayLike) -> bool:
        """Check if the input is a torch tensor."""
        return torch.is_tensor(x)

    def convert_to_ndarray(self, x: Any) -> np.ndarray:
        """Convert input to numpy ndarray."""
        return np.array(x)

    def convert_to_torch_tensor(self, x: Any) -> torch.Tensor:
        """Convert input to torch tensor."""
        return torch.tensor(x)
