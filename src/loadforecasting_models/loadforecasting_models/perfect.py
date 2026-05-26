from typing import Optional, Callable, Union
import numpy as np
import torch
from .normalizer import Normalizer

# Define a type that can be either a torch Tensor or a numpy ndarray
ArrayLike = Union[torch.Tensor, np.ndarray]

class Perfect():
    """
    Trivial 'model': Just gets and returns the perfect profile (used for reference).
    """

    def __init__(self,
            normalizer: Optional[Normalizer] = None,
            ) -> None:
        """
        Args:
            normalizer (Normalizer): Used for X and Y normalization and denormalization.
        """
        self.normalizer = normalizer

    def predict(self,
                y_real: ArrayLike
                ) -> ArrayLike:
        """Gets and return the perfect profile."""

        y_pred = y_real

        return y_pred

    def train_model(self) -> dict:
        """No training necessary for the perfect model."""

        history = {}
        history['loss'] = [0.0]

        return history

    def evaluate(
        self,
        y_test: ArrayLike,
        results: Union[dict, None] = None,
        de_normalize: bool = False,
        eval_fn: Callable[..., torch.Tensor] = torch.nn.L1Loss(),
        loss_relative_to: str = "",
        ) -> dict:
        """
        Evaluate the model on the given x_test and y_test.
        """

        # Unnormalize the target variable, if wished.
        if de_normalize:
            assert self.normalizer is not None, "No normalizer given."
            y_test = self.normalizer.de_normalize_y(y_test)
            assert isinstance(y_test, torch.Tensor), "Denormalized output is not a torch.Tensor"

        if results is None:
            results = {}

        results['test_loss'] = [0.0]
        results['test_loss_relative'] = [0.0]
        results['predicted_profile'] = y_test

        return results

    def state_dict(self) -> dict:
        """No persistent parameter needed for this trivial model."""
        state_dict = {}
        return state_dict

    def load_state_dict(self, state_dict) -> None:
        """No persistent parameter needed for this trivial model."""
