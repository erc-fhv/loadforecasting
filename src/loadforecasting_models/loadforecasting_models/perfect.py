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
        loss_relative_to: str = "mean",
        ) -> dict:
        """
        Evaluate the model on the given x_test and y_test.
        """

        # Convert numpy to torch if needed
        if isinstance(y_test, np.ndarray):
            y_tensor  = torch.from_numpy(y_test).float()
        else:
            y_tensor  = y_test.float()

        if results is None:
            results = {}

        output = self.predict(y_tensor)   # pass Y to get perfect prediction

        assert output.shape == y_tensor.shape, \
            f"Shape mismatch: got {output.shape}, expected {y_tensor.shape})"
        # Unnormalize the target variable, if wished.
        if de_normalize:
            assert self.normalizer is not None, "No normalizer given."
            y_tensor = self.normalizer.de_normalize_y(y_tensor)
            output = self.normalizer.de_normalize_y(output)
            assert isinstance(y_tensor, torch.Tensor), "Denormalized y_tensor is not a torch.Tensor"

        # Compute Loss
        if loss_relative_to == "mean":
            reference = float(torch.abs(torch.mean(y_tensor)))
        elif loss_relative_to == "max":
            reference = float(torch.abs(torch.max(y_tensor)))
        elif loss_relative_to == "range":
            reference = float(torch.max(y_tensor) - torch.min(y_tensor))
        else:
            raise ValueError(f"Unexpected parameter: loss_relative_to = {loss_relative_to}")
        loss = eval_fn(output, y_tensor)
        results['test_loss'] = [loss.item()]
        results['test_loss_relative'] = [100.0*loss.item()/reference]            
        results['predicted_profile'] = output

        return results

    def state_dict(self) -> dict:
        """No persistent parameter needed for this trivial model."""
        state_dict = {}
        return state_dict

    def load_state_dict(self, state_dict) -> None:
        """No persistent parameter needed for this trivial model."""
