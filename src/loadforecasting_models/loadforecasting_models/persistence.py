from typing import Optional, Callable, Union
import numpy as np
import torch
from .normalizer import Normalizer

# Define a type that can be either a torch Tensor or a numpy ndarray
ArrayLike = Union[torch.Tensor, np.ndarray]

class Persistence:
    """
    Predict the load accord to the load last week.
    """

    def __init__(self,
            lagged_load_feature: int,
            normalizer: Normalizer,
            ) -> None:
        """
        Args:
            lagged_load_feature (int): The feature index in the input tensor that
                contains the lagged load to be used for prediction.
            normalizer (Normalizer): Used for X and Y normalization and denormalization.
        """
        self.normalizer = normalizer
        self.lagged_load_feature = lagged_load_feature

    def predict(self,
            x: ArrayLike,
            ) -> ArrayLike:
        """
        Upcoming load profile = load profile 7 days ago.

        Args:
            x (ArrayLike): Normalised model input tensor of shape (batch_len, 
                sequence_len, features), where the feature at index `lagged_load_feature`
                contains the lagged load values.

        Returns:
            ArrayLike: Predicted y tensor of shape (batch_len, sequence_len, 1).
        """

        # De-normalize all inputs
        x = self.normalizer.de_normalize_x(x)

        # Take the chosen lagged loads as predictions
        y_pred = x[:,:, self.lagged_load_feature]

        # Add axis and normalize y_pred again, to compare it to other models.
        y_pred = y_pred[:,:,np.newaxis]
        y_pred = self.normalizer.normalize_y(y_pred, training=False)

        return y_pred

    def train_model(self) -> dict:
        """No training necessary for the persistence model."""

        history = {}
        history['loss'] = [0.0]

        return history

    def evaluate(
        self,
        x_test: ArrayLike,
        y_test: ArrayLike,
        results: Optional[dict] = None,
        de_normalize: bool = False,
        eval_fn: Callable[..., torch.Tensor] = torch.nn.L1Loss(),
        loss_relative_to: str = "mean",
        ) -> dict:
        """
        Evaluate the model on the given x_test and y_test.
        """

        if results is None:
            results = {}

        # Convert numpy to torch if needed
        if isinstance(x_test, np.ndarray):
            x_tensor  = torch.from_numpy(x_test).float()
        else:
            x_tensor  = x_test.float()
        if isinstance(y_test, np.ndarray):
            y_tensor  = torch.from_numpy(y_test).float()
        else:
            y_tensor  = y_test.float()

        output = self.predict(x_tensor)

        assert output.shape == y_tensor.shape, \
            f"Shape mismatch: got {output.shape}, expected {y_tensor.shape})"

        # Unnormalize the target variable, if wished.
        if de_normalize:
            assert self.normalizer is not None, "No model_adapter given."
            y_tensor = self.normalizer.de_normalize_y(y_tensor)
            assert isinstance(y_tensor, torch.Tensor), "Denormalized y_tensor is not a torch.Tensor"
            output = self.normalizer.de_normalize_y(output)
            assert isinstance(output, torch.Tensor), "Denormalized output is not a torch.Tensor"

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
