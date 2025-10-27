from typing import Optional, Callable
import numpy as np
import torch
from .normalizer import Normalizer

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
            lagged_load_feature (int): The feature index in the input tensor
                that contains the lagged load to be used for prediction.
            normalizer (Normalizer): Used for X and Y normalization and denormalization.
        """
        self.normalizer = normalizer
        self.lagged_load_feature = lagged_load_feature

    def predict(self,
            x: torch.Tensor,
            ) -> torch.Tensor:
        """
        Upcoming load profile = load profile 7 days ago.

        Args:
            x (torch.Tensor): Normalised model input tensor of shape (batch_len, 
                sequence_len, features), where the feature at index `lagged_load_feature`
                contains the lagged load values.

        Returns:
            torch.Tensor: Predicted y tensor of shape (batch_len, sequence_len, 1).
        """

        x = self.normalizer.de_normalize_x(x)    # de-normalize all inputs

        # Take the chosen lagged loads as predictions
        #
        y_pred = x[:,:, self.lagged_load_feature]

        # Add axis and normalize y_pred again, to compare it to other models.
        #
        y_pred = y_pred[:,:,np.newaxis]
        y_pred = self.normalizer.normalize_y(y_pred, training=False)
        assert y_pred.shape == (x.size(0), x.size(1), 1), \
            f"Shape mismatch: got {y_pred.shape}, expected ({x.size(0)}, {x.size(1)}, 1)"

        return y_pred

    def train_model(self) -> dict:
        """No training necessary for the persistence model."""

        history = {}
        history['loss'] = [0.0]

        return history

    def evaluate(
        self,
        x_test: torch.Tensor,
        y_test: torch.Tensor,
        results: Optional[dict] = None,
        de_normalize: bool = False,
        eval_fn: Callable[..., torch.Tensor] = torch.nn.L1Loss(),
        ) -> dict:
        """
        Evaluate the model on the given x_test and y_test.
        """

        if results is None:
            results = {}

        output = self.predict(x_test)

        assert output.shape == y_test.shape, \
            f"Shape mismatch: got {output.shape}, expected {y_test.shape})"

        # Unnormalize the target variable, if wished.
        if de_normalize:
            assert self.normalizer is not None, "No model_adapter given."
            y_test = self.normalizer.de_normalize_y(y_test)
            output = self.normalizer.de_normalize_y(output)

        # Compute Loss
        loss = eval_fn(output, y_test)
        results['test_loss'] = [loss.item()]
        reference = float(torch.mean(y_test))
        results['test_loss_relative'] = [100.0*loss.item()/reference]            
        results['predicted_profile'] = output

        return results

    def state_dict(self) -> dict:
        """No persistent parameter needed for this trivial model."""
        state_dict = {}
        return state_dict

    def load_state_dict(self, state_dict) -> None:
        """No persistent parameter needed for this trivial model."""
