import torch
import numpy as np
from loadforecasting_models.interfaces import ModelAdapterProtocol

class Persistence:
    """
    Predict the load accord to the load last week.
    """

    def __init__(self,
            model_adapter: ModelAdapterProtocol,
            ) -> None:
        """
        Args:
            model_adapter (ModelAdapterProtocol): Custom model adapter, especially
                used for X and Y normalization and denormalization.
        """
        self.model_adapter = model_adapter

    def predict(self,
            x: torch.Tensor,
            lagged_load_feature: int,
            ) -> torch.Tensor:
        """
        Upcoming load profile = load profile 7 days ago.

        Args:
            x (torch.Tensor): Normalised model input tensor of shape (batch_len, 
                sequence_len, features), where the feature at index `lagged_load_feature`
                contains the lagged load values.
            lagged_load_feature (int): The feature index in the input tensor
                that contains the lagged load to be used for prediction.

        Returns:
            torch.Tensor: Predicted y tensor of shape (batch_len, sequence_len, 1).
        """

        x = self.model_adapter.de_normalize_x(x)    # de-normalize all inputs

        # Take the chosen lagged loads as predictions
        #
        y_pred = x[:,:, lagged_load_feature]

        # Add axis and normalize y_pred again, to compare it to other models.
        #
        y_pred = y_pred[:,:,np.newaxis]
        y_pred = self.model_adapter.normalize_y(y_pred)
        assert y_pred.shape == (x.size(0), x.size(1), 1), \
            f"Shape mismatch: got {y_pred.shape}, expected ({x.size(0)}, {x.size(1)}, 1)"

        return y_pred

    def train_model(self) -> dict:
        """No training necessary for the persistence model."""

        history = {}
        history['loss'] = [0.0]

        return history

    def state_dict(self) -> dict:
        """No persistent parameter needed for this trivial model."""
        state_dict = {}
        return state_dict

    def load_state_dict(self, state_dict) -> None:
        """No persistent parameter needed for this trivial model."""
