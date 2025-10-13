import torch
from loadforecasting_models.interfaces import PersistenceModelInitParams, PersistencePredictionParams, ModelParams

class Persistence:
    """
    Predict the load accord to the load last week.
    """

    def __init__(self,
                 params: PersistenceModelInitParams
                 ) -> None:
        self.model_adapter = params.model_adapter

    def forward(self,
            params: PersistencePredictionParams
            ) -> torch.Tensor:
        """
        Upcoming load profile = load profile 7 days ago.
        """

        x = self.model_adapter.de_normalize_x(params.x)    # de-normalize all inputs

        # Take the chosen lagged loads as predictions
        #
        lagged_load_feature = 11
        y_pred = x[:,:, lagged_load_feature]

        # Add axis and normalize y_pred again, to compare it to other models.
        #
        y_pred = y_pred[:,:,np.newaxis]
        y_pred = self.model_adapter.normalize_y(y_pred)
        assert y_pred.shape == (x.size(0), 24, 1), \
            f"Shape mismatch: got {y_pred.shape}, expected ({x.size(0)}, 24, 1)"

        return y_pred

    def train_model(
        self,
        params: ModelParams
        ) -> dict:
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
