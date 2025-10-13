import torch
from loadforecasting_models.interfaces import ModelParams, PerfectPredictionParams

class Perfect():
    """
    Trivial 'model': Just gets and returns the perfect profile (used for reference).
    """

    def __init__(self,
                 params: ModelParams
                 ) -> None:
        pass

    def forward(self,
                params: PerfectPredictionParams
                ) -> torch.Tensor:
        """Gets and return the perfect profile."""

        y_pred = params.y_real

        return y_pred

    def train_model(
        self,
        params: ModelParams
        ) -> dict:
        """No training necessary for the perfect model."""

        history = {}
        history['loss'] = [0.0]

        return history

    def state_dict(self) -> dict:
        """No persistent parameter needed for this trivial model."""
        state_dict = {}
        return state_dict

    def load_state_dict(self, state_dict) -> None:
        """No persistent parameter needed for this trivial model."""

