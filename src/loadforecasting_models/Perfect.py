import torch

class Perfect():
    """
    Trivial 'model': Just gets and returns the perfect profile (used for reference).
    """

    def __init__(self) -> None:
        pass

    def forward(self,
                y_real: torch.Tensor
                ) -> torch.Tensor:
        """Gets and return the perfect profile."""

        y_pred = y_real

        return y_pred

    def train_model(self) -> dict:
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

