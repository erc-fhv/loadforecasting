from typing import Optional, Callable
import torch
from loadforecasting_models import Normalizer

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

    def evaluate(
        self,
        y_test: torch.Tensor,
        results: dict | None = None,
        de_normalize: bool = False,
        eval_fn: Callable[..., torch.Tensor] = torch.nn.L1Loss(),
        ) -> dict:
        """
        Evaluate the model on the given x_test and y_test.
        """

        if results is None:
            results = {}

        output = self.predict(y_test)   # pass Y to get perfect prediction

        assert output.shape == y_test.shape, \
            f"Shape mismatch: got {output.shape}, expected {y_test.shape})"

        # Unnormalize the target variable, if wished.
        if de_normalize:
            assert self.normalizer is not None, "No normalizer given."
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
