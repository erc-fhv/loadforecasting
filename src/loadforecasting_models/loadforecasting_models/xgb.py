from typing import Callable, Optional, Union

import numpy as np
import torch
from xgboost import XGBRegressor

from .normalizer import Normalizer

ArrayLike = Union[torch.Tensor, np.ndarray]


class XGBoost:
    """
    XGBoost model for timeseries prediction.

    Requires the ``xgboost`` package: ``pip install xgboost``.
    """

    def __init__(
        self,
        n_estimators: int = 100,
        max_depth: int = 6,
        learning_rate: float = 0.1,
        subsample: float = 1.0,
        n_jobs: int = -1,
        random_state: Optional[int] = None,
        normalizer: Optional[Normalizer] = None,
    ) -> None:
        """
        Args:
            n_estimators:
                Number of boosting rounds. Default: 100.
            max_depth:
                Maximum depth of each tree. Default: 6.
            learning_rate:
                Step size shrinkage used to prevent overfitting. Default: 0.1.
            subsample:
                Fraction of training samples used per boosting round.
                Values in (0, 1] reduce overfitting. Default: 1.0.
            n_jobs:
                Number of parallel threads. -1 uses all available cores. Default: -1.
            random_state:
                Seed for reproducibility. Default: None.
            normalizer:
                Used for X and Y normalization / denormalization.
        """
        self.normalizer = normalizer
        self.model = XGBRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            subsample=subsample,
            n_jobs=n_jobs,
            random_state=random_state,
            verbosity=0,
        )
        self.x_train: torch.Tensor = torch.Tensor([])
        self.y_train: torch.Tensor = torch.Tensor([])

    def predict(self, x: ArrayLike) -> ArrayLike:
        """
        Given an input x, return the predicted y.

        Args:
            x: Input features of shape (batch_len, sequence_len, features).

        Returns:
            Predicted y tensor of shape (batch_len, sequence_len, 1).
        """
        input_was_numpy = isinstance(x, np.ndarray)
        if input_was_numpy:
            x_tensor = torch.from_numpy(x).float()
        else:
            x_tensor = x.float()

        output_shape = (x_tensor.shape[0], x_tensor.shape[1], 1)
        x_flat = x_tensor.reshape(-1, x_tensor.shape[2]).numpy()

        y_pred = self.model.predict(x_flat)
        y_pred = torch.tensor(y_pred, dtype=torch.float32).reshape(output_shape)

        if input_was_numpy:
            return y_pred.numpy()
        return y_pred

    def train_model(self, x_train: ArrayLike, y_train: ArrayLike) -> dict:
        """
        Train this model.

        Args:
            x_train: Input features of shape (batch_len, sequence_len, features).
            y_train: Target values of shape (batch_len, sequence_len, 1).

        Returns:
            dict: Training history containing loss values.
        """
        if isinstance(x_train, np.ndarray):
            x_train = torch.from_numpy(x_train).float()
        if isinstance(y_train, np.ndarray):
            y_train = torch.from_numpy(y_train.copy()).float()

        if x_train.ndim == 3:
            self.x_train = x_train.reshape(-1, x_train.shape[2])
        elif x_train.ndim == 2:
            self.x_train = x_train
        else:
            raise ValueError(f"Unexpected number of dimensions for x_train: {x_train.ndim}")

        if y_train.ndim in [2, 3]:
            self.y_train = y_train.flatten()
        elif y_train.ndim == 1:
            self.y_train = y_train
        else:
            raise ValueError(f"Unexpected number of dimensions for y_train: {y_train.ndim}")

        self.model.fit(self.x_train.numpy(), self.y_train.numpy())

        history = {}
        history['loss'] = self.evaluate(x_train, y_train)['test_loss']
        return history

    def evaluate(
        self,
        x_test: ArrayLike,
        y_test: ArrayLike,
        results: Optional[dict] = None,
        de_normalize: bool = False,
        eval_fn: Callable[..., torch.Tensor] = torch.nn.L1Loss(),
        loss_relative_to: str = "range",
    ) -> dict:
        """
        Evaluate the model on the given test data.
        """
        if results is None:
            results = {}

        if isinstance(x_test, np.ndarray):
            x_tensor = torch.from_numpy(x_test).float()
        else:
            x_tensor = x_test.float()
        if isinstance(y_test, np.ndarray):
            y_tensor = torch.from_numpy(y_test).float()
        else:
            y_tensor = y_test.float()

        output = self.predict(x_tensor)

        assert output.shape == y_tensor.shape, (
            f"Shape mismatch: got {output.shape}, expected {y_tensor.shape}"
        )

        if de_normalize:
            assert self.normalizer is not None, "No normalizer given."
            y_tensor = self.normalizer.de_normalize_y(y_tensor)
            output = self.normalizer.de_normalize_y(output)
            assert isinstance(y_tensor, torch.Tensor), "Denormalized y_tensor is not a torch.Tensor"

        if loss_relative_to == "mean":
            reference = float(torch.abs(torch.mean(y_tensor)))
        elif loss_relative_to == "max":
            reference = float(torch.abs(torch.max(y_tensor)))
        elif loss_relative_to == "range":
            reference = float(torch.max(y_tensor) - torch.min(y_tensor))
        else:
            raise ValueError(f"Unexpected parameter: loss_relative_to = {loss_relative_to}")

        output = torch.Tensor(output)
        loss = eval_fn(output, y_tensor)
        results['test_loss'] = [loss.item()]
        results['test_loss_relative'] = [100.0 * loss.item() / reference]
        results['predicted_profile'] = output
        return results

    def state_dict(self) -> dict:
        """Store the persistent parameters of this model."""
        return {'x_train': self.x_train, 'y_train': self.y_train}

    def load_state_dict(self, state_dict: dict) -> None:
        """Load the persistent parameters of this model and re-trigger the fitting."""
        self.x_train = state_dict['x_train']
        self.y_train = state_dict['y_train']
        self.model.fit(self.x_train.numpy(), self.y_train.numpy())
