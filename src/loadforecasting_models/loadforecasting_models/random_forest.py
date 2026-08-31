from typing import Callable, Optional, Sequence, Union

import numpy as np
import torch
from sklearn.ensemble import RandomForestRegressor as SklearnRandomForest

from .helpers import SklearnOptunaHelper
from .normalizer import Normalizer

ArrayLike = Union[torch.Tensor, np.ndarray]


class RandomForest:
    """
    Random Forest model for timeseries prediction.
    """

    def __init__(
        self,
        n_estimators: int = 100,
        max_depth: Optional[int] = None,
        min_samples_leaf: int = 1,
        n_jobs: int = -1,
        random_state: Optional[int] = None,
        normalizer: Optional[Normalizer] = None,
        loss_relative_to: str = "",
        ) -> None:
        """
        Args:
            n_estimators:
                Number of trees in the forest. Default: 100.
            max_depth:
                Maximum depth of each tree. None means nodes are expanded until all
                leaves are pure or contain fewer than min_samples_leaf samples.
            min_samples_leaf:
                Minimum number of samples required at a leaf node. Default: 1.
            n_jobs:
                Number of parallel jobs for training and prediction.
                -1 uses all available CPU cores. Default: -1.
            random_state:
                Seed for reproducibility. Default: None.
            normalizer:
                Used for X and Y normalization / denormalization.
            loss_relative_to:
                String indicating the reference value for relative loss calculation.
        """
        self.normalizer = normalizer
        self.loss_relative_to = loss_relative_to
        self.model = SklearnRandomForest(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_leaf=min_samples_leaf,
            n_jobs=n_jobs,
            random_state=random_state,
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

    def train_model_auto(
        self,
        x_train: ArrayLike,
        y_train: ArrayLike,
        n_trials: int = 50,
        k_folds: int = 3,
        feature_index_groups: Optional[Sequence[Sequence[int]]] = None,
        verbose: int = 1,
        ) -> dict:
        """
        Tune this model's hyperparameters (n_estimators, max_depth, min_samples_leaf),
        optionally including which feature groups to use, with Optuna and
        TimeSeriesSplit cross-validation, then refit on the full training data with
        the best settings found.

        Args:
            x_train: Input features of shape (batch_len, sequence_len, features).
            y_train: Target values of shape (batch_len, sequence_len, 1).
            n_trials (int): Number of Optuna trials for hyperparameter search.
            k_folds (int): Number of TimeSeriesSplit folds used for cross-validation.
            feature_index_groups: Optional list of column-index groups (one group per
                named feature) to choose from during tuning.
            verbose (int): Verbosity level. 0: silent, 1: dots, 2: full.

        Returns:
            dict: Training history and best hyperparameters.
        """

        tuner = SklearnOptunaHelper(self)
        return tuner.train_auto(
            x_train=x_train,
            y_train=y_train,
            n_trials=n_trials,
            k_folds=k_folds,
            feature_index_groups=feature_index_groups,
            verbose=verbose,
            )

    @staticmethod
    def suggest_params(trial) -> dict:
        """Optuna search space for this model's hyperparameters."""
        return {
            'n_estimators': trial.suggest_int('n_estimators', 50, 500, log=True),
            'max_depth': trial.suggest_categorical(
                'max_depth', [None, 3, 5, 8, 12, 16, 24, 32]),
            'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 20, log=True),
        }

    def evaluate(
        self,
        x_test: ArrayLike,
        y_test: ArrayLike,
        results: Optional[dict] = None,
        de_normalize: bool = False,
        eval_fn: Callable[..., torch.Tensor] = torch.nn.L1Loss(),
        loss_relative_to: str = "",
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

        # Set reference for relative loss if 'loss_relative_to' is an empty string.
        if loss_relative_to == "":
            if self.loss_relative_to != "":
                loss_relative_to = self.loss_relative_to
            else:
                loss_relative_to = "mean"

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
