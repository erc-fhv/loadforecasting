from typing import Callable, Literal, Union
from sklearn.neighbors import KNeighborsRegressor
import numpy as np
import torch
from .normalizer import Normalizer

# Define a type that can be either a torch Tensor or a numpy ndarray
ArrayLike = Union[torch.Tensor, np.ndarray]

class Gam():
    """
    GAM (Generalized Additive Model) model for timeseries prediction.
    """

    def __init__(
        self,
        k: int,
        weights: Union[Literal['uniform', 'distance'], Callable] = 'distance',
        normalizer: Union[Normalizer, None] = None,
        ) -> None:
        """
        Args:
            k (int): Number of neighbors to use.
            weights: Weight function used in prediction. Possible values: 'uniform',
                'distance' or a callable distance function.
            normalizer (Normalizer): Used for X and Y normalization and denormalization.
        """

        self.normalizer = normalizer
        all_gam_terms = (
            # s(features.index("time_of_day"), basis="cp", n_splines=24) +
            # s(features.index("temperature")) +
            # s(features.index("irradiance")) +
            s(features.index("lagged_power_7d")) +
            # s(features.index("wind_speed_10m")) +
            f(features.index("day_of_week")) +
            # f(features.index("is_holiday")) +
            # s(features.index("day_of_year"), basis="cp"),
            te(
                s(features.index("time_of_day"), n_splines=5),
                s(features.index("temperature"), n_splines=5),
            ) +
            te(
                s(features.index("time_of_day"), n_splines=10, lam=0.3),
                s(features.index("irradiance"), n_splines=10, lam=0.15),
            ) +
            te(
                s(features.index("temperature"), n_splines=10, lam=0.3),
                s(features.index("wind_speed_10m"), n_splines=10, lam=0.15),
            )
            # s(features.index("direct_normal_irradiance")),
            # s(features.index("diffuse_radiation")),
            # s(features.index("direct_radiation")),
            # s(features.index("shortwave_radiation")),
            # s(features.index("nr_of_households")),
            # s(features.index("pv_installed")),
            # s(features.index("wind_chill")),
            # s(features.index("heating_degree")),
            # s(features.index("cooling_degree")),

            self.gam = LinearGAM(
                all_gam_terms,
                lam=0.5,
                fit_intercept=True
            )

    def predict(
        self, x: ArrayLike,
        ) -> ArrayLike:
        """
        Given an input x, find the closest neighbors from the training data x_train
        and return the corresponding y_train.
        Args:
            x: Input features of shape (batch_len, sequence_len, features).
        Returns:
            ArrayLike: Predicted y tensor of shape (batch_len, sequence_len, 1).
        """

        # Convert numpy to torch if needed
        input_was_numpy = isinstance(x, np.ndarray)
        if input_was_numpy:
            x_tensor  = torch.from_numpy(x).float()
        else:
            x_tensor  = x.float()

        # Prediction on new data
        y_pred = self.gam.predict(x_tensor)

        # Convert back to numpy if needed
        if input_was_numpy:
            y_pred = y_pred.numpy()

        return y_pred

    def train_model(
        self,
        x_train: ArrayLike,
        y_train: ArrayLike,
        ) -> dict:
        """
        Train this model.
        Args:
            X_train (torch.Tensor): Training input features of shape (batch_len, sequence_len,
                features).
            Y_train (torch.Tensor): Training labels of shape (batch_len, sequence_len, 1).
        Returns:
            dict: Training history containing loss values.
        """

        # Convert numpy to torch if needed
        if isinstance(x_train, np.ndarray):
            x_train  = torch.from_numpy(x_train).float()
        if isinstance(y_train, np.ndarray):
            y_train  = torch.from_numpy(y_train).float()

        self.gam.fit(x_train, y_train)

        history = {}
        history['loss'] = [0.0]

        return history

    def evaluate(
        self,
        x_test: ArrayLike,
        y_test: ArrayLike,
        results: Union[dict, None] = None,
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

        # Get model output
        output = self.predict(x_tensor)

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
        results['test_loss_relative'] = [100.0 * loss.item() / reference]
        results['predicted_profile'] = output

        return results

    def state_dict(self):
        """Store the persistent parameters of this model."""
        state_dict = {}
        state_dict['gam'] = self.gam
        return state_dict

    def load_state_dict(self, state_dict):
        """Load the persistent parameters of this model and re-trigger the KNN fitting."""
        self.gam = state_dict['gam']
