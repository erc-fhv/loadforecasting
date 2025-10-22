from typing import Callable, Literal
from sklearn.neighbors import KNeighborsRegressor
import torch
from loadforecasting_models.interfaces import ModelAdapterProtocol

class KNN():
    """
    KNN model for timeseries prediction.
    """

    def __init__(
        self,
        k: int,
        weights: Literal['uniform', 'distance'] | Callable = 'distance',
            model_adapter: ModelAdapterProtocol | None = None,
        ) -> None:
        """
        Args:
            k (int): Number of neighbors to use.
            weights: Weight function used in prediction. Possible 
                values: 'uniform', 'distance' or a callable distance function.
            model_adapter (ModelAdapterProtocol): Custom model adapter, especially
                used for X and Y normalization and denormalization.
        """

        self.knn = KNeighborsRegressor(n_neighbors = k, weights=weights)
        self.x_train = torch.Tensor([])
        self.y_train = torch.Tensor([])
        self.model_adapter = model_adapter

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """
        Given an input x, find the closest neighbors from the training data x_train
        and return the corresponding y_train.
        Args:
            x: Input features of shape (batch_len, sequence_len, features).
        Returns:
            torch.Tensor: Predicted y tensor of shape (batch_len, sequence_len, 1).
        """

        # Prediction on new hourly data
        #
        batches, timesteps, num_features = x.shape
        x_hourly = x.view(batches * timesteps, num_features).numpy()
        y_pred = self.knn.predict(x_hourly)
        y_pred = torch.tensor(y_pred).view(batches, timesteps, 1)

        return y_pred

    def train_model(
        self,
        x_train: torch.Tensor,
        y_train: torch.Tensor,
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

        self.x_train = x_train
        self.y_train = y_train
        self.knn_fit()
        history = {}
        history['loss'] = [0.0]

        return history

    def knn_fit(self) -> None:
        """Fit the model with hourly training data."""

        batches, timesteps, num_features = self.x_train.shape
        x_hourly = self.x_train.view(batches * timesteps, num_features).numpy()
        y_hourly = self.y_train.view(batches * timesteps, 1).numpy()
        self.knn.fit(x_hourly, y_hourly)

    def evaluate(
        self,
        x_test: torch.Tensor,
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

        output = self.predict(x_test)

        assert output.shape == y_test.shape, \
            f"Shape mismatch: got {output.shape}, expected {y_test.shape})"

        # Unnormalize the target variable, if wished.
        if de_normalize:
            assert self.model_adapter is not None, "No model_adapter given."
            y_test = self.model_adapter.de_normalize_y(y_test)
            output = self.model_adapter.de_normalize_y(output)

        # Compute Loss
        loss = eval_fn(output, y_test)
        results['test_loss'] = [loss.item()]
        reference = float(torch.mean(y_test))
        results['test_loss_relative'] = [100.0*loss.item()/reference]            
        results['predicted_profile'] = output

        return results

    def state_dict(self):
        """Store the persistent parameters of this model."""
        state_dict = {}
        state_dict['x_train'] = self.x_train
        state_dict['y_train'] = self.y_train
        return state_dict

    def load_state_dict(self, state_dict):
        """Load the persistent parameters of this model and re-trigger the KNN fitting."""
        self.x_train = state_dict['x_train']
        self.y_train = state_dict['y_train']

        self.knn_fit()
