from sklearn.neighbors import KNeighborsRegressor
import torch
from loadforecasting_models.interfaces import KnnModelInitParams, KnnTrainParams

class KNN():

    def __init__(
        self,
        params: KnnModelInitParams
        ) -> None:

        self.knn = KNeighborsRegressor(n_neighbors = params.k, weights=params.weights)
        self.x_train = torch.Tensor([])
        self.y_train = torch.Tensor([])

    def forward(self, x):
        """
        Given an input x, find the closest neighbors from the training data x_train
        and return the corresponding y_train.
        """

        # Prediction on new hourly data
        #
        batches, timesteps, num_features = x.shape
        x_hourly = x.view(batches * timesteps, num_features)
        y_pred = self.knn.predict(x_hourly)
        y_pred = torch.tensor(y_pred).view(batches, timesteps, 1)

        return y_pred

    def train_model(
        self,
        params: KnnTrainParams
        ) -> dict:
        """For our KNN model the training means just to store the training data."""

        self.x_train = params.x_train
        self.y_train = params.y_train
        self.knn_fit()
        history = {}
        history['loss'] = [0.0]

        return history

    def knn_fit(self) -> None:
        """Fit the model with hourly training data."""

        batches, timesteps, num_features = self.x_train.shape
        x_hourly = self.x_train.view(batches * timesteps, num_features)
        y_hourly = self.y_train.view(batches * timesteps, 1)
        self.knn.fit(x_hourly, y_hourly)

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
