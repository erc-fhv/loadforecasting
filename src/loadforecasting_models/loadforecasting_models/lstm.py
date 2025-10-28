from typing import Optional, Callable, Sequence
import torch
from .helpers import PytorchHelper
from .normalizer import Normalizer

class Lstm(torch.nn.Module):
    """
    LSTM model for timeseries prediction.
    """

    def __init__(
        self,
        model_size: str,
        num_of_features: int,
        loss_fn: Optional[Callable[..., torch.Tensor]] = torch.nn.L1Loss(),
        normalizer: Optional[Normalizer] = None,
        ) -> None:
        """
        Args:
            model_size (str): The model parameter count, e.g. '0.1k', '0.2k', '0.5k', '1k',
                '2k', '5k', '10k', '20k', '40k', '80k'.
            num_of_features (int): Number of model input features.
            loss_fn (Callable[..., torch.Tensor]): Loss function to be used during 
                training. E.g., torch.nn.L1Loss(), torch.nn.MSELoss(), pytorch_helpers.smape, ...
            normalizer (Normalizer): Used for X and Y normalization and denormalization.
        """

        super().__init__()

        self.loss_fn = loss_fn
        self.normalizer = normalizer

        # LSTM configuration based on model size
        if model_size == "0.1k":
            bidirectional=False
            hidden_dimension_lstm1 = 1
            hidden_dimension_lstm2 = 1
            hidden_dimension_dense1 = 4
            hidden_dimension_dense2 = 4
        elif  model_size == "0.2k":
            bidirectional=True
            hidden_dimension_lstm1 = 1
            hidden_dimension_lstm2 = 1
            hidden_dimension_dense1 = 4
            hidden_dimension_dense2 = 4
        elif model_size == "0.5k":
            bidirectional=True
            hidden_dimension_lstm1 = 2
            hidden_dimension_lstm2 = 2
            hidden_dimension_dense1 = 5
            hidden_dimension_dense2 = 5
        elif model_size == "1k":
            bidirectional=True
            hidden_dimension_lstm1 = 3
            hidden_dimension_lstm2 = 3
            hidden_dimension_dense1 = 10
            hidden_dimension_dense2 = 10
        elif model_size == "2k":
            bidirectional=True
            hidden_dimension_lstm1 = 5
            hidden_dimension_lstm2 = 5
            hidden_dimension_dense1 = 15
            hidden_dimension_dense2 = 10
        elif model_size == "5k":
            bidirectional=True
            hidden_dimension_lstm1 = 8
            hidden_dimension_lstm2 = 9
            hidden_dimension_dense1 = 30
            hidden_dimension_dense2 = 20
        elif model_size == "10k":
            bidirectional=True
            hidden_dimension_lstm1 = 10
            hidden_dimension_lstm2 = 18
            hidden_dimension_dense1 = 30
            hidden_dimension_dense2 = 20
        elif model_size == "20k":
            bidirectional=True
            hidden_dimension_lstm1 = 22
            hidden_dimension_lstm2 = 20
            hidden_dimension_dense1 = 30
            hidden_dimension_dense2 = 20
        elif model_size == "40k":
            bidirectional=True
            hidden_dimension_lstm1 = 42
            hidden_dimension_lstm2 = 20
            hidden_dimension_dense1 = 30
            hidden_dimension_dense2 = 20
        elif model_size == "80k":
            bidirectional=True
            hidden_dimension_lstm1 = 70
            hidden_dimension_lstm2 = 21
            hidden_dimension_dense1 = 30
            hidden_dimension_dense2 = 20
        else:
            assert False, f"Unimplemented params.model_size parameter given: {model_size}"

        if bidirectional:
            bidirectional_factor = 2
        else:
            bidirectional_factor = 1

        self.lstm1 = torch.nn.LSTM(input_size=num_of_features, hidden_size=hidden_dimension_lstm1,
                                   batch_first=True, bidirectional=bidirectional)
        self.lstm2 = torch.nn.LSTM(input_size=hidden_dimension_lstm1*bidirectional_factor,
                                   hidden_size=hidden_dimension_lstm2, batch_first=True,
                                   bidirectional=bidirectional)

        # Adding additional dense layers
        self.activation = torch.nn.ReLU()
        self.dense1 = torch.nn.Linear(hidden_dimension_lstm2*bidirectional_factor,
                                      hidden_dimension_dense1)
        self.dense2 = torch.nn.Linear(hidden_dimension_dense1, hidden_dimension_dense2)
        self.output_layer = torch.nn.Linear(hidden_dimension_dense2, 1)

        # Setup Pytorch helper for training and evaluation
        self.my_pytorch_helper = PytorchHelper(self)

    def forward(self, x) -> torch.Tensor:
        """Model forward pass."""

        x, _ = self.lstm1(x.float())
        x, _ = self.lstm2(x)
        x = self.activation(self.dense1(x))
        x = self.activation(self.dense2(x))
        x = self.output_layer(x)

        return x

    def train_model(
        self,
        x_train: torch.Tensor,
        y_train: torch.Tensor,
        x_dev: Optional[torch.Tensor] = None,
        y_dev: Optional[torch.Tensor] = None,
        pretrain_now: bool = False,
        finetune_now: bool = False,
        epochs: int = 100,
        learning_rates: Optional[Sequence[float]] = None,
        batch_size: int = 256,
        verbose: int = 0,
        ) -> dict:
        """
        Train this model.
        Args:
            X_train (torch.Tensor): Training input features of shape (batch_len, sequence_len, 
                features).
            Y_train (torch.Tensor): Training labels of shape (batch_len, sequence_len, 1).
            X_dev (torch.Tensor, optional): Validation input features of shape (batch_len, 
                sequence_len, features).
            Y_dev (torch.Tensor, optional): Validation labels of shape (batch_len, 
                sequence_len, 1).
            pretrain_now (bool): Whether to run a pretraining phase.
            finetune_now (bool): Whether to run fine-tuning.
            epochs (int): Number of training epochs.
            learning_rates (Sequence[float], optional): Learning rates schedule.
            batch_size (int): Batch size for training.
            verbose (int): Verbosity level.
        Returns:
            dict: Training history containing loss values.
        """

        if x_dev is None:
            x_dev = torch.Tensor([])
        if y_dev is None:
            y_dev = torch.Tensor([])
        if learning_rates is None:
            learning_rates = [0.01, 0.005, 0.001, 0.0005]

        history = self.my_pytorch_helper.train(
            x_train,
            y_train,
            x_dev,
            y_dev,
            pretrain_now,
            finetune_now,
            epochs,
            learning_rates,
            batch_size, verbose,
            )

        return history

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """
        Predict y from the given x.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_len, sequence_len, features) 
                containing the features for which predictions are to be made.

        Returns:
            torch.Tensor: Predicted y tensor of shape (batch_len, sequence_len, 1).
        """

        self.eval()
        with torch.no_grad():
            y = self(x)

        return y

    def evaluate(
        self,
        x_test: torch.Tensor,
        y_test: torch.Tensor,
        results: Optional[dict] = None,
        de_normalize: bool = False,
        ) -> dict:
        """
        Evaluate the model on the given x_test and y_test.
        """

        if results is None:
            results = {}

        results = self.my_pytorch_helper.evaluate(
            x_test,
            y_test,
            results,
            de_normalize,
            )

        return results

    def get_nr_of_parameters(self, do_print=True):
        """
        Return and optionally print the number of parameters of this owned model
        """

        total_params = sum(p.numel() for p in self.parameters())

        if do_print:
            print(f"Total number of parameters: {total_params}")

        return total_params
