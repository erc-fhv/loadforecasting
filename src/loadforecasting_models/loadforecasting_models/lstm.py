from typing import Optional, Callable, Sequence, Union
import numpy as np
import torch
from .helpers import PytorchHelper, OptunaHelper
from .normalizer import Normalizer

# Define a type that can be either a torch Tensor or a numpy ndarray
ArrayLike = Union[torch.Tensor, np.ndarray]

class Lstm(torch.nn.Module):
    """
    LSTM model for timeseries prediction.
    """

    def __init__(
        self,
        model_size: str = '5k',
        loss_fn: Callable[..., torch.Tensor] = torch.nn.L1Loss(),
        normalizer: Optional[Normalizer] = None,
        ) -> None:
        """
            model_size (str): The model parameter count, e.g. '0.1k',
                '0.2k', '0.5k', '1k', '2k', '5k', '10k', '20k', '40k',
                '80k'. Default is '5k', which is a good trade-off
                between performance and speed, see
                https://arxiv.org/abs/2501.05000.
            loss_fn (Callable[..., torch.Tensor]): Loss function to be
                used during training. E.g., torch.nn.L1Loss(),
                torch.nn.MSELoss(), pytorch_helpers.smape, ...
            normalizer (Normalizer): Used for X and Y normalization and
                denormalization.
        """

        super().__init__()
        self.loss_fn = loss_fn
        self.normalizer = normalizer
        self.create_model(model_size)

    def create_model(self, model_size) -> None:
        """ Create the Transformer model based on the specified model size. """

        if model_size == "0.1k":
            bidirectional=False
            d_model = 1
            hidden_dimension_dense1 = 4
            hidden_dimension_dense2 = 4
        elif  model_size == "0.2k":
            bidirectional=True
            d_model = 2
            hidden_dimension_dense1 = 4
            hidden_dimension_dense2 = 4
        elif model_size == "0.5k":
            bidirectional=True
            d_model = 3
            hidden_dimension_dense1 = 5
            hidden_dimension_dense2 = 5
        elif model_size == "1k":
            bidirectional=True
            d_model = 4
            hidden_dimension_dense1 = 10
            hidden_dimension_dense2 = 10
        elif model_size == "2k":
            bidirectional=True
            d_model = 6
            hidden_dimension_dense1 = 15
            hidden_dimension_dense2 = 10
        elif model_size == "5k":
            bidirectional=True
            d_model = 9
            hidden_dimension_dense1 = 30
            hidden_dimension_dense2 = 20
        elif model_size == "10k":
            bidirectional=True
            d_model = 14
            hidden_dimension_dense1 = 30
            hidden_dimension_dense2 = 20
        elif model_size == "20k":
            bidirectional=True
            d_model = 21
            hidden_dimension_dense1 = 30
            hidden_dimension_dense2 = 20
        elif model_size == "40k":
            bidirectional=True
            d_model = 30
            hidden_dimension_dense1 = 30
            hidden_dimension_dense2 = 20
        elif model_size == "80k":
            bidirectional=True
            d_model = 43
            hidden_dimension_dense1 = 30
            hidden_dimension_dense2 = 20
        else:
            raise ValueError(f"Unimplemented params.model_size parameter given: {model_size}")

        if bidirectional:
            bidirectional_factor = 2
        else:
            bidirectional_factor = 1

        self.input_projection = torch.nn.LazyLinear(d_model)
        self.lstm1 = torch.nn.LSTM(input_size=d_model, hidden_size=d_model,
                                   batch_first=True, bidirectional=bidirectional)
        self.lstm2 = torch.nn.LSTM(input_size=d_model*bidirectional_factor,
                                   hidden_size=d_model, batch_first=True,
                                   bidirectional=bidirectional)

        # Adding additional dense layers
        self.activation = torch.nn.ReLU()
        self.dense1 = torch.nn.Linear(d_model*bidirectional_factor,
                                      hidden_dimension_dense1)
        self.dense2 = torch.nn.Linear(hidden_dimension_dense1, hidden_dimension_dense2)
        self.output_layer = torch.nn.Linear(hidden_dimension_dense2, 1)

        # Setup Pytorch helper for training and evaluation
        self.my_pytorch_helper = PytorchHelper(self)

    def forward(
        self,
        x: ArrayLike,
        ) -> ArrayLike:
        """Model forward pass."""

        # Convert numpy to torch if needed
        input_was_numpy = isinstance(x, np.ndarray)
        if input_was_numpy:
            x_tensor  = torch.from_numpy(x).float()
        else:
            x_tensor  = x.float()

        # Forward pass through LSTM and dense layers
        x_tensor = self.input_projection(x_tensor)
        x_tensor, _ = self.lstm1(x_tensor)
        x_tensor, _ = self.lstm2(x_tensor)
        x_tensor = self.activation(self.dense1(x_tensor))
        x_tensor = self.activation(self.dense2(x_tensor))
        out = self.output_layer(x_tensor)

        # Convert back to numpy if needed
        if input_was_numpy:
            out = out.numpy()

        return out

    def train_model(
        self,
        x_train: ArrayLike,
        y_train: ArrayLike,
        x_dev: Optional[ArrayLike] = None,
        y_dev: Optional[ArrayLike] = None,
        pretrain_now: bool = False,
        finetune_now: bool = False,
        epochs: int = 100,
        learning_rates: Optional[Sequence[float]] = None,
        batch_size: int = 256,
        verbose: int = 1,
        ) -> dict:
        """
        Train this model.
        Args:
            X_train (ArrayLike): Training input features of shape (batch_len, sequence_len, 
                features).
            Y_train (ArrayLike): Training labels of shape (batch_len, sequence_len, 1).
            X_dev (ArrayLike, optional): Validation input features of shape (batch_len, 
                sequence_len, features).
            Y_dev (ArrayLike, optional): Validation labels of shape (batch_len, 
                sequence_len, 1).
            pretrain_now (bool): Whether to run a pretraining phase.
            finetune_now (bool): Whether to run fine-tuning.
            epochs (int): Number of training epochs.
            learning_rates (Sequence[float], optional): Learning rates schedule.
            batch_size (int): Batch size for training.
            verbose (int): Verbosity level. 0: silent, 1: dots, 2: full.
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
            x_train=x_train,
            y_train=y_train,
            x_dev=x_dev,
            y_dev=y_dev,
            pretrain_now=pretrain_now,
            finetune_now=finetune_now,
            epochs=epochs,
            learning_rates=learning_rates,
            batch_size=batch_size,
            verbose=verbose,
            )

        return history

    def train_model_auto(
        self,
        x_train: ArrayLike,
        y_train: ArrayLike,
        n_trials: int = 50,
        k_folds: int = 1,
        val_ratio: float = 0.2,
        verbose: int = 1,
        ) -> dict:
        """
        Train this model with automatic hyperparameter optimization using
        Optuna and expanding window cross-validation.

        Args:
            x_train (ArrayLike): Training input features of
                shape (batch_len, sequence_len, features).
            y_train (ArrayLike): Training labels of
                shape (batch_len, sequence_len, 1).
            n_trials (int): Number of Optuna trials for hyperparameter
                search.
            k_folds (int): Number of folds for the timeseries cross-validation. If set to 1,
                this is the same as a static train-dev-split.
            val_ratio (float, optional): Proportion of data for validation
                compared to the total training data.
            verbose (int): Verbosity level. 0: silent, 1: dots, 2: full.
                optuna_study_name (str, optional): Name for the Optuna study.

        Returns:
            dict: Training history and best hyperparameters.
        """

        optuna_helper = OptunaHelper(self)
        history = optuna_helper.train_auto(
            x_train=x_train,
            y_train=y_train,
            n_trials=n_trials,
            k_folds=k_folds,
            val_ratio=val_ratio,
            verbose=verbose,
            )

        return history

    def predict(
        self,
        x: ArrayLike,
        ) -> ArrayLike:
        """
        Predict y from the given x.

        Args:
            x (ArrayLike): Input tensor of shape (batch_len, sequence_len, features) 
                containing the features for which predictions are to be made.

        Returns:
            ArrayLike: Predicted y tensor of shape (batch_len, sequence_len, 1).
        """

        self.eval()
        with torch.no_grad():
            y = self(x)

        return y

    def evaluate(
        self,
        x_test: ArrayLike,
        y_test: ArrayLike,
        results: Optional[dict] = None,
        de_normalize: bool = False,
        loss_relative_to: str = "mean",
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
            loss_relative_to,
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
