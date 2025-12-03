from typing import Optional, Callable, Sequence, Union
import numpy as np
import torch
from .helpers import PytorchHelper, PositionalEncoding, OptunaHelper
from .normalizer import Normalizer

# Define a type that can be either a torch Tensor or a numpy ndarray
ArrayLike = Union[torch.Tensor, np.ndarray]

class TransformerFull(torch.nn.Module):
    """
    Full pytorch transformer for timeseries prediction.
    """

    def __init__(
        self,
        model_size: str = '5k',
        loss_fn: Callable[..., torch.Tensor] = torch.nn.L1Loss(),
        normalizer: Optional[Normalizer] = None,
        lagged_load_feature: int = 11,
        ) -> None:
        """
        Args:
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
            lagged_load_feature (int): Index of the lagged load feature in the input tensor.
        """

        super().__init__()
        self.loss_fn = loss_fn
        self.normalizer = normalizer
        self.lagged_load_feature = lagged_load_feature
        self.create_model(model_size)

    def create_model(self, model_size) -> None:
        """ Create the Transformer model based on the specified model size. """

        # Configuration based on model size
        if model_size == "1k":
            num_heads=2
            num_layers=1
            dim_feedforward=6
            d_model=6
        elif model_size == "2k":
            num_heads=2
            num_layers=1
            dim_feedforward=10
            d_model=10
        elif model_size == "5k":
            num_heads=2
            num_layers=1
            dim_feedforward=16
            d_model=14
        elif model_size == "10k":
            num_heads=4
            num_layers=1
            dim_feedforward=90
            d_model=20
        elif model_size == "20k":
            num_heads=4
            num_layers=1
            dim_feedforward=200
            d_model=20
        elif model_size == "40k":
            num_heads=4
            num_layers=1
            dim_feedforward=400
            d_model=20
        elif model_size == "80k":
            num_heads=4
            num_layers=1
            dim_feedforward=400
            d_model=40
        else:
            raise ValueError(f"Unimplemented params.model_size parameter given: {model_size}")

        # Project input features to transformer dimension
        self.input_projection = torch.nn.LazyLinear(d_model)
        self.tgt_projection = torch.nn.Linear(1, d_model)
        self.positional_encoding = PositionalEncoding(d_model)

        # Transformer Encoder-Decoder
        self.transformer = torch.nn.Transformer(
            d_model=d_model,
            nhead=num_heads,
            num_encoder_layers=num_layers,
            num_decoder_layers=num_layers,
            dim_feedforward=dim_feedforward,
            batch_first=True
        )

        # Final output projection (to 1)
        self.output_layer = torch.nn.Linear(d_model, 1)

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

        # Take the latest available lagged loads as target-input
        x_tensor = x_tensor.float()
        tgt = x_tensor[:,:, self.lagged_load_feature].unsqueeze(-1)

        # Input and tgt projection
        x_tensor = self.input_projection(x_tensor)
        x_tensor = self.positional_encoding(x_tensor)
        tgt = self.tgt_projection(tgt)
        tgt = self.positional_encoding(tgt)

        # Run the full transformer model
        out = self.transformer(x_tensor, tgt)
        out = self.output_layer(out)

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
        n_splits: int = 5,
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
            n_splits (int): Number of expanding window splits for
                cross-validation.
            min_train_size (float): Minimum fraction of data to use for
                training in first split.
            pretrain_now (bool): Whether to run a pretraining phase.
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
            n_splits=n_splits,
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
