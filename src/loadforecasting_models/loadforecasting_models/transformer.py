from typing import Optional, Callable, Sequence
import torch
from .helpers import PytorchHelper, PositionalEncoding
from .normalizer import Normalizer


class Transformer(torch.nn.Module):
    """
    Encoder-only Transformer inspired by "A Time Series is
    Worth 64 Words" (https://arxiv.org/abs/2211.14730)
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

        # Configuration based on model size
        if model_size == "0.1k":
            num_layers=1
            num_heads=2
            dim_feedforward=5
            d_model = 2
        elif model_size == "0.2k":
            num_layers=1
            num_heads=2
            dim_feedforward=5
            d_model=4
        elif model_size == "0.5k":
            num_layers=1
            num_heads=2
            dim_feedforward=6
            d_model=6
        elif model_size == "1k":
            num_layers=1
            num_heads=2
            dim_feedforward=10
            d_model=10
        elif model_size == "2k":
            num_layers=1
            num_heads=2
            dim_feedforward=16
            d_model=14
        elif model_size == "5k":
            num_layers=1
            num_heads=4
            dim_feedforward=90
            d_model=20
        elif model_size == "10k":
            num_layers=1
            num_heads=4
            dim_feedforward=200
            d_model=20
        elif model_size == "20k":
            num_layers=1
            num_heads=4
            dim_feedforward=400
            d_model=20
        elif model_size == "40k":
            num_layers=1
            num_heads=4
            dim_feedforward=400
            d_model=40
        elif model_size == "80k":
            num_layers=2
            num_heads=8
            dim_feedforward=400
            d_model=40
        else:
            assert False, f"Unimplemented model_size parameter given: {model_size}"

        # Transformer Encoder Layers
        self.input_projection = torch.nn.Linear(num_of_features, d_model)
        self.positional_encoding = PositionalEncoding(d_model)
        encoder_layer = torch.nn.TransformerEncoderLayer(d_model=d_model, nhead=num_heads,
            dim_feedforward=dim_feedforward, batch_first=True)
        self.transformer = torch.nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.output_layer = torch.nn.Linear(d_model, 1)

        # Setup Pytorch helper for training and evaluation
        self.my_pytorch_helper = PytorchHelper(self)

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

    def forward(self, x) -> torch.Tensor:
        """Model forward pass."""

        x = self.input_projection(x.float())
        x = self.positional_encoding(x)
        x = self.transformer(x)
        x = self.output_layer(x)

        return x

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
