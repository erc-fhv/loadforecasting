from typing import Optional, Callable, Sequence
import torch
from xlstm import (
    xLSTMBlockStack,
    xLSTMBlockStackConfig,
    mLSTMBlockConfig,
    mLSTMLayerConfig,
    sLSTMBlockConfig,
    sLSTMLayerConfig,
    FeedForwardConfig,
)
from .helpers import PytorchHelper, PositionalEncoding
from .normalizer import Normalizer


class xLstm(torch.nn.Module):
    """xLSTM configuration as provided by the xLSTM authors."""

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

        # The following xLSTM config variables are overtaken from the xLSTM authors
        conv1d_kernel_size=4
        num_heads=4
        qkv_proj_blocksize=4
        proj_factor=1.3
        num_blocks=7
        slstm_at=[1]

        # Finetune the XLSTM config variables based on model size
        if model_size == "0.1k":
            num_blocks=1
            num_heads=1
            d_model=1
            slstm_at=[0]
        elif  model_size == "0.2k":
            num_blocks=1
            num_heads=1
            d_model=1
            slstm_at=[0]
        elif model_size == "0.5k":
            num_blocks=1
            num_heads=2
            d_model=2
            slstm_at=[0]
        elif model_size == "1k":
            num_blocks=1
            num_heads=2
            d_model=4
            slstm_at=[0]
        elif model_size == "2k":
            num_blocks=1
            num_heads=4
            d_model=8
            slstm_at=[0]
        elif model_size == "5k":
            num_blocks=2
            num_heads=4
            d_model=8
            slstm_at=[1]
        elif model_size == "10k":
            num_blocks=2
            num_heads=4
            d_model=16
            slstm_at=[1]
        elif model_size == "20k":
            num_blocks=2
            num_heads=4
            d_model=32
            slstm_at=[1]
        elif model_size == "40k":
            num_blocks=4
            num_heads=4
            d_model=32
            slstm_at=[1]
        elif model_size == "80k":
            num_blocks=4
            num_heads=8
            d_model=40
            slstm_at=[1]
        else:
            assert False, f"Unimplemented model_size parameter given: {model_size}"

        # Configuration for the xLSTM Block
        self.cfg = xLSTMBlockStackConfig(
            mlstm_block=mLSTMBlockConfig(
                mlstm=mLSTMLayerConfig(
                    conv1d_kernel_size=conv1d_kernel_size, qkv_proj_blocksize=qkv_proj_blocksize, num_heads=num_heads
                )
            ),
            slstm_block=sLSTMBlockConfig(
                slstm=sLSTMLayerConfig(
                    backend="vanilla",  # For now run at CPU. Changed from "cuda".
                    num_heads=num_heads,
                    conv1d_kernel_size=conv1d_kernel_size,
                    bias_init="powerlaw_blockdependent",
                ),
                feedforward=FeedForwardConfig(proj_factor=proj_factor, act_fn="gelu"),
            ),
            context_length=256,
            num_blocks=num_blocks,
            embedding_dim=d_model,
            slstm_at=slstm_at,
        )
        self.xlstm_stack = xLSTMBlockStack(self.cfg)

        # Adding none-xlstm layers
        self.input_projection = torch.nn.Linear(num_of_features, d_model)
        self.positional_encoding = PositionalEncoding(d_model)
        self.output_layer = torch.nn.Linear(d_model, 1)

        # Setup Pytorch helper for training and evaluation
        self.my_pytorch_helper = PytorchHelper(self)

    def forward(self, x) -> torch.Tensor:
        """Model forward pass."""

        x = self.input_projection(x.float())
        x = self.positional_encoding(x)
        x = self.xlstm_stack(x)
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
