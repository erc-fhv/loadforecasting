from typing import Optional, Callable, Sequence
import torch
from loadforecasting_models.Helpers import PytorchHelper, PositionalEncoding
from loadforecasting_models.interfaces import ModelAdapterProtocol


class TransformerFull(torch.nn.Module):
    """
    Full pytorch transformer for timeseries prediction.
    """    

    def __init__(
        self,
        model_size: int,
        num_of_features: int,
        loss_fn: Optional[Callable[..., torch.Tensor]] = torch.nn.L1Loss(),
        model_adapter: Optional[ModelAdapterProtocol] = None,
        ) -> None:
        """
        Args:
            model_size (str): The model parameter count, e.g. '0.1k', '0.2k', '0.5k', '1k',
                '2k', '5k', '10k', '20k', '40k', '80k'.
            num_of_features (int): Number of model input features.
            loss_fn (Callable[..., torch.Tensor]): Loss function to be used during 
                training. E.g., torch.nn.L1Loss(), torch.nn.MSELoss(), pytorch_helpers.smape, ...
            model_adapter (ModelAdapterProtocol, optional): Custom model adapter, especially
                used for X and Y normalization and denormalization.    
        """

        super().__init__()

        self.loss_fn = loss_fn
        self.model_adapter = model_adapter

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
            assert False, f"Unimplemented model_size parameter given: {model_size}"

        # Project input features to transformer dimension
        self.input_projection = torch.nn.Linear(num_of_features, d_model)
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

    def forward(self, x) -> torch.Tensor:
        """Model forward pass."""

        # Take the latest available lagged loads as target-input
        lagged_load_feature = 11
        x = x.float()
        tgt = x[:,:, lagged_load_feature].unsqueeze(-1)

        # Input and tgt projection
        x = self.input_projection(x)
        x = self.positional_encoding(x)
        tgt = self.tgt_projection(tgt)
        tgt = self.positional_encoding(tgt)

        # Run the full transformer model
        out = self.transformer(x, tgt)
        out = self.output_layer(out)

        return out

    def train_model(
        self,
        x_train: torch.Tensor,
        y_train: torch.Tensor,
        x_dev: Optional[torch.Tensor] = None,
        y_dev: Optional[torch.Tensor] = None,
        pretrain_now: bool = False,
        finetune_now: bool = True,
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

        self.my_model.eval()
        with torch.no_grad():
            y = self.my_model(x)

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
