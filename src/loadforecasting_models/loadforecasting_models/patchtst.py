from typing import Optional, Callable, Sequence, Union
import numpy as np
import torch
from .helpers import PytorchHelper, PositionalEncoding, OptunaHelper
from .normalizer import Normalizer

# Define a type that can be either a torch Tensor or a numpy ndarray
ArrayLike = Union[torch.Tensor, np.ndarray]

class PatchTST(torch.nn.Module):
    """
    Patch-based encoder-only Transformer, following "A Time Series is
    Worth 64 Words: Long-term Forecasting with Transformers"
    (https://arxiv.org/abs/2211.14730). Unlike Transformer, which tokenizes
    every single timestep, the input sequence is first split into
    non-overlapping patches along time, so each attended token represents a
    short time window - fewer, richer tokens for the same context length.
    """

    def __init__(
        self,
        model_size: str = '5k',
        loss_fn: Callable[..., torch.Tensor] = torch.nn.L1Loss(),
        normalizer: Optional[Normalizer] = None,
        patch_len: int = 4,
        loss_relative_to: str = "",
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
            patch_len (int): Number of timesteps per patch/token. Default
                4, e.g. 1h patches on 15min-resolution data.
            loss_relative_to (str): Reference for relative loss calculation. Default: "".
        """

        super().__init__()
        self.loss_fn = loss_fn
        self.normalizer = normalizer
        self.loss_relative_to = loss_relative_to
        self.patch_len = patch_len
        self.create_model(model_size)

    def create_model(self, model_size) -> None:
        """ Create the PatchTST model based on the specified model size. """

        if model_size == "0.1k":
            num_layers=1
            num_heads=1
            dim_feedforward=2
            d_model = 1
        elif model_size == "0.2k":
            num_layers=1
            num_heads=2
            dim_feedforward=2
            d_model=2
        elif model_size == "0.5k":
            num_layers=1
            num_heads=2
            dim_feedforward=6
            d_model=4
        elif model_size == "1k":
            num_layers=1
            num_heads=2
            dim_feedforward=22
            d_model=6
        elif model_size == "2k":
            num_layers=1
            num_heads=2
            dim_feedforward=31
            d_model=10
        elif model_size == "5k":
            num_layers=1
            num_heads=4
            dim_feedforward=37
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
            raise ValueError(f"Unimplemented params.model_size parameter given: {model_size}")

        # Patch embedding (a flattened patch of patch_len timesteps -> one token) +
        # Transformer Encoder Layers
        self.patch_embedding = torch.nn.LazyLinear(d_model)
        self.positional_encoding = PositionalEncoding(d_model)
        encoder_layer = torch.nn.TransformerEncoderLayer(d_model=d_model, nhead=num_heads,
            dim_feedforward=dim_feedforward, batch_first=True)
        self.transformer = torch.nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        # Decode each patch token back to patch_len predictions
        self.output_layer = torch.nn.Linear(d_model, self.patch_len)

        # Setup Pytorch helper for training and evaluation
        self.my_pytorch_helper = PytorchHelper(self)

    def _patchify(self, x: torch.Tensor) -> tuple[torch.Tensor, int]:
        """
        Split x of shape (batch, seq_len, features) into non-overlapping patches of
        patch_len timesteps, flattened to one token per patch. If seq_len is not a
        multiple of patch_len, the first timestep is repeated at the front until it
        is - so that after decoding, dropping that many leading predictions restores
        alignment with the original, unpadded timesteps.
        """

        batch_size, seq_len, n_features = x.shape
        pad_len = (-seq_len) % self.patch_len
        if pad_len > 0:
            pad = x[:, :1, :].expand(batch_size, pad_len, n_features)
            x = torch.cat([pad, x], dim=1)

        num_patches = x.shape[1] // self.patch_len
        patches = x.reshape(batch_size, num_patches, self.patch_len * n_features)
        return patches, pad_len

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

        batch_size = x_tensor.shape[0]
        patches, pad_len = self._patchify(x_tensor)

        # Forward pass through the patch Transformer Encoder
        tokens = self.patch_embedding(patches)
        tokens = self.positional_encoding(tokens)
        tokens = self.transformer(tokens)

        # Decode each patch token back into patch_len timesteps and flatten to a
        # per-timestep sequence, then drop the leading padding added in _patchify.
        out = self.output_layer(tokens)
        out = out.reshape(batch_size, -1, 1)
        out = out[:, pad_len:, :]

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
            X_train (ArrayLike): Training input features of
                shape (batch_len, sequence_len, features).
            Y_train (ArrayLike): Training labels of
                shape (batch_len, sequence_len, 1).
            X_dev (ArrayLike, optional): Validation input features of
                shape (batch_len, sequence_len, features).
            Y_dev (ArrayLike, optional): Validation labels of
                shape (batch_len, sequence_len, 1).
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
        k_folds: int = 3,
        feature_index_groups: Union[Sequence[Sequence[int]], None] = None,
        default_feature_group_ids: Union[Sequence[int], None] = None,
        default_hyperparams: Union[dict, None] = None,
        storage_path: Union[str, None] = None,
        study_name: Union[str, None] = None,
        verbose: int = 1,
        ) -> dict:
        """
        Train this model with automatic hyperparameter optimization using
        Optuna and TimeSeriesSplit cross-validation.

        Args:
            x_train (ArrayLike): Training input features of
                shape (batch_len, sequence_len, features).
            y_train (ArrayLike): Training labels of
                shape (batch_len, sequence_len, 1).
            n_trials (int): Number of Optuna trials for hyperparameter
                search.
            k_folds (int): Number of TimeSeriesSplit folds used for cross-validation.
            feature_index_groups: Optional list of column-index groups (one group per
                named feature) to choose from during tuning.
            default_feature_group_ids: Optional indices into feature_index_groups to
                seed as one guaranteed trial - see OptunaHelper.train_auto.
            default_hyperparams: Optional previously-tuned hyperparameters to seed as
                one guaranteed trial - see OptunaHelper.train_auto.
            storage_path: Optional sqlite file path for the Optuna study storage.
            verbose (int): Verbosity level. 0: silent, 1: dots, 2: full.

        Returns:
            dict: Training history and best hyperparameters.
        """

        optuna_helper = OptunaHelper(self)
        history = optuna_helper.train_auto(
            x_train=x_train,
            y_train=y_train,
            n_trials=n_trials,
            k_folds=k_folds,
            feature_index_groups=feature_index_groups,
            default_feature_group_ids=default_feature_group_ids,
            default_hyperparams=default_hyperparams,
            storage_path=storage_path,
            study_name=study_name,
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
            x (ArrayLike): Input tensor of shape (batch_len, sequence_len,
                features) containing the features for which predictions are to be made.

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
        loss_relative_to: str = "",
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
