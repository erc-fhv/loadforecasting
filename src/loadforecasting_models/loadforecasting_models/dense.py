from typing import Optional, Callable, Sequence, Union
import numpy as np
import torch
from torch.nn.modules.lazy import LazyModuleMixin
from torch.nn.parameter import UninitializedParameter
from .helpers import PytorchHelper, OptunaHelper
from .normalizer import Normalizer

# Define a type that can be either a torch Tensor or a numpy ndarray
ArrayLike = Union[torch.Tensor, np.ndarray]

# Sequence length the model_size table below is calibrated for (1 day at 15min).
CALIBRATED_SEQUENCE_LEN = 96

class _LazySequenceHead(LazyModuleMixin, torch.nn.Linear):
    """Linear output head whose in_features and out_features (= sequence_len) are
    inferred on the first forward pass."""

    cls_to_become = None
    weight: UninitializedParameter
    bias: UninitializedParameter

    def __init__(self) -> None:
        super().__init__(0, 0, False)
        self.weight = UninitializedParameter()
        self.bias = UninitializedParameter()

    def reset_parameters(self) -> None:
        if not self.has_uninitialized_params() and self.in_features != 0:
            super().reset_parameters()

    def initialize_parameters(self, x: torch.Tensor, sequence_len: int) -> None:
        if self.has_uninitialized_params():
            if sequence_len != CALIBRATED_SEQUENCE_LEN:
                print(f"Warning: Dense got {sequence_len} timesteps instead of "
                      f"{CALIBRATED_SEQUENCE_LEN}, so the parameter count deviates from model_size.")
            with torch.no_grad():
                self.in_features = x.shape[-1]
                self.out_features = sequence_len
                self.weight.materialize((self.out_features, self.in_features))
                self.bias.materialize((self.out_features,))
                self.reset_parameters()

    def forward(self, x: torch.Tensor, sequence_len: int) -> torch.Tensor:
        out = super().forward(x)
        assert out.shape[-1] == sequence_len, \
            f"Dense was built for sequence_len {out.shape[-1]}, got {sequence_len}."
        return out

class Dense(torch.nn.Module):
    """
    Fully connected (MLP) model for timeseries prediction, intended as a simple
    deep learning benchmark for the sequence models (Lstm, Transformer, ...).
    Each timestep's features are first projected to a small dimension, then the
    whole sequence is flattened and mapped by dense layers to all output timesteps
    at once, so every prediction sees the full input sequence.
    """

    def __init__(
        self,
        model_size: str = '5k',
        loss_fn: Callable[..., torch.Tensor] = torch.nn.L1Loss(),
        normalizer: Optional[Normalizer] = None,
        loss_relative_to: str = "",
        ) -> None:
        """
            model_size (str): The model parameter count, e.g. '0.1k',
                '0.2k', '0.5k', '1k', '2k', '5k', '10k', '20k', '40k',
                '80k', calibrated for 96 timesteps and ~30 features. Sizes
                not reachable there ('0.1k', '0.2k') predict NaN.
            loss_fn (Callable[..., torch.Tensor]): Loss function to be
                used during training. E.g., torch.nn.L1Loss(),
                torch.nn.MSELoss(), pytorch_helpers.smape, ...
            normalizer (Normalizer): Used for X and Y normalization and
                denormalization.
            loss_relative_to (str): Reference for relative loss calculation. Default: "".
        """

        super().__init__()
        self.loss_fn = loss_fn
        self.normalizer = normalizer
        self.loss_relative_to = loss_relative_to
        self.create_model(model_size)

    def create_model(self, model_size) -> None:
        """ Create the Dense model based on the specified model size. """

        # (per-timestep projection dim, hidden dim 1, hidden dim 2); None = not reachable
        sizes = {
            "0.1k": None,
            "0.2k": None,
            "0.5k": (1, 2, 2),
            "1k":   (1, 4, 4),
            "2k":   (2, 6, 8),
            "5k":   (3, 12, 14),
            "10k":  (4, 20, 22),
            "20k":  (5, 34, 30),
            "40k":  (6, 58, 48),
            "80k":  (8, 88, 64),
        }
        if model_size not in sizes:
            raise ValueError(f"Unimplemented params.model_size parameter given: {model_size}")
        self.is_size_reachable = sizes[model_size] is not None
        if not self.is_size_reachable:
            print(f"Warning: Dense cannot reach model_size {model_size}, so it predicts NaN.")
        d_model, hidden_dimension_dense1, hidden_dimension_dense2 = sizes[model_size] or (1, 1, 1)

        self.input_projection = torch.nn.LazyLinear(d_model)
        self.activation = torch.nn.ReLU()
        self.dense1 = torch.nn.LazyLinear(hidden_dimension_dense1)
        self.dense2 = torch.nn.Linear(hidden_dimension_dense1, hidden_dimension_dense2)
        self.output_layer = _LazySequenceHead()

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

        # (batch, seq, features) -> (batch, seq*d_model) -> (batch, seq, 1)
        batch_len, sequence_len, _ = x_tensor.shape
        x_tensor = self.input_projection(x_tensor)
        x_tensor = x_tensor.reshape(batch_len, -1)
        x_tensor = self.activation(self.dense1(x_tensor))
        x_tensor = self.activation(self.dense2(x_tensor))
        out = self.output_layer(x_tensor, sequence_len).unsqueeze(-1)
        if not self.is_size_reachable:
            out = out * float('nan')    # keeps the graph, so training does not crash

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
