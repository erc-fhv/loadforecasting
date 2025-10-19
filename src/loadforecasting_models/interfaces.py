"""
This module specifies all parameters of the prediction models.
"""


from typing import Optional, Protocol, Callable, Sequence, Literal
from dataclasses import dataclass
import torch


class ModelAdapterProtocol(Protocol):
    """Defines the minimal interface for model adapters."""

    def normalize_x(
        self,
        x: torch.Tensor,
        training: bool = False,
        ) -> torch.Tensor:
        """Normalizes x."""
        ...

    def normalize_y(
        self,
        y: torch.Tensor,
        training: bool = False,
        ) -> torch.Tensor:
        """Normalizes y."""
        ...

    def de_normalize_x(
        self,
        x: torch.Tensor,
        ) -> torch.Tensor:
        """De-Normalizes X."""
        ...

    def de_normalize_y(
        self,
        y: torch.Tensor,
        ) -> torch.Tensor:
        """De-Normalizes X."""
        ...


@dataclass
class ModelParams:
    """
    Base class for all parameter(s) of all prediction models.
    Args:
        model_type (str): The type of the model, e.g. 'Transformer', 'LSTM', 'xLSTM', 
            'Transformer_Full', 'KNN', 'Persistence', 'Perfect'.
    """

    # Mandatory parameters
    model_type: str
    loss_fn: Optional[Callable[..., torch.Tensor]] # Options: nn.L1Loss(), nn.MSELoss(), pytorch_helpers.smape, ...


@dataclass
class MachineLearningModelInitParams(ModelParams):
    """
    Defines the specific init parameter for all machine learning prediction models.
    Args:
        model_size (str): The model parameter count, e.g. '0.1k', '0.2k', '0.5k', '1k',
            '2k', '5k', '10k', '20k', '40k', '80k'.
        num_of_features (int): Number of model input features.
        model_adapter (ModelAdapterProtocol, optional): Custom model adapter, especially
            used for X and Y normalization and denormalization.    
    """

    # Mandatory parameters
    model_size: int
    num_of_features: int

    # Optional parameter
    model_adapter: Optional[Callable[[torch.Tensor, torch.Tensor], torch.Tensor]] \
        = torch.nn.L1Loss() 
    forecast_horizon: int = 24

@dataclass
class PersistenceModelInitParams(ModelParams):
    """Defines the specific init parameter for the persistence prediction model."""

    # Mandatory parameter
    model_adapter: ModelAdapterProtocol

@dataclass
class KnnModelInitParams(ModelParams):
    """Defines the specific init parameter for the KNN prediction model."""

    # Optional parameters
    k: int = 40
    weights: Literal['uniform', 'distance'] | Callable | None = 'distance'


@dataclass
class BaselineModelInitParams(ModelParams):
    """Defines the specific init parameter for the baseline prediction models."""

    # Optional parameters
    model_adapter: Optional[ModelAdapterProtocol] = None

@dataclass
class MachineLearningModelTrainParams(ModelParams):
    """
    Defines the specific training parameter for all machine learning prediction models.
    
    Args:
        X_train (torch.Tensor): Training input features of shape (batch_len, sequence_len, features).
        Y_train (torch.Tensor): Training labels of shape (batch_len, sequence_len, 1).
        X_dev (torch.Tensor, optional): Validation input features of shape (batch_len, sequence_len, features).
        Y_dev (torch.Tensor, optional): Validation labels of shape (batch_len, sequence_len, 1).
        pretrain_now (bool): Whether to run a pretraining phase.
        finetune_now (bool): Whether to run fine-tuning.
        epochs (int): Number of training epochs.
        learning_rates (Sequence[float], optional): Learning rates schedule.
        batch_size (int): Batch size for training.
        verbose (int): Verbosity level.
    
    """

    # Mandatory parameters
    x_train: torch.Tensor
    y_train: torch.Tensor
    x_dev: Optional[torch.Tensor] = torch.Tensor([])
    y_dev: Optional[torch.Tensor] = torch.Tensor([])
    pretrain_now: bool = False
    finetune_now: bool = True
    epochs: int = 100
    learning_rates: Optional[Sequence[float]] = [0.01, 0.005, 0.001, 0.0005]
    batch_size: int = 256
    verbose: int = 0

@dataclass
class KnnTrainParams(ModelParams):
    """Defines the specific training parameter for the baseline prediction models."""

    # Mandatory parameters
    x_train: torch.Tensor
    y_train: torch.Tensor

@dataclass
class PersistencePredictionParams(ModelParams):
    """Defines the specific prediction parameter for the persistence model."""

    # Mandatory parameter
    x: torch.Tensor

@dataclass
class PerfectPredictionParams(ModelParams):
    """Defines the specific prediction parameter for the perfect model."""

    # Mandatory parameter
    y_real: torch.Tensor

@dataclass
class MachineLearningPredictionParams(ModelParams):
    """Defines the specific prediction parameter for the machine learning model."""

    # Mandatory parameter
    x: torch.Tensor

@dataclass
class EvaluationParams(ModelParams):
    """Defines the specific evaluation parameter for the machine learning model."""

    # Mandatory parameter
    x_test: torch.Tensor
    y_test: torch.Tensor

    # Optional parameters
    results: dict = {}
    de_normalize: bool = False
