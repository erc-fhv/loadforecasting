"""
This module contains common (mainly pytorch) code for the forecasting models.
"""

from pathlib import Path
from typing import Sequence, Union, TYPE_CHECKING
import datetime
import math
import numpy as np
import torch
import optuna
from sklearn.model_selection import TimeSeriesSplit
from torch import optim
from torch.utils.data import DataLoader, Dataset

# The following modules are only imported during type checking
if TYPE_CHECKING:
    from loadforecasting_models import Lstm, xLstm, Transformer, TransformerFull

# Define a type that can be either a torch Tensor or a numpy ndarray
ArrayLike = Union[torch.Tensor, np.ndarray]


def _time_series_splits(n_samples: int, k_folds: int):
    """
    Yield (train_idx, val_idx) index arrays for cross-validation. For k_folds <= 1,
    yields a single static 80/20 train/validation split (matching the historical
    meaning of k_folds=1: "the same as a static train-dev-split"). For k_folds >= 2,
    uses sklearn's TimeSeriesSplit(n_splits=k_folds).
    """

    if k_folds <= 1:
        split_point = min(max(1, round(n_samples * 0.8)), n_samples - 1)
        yield np.arange(0, split_point), np.arange(split_point, n_samples)
    else:
        splitter = TimeSeriesSplit(n_splits=k_folds)
        yield from splitter.split(np.arange(n_samples))


class SequenceDataset(Dataset):
    """Custom Dataset for sequence data."""

    def __init__(self, x: ArrayLike, y: ArrayLike):
        self.x = x
        self.y = y

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]

class CustomLRScheduler:
    """Custom learning rate scheduler for PyTorch optimizers."""

    def __init__(self, optimizer, set_learning_rates, max_epochs):
        self.optimizer = optimizer
        self.set_learning_rates = set_learning_rates
        self.max_epochs = max_epochs
        self.lr_switching_points = np.flip(np.linspace(1, 0, len(self.set_learning_rates),
            endpoint=False))

    def adjust_learning_rate(self, epoch):
        """Adjust the learning rate based on the current epoch."""

        # Calculate the progress through the epochs (0 to 1)
        progress = epoch / self.max_epochs

        # Determine the current learning rate based on progress
        for i, boundary in enumerate(self.lr_switching_points):
            if progress < boundary:
                new_lr = self.set_learning_rates[i]
                break
            else:
                # If progress is >= 1, use the last learning rate
                new_lr = self.set_learning_rates[-1]

        # Update the optimizer's learning rate
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = new_lr


class PytorchHelper():
    """Helper class for Pytorch models."""

    def __init__(self, my_model: "Union[Lstm, xLstm, Transformer, TransformerFull]"):
        self.my_model = my_model

    def train(
        self,
        x_train: ArrayLike,
        y_train: ArrayLike,
        x_dev: ArrayLike,
        y_dev: ArrayLike,
        pretrain_now: bool,
        finetune_now: bool,
        epochs: int,
        learning_rates: Sequence[float],
        batch_size: int,
        verbose: int,
        ) -> dict:
        """
        Train a pytorch model.
        Args:
            X_train (torch.Tensor or np.ndarray): Training input features of
                shape (batch_len, sequence_len, features).
            Y_train (torch.Tensor or np.ndarray): Training labels of
                shape (batch_len, sequence_len, 1).
            X_dev (torch.Tensor or np.ndarray, optional): Validation input features of
                shape (batch_len, sequence_len, features).
            Y_dev (torch.Tensor or np.ndarray, optional): Validation labels of
                shape (batch_len, sequence_len, 1).
            pretrain_now (bool): Whether to run a pretraining phase.
            finetune_now (bool): Whether to run fine-tuning.
            epochs (int): Number of training epochs.
            learning_rates (Sequence[float], optional): Learning rates schedule.
            batch_size (int): Batch size for training.
            verbose (int): Verbosity level. 0: silent, 1: dots, 2: full.
        """

        # Convert numpy to torch if needed
        if isinstance(x_train, np.ndarray):
            x_train  = torch.from_numpy(x_train)
        if isinstance(y_train, np.ndarray):
            y_train  = torch.from_numpy(y_train)
        if isinstance(x_dev, np.ndarray):
            x_dev  = torch.from_numpy(x_dev)
        if isinstance(y_dev, np.ndarray):
            y_dev  = torch.from_numpy(y_dev)

        # Prepare Optimization
        train_dataset = SequenceDataset(x_train, y_train)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        my_optimizer = optim.Adam(self.my_model.parameters(), lr=learning_rates[0])
        lr_scheduler = CustomLRScheduler(my_optimizer, learning_rates, epochs)
        history = {"loss": []}

        # Load pretrained weights
        if finetune_now:
            filename = f'pretrained_weights_{self.my_model.__class__.__name__}.pth'
            load_path = Path.home() / ".loadforecasting_models" / filename
            if not load_path.exists():
                raise FileNotFoundError(f"No weights found at {load_path}")
            self.my_model.load_state_dict(torch.load(load_path))

        # Start training
        self.my_model.train()   # Switch on the training flags
        for epoch in range(epochs):
            loss_sum = 0
            total_samples = 0
            batch_losses = []

            # Optimize over one epoch
            for batch_x, batch_y in train_loader:
                my_optimizer.zero_grad()
                output = self.my_model(batch_x.float())
                loss = self.my_model.loss_fn(output, batch_y)
                batch_losses.append(loss.item())
                loss.backward()
                my_optimizer.step()
                loss_sum += loss.item() * batch_x.size(0)
                total_samples += batch_x.size(0)

            # Adjust learning rate once per epoch
            lr_scheduler.adjust_learning_rate(epoch)

            # Calculate average loss for the epoch
            epoch_loss = loss_sum / total_samples
            history['loss'].append(epoch_loss)

            if verbose == 0:
                pass    # silent
            elif verbose == 1:
                print(".", end="", flush=True)
            elif verbose == 2:
                if x_dev.shape[0] == 0 or y_dev.shape[0] == 0:
                    dev_loss = -1.0
                else:
                    eval_value = self.evaluate(x_dev, y_dev, results={}, de_normalize=False)
                    dev_loss = float(eval_value['test_loss'][-1])
                    self.my_model.train()  # Switch back to training mode after evaluation
                print(f"Epoch {epoch + 1}/{epochs} - " +
                    f"Loss = {epoch_loss:.4f} - " +
                    f"Dev_Loss = {dev_loss:.4f} - " +
                    f"LR = {my_optimizer.param_groups[0]['lr']}",
                    flush=True)
            else:
                raise ValueError(f"Unexpected parameter value: verbose = {verbose}")

        # Save the trained weights
        if pretrain_now:
            filename = f'pretrained_weights_{self.my_model.__class__.__name__}.pth'
            save_dir = Path.home() / ".loadforecasting_models"
            save_dir.mkdir(exist_ok=True)
            pretrained_weights_path = save_dir / filename
            torch.save(self.my_model.state_dict(), pretrained_weights_path)

        return history

    def s_mape(self, y_true, y_pred, dim=None):
        """
        Compute the Symmetric Mean Absolute Percentage Error (sMAPE).
        """

        numerator = torch.abs(y_pred - y_true)
        denominator = (torch.abs(y_true) + torch.abs(y_pred))
        eps = 1e-8 # To avoid division by zero
        smape_values = torch.mean(numerator / (denominator + eps), dim=dim) * 2 * 100
        return smape_values

    def evaluate(
        self,
        x_test: ArrayLike,
        y_test: ArrayLike,
        results: dict,
        de_normalize: bool = False,
        loss_relative_to: str = "",
        ) -> dict:
        """
        Evaluate the model on the given x_test and y_test.
        """

        # Convert numpy to torch if needed
        if isinstance(x_test, np.ndarray):
            x_test  = torch.from_numpy(x_test)
        if isinstance(y_test, np.ndarray):
            y_test  = torch.from_numpy(y_test)

        # Initialize metrics
        prediction = torch.zeros(size=(0, y_test.size(1), y_test.size(2)))

        # Unnormalize the target variable, if wished.
        if de_normalize:
            assert self.my_model.normalizer is not None, "No normalizer given."
            y_test = self.my_model.normalizer.de_normalize_y(y_test)

        # Create DataLoader
        batch_size=256
        val_dataset = SequenceDataset(x_test, y_test)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

        self.my_model.eval()       # Switch off the training flags
        with torch.no_grad():  # No gradient calculation
            for batch_x, _ in val_loader:

                # Predict
                output: torch.Tensor
                output = self.my_model(batch_x.float())

                prediction = torch.cat([prediction, output], dim=0)

        # Unnormalize the predictions, if wished.
        if de_normalize:
            assert self.my_model.normalizer is not None, "No normalizer given."
            prediction = self.my_model.normalizer.de_normalize_y(prediction)

        # Calculate average test loss
        if prediction.size(0) > 0:
            test_loss = float(self.my_model.loss_fn(prediction, y_test.float()))

            # Set default reference for relative loss if not given as argument and as attribute.
            if loss_relative_to == "" and self.my_model.loss_relative_to != "":
                loss_relative_to = self.my_model.loss_relative_to
            else:
                loss_relative_to = "mean"

            if loss_relative_to == "mean":
                reference = float(torch.abs(torch.mean(y_test)))
            elif loss_relative_to == "max":
                reference = float(torch.abs(torch.max(y_test)))
            elif loss_relative_to == "range":
                reference = float(torch.max(y_test) - torch.min(y_test))
            else:
                raise ValueError(f"Unexpected parameter: loss_relative_to = {loss_relative_to}")
            results['test_loss'] = [test_loss]
            results['test_loss_relative'] = [100.0 * test_loss / reference]
            results['predicted_profile'] = prediction
        else:
            results['test_loss'] = [0.0]
            results['test_loss_relative'] = [0.0]
            results['predicted_profile'] = [0.0]

        return results


class PositionalEncoding(torch.nn.Module):
    """
    Implements sinusoidal positional encoding as used in Transformer models.

    Positional encodings provide information about the relative or absolute position
    of tokens in a sequence, allowing the model to capture order without recurrence.

    This implementation is adapted from:
    https://stackoverflow.com/questions/77444485/using-positional-encoding-in-pytorch
    or respectively:
    https://pytorch-tutorials-preview.netlify.app/beginner/transformer_tutorial.html
    """

    def __init__(self, d_model: int, dropout: float = 0.0, max_len: int = 5000):
        super().__init__()
        self.dropout = torch.nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Arguments:
            x: Tensor, shape ``[seq_len, batch_size, embedding_dim]``
        """
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)


class OptunaHelper:
    """Helper class for Optuna hyperparameter optimization."""

    def __init__(self, my_model: "Union[Lstm, xLstm, Transformer]"):

        self.my_model = my_model

        # Initialize attributes for training
        self.lr_schedules: dict
        self.x_train: ArrayLike
        self.y_train: ArrayLike
        self.k_folds: int
        self.feature_index_groups: Union[Sequence[Sequence[int]], None]
        self.verbose_level: int

    def train_auto(
        self,
        x_train: ArrayLike,
        y_train: ArrayLike,
        n_trials: int = 50,
        k_folds: int = 3,
        feature_index_groups: Union[Sequence[Sequence[int]], None] = None,
        verbose: int = 1,
        ) -> dict:
        """
        Train the model with automatic hyperparameter optimization.
        Args:
            x_train (ArrayLike): Training input features of the model.
            y_train (ArrayLike): Training target values of the model.
            n_trials (int, optional): Number of Optuna trials.
            k_folds (int): Number of TimeSeriesSplit folds used for cross-validation.
            feature_index_groups: Optional list of column-index groups (one group per
                named feature, since one named feature can expand to several encoded
                columns). If given, Optuna additionally chooses which groups to keep,
                and the returned history contains the winning 'selected_feature_indices'.
            verbose (int, optional): Verbosity level. 0: silent, 1: dots, 2: full.
        Returns:
            dict: Training history containing loss values.
        """

        # Store training parameters as instance attributes
        self.x_train = x_train
        self.y_train = y_train
        self.k_folds = k_folds
        self.feature_index_groups = feature_index_groups
        self.verbose_level = verbose

        # Create and run Optuna study
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        study = optuna.create_study(
            direction='minimize',
            study_name=f"loadforecasting_{self.my_model.__class__.__name__}_{timestamp}",
            storage="sqlite:///optuna_study.db",
            load_if_exists=False,
        )

        if verbose > 0:
            print(f"Starting Optuna optimization with {n_trials} trials "
                  f"and {k_folds} TimeSeriesSplit folds...")

        study.optimize(
            self.objective, n_trials=n_trials, show_progress_bar=(verbose > 0)
        )

        # Get best hyperparameters
        best_params = dict(study.best_params)
        selected_feature_indices = self._extract_selected_indices(best_params)
        best_learning_rates = self.lr_schedules[best_params['lr_schedule_name']]
        best_epochs = best_params['epochs']
        best_batch_size = best_params['batch_size']
        best_model_size = best_params['model_size']
        # Resolve the schedule name to its actual learning rates, so callers that reuse
        # best_params later (e.g. to reconstruct training) don't need self.lr_schedules.
        best_params['learning_rates'] = best_learning_rates

        x_train_final = x_train
        if selected_feature_indices is not None:
            x_train_final = x_train[..., selected_feature_indices]

        if verbose > 0:
            print("\nBest hyperparameters found:")
            print(f"  model_size: {best_model_size}")
            print(f"  learning_rate_schedule: {best_learning_rates}")
            print(f"  epochs: {best_epochs}")
            print(f"  batch_size: {best_batch_size}")
            if selected_feature_indices is not None:
                print(f"  selected_feature_indices: {selected_feature_indices}")
            print(f"  Best CV loss: {study.best_value:.6f}")

        # Reinitialize model with best model size if different
        self.my_model.create_model(model_size = best_model_size)

        # Train final model with best hyperparameters on full training data
        history = self.my_model.train_model(
            x_train = x_train_final,
            y_train = y_train,
            epochs = best_epochs,
            learning_rates = best_learning_rates,
            batch_size = best_batch_size,
            verbose = verbose,
            )

        # Add best params to history
        history['best_params'] = best_params
        history['best_cv_loss'] = study.best_value
        history['optuna_study'] = study
        history['selected_feature_indices'] = selected_feature_indices

        return history

    def objective(self, trial: optuna.Trial) -> float:
        """Objective function for Optuna optimization."""

        # Hyperparameters to choose from.
        #

        # Learning rate schedules. Learning rates will step through these values during training.
        self.lr_schedules = {
            "default": [0.01, 0.005, 0.001, 0.0005], # Default for this framework
            "constant": [0.001],    # Default Adam Parameter as Baseline
            "moderate_decay": [0.005, 0.0025, 0.001, 0.0005],
            "conservative": [0.001, 0.0007, 0.0005, 0.0003],
        }
        learning_rates = self.lr_schedules[trial.suggest_categorical(
            "lr_schedule_name", list(self.lr_schedules.keys())
            )]
        trial_epochs = trial.suggest_int(
            "epochs", low=30, high=300, log=True,
            )
        trial_batch_size = trial.suggest_categorical(
            'batch_size', [32, 64, 128, 256]
            )
        trial_model_size = trial.suggest_categorical(
            'model_size', ['0.1k', '0.2k', '0.5k', '1k', '2k', '5k', '10k', '20k', '40k', '80k']
            )
        selected_feature_indices = self._suggest_feature_indices(trial)

        # Time series cross-validation
        #
        cv_losses = []
        n_samples = self.x_train.shape[0]

        for train_idx, val_idx in _time_series_splits(n_samples, self.k_folds):

            # Split data
            x_fold_train = self.x_train[train_idx]
            y_fold_train = self.y_train[train_idx]
            x_fold_val = self.x_train[val_idx]
            y_fold_val = self.y_train[val_idx]

            if selected_feature_indices is not None:
                x_fold_train = x_fold_train[..., selected_feature_indices]
                x_fold_val = x_fold_val[..., selected_feature_indices]

            # Create a fresh model copy for this fold
            self.my_model.create_model(trial_model_size)

            # Train on this fold
            _ = self.my_model.train_model(
                x_train = x_fold_train,
                y_train = y_fold_train,
                epochs = trial_epochs,
                learning_rates = learning_rates,
                batch_size = trial_batch_size,
                verbose = self.verbose_level,
            )

            # Evaluate on validation set
            eval_value = self.my_model.evaluate(x_fold_val, y_fold_val, results={},
                de_normalize=False)
            test_loss = float(eval_value['test_loss'][-1])
            cv_losses.append(test_loss)

        # Return mean validation loss across folds
        return float(np.mean(cv_losses))

    def _suggest_feature_indices(self, trial: optuna.Trial) -> Union[list, None]:
        """Ask the trial which feature groups to keep in, if feature search is enabled."""

        if not self.feature_index_groups:
            return None
        selected = []
        for group_id, indices in enumerate(self.feature_index_groups):
            if trial.suggest_categorical(f"use_feature_group_{group_id}", [True, False]):
                selected.extend(indices)
        if not selected:
            raise optuna.TrialPruned("No feature group selected.")
        return sorted(selected)

    def _extract_selected_indices(self, best_params: dict) -> Union[list, None]:
        """Pop the feature-group flags out of best_params, returning the winning indices."""

        if not self.feature_index_groups:
            return None
        selected = []
        for group_id, indices in enumerate(self.feature_index_groups):
            if best_params.pop(f"use_feature_group_{group_id}"):
                selected.extend(indices)
        return sorted(selected)


class SklearnOptunaHelper:
    """
    Helper class for Optuna hyperparameter optimization of scikit-learn-style models
    (Knn, Ridge, RandomForest, XGBoost, Gam), optionally including feature-group
    selection. Mirrors the calling convention of OptunaHelper, but re-creates a fresh
    model instance per trial/fold instead of relying on a create_model() method.
    """

    def __init__(self, my_model):
        self.my_model = my_model
        self.model_class = type(my_model)

        # Initialize attributes for training
        self.x_train: ArrayLike
        self.y_train: ArrayLike
        self.k_folds: int
        self.feature_index_groups: Union[Sequence[Sequence[int]], None]
        self.fixed_kwargs: dict
        self.verbose_level: int

    def train_auto(
        self,
        x_train: ArrayLike,
        y_train: ArrayLike,
        n_trials: int = 50,
        k_folds: int = 3,
        feature_index_groups: Union[Sequence[Sequence[int]], None] = None,
        fixed_kwargs: Union[dict, None] = None,
        verbose: int = 1,
        ) -> dict:
        """
        Tune this model's hyperparameters (and, if feature_index_groups is given, which
        feature groups to use) with Optuna and time series cross-validation, then
        reinitialize the given model instance in place with the best settings and fit
        it on the full training data.

        Args:
            x_train (ArrayLike): Training input features of the model.
            y_train (ArrayLike): Training target values of the model.
            n_trials (int): Number of Optuna trials.
            k_folds (int): Number of TimeSeriesSplit folds used for cross-validation.
            feature_index_groups: Optional list of column-index groups (one group per
                named feature, since one named feature can expand to several encoded
                columns) to choose from.
            fixed_kwargs: Extra constructor kwargs that must stay the same across all
                trials (e.g. Gam's all_gam_terms).
            verbose (int): Verbosity level. 0: silent, 1: dots, 2: full.

        Returns:
            dict: Training history of the final, refit model, including 'best_params'
            and 'selected_feature_indices'.
        """

        self.x_train = x_train
        self.y_train = y_train
        self.k_folds = k_folds
        self.feature_index_groups = feature_index_groups
        self.fixed_kwargs = fixed_kwargs or {}
        self.verbose_level = verbose

        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        study = optuna.create_study(
            direction='minimize',
            study_name=f"loadforecasting_{self.model_class.__name__}_{timestamp}",
            storage="sqlite:///optuna_study.db",
            load_if_exists=False,
        )

        if verbose > 0:
            print(f"Starting Optuna optimization with {n_trials} trials "
                  f"and {k_folds} TimeSeriesSplit folds...")

        study.optimize(self.objective, n_trials=n_trials, show_progress_bar=(verbose > 0))

        best_params = dict(study.best_params)
        selected_feature_indices = self._extract_selected_indices(best_params)
        model_params = best_params

        x_train_final = x_train
        if selected_feature_indices is not None:
            x_train_final = x_train[..., selected_feature_indices]

        if verbose > 0:
            print("\nBest hyperparameters found:")
            for key, value in model_params.items():
                print(f"  {key}: {value}")
            if selected_feature_indices is not None:
                print(f"  selected_feature_indices: {selected_feature_indices}")
            print(f"  Best CV loss: {study.best_value:.6f}")

        # Reinitialize the given model instance in place with the best settings, then
        # fit it on the full training data.
        self.my_model.__init__(
            normalizer=self.my_model.normalizer,
            loss_relative_to=self.my_model.loss_relative_to,
            **self.fixed_kwargs,
            **model_params,
            )
        history = self.my_model.train_model(x_train_final, y_train)

        history['best_params'] = model_params
        history['best_cv_loss'] = study.best_value
        history['optuna_study'] = study
        history['selected_feature_indices'] = selected_feature_indices

        return history

    def objective(self, trial: optuna.Trial) -> float:
        """Objective function for Optuna optimization."""

        model_params = self.model_class.suggest_params(trial)
        selected_feature_indices = self._suggest_feature_indices(trial)

        n_samples = self.x_train.shape[0]

        cv_losses = []
        for train_idx, val_idx in _time_series_splits(n_samples, self.k_folds):

            x_fold_train = self.x_train[train_idx]
            y_fold_train = self.y_train[train_idx]
            x_fold_val = self.x_train[val_idx]
            y_fold_val = self.y_train[val_idx]

            if selected_feature_indices is not None:
                x_fold_train = x_fold_train[..., selected_feature_indices]
                x_fold_val = x_fold_val[..., selected_feature_indices]

            trial_model = self.model_class(
                normalizer=self.my_model.normalizer,
                loss_relative_to=self.my_model.loss_relative_to,
                **self.fixed_kwargs,
                **model_params,
                )
            try:
                trial_model.train_model(x_fold_train, y_fold_train)
                eval_result = trial_model.evaluate(x_fold_val, y_fold_val, results={},
                    de_normalize=False)
            except ValueError as error:
                # E.g. a sampled hyperparameter (such as Knn's k) that is only valid for
                # larger folds than the early, small TimeSeriesSplit folds provide.
                raise optuna.TrialPruned(
                    f"Fold training/evaluation failed for params {model_params}: {error}"
                    ) from error
            cv_losses.append(float(eval_result['test_loss'][-1]))

        return float(np.mean(cv_losses))

    def _suggest_feature_indices(self, trial: optuna.Trial) -> Union[list, None]:
        """Ask the trial which feature groups to keep in, if feature search is enabled."""

        if not self.feature_index_groups:
            return None
        selected = []
        for group_id, indices in enumerate(self.feature_index_groups):
            if trial.suggest_categorical(f"use_feature_group_{group_id}", [True, False]):
                selected.extend(indices)
        if not selected:
            raise optuna.TrialPruned("No feature group selected.")
        return sorted(selected)

    def _extract_selected_indices(self, best_params: dict) -> Union[list, None]:
        """Pop the feature-group flags out of best_params, returning the winning indices."""

        if not self.feature_index_groups:
            return None
        selected = []
        for group_id, indices in enumerate(self.feature_index_groups):
            if best_params.pop(f"use_feature_group_{group_id}"):
                selected.extend(indices)
        return sorted(selected)
