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


def _require_completed_trial(study: optuna.Study, model_class_name: str) -> None:
    """
    Raise a clear error if every trial in the study was pruned or failed (e.g. every
    fold of every trial hit an incompatible hyperparameter/data combination), rather
    than letting study.best_params fail with an opaque backend "Record does not
    exist" error.
    """

    if not any(trial.state == optuna.trial.TrialState.COMPLETE for trial in study.trials):
        raise RuntimeError(
            f"Optuna tuning for {model_class_name} produced no completed trial (all "
            f"{len(study.trials)} were pruned or failed) - see the trial warnings "
            "above for the underlying cause. Try more trials, fewer/looser search "
            "bounds, or check that the training data is large/varied enough for the "
            "chosen k_folds.")


class _ParamNameRecorder:
    """
    Stands in for an optuna.Trial to discover which parameter names (and bounds)
    a model class's suggest_params(trial) asks for, without needing a real trial
    or study. Used to build a safe seed trial from a model's current (pre-tuning)
    attribute values - see SklearnOptunaHelper._seed_default_hyperparams.
    """

    def __init__(self):
        # name -> ("range", low, high) | ("categorical", choices)
        self.bounds: dict = {}

    def suggest_int(self, name, low, high, *, step=1, log=False):
        self.bounds[name] = ("range", low, high)
        return low

    def suggest_float(self, name, low, high, *, step=None, log=False):
        self.bounds[name] = ("range", low, high)
        return low

    def suggest_categorical(self, name, choices):
        self.bounds[name] = ("categorical", tuple(choices))
        return choices[0]


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
        self.datasets: Sequence[tuple]
        self.k_folds: int
        self.feature_index_groups: Union[Sequence[Sequence[int]], None]
        self.default_feature_group_ids: Union[Sequence[int], None]
        self.verbose_level: int

    def train_auto(
        self,
        x_train: Union[ArrayLike, Sequence[ArrayLike]],
        y_train: Union[ArrayLike, Sequence[ArrayLike]],
        n_trials: int = 50,
        k_folds: int = 3,
        feature_index_groups: Union[Sequence[Sequence[int]], None] = None,
        default_feature_group_ids: Union[Sequence[int], None] = None,
        storage_path: Union[str, None] = None,
        study_name: Union[str, None] = None,
        verbose: int = 1,
        ) -> dict:
        """
        Train the model with automatic hyperparameter optimization.

        x_train/y_train can either be a single dataset, or a list of datasets (e.g.
        one per community) - see SklearnOptunaHelper.train_auto for the exact
        single-vs-pooled semantics (this class mirrors it). In pooled mode, no final
        fit is performed; my_model is left with its best model_size, but unfitted.

        Args:
            x_train: Training input features, or a list of them (one per dataset).
            y_train: Training target values, or a list of them (one per dataset).
            n_trials (int, optional): Number of Optuna trials.
            k_folds (int): Number of TimeSeriesSplit folds used for cross-validation
                (per dataset in single-dataset mode; across datasets in pooled mode).
            feature_index_groups: Optional list of column-index groups (one group per
                named feature, since one named feature can expand to several encoded
                columns). If given, Optuna additionally chooses which groups to keep,
                and the returned history contains the winning 'selected_feature_indices'.
            default_feature_group_ids: Optional list of indices into
                feature_index_groups (by position) to enable in one seeded trial
                (via study.enqueue_trial), so a chosen "known-good" default feature
                selection is always evaluated at least once, and a combinatorial
                feature-group search space can't make tuning end up worse than that
                default. Has no effect if feature_index_groups is not given.
            storage_path: Optional sqlite file path for the Optuna study storage (e.g.
                '<outputs_dir>/optuna_study.db'). Defaults to 'optuna_study.db' in the
                current working directory.
            study_name: Optional stable study name. If given, the study is resumed
                (load_if_exists=True) rather than created fresh - repeated calls with
                the same study_name+storage_path chain onto the same study. If None
                (default), a fresh, uniquely-timestamped study is created.
            verbose (int, optional): Verbosity level. 0: silent, 1: dots, 2: full.
        Returns:
            dict: Training history containing loss values.
        """

        # Store training parameters as instance attributes
        self.datasets = SklearnOptunaHelper._as_dataset_list(x_train, y_train)
        self.k_folds = k_folds
        self.feature_index_groups = feature_index_groups
        self.default_feature_group_ids = default_feature_group_ids
        self.verbose_level = verbose

        # Create and run Optuna study
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        study = optuna.create_study(
            direction='minimize',
            study_name=study_name or f"loadforecasting_{self.my_model.__class__.__name__}_{timestamp}",
            storage=f"sqlite:///{storage_path or 'optuna_study.db'}",
            load_if_exists=study_name is not None,
        )

        if verbose > 0:
            pooling_note = f", pooled across {len(self.datasets)} datasets" \
                if len(self.datasets) > 1 else ""
            print(f"Starting Optuna optimization with {n_trials} trials "
                  f"and {k_folds} TimeSeriesSplit folds{pooling_note}...")

        if self.feature_index_groups and self.default_feature_group_ids is not None:
            default_ids = set(self.default_feature_group_ids)
            study.enqueue_trial({f"use_feature_group_{i}": (i in default_ids)
                for i in range(len(self.feature_index_groups))})

        study.optimize(
            self.objective, n_trials=n_trials, show_progress_bar=(verbose > 0)
        )
        _require_completed_trial(study, self.my_model.__class__.__name__)

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

        history = {}
        if len(self.datasets) == 1:
            # Single dataset: also fit on the full (feature-sliced) training data,
            # so the given model instance is immediately usable.
            x_data, y_data = self.datasets[0]
            x_train_final = x_data
            if selected_feature_indices is not None:
                x_train_final = x_data[..., selected_feature_indices]
            history = self.my_model.train_model(
                x_train = x_train_final,
                y_train = y_data,
                epochs = best_epochs,
                learning_rates = best_learning_rates,
                batch_size = best_batch_size,
                verbose = verbose,
                )
        # else: pooled mode - no single "full training set" to fit on. The model is
        # left with the best model_size (create_model above), but unfitted.

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

        cv_losses = []
        if len(self.datasets) == 1:
            # Single dataset: average over all k_folds within it.
            x_data, y_data = self.datasets[0]
            for train_idx, val_idx in _time_series_splits(x_data.shape[0], self.k_folds):
                cv_losses.append(self._fit_and_eval(x_data, y_data, train_idx, val_idx,
                    selected_feature_indices, trial_model_size, trial_epochs,
                    learning_rates, trial_batch_size))
        else:
            # Multiple datasets (e.g. communities): each contributes exactly one
            # fold, cycling through the k_folds fold positions across datasets - see
            # SklearnOptunaHelper.objective for the rationale.
            for dataset_index, (x_data, y_data) in enumerate(self.datasets):
                splits = list(_time_series_splits(x_data.shape[0], self.k_folds))
                train_idx, val_idx = splits[dataset_index % len(splits)]
                cv_losses.append(self._fit_and_eval(x_data, y_data, train_idx, val_idx,
                    selected_feature_indices, trial_model_size, trial_epochs,
                    learning_rates, trial_batch_size))

        # Return mean validation loss across folds/datasets
        return float(np.mean(cv_losses))

    def _fit_and_eval(self, x_data, y_data, train_idx, val_idx, selected_feature_indices,
            trial_model_size, trial_epochs, learning_rates, trial_batch_size) -> float:
        """Fit one trial's model on one fold and return its validation loss."""

        x_fold_train = x_data[train_idx]
        y_fold_train = y_data[train_idx]
        x_fold_val = x_data[val_idx]
        y_fold_val = y_data[val_idx]

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
        return float(eval_value['test_loss'][-1])

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
        self.datasets: Sequence[tuple]
        self.k_folds: int
        self.feature_index_groups: Union[Sequence[Sequence[int]], None]
        self.covariate_param_name: Union[str, None]
        self.fixed_kwargs: dict
        self.suggest_params_kwargs: dict
        self.param_resolver: Union[callable, None]
        self.verbose_level: int

    def train_auto(
        self,
        x_train: Union[ArrayLike, Sequence[ArrayLike]],
        y_train: Union[ArrayLike, Sequence[ArrayLike]],
        n_trials: int = 50,
        k_folds: int = 3,
        feature_index_groups: Union[Sequence[Sequence[int]], None] = None,
        covariate_param_name: Union[str, None] = None,
        fixed_kwargs: Union[dict, None] = None,
        suggest_params_kwargs: Union[dict, None] = None,
        param_resolver: Union[callable, None] = None,
        storage_path: Union[str, None] = None,
        study_name: Union[str, None] = None,
        verbose: int = 1,
        ) -> dict:
        """
        Tune this model's hyperparameters (and, if feature_index_groups is given, which
        feature groups to use) with Optuna and time series cross-validation.

        x_train/y_train can either be a single dataset, or a list of datasets (e.g.
        one per community). With a single dataset, each trial is evaluated by
        averaging over all k_folds TimeSeriesSplit folds of it (as usual), and the
        model instance given at construction is reinitialized in place with the best
        settings and fit on the full dataset before returning.

        With multiple datasets, each trial is instead evaluated by giving each
        dataset exactly one TimeSeriesSplit fold - cycling through the k_folds fold
        positions across the datasets (dataset i gets fold i % k_folds) - and
        averaging the resulting per-dataset losses. This way, every trial's loss
        reflects generalization across all given datasets (e.g. communities) at
        once, rather than just one, without paying k_folds evaluations per dataset;
        and cycling through fold positions still gives every trial some data from
        each part of the time range (e.g. season) covered by the folds, spread
        across the pooled datasets instead of repeated within each one. In this
        mode, no final fit is performed (there is no single "full training set" to
        fit on) - the given model instance is left reinitialized with the best
        settings, but unfitted. Call my_model.train_model(...) yourself afterwards
        on whichever dataset you want an actual fitted model for.

        Args:
            x_train: Training input features, or a list of them (one per dataset).
            y_train: Training target values, or a list of them (one per dataset).
            n_trials (int): Number of Optuna trials.
            k_folds (int): Number of TimeSeriesSplit folds used for cross-validation
                (per dataset in single-dataset mode; across datasets in pooled mode).
            feature_index_groups: Optional list of column-index groups (one group per
                named feature, since one named feature can expand to several encoded
                columns) to choose from.
            covariate_param_name: If given, the selected feature indices are NOT used
                to slice x (the default). Instead, they are passed as a constructor
                kwarg of this name (e.g. Tirex2/Chronos2's future_covariate_indices),
                for models that only consume a named subset of x's columns rather than
                all of them. An empty selection is then a valid choice (univariate
                forecast), unlike the default slicing mode where it would leave the
                model with zero input features.
            fixed_kwargs: Extra constructor kwargs that must stay the same across all
                trials (e.g. PhysicsPvForecast-style fixed feature indices).
            suggest_params_kwargs: Extra keyword arguments forwarded to the model
                class's suggest_params(trial, **suggest_params_kwargs), for models
                whose search space depends on external context (e.g. Gam's list of
                candidate all_gam_terms).
            param_resolver: Optional callable(dict) -> dict, applied to the raw
                (JSON-serializable) suggested/stored params right before constructing
                a model, to turn e.g. an index into the actual object it refers to
                (e.g. Gam's term_set_index -> all_gam_terms). The raw, unresolved
                params are still what's returned in history['best_params'].
            storage_path: Optional sqlite file path for the Optuna study storage
                (e.g. '<outputs_dir>/optuna_study.db'). Defaults to 'optuna_study.db'
                in the current working directory.
            study_name: Optional stable study name. If given, the study is resumed
                (load_if_exists=True) rather than created fresh - repeated calls with
                the same study_name+storage_path chain onto the same study, so its
                sampler keeps learning across calls. If None (default), a fresh,
                uniquely-timestamped study is created.
            verbose (int): Verbosity level. 0: silent, 1: dots, 2: full.

        Returns:
            dict: Training history (of the final, refit model in single-dataset mode;
            otherwise just the tuning metadata), including 'best_params' and
            'selected_feature_indices'.
        """

        self.datasets = self._as_dataset_list(x_train, y_train)
        self.k_folds = k_folds
        self.feature_index_groups = feature_index_groups
        self.covariate_param_name = covariate_param_name
        self.fixed_kwargs = fixed_kwargs or {}
        self.suggest_params_kwargs = suggest_params_kwargs or {}
        self.param_resolver = param_resolver
        self.verbose_level = verbose

        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        study = optuna.create_study(
            direction='minimize',
            study_name=study_name or f"loadforecasting_{self.model_class.__name__}_{timestamp}",
            storage=f"sqlite:///{storage_path or 'optuna_study.db'}",
            load_if_exists=study_name is not None,
        )

        if verbose > 0:
            pooling_note = f", pooled across {len(self.datasets)} datasets" \
                if len(self.datasets) > 1 else ""
            print(f"Starting Optuna optimization with {n_trials} trials "
                  f"and {k_folds} TimeSeriesSplit folds{pooling_note}...")

        # Seed one trial with my_model's current (pre-tuning) configuration, so a
        # combinatorial/high-dimensional search space (e.g. many feature groups)
        # can never make the tuned result worse than not tuning at all - Optuna
        # will only move away from this configuration if it finds something that
        # actually scores better on the CV loss.
        seed_params = {**self._seed_default_hyperparams(), **self._seed_default_feature_flags()}
        if seed_params:
            study.enqueue_trial(seed_params)

        study.optimize(self.objective, n_trials=n_trials, show_progress_bar=(verbose > 0))
        _require_completed_trial(study, self.model_class.__name__)

        best_params = dict(study.best_params)
        selected_feature_indices = self._extract_selected_indices(best_params)
        model_params = best_params

        resolved_params = self.param_resolver(model_params) if self.param_resolver \
            else model_params
        if selected_feature_indices is not None and self.covariate_param_name is not None:
            resolved_params = dict(resolved_params)
            resolved_params[self.covariate_param_name] = selected_feature_indices

        if verbose > 0:
            print("\nBest hyperparameters found:")
            for key, value in model_params.items():
                print(f"  {key}: {value}")
            if selected_feature_indices is not None:
                print(f"  selected_feature_indices: {selected_feature_indices}")
            print(f"  Best CV loss: {study.best_value:.6f}")

        # Reinitialize the given model instance in place with the best settings.
        ctor_kwargs = {
            'normalizer': self.my_model.normalizer,
            'loss_relative_to': self.my_model.loss_relative_to,
            **self.fixed_kwargs,
            }
        ctor_kwargs.update(resolved_params)
        self.my_model.__init__(**ctor_kwargs)

        history = {}
        if len(self.datasets) == 1:
            # Single dataset: also fit on the full (feature-sliced) training data,
            # so the given model instance is immediately usable.
            x_data, y_data = self.datasets[0]
            x_train_final = x_data
            if selected_feature_indices is not None and self.covariate_param_name is None:
                x_train_final = x_data[..., selected_feature_indices]
            history = self.my_model.train_model(x_train_final, y_data)
        # else: pooled mode - no single "full training set" to fit on. The model
        # instance is left reinitialized with the best settings, but unfitted.

        history['best_params'] = model_params
        history['best_cv_loss'] = study.best_value
        history['optuna_study'] = study
        history['selected_feature_indices'] = selected_feature_indices

        return history

    @staticmethod
    def _as_dataset_list(x_train, y_train) -> list:
        """Normalize x_train/y_train into a list of (x, y) dataset tuples."""

        if isinstance(x_train, (list, tuple)):
            assert isinstance(y_train, (list, tuple)) and len(y_train) == len(x_train), \
                "y_train must be a list of the same length as x_train."
            return list(zip(x_train, y_train))
        return [(x_train, y_train)]

    def objective(self, trial: optuna.Trial) -> float:
        """Objective function for Optuna optimization."""

        model_params = self.model_class.suggest_params(trial, **self.suggest_params_kwargs)
        selected_feature_indices = self._suggest_feature_indices(trial)
        resolved_params = self.param_resolver(model_params) if self.param_resolver \
            else model_params

        cv_losses = []
        if len(self.datasets) == 1:
            # Single dataset: average over all k_folds within it.
            x_data, y_data = self.datasets[0]
            for train_idx, val_idx in _time_series_splits(x_data.shape[0], self.k_folds):
                cv_losses.append(self._fit_and_eval(x_data, y_data, train_idx, val_idx,
                    resolved_params, selected_feature_indices, model_params))
        else:
            # Multiple datasets (e.g. communities): each contributes exactly one
            # fold, cycling through the k_folds fold positions across datasets, so
            # the pool as a whole still covers all fold positions (e.g. seasons)
            # without paying k_folds evaluations per dataset.
            for dataset_index, (x_data, y_data) in enumerate(self.datasets):
                splits = list(_time_series_splits(x_data.shape[0], self.k_folds))
                train_idx, val_idx = splits[dataset_index % len(splits)]
                cv_losses.append(self._fit_and_eval(x_data, y_data, train_idx, val_idx,
                    resolved_params, selected_feature_indices, model_params))

        return float(np.mean(cv_losses))

    def _fit_and_eval(self, x_data, y_data, train_idx, val_idx, resolved_params,
            selected_feature_indices, model_params_for_error) -> float:
        """Fit one trial's model on one fold and return its validation loss."""

        x_fold_train = x_data[train_idx]
        y_fold_train = y_data[train_idx]
        x_fold_val = x_data[val_idx]
        y_fold_val = y_data[val_idx]

        fold_resolved_params = resolved_params
        if selected_feature_indices is not None:
            if self.covariate_param_name is not None:
                fold_resolved_params = dict(resolved_params)
                fold_resolved_params[self.covariate_param_name] = selected_feature_indices
            else:
                x_fold_train = x_fold_train[..., selected_feature_indices]
                x_fold_val = x_fold_val[..., selected_feature_indices]

        ctor_kwargs = {
            'normalizer': self.my_model.normalizer,
            'loss_relative_to': self.my_model.loss_relative_to,
            **self.fixed_kwargs,
            }
        ctor_kwargs.update(fold_resolved_params)
        trial_model = self.model_class(**ctor_kwargs)
        try:
            trial_model.train_model(x_fold_train, y_fold_train)
            eval_result = trial_model.evaluate(x_fold_val, y_fold_val, results={},
                de_normalize=False)
        except ValueError as error:
            # E.g. a sampled hyperparameter (such as Knn's k) that is only valid for
            # larger folds than the early, small TimeSeriesSplit folds provide.
            raise optuna.TrialPruned(
                f"Fold training/evaluation failed for params {model_params_for_error}: {error}"
                ) from error
        return float(eval_result['test_loss'][-1])

    def _suggest_feature_indices(self, trial: optuna.Trial) -> Union[list, None]:
        """Ask the trial which feature groups to keep in, if feature search is enabled."""

        if not self.feature_index_groups:
            return None
        selected = []
        for group_id, indices in enumerate(self.feature_index_groups):
            if trial.suggest_categorical(f"use_feature_group_{group_id}", [True, False]):
                selected.extend(indices)
        if not selected and self.covariate_param_name is None:
            # An empty selection is meaningless for a model that consumes all of x's
            # columns as its input (it would be left with zero features). It is a
            # valid choice (e.g. a univariate forecast) in covariate_param_name mode.
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

    def _seed_default_feature_flags(self) -> dict:
        """
        Build the use_feature_group_* flags reproducing my_model's pre-tuning
        default feature selection: every group, for column-slicing models (which
        had no feature selection before tuning existed - all of x was used); or
        only the group(s) covering my_model's current covariate_param_name value,
        for covariate-passing models (e.g. Tirex2/Chronos2's
        future_covariate_indices default of a single named covariate).
        """

        if not self.feature_index_groups:
            return {}
        if self.covariate_param_name is None:
            return {f"use_feature_group_{i}": True
                for i in range(len(self.feature_index_groups))}

        default_indices = set(getattr(self.my_model, self.covariate_param_name, None) or [])
        return {f"use_feature_group_{i}": bool(set(indices) & default_indices)
            for i, indices in enumerate(self.feature_index_groups)}

    def _seed_default_hyperparams(self) -> dict:
        """
        Build a {param_name: value} seed from my_model's current (pre-tuning)
        attribute values, restricted to the parameters suggest_params() actually
        searches over and to values that fall within its declared bounds - so a
        stale or out-of-range default can never break study.optimize(). A model
        whose tunable constructor args aren't mirrored as same-named attributes
        (nothing to introspect) simply seeds nothing for this part.
        """

        recorder = _ParamNameRecorder()
        self.model_class.suggest_params(recorder, **self.suggest_params_kwargs)

        seed = {}
        for name, spec in recorder.bounds.items():
            if not hasattr(self.my_model, name):
                continue
            value = getattr(self.my_model, name)
            kind = spec[0]
            if kind == "range":
                _, low, high = spec
                # e.g. Tirex2/Chronos2's context_length defaults to None (no cap) -
                # not a valid value for its own suggest_int range, so skip it.
                if isinstance(value, (int, float)) and not isinstance(value, bool) \
                        and low <= value <= high:
                    seed[name] = value
            else:
                _, choices = spec
                if value in choices:
                    seed[name] = value
        return seed
