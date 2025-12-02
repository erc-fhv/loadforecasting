"""
This module contains common (mainly pytorch) code for the forecasting models.
"""

from pathlib import Path
from typing import Sequence, Union
import math
import numpy as np
import torch
import optuna
from torch import optim
from torch.utils.data import DataLoader, Dataset
from loadforecasting_models import Transformer, Lstm, xLstm

# Define own types
ArrayLike = Union[torch.Tensor, np.ndarray]
MachineLearningModels = Union[Transformer, Lstm, xLstm]

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

    def __init__(self, my_model: torch.nn.Module):
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
        loss_relative_to: str = "mean",
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
        loss_sum = 0
        total_samples = 0
        prediction = torch.zeros(size=(y_test.size(0), 0, y_test.size(2)))

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
            for batch_x, batch_y in val_loader:

                # Predict
                output = self.my_model(batch_x.float())

                # Unnormalize the target variable, if wished.
                if de_normalize:
                    output = self.my_model.normalizer.de_normalize_y(output)

                # Compute Metrics
                loss = self.my_model.loss_fn(output, batch_y.float())
                loss_sum += loss.item() * batch_x.size(0)
                total_samples += batch_x.size(0)

                prediction = torch.cat([prediction, output], dim=1)

        # Calculate average test loss
        if total_samples > 0:
            if loss_relative_to == "mean":
                reference = float(torch.abs(torch.mean(y_test)))
            elif loss_relative_to == "max":
                reference = float(torch.abs(torch.max(y_test)))
            elif loss_relative_to == "range":
                reference = float(torch.max(y_test) - torch.min(y_test))
            else:
                raise ValueError(f"Unexpected parameter: loss_relative_to = {loss_relative_to}")
            test_loss = loss_sum / total_samples
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

    def __init__(self, my_model: MachineLearningModels):
        self.my_model = my_model

    def train_auto(
        self,
        x_train,
        y_train,
        n_trials=50,
        n_splits=5,
        val_size = 31,  # days for validation in each split
        verbose=1,
        ) -> dict:

        n_samples = x_train.shape[0]

        def objective(trial: 'optuna.Trial') -> float:
            """Objective function for Optuna optimization."""

            # Suggest hyperparameters
            learning_rate = trial.suggest_float(
                'learning_rate', 1e-4, 1e-1, log=True
                )
            trial_finetune = trial.suggest_categorical(
                'finetune_now', [True, False]
                )
            trial_epochs = trial.suggest_categorical(
                'epochs', [30, 50, 80, 100, 150, 200, 250, 300]
                )
            trial_batch_size = trial.suggest_categorical(
                'batch_size', [32, 64, 128, 256, 512]
                )
            trial_model_size = trial.suggest_categorical(
                'model_size',
                ['0.1k', '0.2k', '0.5k', '1k', '2k', '5k', '10k', '20k',
                 '40k', '80k']
                )

            # Expanding window cross-validation
            cv_losses = []

            for split_idx in range(n_splits):

                # Determine train and validation indices
                step_size = (n_samples - val_size) // n_splits
                train_end = (split_idx + 1) * step_size
                val_end = train_end + val_size
                assert val_end < n_samples, "Validation end index exceeds number of samples."

                # Split data
                x_fold_train = x_train[0:train_end]
                y_fold_train = y_train[0:train_end]
                x_fold_val = x_train[train_end:val_end]
                y_fold_val = y_train[train_end:val_end]
                assert y_fold_val.shape[0] == val_size, "Validation size mismatch."

                # Create a fresh model copy for this fold
                self.my_model.create_model(trial_model_size)

                # Train on this fold
                _ = self.my_model.train_model(
                    x_train = x_fold_train,
                    y_train = y_fold_train,
                    x_dev = torch.Tensor([]),
                    y_dev = torch.Tensor([]),
                    pretrain_now=False,
                    finetune_now = trial_finetune,
                    epochs = trial_epochs,
                    learning_rates = [learning_rate],
                    batch_size = trial_batch_size,
                    verbose = 0,  # silent
                )

                # Evaluate on validation set
                eval_value = self.my_model.evaluate(x_fold_val, y_fold_val, results={},
                    de_normalize=False)
                test_loss = float(torch.squeeze(eval_value['test_loss']))
                cv_losses.append(test_loss)

            # Return mean validation loss across folds
            return float(np.mean(cv_losses))

        # Create and run Optuna study
        study = optuna.create_study(
            direction='minimize',
            load_if_exists=True
        )

        if verbose > 0:
            print(f"Starting Optuna optimization with {n_trials} trials "
                  f"and {n_splits} expanding window splits...")

        study.optimize(
            objective, n_trials=n_trials, show_progress_bar=(verbose > 0)
        )

        # Get best hyperparameters
        best_params = study.best_params
        best_learning_rate = best_params['learning_rate']
        best_finetune = best_params['finetune_now']
        best_epochs = best_params['epochs']
        best_batch_size = best_params['batch_size']
        best_model_size = best_params['model_size']

        if verbose > 0:
            print("\nBest hyperparameters found:")
            print(f"  model_size: {best_model_size}")
            print(f"  learning_rate: {best_learning_rate:.6f}")
            print(f"  finetune_now: {best_finetune}")
            print(f"  epochs: {best_epochs}")
            print(f"  batch_size: {best_batch_size}")
            print(f"  Best CV loss: {study.best_value:.6f}")

        # Reinitialize model with best model size if different
        self.my_model.create_model(model_size = best_model_size)

        # Train final model with best hyperparameters on full training data
        history = self.my_model.train_model(
            x_train = x_train,
            y_train = y_train,
            finetune_now = best_finetune,
            epochs = best_epochs,
            learning_rates = [best_learning_rate],
            batch_size = best_batch_size,
            verbose = 0,  # silent
            )

        # Add best params to history
        history['best_params'] = best_params
        history['best_cv_loss'] = study.best_value
        history['optuna_study'] = study

        return history
