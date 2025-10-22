"""
This module contains common (mainly pytorch) code for the machine learning models.
"""

import os
from typing import Sequence
import numpy as np
import torch
from torch import nn
from torch import optim
from torch.utils.data import DataLoader, Dataset


class SequenceDataset(Dataset):
    def __init__(self, X, Y):
        self.X = X
        self.Y = Y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]


class CustomLRScheduler:
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

    def __init__(self, my_model: nn.Module):
        self.my_model = my_model

    def train(
        self,
        x_train: torch.Tensor,
        y_train: torch.Tensor,
        x_dev: torch.Tensor,
        y_dev: torch.Tensor,
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

        # Prepare Optimization
        train_dataset = SequenceDataset(x_train, y_train)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)            
        my_optimizer = optim.Adam(self.my_model.parameters(), lr=learning_rates[0])
        lr_scheduler = CustomLRScheduler(my_optimizer, learning_rates, epochs)
        history = {"loss": []}

        # Load pretrained weights
        if finetune_now:
            pretrained_weights_path = f'{os.path.dirname(__file__)}/outputs/pretrained_weights_{self.my_model.__class__.__name__}.pth'
            self.my_model.load_state_dict(torch.load(pretrained_weights_path))

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
                print(".", end="", flush=True)
            elif verbose == 1:
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
            elif verbose == 2:
                pass    # silent
            else:
                raise ValueError(f"Unexpected parameter value: verbose = {verbose}")

        # Save the trained weights
        if pretrain_now:
            pretrained_weights_path = f'{os.path.dirname(__file__)}/outputs/pretrained_weights_{self.my_model.__class__.__name__}.pth'
            torch.save(self.my_model.state_dict(), pretrained_weights_path)

        return history

    def smape(self, y_true, y_pred, dim=None):
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
        x_test: torch.Tensor,
        y_test: torch.Tensor,
        results: dict,
        de_normalize: bool = False,
        ) -> dict:
        """
        Evaluate the model on the given x_test and y_test.
        """

        # Initialize metrics
        loss_sum = 0
        smape_sum = 0
        total_samples = 0
        prediction = torch.zeros(size=(y_test.size(0), 0, y_test.size(2)))

        # Unnormalize the target variable, if wished.
        if de_normalize:
            assert self.my_model.model_adapter is not None, "No model_adapter given."
            y_test = self.my_model.model_adapter.deNormalizeY(y_test)

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
                    output = self.my_model.model_adapter.deNormalizeY(output)

                # Compute Metrics
                loss = self.my_model.loss_fn(output, batch_y.float())
                loss_sum += loss.item() * batch_x.size(0)
                smape_val = self.smape(batch_y.float(), output)
                smape_sum += smape_val * batch_x.size(0)
                total_samples += batch_x.size(0)

                prediction = torch.cat([prediction, output], dim=1)

        # Calculate average loss and sMAPE
        if total_samples > 0:
            test_loss = loss_sum / total_samples
            reference = float(torch.mean(y_test))
            results['test_loss'] = [test_loss]
            results['test_loss_relative'] = [100.0 * test_loss / reference]
            results['test_sMAPE'] = [smape_sum / total_samples]
            results['predicted_profile'] = prediction
        else:
            results['test_loss'] = [0.0]
            results['test_loss_relative'] = [0.0]
            results['test_sMAPE'] = [0.0]
            results['predicted_profile'] = [0.0]

        return results


class PositionalEncoding(nn.Module):
    """    
    This implementation of positional encoding is based on the
    "Attention Is All You Need" paper, and is conceptually similar to:
    https://stackoverflow.com/questions/77444485/using-positional-encoding-in-pytorch
    """

    def __init__(self, d_model, timesteps=5000):
        super().__init__()

        pe = torch.zeros(timesteps, d_model)  # [timesteps, d_model]
        position = torch.arange(0, timesteps, dtype=torch.float).unsqueeze(1)  # [timesteps, 1]
        _2i = torch.arange(0, d_model, 2).float()
        div_term = torch.exp(_2i * (-math.log(10000.0) / d_model))  # [d_model/2]

        pe[:, 0::2] = torch.sin(position * div_term)  # Apply sin to even indices in the array
        pe[:, 1::2] = torch.cos(position * div_term)  # Apply cos to odd indices in the array

        pe = pe.unsqueeze(0)  # [1, timesteps, d_model]
        self.register_buffer('pe', pe)  # Save as a non-learnable buffer

    def forward(self, x):
        """Add positional encoding to input tensor x."""

        _, timesteps, features = x.shape
        assert (self.pe.size(1) == timesteps), f"Expected timesteps: {self.pe.size(1)}, received timesteps: {timesteps}"
        assert (self.pe.size(2) == features), f"Expected features: {self.pe.size(2)}, received features: {features}"

        x = x + self.pe

        return x


class PositionalEncoding(nn.Module):
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
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: Tensor) -> Tensor:
        """
        Arguments:
            x: Tensor, shape ``[seq_len, batch_size, embedding_dim]``
        """
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)



