"""
This module contains common code for the (pytorch) machine learning models.
"""

import os
from typing import Optional, Sequence
import numpy as np
import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader, Dataset
from loadforecasting_models.interfaces import MachineLearningModelTrainParams

class PytorchHelper():
    
    @staticmethod
    def train(
        my_model: nn.Module,
        params: MachineLearningModelTrainParams
        ) -> dict:

        # Set default parameters
        if params.learning_rates is None:
            params.learning_rates = [0.01, 0.005, 0.001, 0.0005]
        if params.x_dev is None:
            params.x_dev = torch.Tensor([])
        if params.y_dev is None:
            params.y_dev = torch.Tensor([])

        # Prepare Optimization
        train_dataset = SequenceDataset(params.x_train, params.y_train)
        train_loader = DataLoader(train_dataset, batch_size=params.batch_size, shuffle=True)            
        my_optimizer = optim.Adam(my_model.parameters(), lr=params.learning_rates[0])
        lr_scheduler = CustomLRScheduler(my_optimizer, params.learning_rates, params.epochs)
        history = {"loss": []}

        # Load pretrained weights
        if params.finetune_now:
            pretrained_weights_path = f'{os.path.dirname(__file__)}/outputs/pretrained_weights_{my_model.__class__.__name__}.pth'
            my_model.load_state_dict(torch.load(pretrained_weights_path))

        # Start training
        my_model.train()   # Switch on the training flags
        for epoch in range(params.epochs):
            loss_sum = 0
            total_samples = 0
            batch_losses = []

            # Optimize over one epoch
            for batch_x, batch_y in train_loader:
                my_optimizer.zero_grad()
                output = my_model(batch_x.float())
                loss = params.loss_fn(output, batch_y)
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

            if params.verbose == 0:
                print(".", end="", flush=True)
            elif params.verbose == 1:
                if params.x_dev.shape[0] == 0 or params.y_dev.shape[0] == 0:
                    dev_loss = -1.0
                else:
                    eval_value = PytorchHelper.evaluate(params.x_dev, params.y_dev)
                    dev_loss = float(eval_value['test_loss'][-1])
                    my_model.train()  # Switch back to training mode after evaluation
                print(f"Epoch {epoch + 1}/{params.epochs} - " + 
                    f"Loss = {epoch_loss:.4f} - " + 
                    f"Dev_Loss = {dev_loss:.4f} - " + 
                    f"LR = {my_optimizer.param_groups[0]['lr']}", 
                    flush=True)
            elif params.verbose == 2:
                pass    # silent
            else:
                raise ValueError(f"Unexpected parameter value: verbose = {params.verbose}")

        # Save the trained weights
        if params.pretrain_now:
            pretrained_weights_path = f'{os.path.dirname(__file__)}/outputs/pretrained_weights_{my_model.__class__.__name__}.pth'
            torch.save(my_model.state_dict(), pretrained_weights_path)

        return history

    @staticmethod
    def smape(y_true, y_pred, dim=None):
        """
        Compute the Symmetric Mean Absolute Percentage Error (sMAPE).
        """

        numerator = torch.abs(y_pred - y_true)
        denominator = (torch.abs(y_true) + torch.abs(y_pred))
        eps = 1e-8 # To avoid division by zero
        smape_values = torch.mean(numerator / (denominator + eps), dim=dim) * 2 * 100
        return smape_values

    @staticmethod
    def evaluate(X_test, Y_test, results={}, deNormalize=False):

        # Initialize metrics
        loss_sum = 0
        smape_sum = 0
        total_samples = 0
        prediction = torch.zeros(size=(Y_test.size(0), 0, Y_test.size(2)))

        # Unnormalize the target variable, if wished.
        if deNormalize == True:
            assert params.model_adapter != None, "No model_adapter given."
            Y_test = params.model_adapter.deNormalizeY(Y_test)

        # Create DataLoader
        batch_size=256
        val_dataset = SequenceDataset(X_test, Y_test)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

        self.my_model.eval()       # Switch off the training flags
        with torch.no_grad():  # No gradient calculation
            for batch_x, batch_y in val_loader:

                # Predict
                output = self.my_model(batch_x.float())

                # Unnormalize the target variable, if wished.
                if deNormalize == True:
                    output = self.model_adapter.deNormalizeY(output)

                # Compute Metrics
                loss = self.loss_fn(output, batch_y.float())
                loss_sum += loss.item() * batch_x.size(0)
                smape_val = PytorchHelper.smape(batch_y.float(), output)
                smape_sum += smape_val * batch_x.size(0)
                total_samples += batch_x.size(0)

                prediction = torch.cat([prediction, output], dim=1)

        # Calculate average loss and sMAPE
        if total_samples > 0:
            test_loss = loss_sum / total_samples
            reference = float(torch.mean(Y_test))
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
        super(PositionalEncoding, self).__init__()

        pe = torch.zeros(timesteps, d_model)  # [timesteps, d_model]
        position = torch.arange(0, timesteps, dtype=torch.float).unsqueeze(1)  # [timesteps, 1]
        _2i = torch.arange(0, d_model, 2).float()
        div_term = torch.exp(_2i * (-math.log(10000.0) / d_model))  # [d_model/2]

        pe[:, 0::2] = torch.sin(position * div_term)  # Apply sin to even indices in the array
        pe[:, 1::2] = torch.cos(position * div_term)  # Apply cos to odd indices in the array

        pe = pe.unsqueeze(0)  # [1, timesteps, d_model]
        self.register_buffer('pe', pe)  # Save as a non-learnable buffer

    def forward(self, x):

        batches, timesteps, features = x.shape
        assert (self.pe.size(1) == timesteps), f"Expected timesteps: {self.pe.size(1)}, received timesteps: {timesteps}"
        assert (self.pe.size(2) == features), f"Expected features: {self.pe.size(2)}, received features: {features}"

        x = x + self.pe # Add positional encoding

        return x


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
        self.lr_switching_points = np.flip(np.linspace(1, 0, len(self.set_learning_rates), endpoint=False))

    # This function adjusts the learning rate based on the epoch
    def adjust_learning_rate(self, epoch):
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
