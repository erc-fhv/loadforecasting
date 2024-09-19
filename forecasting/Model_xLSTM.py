import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, StackDataset, Dataset
from torch.autograd import Variable
import torch.optim.lr_scheduler as lr_scheduler
import numpy as np

from xlstm import (
    xLSTMBlockStack,
    xLSTMBlockStackConfig,
    mLSTMBlockConfig,
    mLSTMLayerConfig,
    sLSTMBlockConfig,
    sLSTMLayerConfig,
    FeedForwardConfig,
)    

class Model(nn.Module):
    def __init__(self, num_of_features):
        super(Model, self).__init__()        

        # Configuration for xLSTMBlockStack
        self.cfg = xLSTMBlockStackConfig(
            mlstm_block=mLSTMBlockConfig(
                mlstm=mLSTMLayerConfig(
                    conv1d_kernel_size=4, qkv_proj_blocksize=4, num_heads=2
                )
            ),
            slstm_block=sLSTMBlockConfig(
                slstm=sLSTMLayerConfig(
                    backend="vanilla",
                    num_heads=2,
                    conv1d_kernel_size=4,
                    bias_init="powerlaw_blockdependent",
                ),
                feedforward=FeedForwardConfig(proj_factor=1.3, act_fn="gelu"),
            ),
            context_length=48,
            num_blocks=7,
            embedding_dim=18,  # Input and Output dimension of the xLSTMBlockStack
            slstm_at=[1],
        )
        
        self.output_dim = 1
        self.xlstm_stack = xLSTMBlockStack(self.cfg)
        self.dense_layer = nn.Linear(self.cfg.embedding_dim, self.output_dim)

    def forward(self, x):
        x = self.xlstm_stack(x)
        x = self.dense_layer(x)
        x[:, :24, :] = 0.0  # Take only the last 24h
        return x

    def train_model(self, 
                    X_train, 
                    Y_train, 
                    epochs=100,
                    loss_fn= nn.MSELoss(), 
                    set_learning_rates=[0.05, 0.01, 0.005, 0.001], 
                    batch_size=None, 
                    verbose=0):
        # Create DataLoader
        if batch_size is None:
            batch_size = X_train.shape[0]  # If batch_size is not provided, use the full batch
        train_dataset = SequenceDataset(X_train, Y_train)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        
        my_optimizer = optim.Adam(self.parameters(), lr=set_learning_rates[0])
        lr_scheduler = CustomLRScheduler(my_optimizer, set_learning_rates, epochs)

        # Initialize storage for training history
        history = {
            "loss": [],
            "epoch_loss": []
        }
        
        self.train()
        for epoch in range(epochs):
            total_loss = 0
            batch_losses = []
            for batch_x, batch_y in train_loader:
                my_optimizer.zero_grad()
                output = self(batch_x.float())
                
                # Reshape output and target
                output = output.view(-1, self.output_dim)  # Shape: (batch_size * seq_length, output_dim)
                batch_y = batch_y.view(-1)  # Shape: (batch_size * seq_length)

                # Compute the loss
                loss = loss_fn(output, batch_y.float())
                batch_losses.append(loss.item())
                
                # Backpropagation
                loss.backward()
                lr_scheduler.adjust_learning_rate(epoch)
                my_optimizer.step()
                
                total_loss += loss.item()

            # Calculate average loss for the epoch
            epoch_loss = total_loss / len(train_loader)
            history['loss'].append(batch_losses)
            history['epoch_loss'].append(epoch_loss)
            
            if verbose > 0:
                print(f"Epoch {epoch + 1}/{epochs} - Loss: {epoch_loss:.4f} - LR = {my_optimizer.param_groups[0]['lr']}")
                
        return history

    def predict(self, X, verbose=False):
        self.eval()  # Set the model to evaluation mode
        with torch.no_grad():
            X = Variable(torch.Tensor(X))
            output = self.forward(X)
        return output.numpy()

    def evaluate(self, X_val, Y_val, loss_fn=nn.MSELoss(), metrics=None, batch_size=None):
        # Create DataLoader
        if batch_size is None:
            batch_size = X_val.shape[0]  # If batch_size is not provided, use the full batch
        val_dataset = SequenceDataset(X_val, Y_val)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

        # Switch to evaluation mode
        self.eval()
        
        total_loss = 0
        metric_results = {metric: 0 for metric in metrics} if metrics else {}

        with torch.no_grad():  # No gradient calculation
            for batch_x, batch_y in val_loader:
                output = self(batch_x.float())
                
                # Reshape output and target
                output = output.view(-1, self.output_dim)
                batch_y = batch_y.view(-1)
                
                # Compute loss
                loss = loss_fn(output, batch_y.float())
                total_loss += loss.item()

                # Compute metrics if provided
                if metrics:
                    for metric_name, metric_fn in metrics.items():
                        metric_results[metric_name] += metric_fn(output, batch_y).item()

        # Calculate average loss and metrics over the validation set
        avg_loss = total_loss / len(val_loader)
        avg_metrics = {metric_name: total / len(val_loader) for metric_name, total in metric_results.items()}
        
        # Return loss and metrics (similar to Keras evaluate)
        return avg_loss, avg_metrics if metrics else avg_loss


    
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
        
        # Compute lr_switching_points
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
