import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader, Dataset

class Model(nn.Module):
    def __init__(self, num_of_features, chosen_model='LSTM', ):
        super(Model, self).__init__()
        
        if chosen_model == 'LSTM':
            self.model_layers = nn.Sequential(
                nn.LSTM(input_size=num_of_features, hidden_size=10, batch_first=True, bidirectional=True),
                # nn.BatchNorm1d(num_of_features * 2),  # Adjust for bidirectional doubling of features
                nn.LSTM(input_size=20, hidden_size=30, batch_first=True, bidirectional=True),
                # nn.BatchNorm1d(num_of_features * 2),
                LambdaLayer(lambda x: x[:, -24:, :]),  # Custom layer to slice last 24 timesteps
                nn.Linear(in_features=30*2, out_features=10),  # Adjust input size for bidirectional LSTM
                nn.ReLU(),
                nn.Linear(10, 10),
                nn.ReLU(),
                nn.Linear(10, 1)
            )
            
        else:
            raise ValueError("Unknown model chosen.")
    
    def forward(self, x):
        i=0
        out, _ = self.model_layers[i](x)  # LSTM layers return both output and hidden state
        # i=i+1
        # out = self.model_layers[i](out)
        i=i+1
        out, _ = self.model_layers[i](out)  # second LSTM
        # i=i+1
        # out = self.model_layers[i](out)
        i=i+1
        out = self.model_layers[i](out)  # custom lambda slicing last 24 steps
        i=i+1
        # out = out.contiguous().view(out.size(0), -1)  # flatten before fully connected
        # i=i+1
        out = self.model_layers[i:](out)  # pass through the rest of the layers
        return out
        
    def train_model(self, 
                    X_train, 
                    Y_train, 
                    epochs=50,
                    loss_fn= nn.MSELoss(), 
                    # set_learning_rates=[0.01, 0.001, 0.0001], 
                    set_learning_rates=[0.001], 
                    batch_size=None, 
                    verbose=0
                    ):
        # Create DataLoader
        if batch_size is None:
            batch_size = X_train.shape[0]  # If batch_size is not provided, use the full batch
        train_dataset = SequenceDataset(X_train, Y_train)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        
        my_optimizer = optim.Adam(self.parameters(), lr=set_learning_rates[0])
        
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
                # output = output.view(-1, self.output_dim)  # Shape: (batch_size * seq_length, output_dim)
                # batch_y = batch_y.view(-1)  # Shape: (batch_size * seq_length)

                # Compute the loss
                loss = loss_fn(output, batch_y.float())
                batch_losses.append(loss.item())
                
                # Backpropagation
                loss.backward()
                my_optimizer.step()
                
                total_loss += loss.item()

            # Calculate average loss for the epoch
            epoch_loss = total_loss / len(train_loader)
            history['loss'].append(batch_losses)
            history['epoch_loss'].append(epoch_loss)
            
            if verbose > 0:
                print(f"Epoch {epoch + 1}/{epochs} - Loss: {epoch_loss:.4f}")
                
        return history
        
    def save_weights(self, filename):
        torch.save(self.state_dict(), filename)

    def load_weights(self, filename):
        self.load_state_dict(torch.load(filename))

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
                # output = output.view(-1, self.output_dim)
                # batch_y = batch_y.view(-1)
                
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
        return avg_loss # avg_metrics if metrics else avg_loss

# Custom lambda layer
class LambdaLayer(nn.Module):
    def __init__(self, lambda_func):
        super(LambdaLayer, self).__init__()
        self.lambda_func = lambda_func

    def forward(self, x):
        return self.lambda_func(x)

    
class SequenceDataset(Dataset):
    def __init__(self, X, Y):
        self.X = X
        self.Y = Y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]