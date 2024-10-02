import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torch.autograd import Variable
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


class Model():
    def __init__(self, num_of_features, model_type= "xLSTM", lstmAdapter=None):
        
        if model_type not in globals():
            raise ValueError(f"Unexpected 'model_type' parameter received: {model_type}")
        else:
            # Instantiate model_type (if it is implemented)
            my_model_class = globals()[model_type]
            self.my_model = my_model_class(num_of_features, lstmAdapter=lstmAdapter)

    def forward(self, x):
        return self.my_model.forward(x)
    
    # Print the number of parameters of this model
    def get_nr_of_parameters(self, do_print=True):
        total_params = sum(p.numel() for p in self.my_model.parameters())
        
        if do_print == True:
            print(f"Total number of parameters: {total_params}")   
            
        return total_params
    
    def train_model(self, 
                    X_train, 
                    Y_train, 
                    X_val = np.array([]), 
                    Y_val = np.array([]),
                    pretraining_mode = 'load_weights',
                    epochs=100,
                    loss_fn= nn.MSELoss(), 
                    set_learning_rates=[0.01, 0.005, 0.001, 0.0005], 
                    batch_size=256,
                    verbose=0):
                
        # if type(self.my_model) == xLSTM:
        #     set_learning_rates=[0.01]
        #     epochs=20   # xLSTM need less epochs.
        
        if type(self.my_model) == KNN or type(self.my_model) == PersistencePrediction:
            self.my_model.train_model(X_train, Y_train)
            history = {}
            history['loss'] = [0.0]
        else:
            # Create DataLoader
            train_dataset = SequenceDataset(X_train, Y_train)
            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
            
            my_optimizer = optim.Adam(self.my_model.parameters(), lr=set_learning_rates[0])
            lr_scheduler = CustomLRScheduler(my_optimizer, set_learning_rates, epochs)

            # Initialize storage for training history
            history = {"loss": []}
            
            # Load pretrained weights
            if pretraining_mode == 'load_weights':
                pretrained_weights_path = f'outputs/pretrained_weights_{self.my_model.__class__.__name__}.pth'
                self.my_model.load_state_dict(torch.load(pretrained_weights_path))

            self.my_model.train()
            for epoch in range(epochs):
                total_loss = 0  
                total_samples = 0
                batch_losses = []  
                
                # Optimize over one epoch
                for batch_x, batch_y in train_loader:
                    my_optimizer.zero_grad()
                    output = self.my_model(batch_x.float())
                    loss = loss_fn(output, batch_y.float())
                    batch_losses.append(loss.item())
                    loss.backward()
                    lr_scheduler.adjust_learning_rate(epoch)
                    my_optimizer.step()                
                    total_loss += loss.item() * batch_x.size(0)
                    total_samples += batch_x.size(0)

                # Calculate average loss for the epoch
                epoch_loss = total_loss / total_samples
                history['loss'].append(epoch_loss)
                
                if verbose > 0:
                    if X_val.shape[0] != 0 and Y_val.shape[0] != 0:
                        eval_value = self.evaluate(X_val, Y_val)
                        val_loss = float(eval_value['val_loss'][-1])
                    else:
                        val_loss = -1.0
                    print(f"Epoch {epoch + 1}/{epochs} - " + 
                        f"Loss = {epoch_loss:.4f} - " + 
                        f"Val_Loss = {val_loss:.4f} - " + 
                        f"LR = {my_optimizer.param_groups[0]['lr']}", 
                        flush=True)
                    
            # Save the trained weights
            if pretraining_mode == 'store_weights':
                pretrained_weights_path = f'outputs/pretrained_weights_{self.my_model.__class__.__name__}.pth'
                torch.save(self.my_model.state_dict(), pretrained_weights_path)

        return history

    def predict(self, X, verbose=False):
        self.my_model.eval()  # Set the model to evaluation mode
        with torch.no_grad():
            X = Variable(torch.Tensor(X))
            output = self.my_model.forward(X)
        return output.numpy()

    def evaluate(self, X_val, Y_val, results={}, loss_fn=nn.MSELoss(), batch_size=256):
        
        if type(self.my_model) == KNN or type(self.my_model) == PersistencePrediction:
            output = self.my_model(torch.Tensor(X_val))
            assert output.shape == Y_val.shape, \
                f"Shape mismatch: got {output.shape}, expected {Y_val.shape})"
            loss = loss_fn(output, torch.Tensor(Y_val))
            results['val_loss'] = [loss]
        else:
            # Create DataLoader
            val_dataset = SequenceDataset(X_val, Y_val)
            val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

            self.my_model.eval()        
            total_loss = 0
            total_samples = 0
            
            with torch.no_grad():  # No gradient calculation
                for batch_x, batch_y in val_loader:
                                    
                    # Compute loss
                    output = self.my_model(batch_x.float())
                    loss = loss_fn(output, batch_y.float())
                    total_loss += loss.item() * batch_x.size(0)
                    total_samples += batch_x.size(0)

            # Calculate average loss
            if total_samples > 0:
                results['val_loss'] = [total_loss / total_samples]
            else:
                results['val_loss'] = [0.0]
        
        return results


class xLSTM(nn.Module):
    def __init__(self, num_of_features, output_dim = 1, lstmAdapter=None, hidden_dim=20):
        super(xLSTM, self).__init__()     

        # Embedding layer to transform input into model dimension
        # self.embedding = nn.Linear(num_of_features, hidden_dim)
        
        # Configuration for the xLSTM
        # molu: The config was left as provided by NX-AI, if not commented by me.
        self.cfg = xLSTMBlockStackConfig(
            mlstm_block=mLSTMBlockConfig(
                mlstm=mLSTMLayerConfig(
                    conv1d_kernel_size=4, qkv_proj_blocksize=4, num_heads=4
                )
            ),
            slstm_block=sLSTMBlockConfig(
                slstm=sLSTMLayerConfig(
                    backend="vanilla",  # For now run at CPU. Changed from "cuda".
                    num_heads=4,
                    conv1d_kernel_size=4,
                    bias_init="powerlaw_blockdependent",
                ),
                feedforward=FeedForwardConfig(proj_factor=1.3, act_fn="gelu"),
            ),
            context_length=48,  # Num of (hourly) input timesteps. Changed from "256".
            num_blocks=7,
            embedding_dim=num_of_features,  # Number of features. Changed from "128"
            slstm_at=[1],
        )
        self.xlstm_stack = xLSTMBlockStack(self.cfg)

        # Adding additional dense layers
        self.lambdaLayer = LambdaLayer(lambda x: x[:, -24:, :])  # Custom layer to slice last 24 timesteps
        self.activation = nn.ReLU()
        self.dense1 = nn.Linear(self.cfg.embedding_dim, 10)
        self.dense2 = nn.Linear(10, 10)
        self.output_dim = output_dim
        self.output_layer = nn.Linear(10, self.output_dim)
        
    def forward(self, x):
        # x = self.embedding(x)
        x = self.xlstm_stack(x)
        x = self.lambdaLayer(x)
        x = self.activation(self.dense1(x))
        x = self.activation(self.dense2(x))
        x = self.output_layer(x)
        return x


class LSTM(nn.Module):
    def __init__(self, num_of_features, output_dim = 1, lstmAdapter=None):
        super(LSTM, self).__init__()

        self.lstm1 = nn.LSTM(input_size=num_of_features, hidden_size=50, batch_first=True, bidirectional=True)
        self.lstm2 = nn.LSTM(input_size=100, hidden_size=30, batch_first=True, bidirectional=True)

        # Adding additional dense layers
        self.lambdaLayer = LambdaLayer(lambda x: x[:, -24:, :])  # Custom layer to slice last 24 timesteps
        self.activation = nn.ReLU()
        self.dense1 = nn.Linear(60, 10)
        self.dense2 = nn.Linear(10, 10)
        self.output_dim = output_dim
        self.output_layer = nn.Linear(10, self.output_dim)        
        
    def forward(self, x):
        x, _ = self.lstm1(x)
        x, _ = self.lstm2(x)
        x = self.lambdaLayer(x)
        x = self.activation(self.dense1(x))
        x = self.activation(self.dense2(x))
        x = self.output_layer(x)        
        return x


class Transformer(nn.Module):
    def __init__(self, num_of_features, output_dim=1, lstmAdapter=None, num_heads=4, num_layers=1, hidden_dim=20):
        super(Transformer, self).__init__()

        # Embedding layer to transform input into model dimension
        # self.embedding = nn.Linear(num_of_features, hidden_dim)
        
        # Transformer Encoder Layers
        encoder_layer = nn.TransformerEncoderLayer(d_model=num_of_features, nhead=num_heads, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Additional dense layers
        self.lambdaLayer = LambdaLayer(lambda x: x[:, -24:, :])  # Custom layer to slice last 24 timesteps
        self.activation = nn.ReLU()
        self.dense1 = nn.Linear(num_of_features, 10)
        self.dense2 = nn.Linear(10, 10)
        
        # Output layer
        self.output_dim = output_dim
        self.output_layer = nn.Linear(10, self.output_dim)

    def forward(self, x):
        # Pass input through embedding layer
        # x = self.embedding(x)
        
        # Pass input through transformer encoder
        x = self.transformer_encoder(x)
        
        # Slice the last 24 timesteps
        x = self.lambdaLayer(x)
        
        # Pass through dense layers with ReLU activation
        x = self.activation(self.dense1(x))
        x = self.activation(self.dense2(x))
        
        # Output layer
        x = self.output_layer(x)
        return x

class KNN(nn.Module):
    def __init__(self, num_of_features, output_dim=1, lstmAdapter=None):
        super(KNN, self).__init__()
        self.X_train = None
        self.Y_train = None
        self.num_of_features = num_of_features
    
    def forward(self, x):
        """
        Given an input x, find the closest neighbor from the training data X_train
        and return the corresponding Y_train.
        """
        
        batch_size = x.size(0)
        x_flat = x.view(batch_size, -1)  # Flatten input to (batch_size, 24*num_of_features)
        assert x_flat.shape == torch.Size([batch_size, 48 * self.num_of_features]), \
            f"Shape mismatch: got {x_flat.shape}, expected ({batch_size}, {48 * self.num_of_features})"
        distances = torch.cdist(x_flat, self.X_train)  # Compute pairwise distances
        assert distances.shape == torch.Size([batch_size, self.Y_train.shape[0]]), \
            f"Shape mismatch: got {distances.shape}, expected ({batch_size}, {self.Y_train.shape[0]})"
        nearest_neighbors = torch.argmin(distances, dim=1)  # Get nearest neighbor index
        assert nearest_neighbors.shape == torch.Size([batch_size]), \
            f"Shape mismatch: got {nearest_neighbors.shape}, expected ({batch_size})"
        y_pred = self.Y_train[nearest_neighbors]  # Fetch corresponding Y_train
        assert y_pred.shape == torch.Size([batch_size, 24, 1]), \
            f"Shape mismatch: got {y_pred.shape}, expected ({torch.Size([batch_size, 24, 1])})"
        return y_pred
    
    def train_model(self, X_train, Y_train):
        
        # Check if the input is a NumPy array and convert it to a torch.Tensor if necessary
        if isinstance(X_train, np.ndarray):
            X_train = torch.tensor(X_train, dtype=torch.float32)
        if isinstance(Y_train, np.ndarray):
            Y_train = torch.tensor(Y_train, dtype=torch.float32)

        # Store the training data as flattened tensors
        self.X_train = X_train.view(X_train.shape[0], -1)  # Flatten X_train
        self.Y_train = Y_train  # Y_train remains unchanged in shape (nr_of_days, 24, 1)



class PersistencePrediction(nn.Module):
    def __init__(self, num_of_features, output_dim=1, lstmAdapter=None):
        super(PersistencePrediction, self).__init__()
        self.num_of_features = num_of_features
        self.lstmAdapter = lstmAdapter
    
    def forward(self, x):
        """
        Estimate the upcoming profile accord to the last profiles.
        This is done with the un-normalized lagged power features.
        """
        
        batch_size = x.size(0)
        assert x.shape == (batch_size, 48, self.num_of_features), \
            f"Shape mismatch: got {x.shape}, expected ({batch_size}, 48, {self.num_of_features})" \
            + "Please check the the following feature dimension."
        
        x = self.lstmAdapter.deNormalizeX(x)    # de-normalize the lagged power feature
        y_pred = (x[:,-24:,11] + x[:,-24:,12] + x[:,-24:,13]) / 3.0
        y_pred = y_pred[:,:,np.newaxis]  # Get Shape: (batch_size, 24, 1)
        y_pred = self.lstmAdapter.normalizeY(y_pred)    # normalize the output, to compare it to other models.
        
        assert y_pred.shape == (batch_size, 24, 1), \
            f"Shape mismatch: got {y_pred.shape}, expected ({batch_size}, 24, 1)"
            
        return y_pred
    
    def train_model(self, X_train, Y_train):        
        pass

    
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


# Custom lambda layer
class LambdaLayer(nn.Module):
    def __init__(self, lambda_func):
        super(LambdaLayer, self).__init__()
        self.lambda_func = lambda_func

    def forward(self, x):
        return self.lambda_func(x)
