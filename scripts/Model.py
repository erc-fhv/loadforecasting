import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torch.autograd import Variable
import numpy as np
import scripts.Simulation_config as config
import pickle
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
    def __init__(self, model_type, model_size, num_of_features, modelAdapter=None):
        
        if model_type not in globals():
            # No class with name model_type is implemented below
            raise ValueError(f"Unexpected 'model_type' parameter received: {model_type}")
        else:
            # Instantiate the model
            my_model_class = globals()[model_type]        
            self.my_model = my_model_class(model_size, num_of_features, modelAdapter)
        
        # Member Variables
        self.loss_fn = nn.L1Loss()   # Optional: nn.L1Loss(), nn.MSE(), self.smape, ...
        self.modelAdapter = modelAdapter

    # Predict Y from the given X.
    #
    def predict(self, X):
        
        if self.my_model.isPytorchModel == True:            
            # Machine Learning Model            
            self.my_model.eval()  
            with torch.no_grad():
                output = self.my_model.forward(X.float())
                
        else:
            # Simple models
            output = self.my_model.forward(X)
            
        return output
    
    def train_model(self,
                    X_train,
                    Y_train,
                    X_dev = torch.Tensor([]),
                    Y_dev = torch.Tensor([]),
                    pretrain_now = False,
                    finetune_now = True,
                    epochs=100,
                    set_learning_rates=[0.01, 0.005, 0.001, 0.0005],
                    batch_size=256,
                    verbose=0):
        
        if self.my_model.isPytorchModel == False:   # Simple, parameter free models    
            
            history = {}
            history['loss'] = [0.0]            
            if pretrain_now:
                # No pretraining possible for these parameter-free models
                pass    
            else:
                self.my_model.train_model(X_train, Y_train)
        
        else:   # Pytorch models            
            
            # Prepare Optimization
            train_dataset = SequenceDataset(X_train.float(), Y_train)
            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)            
            my_optimizer = optim.Adam(self.my_model.parameters(), lr=set_learning_rates[0])
            lr_scheduler = CustomLRScheduler(my_optimizer, set_learning_rates, epochs)
            history = {"loss": []}
            
            # Load pretrained weights
            if finetune_now:
                pretrained_weights_path = f'scripts/outputs/pretrained_weights_{self.my_model.__class__.__name__}.pth'
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
                    output = self.my_model(batch_x)
                    loss = self.loss_fn(output, batch_y)
                    batch_losses.append(loss)
                    loss.backward()
                    my_optimizer.step()
                    loss_sum += loss * batch_x.size(0)
                    total_samples += batch_x.size(0)
                
                # Adjust learning rate once per epoch
                lr_scheduler.adjust_learning_rate(epoch)
                
                # Calculate average loss for the epoch
                epoch_loss = loss_sum / total_samples
                history['loss'].append(epoch_loss)
                
                if verbose > 0:
                    if X_dev.shape[0] == 0 or Y_dev.shape[0] == 0:
                        dev_loss = -1.0
                    else:
                        eval_value = self.evaluate(X_dev, Y_dev)
                        dev_loss = float(eval_value['test_loss'][-1])
                        self.my_model.train()  # Switch back to training mode after evaluation
                    print(f"Epoch {epoch + 1}/{epochs} - " + 
                        f"Loss = {epoch_loss:.4f} - " + 
                        f"Dev_Loss = {dev_loss:.4f} - " + 
                        f"LR = {my_optimizer.param_groups[0]['lr']}", 
                        flush=True)
                else:
                    print(".", end="", flush=True)
                
            # Save the trained weights
            if pretrain_now:
                pretrained_weights_path = f'scripts/outputs/pretrained_weights_{self.my_model.__class__.__name__}.pth'
                torch.save(self.my_model.state_dict(), pretrained_weights_path)

        return history
    
    # Compute the Symmetric Mean Absolute Percentage Error (sMAPE).
    #
    def smape(self, y_true, y_pred, dim=None):
        numerator = torch.abs(y_pred - y_true)
        denominator = (torch.abs(y_true) + torch.abs(y_pred))
        eps = 1e-8 # To avoid division by zero
        smape_values = torch.mean(numerator / (denominator + eps), dim=dim) * 2 * 100
        return smape_values

    def evaluate(self, X_test, Y_test, results={}, deNormalize=False, batch_size=256):
        
        if self.my_model.isPytorchModel == False:   # Simple, parameter free models    
            
            # Predict
            output = self.predict(X_test)
            assert output.shape == Y_test.shape, \
                f"Shape mismatch: got {output.shape}, expected {Y_test.shape})"
            
            # Unnormalize the target variable, if wished.
            if deNormalize == True:
                assert self.modelAdapter != None, "No modelAdapter given."
                Y_test = self.modelAdapter.deNormalizeY(Y_test)
                output = self.modelAdapter.deNormalizeY(output)
            
            # Compute Loss
            loss = self.loss_fn(output, Y_test)
            results['test_loss'] = [loss.item()]
            metric = self.smape(output, Y_test)
            results['test_sMAPE'] = [metric]
            
        else:   # Pytorch models            
            
            # Initialize metrics
            loss_sum = 0
            smape_sum = 0
            total_samples = 0
        
            # Unnormalize the target variable, if wished.
            if deNormalize == True:
                assert self.modelAdapter != None, "No modelAdapter given."
                Y_test = self.modelAdapter.deNormalizeY(Y_test)
            
            # Create DataLoader
            val_dataset = SequenceDataset(X_test.float(), Y_test)
            val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

            self.my_model.eval()       # Switch off the training flags
            with torch.no_grad():  # No gradient calculation
                for batch_x, batch_y in val_loader:

                    # Predict
                    output = self.my_model(batch_x)
                    
                    # Unnormalize the target variable, if wished.
                    if deNormalize == True:
                        output = self.modelAdapter.deNormalizeY(output)
                    
                    # Compute Metrics
                    loss = self.loss_fn(output, batch_y)
                    loss_sum += loss.item() * batch_x.size(0)
                    smape_val = self.smape(batch_y, output)
                    smape_sum += smape_val * batch_x.size(0)
                    total_samples += batch_x.size(0)

            # Calculate average loss and sMAPE
            if total_samples > 0:
                results['test_loss'] = [loss_sum / total_samples]
                results['test_sMAPE'] = [smape_sum / total_samples]
            else:
                results['test_loss'] = [0.0]
                results['test_sMAPE'] = [0.0]
        
        return results
    
    # Print the number of parameters of this model
    def get_nr_of_parameters(self, do_print=True):
        total_params = sum(p.numel() for p in self.my_model.parameters())
        
        if do_print == True:
            print(f"Total number of parameters: {total_params}")   
            
        return total_params


class xLSTM(nn.Module):
    def __init__(self, model_size, num_of_features, modelAdapter):
        super(xLSTM, self).__init__()
        self.isPytorchModel = True
        self.forecast_horizon = 24
        
        # The following xLSTM config variables are as as provided by NX-AI.
        conv1d_kernel_size=4
        num_heads=4
        qkv_proj_blocksize=4
        proj_factor=1.3
        num_blocks=7
        
        # Finetune the config variables
        if model_size == "SMALL":
            qkv_proj_blocksize=4
            num_blocks=3
            d_model=20
        elif model_size == "MEDIUM":
            num_blocks=4
            d_model=20
        elif model_size == "LARGE":
            num_blocks=5
            d_model=40
        else:
            assert False, f"Unimplemented model_size parameter given: {model_size}"
        
        # Configuration for the xLSTM Block
        self.cfg = xLSTMBlockStackConfig(
            mlstm_block=mLSTMBlockConfig(
                mlstm=mLSTMLayerConfig(
                    conv1d_kernel_size=conv1d_kernel_size, qkv_proj_blocksize=qkv_proj_blocksize, num_heads=num_heads
                )
            ),
            slstm_block=sLSTMBlockConfig(
                slstm=sLSTMLayerConfig(
                    backend="vanilla",  # For now run at CPU. Changed from "cuda".
                    num_heads=num_heads,
                    conv1d_kernel_size=conv1d_kernel_size,
                    bias_init="powerlaw_blockdependent",
                ),
                feedforward=FeedForwardConfig(proj_factor=proj_factor, act_fn="gelu"),
            ),
            context_length=600,
            num_blocks=num_blocks,
            embedding_dim=d_model,
            slstm_at=[1,],
        )
        self.xlstm_stack = xLSTMBlockStack(self.cfg)

        # Adding none-xlstm layers
        self.input_projection = nn.Linear(num_of_features, d_model)
        self.activation = nn.ReLU()
        self.dense1 = nn.Linear(d_model, 30)
        self.dense2 = nn.Linear(30, 20)
        self.output_layer = nn.Linear(20, 1)
        
    def forward(self, x):
        x = self.input_projection(x)
        x = self.xlstm_stack(x)
        x = self.activation(self.dense1(x))
        x = self.activation(self.dense2(x))
        x = self.output_layer(x)
        return x


class LSTM(nn.Module):
    def __init__(self, model_size, num_of_features, modelAdapter):
        super(LSTM, self).__init__()
        self.isPytorchModel = True
        self.forecast_horizon = 24
        
        if model_size == "SMALL":
            self.lstm1 = nn.LSTM(input_size=num_of_features, hidden_size=28, batch_first=True, bidirectional=True)
            self.lstm2 = nn.LSTM(input_size=56, hidden_size=10, batch_first=True, bidirectional=True)
        elif model_size == "MEDIUM":
            self.lstm1 = nn.LSTM(input_size=num_of_features, hidden_size=55, batch_first=True, bidirectional=True)
            self.lstm2 = nn.LSTM(input_size=110, hidden_size=10, batch_first=True, bidirectional=True)
        elif model_size == "LARGE":
            self.lstm1 = nn.LSTM(input_size=num_of_features, hidden_size=95, batch_first=True, bidirectional=True)
            self.lstm2 = nn.LSTM(input_size=190, hidden_size=10, batch_first=True, bidirectional=True)
        else:
            assert False, f"Unimplemented model_size parameter given: {model_size}"

        # Adding additional dense layers
        self.activation = nn.ReLU()
        self.dense1 = nn.Linear(20, 30)
        self.dense2 = nn.Linear(30, 20)
        self.output_layer = nn.Linear(20, 1)          
        
    def forward(self, x):
        x, _ = self.lstm1(x)
        x, _ = self.lstm2(x)
        x = self.activation(self.dense1(x))
        x = self.activation(self.dense2(x))
        x = self.output_layer(x)
        return x


class Transformer(nn.Module):
    def __init__(self, model_size, num_of_features, modelAdapter):
        super(Transformer, self).__init__()
        self.isPytorchModel = True
        self.num_of_features = num_of_features
        self.forecast_horizon = 24
        
        if model_size == "SMALL":
            num_heads=4
            num_layers=1
            dim_feedforward=512
            d_model=20
        elif model_size == "MEDIUM":
            num_heads=4
            num_layers=1
            dim_feedforward=512
            d_model=40
        elif model_size == "LARGE":
            num_heads=10
            num_layers=2
            dim_feedforward=600
            d_model=40
        else:
            assert False, f"Unimplemented model_size parameter given: {model_size}"

        # Transformer Encoder Layers
        self.input_projection = nn.Linear(num_of_features, d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=num_heads, 
                                                   dim_feedforward=dim_feedforward, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.activation = nn.ReLU()
        self.dense1 = nn.Linear(d_model, 30)
        self.dense2 = nn.Linear(30, 20)
        self.output_layer = nn.Linear(20, 1)    

    def forward(self, x):
        x = self.input_projection(x)
        x = self.transformer(x)
        x = self.activation(self.dense1(x))
        x = self.activation(self.dense2(x))
        x = self.output_layer(x)
        return x


class KNN():
    def __init__(self, model_size, num_of_features, modelAdapter):
        super(KNN, self).__init__()
        self.isPytorchModel = False
        self.X_train = None
        self.Y_train = None
        self.num_of_features = num_of_features
    
    def train_model(self, X_train, Y_train):

        # Store the training data as flattened tensors
        self.X_train = X_train.view(X_train.shape[0], -1)  # Flatten X_train from (nr_of_batches, timesteps, features) to (nr_of_batches, timesteps * features)
        self.Y_train = Y_train  # Y_train remains unchanged in shape (nr_of_days, timesteps, 1)
    
    # Given an input x, find the closest neighbor from the training data X_train
    # and return the corresponding Y_train.
    #
    def forward(self, x):
        
        batch_size = x.size(0)
        nr_of_timesteps = x.size(1)
        x_flat = x.view(batch_size, -1)  # Flatten input to (batch_size, 24*num_of_features)
        assert x_flat.shape == torch.Size([batch_size, nr_of_timesteps * self.num_of_features]), \
            f"Shape mismatch: got {x_flat.shape}, expected ({batch_size}, {nr_of_timesteps * self.num_of_features})"
        distances = torch.cdist(x_flat, self.X_train)  # Compute pairwise distances
        assert distances.shape == torch.Size([batch_size, self.X_train.shape[0]]), \
            f"Shape mismatch: got {distances.shape}, expected ({batch_size}, {self.Y_train.shape[0]})"
        nearest_neighbors = torch.argmin(distances, dim=1)  # Get nearest neighbor index
        assert nearest_neighbors.shape == torch.Size([batch_size]), \
            f"Shape mismatch: got {nearest_neighbors.shape}, expected ({batch_size})"
        y_pred = self.Y_train[nearest_neighbors]  # Fetch corresponding Y_train
        assert y_pred.shape == torch.Size([batch_size, 24, 1]), \
            f"Shape mismatch: got {y_pred.shape}, expected ({torch.Size([batch_size, 24, 1])})"
        return y_pred
    
    def state_dict(self):
        state_dict = {}
        state_dict['X_train'] = self.X_train
        state_dict['Y_train'] = self.Y_train
        return state_dict

    def load_state_dict(self, state_dict):
        self.X_train = state_dict['X_train']
        self.Y_train = state_dict['Y_train']


# Prediction with open-access synthetic load profiles.
#
class SyntheticLoadProfile():
    def __init__(self, model_size, num_of_features, modelAdapter):
        super(SyntheticLoadProfile, self).__init__()
        self.isPytorchModel = False

        # Load standard loadprofile
        pretraining_filename = 'scripts/outputs/standard_loadprofile.pkl'
        with open(pretraining_filename, 'rb') as f:
            (_, Y_standardload, _) = pickle.load(f)
            self.Y_standardload_test = Y_standardload['test']
    
    def train_model(self, X_train, Y_train):
        pass
    
    def forward(self, x):
        
        # Predict the next days, using the standard profile.
        # In this project the predicted testset has the same shape as standard-load-profile.
        #
        nr_of_days = x.shape[0]
        if nr_of_days == self.Y_standardload_test.shape[0]:
            y_pred = self.Y_standardload_test
        else:
            y_pred = np.zeros(shape=(nr_of_days, *self.Y_standardload_test.shape[1:]))
        
        return y_pred
    
    def state_dict(self):
        state_dict = {}
        state_dict['Y_standardload_test'] = self.Y_standardload_test
        return state_dict

    def load_state_dict(self, state_dict):
        self.Y_standardload_test = state_dict['Y_standardload_test']


class PersistencePrediction():
    def __init__(self, model_size, num_of_features, modelAdapter):
        super(PersistencePrediction, self).__init__()
        self.isPytorchModel = False
        self.modelAdapter = modelAdapter
    
    def forward(self, x):
        """
        Upcoming load profile = load profile 7 days ago.
        Assumption: The training load profile immediately precedes the given test load profile (to ensure accurate 
        prediction of the initial days in the test set).
        """

        x = self.modelAdapter.deNormalizeX(x)    # de-normalize especially the lagged power feature

        # Take the latest available lagged loads as predictions
        # 
        lagged_load_feature = 11
        y_pred = x[:,:, lagged_load_feature]
        
        # Add axis and normalize y_pred again, to compare it to other models.
        #
        y_pred = y_pred[:,:,np.newaxis]
        y_pred = self.modelAdapter.normalizeY(y_pred)
        assert y_pred.shape == (x.size(0), 24, 1), \
            f"Shape mismatch: got {y_pred.shape}, expected ({x.size(0)}, 24, 1)"
        
        return y_pred
    
    def train_model(self, X_train, Y_train):
        self.Y_train = Y_train
    
    def state_dict(self):
        state_dict = {}
        state_dict['Y_train'] = self.Y_train
        return state_dict

    def load_state_dict(self, state_dict):
        self.Y_train = state_dict['Y_train']


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
