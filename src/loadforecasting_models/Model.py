import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import importlib
import numpy as np

from loadforecasting_models.Perfect import Perfect

# This class wraps a single machine learning or benchmark model,
# and provides commonly used utilities and shared functionality to work with that model.
# 
class Model():
    
    def __init__(self, model_type, model_size, num_of_features, modelAdapter=None):
        
        # Import and instantiate the given model
        try:                  
            model = importlib.import_module(f"loadforecasting_models.{model_type}")
            my_model_class = getattr(model, model_type)
            self.my_model = my_model_class(model_size, num_of_features, modelAdapter)        
            
        except AttributeError as e:
            
            # No class with name model_type was found
            print(f"Unexpected 'model_type' parameter received: {model_type}")
            print(f"Detailed error description : {e}")
    
        self.loss_fn = nn.L1Loss()   # Optional: nn.L1Loss(), nn.MSELoss(), self.smape, ...
        self.modelAdapter = modelAdapter

    # Predict Y from the given X.
    #
    def predict(self, X):
        
        if self.my_model.isPytorchModel == True:            
            # Machine Learning Model            
            self.my_model.eval()  
            with torch.no_grad():
                output = self.my_model(X.float())
                
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
            train_dataset = SequenceDataset(X_train, Y_train)
            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)            
            my_optimizer = optim.Adam(self.my_model.parameters(), lr=set_learning_rates[0])
            lr_scheduler = CustomLRScheduler(my_optimizer, set_learning_rates, epochs)
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
                    loss = self.loss_fn(output, batch_y.float())
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
                pretrained_weights_path = f'{os.path.dirname(__file__)}/outputs/pretrained_weights_{self.my_model.__class__.__name__}.pth'
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
            if isinstance(self.my_model, Perfect):
                # The Perfect prediction model just gets and returns 
                # the perfect profile (used for reference).
                output = self.predict(Y_test)
            else:
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
            reference = float(torch.mean(Y_test))
            results['test_loss_relative'] = [100.0*loss.item()/reference]            
            results['predicted_profile'] = output
            
        else:   # Pytorch models            
            
            # Initialize metrics
            loss_sum = 0
            smape_sum = 0
            total_samples = 0
            prediction = torch.zeros(size=(Y_test.size(0), 0, Y_test.size(2)))
        
            # Unnormalize the target variable, if wished.
            if deNormalize == True:
                assert self.modelAdapter != None, "No modelAdapter given."
                Y_test = self.modelAdapter.deNormalizeY(Y_test)
            
            # Create DataLoader
            val_dataset = SequenceDataset(X_test, Y_test)
            val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

            self.my_model.eval()       # Switch off the training flags
            with torch.no_grad():  # No gradient calculation
                for batch_x, batch_y in val_loader:

                    # Predict
                    output = self.my_model(batch_x.float())
                    
                    # Unnormalize the target variable, if wished.
                    if deNormalize == True:
                        output = self.modelAdapter.deNormalizeY(output)
                    
                    # Compute Metrics
                    loss = self.loss_fn(output, batch_y.float())
                    loss_sum += loss.item() * batch_x.size(0)
                    smape_val = self.smape(batch_y.float(), output)
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
    
    # Print the number of parameters of this model
    def get_nr_of_parameters(self, do_print=True):
        total_params = sum(p.numel() for p in self.my_model.parameters())
        
        if do_print == True:
            print(f"Total number of parameters: {total_params}")
            
        return total_params


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
