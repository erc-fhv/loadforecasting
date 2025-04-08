import torch

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
        