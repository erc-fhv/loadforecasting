from sklearn.neighbors import KNeighborsRegressor
import torch

class KNN():
    def __init__(self, model_size, num_of_features, modelAdapter):
        self.isPytorchModel = False
        self.X_train = None
        self.Y_train = None
        self.num_of_features = num_of_features
    
        # Configure the KNN model.
        # This was done empirically during developement 
        # phase to the values with best forecast error.
        # 
        self.k = 40
        self.weights = 'distance'
        
    # Store the training data
    #
    def train_model(self, X_train, Y_train):
        
        self.X_train = X_train
        self.Y_train = Y_train  
    
    # Given an input x, find the closest neighbors from the training data X_train
    # and return the corresponding Y_train.
    #
    def forward(self, x):
        
        # Fit the model with hourly training data
        #
        knn = KNeighborsRegressor(n_neighbors = self.k, weights=self.weights)
        batches, timesteps, num_features = self.X_train.shape
        X_hourly = self.X_train.view(batches * timesteps, num_features)
        Y_hourly = self.Y_train.view(batches * timesteps, 1)
        knn.fit(X_hourly, Y_hourly)
        
        # Prediction on new hourly data
        #
        batches, timesteps, num_features = x.shape
        x_hourly = x.view(batches * timesteps, num_features)
        y_pred = knn.predict(x_hourly)
        y_pred = torch.tensor(y_pred).view(batches, timesteps, 1)

        return y_pred
    
    def state_dict(self):
        state_dict = {}
        state_dict['X_train'] = self.X_train
        state_dict['Y_train'] = self.Y_train
        return state_dict

    def load_state_dict(self, state_dict):
        self.X_train = state_dict['X_train']
        self.Y_train = state_dict['Y_train']
