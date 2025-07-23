import numpy as np

class Perfect():
    def __init__(self, model_size, num_of_features, modelAdapter):
        self.isPytorchModel = False
        self.modelAdapter = modelAdapter
    
    def forward(self, y):
        """
        Trivial 'model': 
        Just gets and returns the perfect profile (used for reference).
        """
        
        y_pred = y
        
        return y_pred
    
    def train_model(self, X_train, Y_train):
        pass
    
    def state_dict(self):
        state_dict = {}
        return state_dict

    def load_state_dict(self, state_dict):
        pass

