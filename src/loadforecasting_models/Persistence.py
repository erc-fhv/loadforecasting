import numpy as np

class Persistence():
    def __init__(self, model_size, num_of_features, modelAdapter):
        super(Persistence, self).__init__()
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

