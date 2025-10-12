import numpy as np

class Persistence():
    def __init__(self, model_size, num_of_features, modelAdapter):
        self.isPytorchModel = False
        self.modelAdapter = modelAdapter
    
    def forward(self, x):
        """
        Upcoming load profile = load profile 7 days ago.
        """

        x = self.modelAdapter.deNormalizeX(x)    # de-normalize all inputs

        # Take the chosen lagged loads as predictions
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
    
    def state_dict(self):
        state_dict = {}
        return state_dict

    def load_state_dict(self, state_dict):
        pass

