import pickle
import numpy as np

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
        