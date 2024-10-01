import numpy as np
from tqdm import tqdm
import pickle
import multiprocessing as mp
import Model as model
import importlib

def optimize_model(model_type, load_profile, pretraining_mode):

    # Load a new powerprofile
    with open(load_profile, 'rb') as f:
        (X, Y, lstmAdapter) = pickle.load(f)

    # Train the model
    num_of_features = X['train'].shape[2]
    myModel = model.Model(num_of_features, model_type, lstmAdapter=lstmAdapter)
    history = myModel.train_model(X['train'], Y['train'], pretraining_mode, verbose=1)
    history = myModel.evaluate(X['test'], Y['test'], history)
    
    # Return the results
    return (model_type, load_profile, history, myModel)

class ModelTrainer:

    def __init__(self, test_config, use_multiprocessing = True):        
        self.use_multiprocessing = use_multiprocessing
        self.test_config = test_config
        importlib.reload(model)

    def optimize_model_wrapper(self, args):
        return optimize_model(*args)

    def run(self):

        if self.use_multiprocessing:
            with mp.Pool() as pool:
                results = list(
                    tqdm(
                        pool.imap(self.optimize_model_wrapper, 
                                  [(model_type, load_profile, self.test_config['pretraining_mode'])
                                  for model_type in self.test_config['models']
                                  for load_profile in self.test_config['data']]),
                        total=len(self.test_config['models'])*len(self.test_config['data']),
                    )
                )
                # pool.close()
                # pool.join()

        else:   # Single Process
            results = []
            for model_type in tqdm(self.test_config['models']):
                for load_profile in tqdm(self.test_config['data']):
                    result = optimize_model(model_type, load_profile, self.test_config['pretraining_mode'])
                    results.append(result)

        # create a dict out of the results
        model_types, load_profiles, histories, returnedModels = zip(*results)
        train_histories = dict(sorted(({(model_type, load_profil): history for model_type, load_profil, history in zip( model_types, load_profiles, histories)}.items())))
        myModels = dict(sorted({(model_type, load_profil): returnedModel for model_type, load_profil, returnedModel in zip( model_types, load_profiles, returnedModels)}.items()))                               

        return train_histories, myModels
