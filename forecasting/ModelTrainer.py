import numpy as np
from tqdm import tqdm
import pickle
import multiprocessing as mp
import controller.model.Model as model
import importlib
import tensorflow as tf

# Only use the chosen features
def select_given_features(X, all_input_features, given_features):

    included_indices = []
    for feature in given_features:
        index = np.argwhere(np.array(all_input_features) == feature)
        included_indices.extend(index.flatten())
    excluded_indices = [i for i in range(X['train'].shape[2]) if i not in included_indices]
    
    X_mod = {}
    X_mod['train'] = np.copy(np.delete(X['train'], excluded_indices, axis=2))
    X_mod['test'] = np.copy(np.delete(X['test'], excluded_indices, axis=2))
    X_mod['dev'] = np.copy(np.delete(X['dev'], excluded_indices, axis=2))
    X_mod['all'] = np.copy(np.delete(X['all'], excluded_indices, axis=2))

    # If all features were deleted: Add a one-vector
    if X_mod['train'].shape[2] == 0:
        X_mod['train'] = np.ones((*X_mod['train'].shape[:2], 3))
        X_mod['test'] = np.ones((*X_mod['test'].shape[:2], 3))
        X_mod['dev'] = np.ones((*X_mod['dev'].shape[:2], 3))
        X_mod['all'] = np.ones((*X_mod['all'].shape[:2], 3))

    return X_mod

# Get X and Y accord to the 'test_config'
def get_input_data(community_id, all_input_features, given_features, model_inputs_file):

    # Load a new powerprofile
    filename = model_inputs_file + '/file_' + str(community_id) + '.pkl' 
    with open(filename, 'rb') as f:
        (X, Y, lstmAdapter) = pickle.load(f)

    assert X['train'].shape[2] == len(all_input_features), \
        f"Undesired number of features included! {X['train'].shape[2]} != {len(all_input_features)}" 

    X_mod = select_given_features(X, all_input_features, given_features)

    return X_mod, Y, lstmAdapter


def optimize_model(test_config, test_id, 
                   chosen_model, all_input_features,
                   model_inputs_file):

    community_id = test_config[test_id][0]
    given_features = test_config[test_id][1]
    X_mod, Y, _ = get_input_data(community_id, all_input_features, given_features, model_inputs_file)

    # Train the model
    optimizer = model.Optimizer(maxEpochs=100, set_learning_rates=[0.015, 0.005, 0.003, 0.002, 0.001, 0.001, 0.001])
    myReducedModel = model.Model(optimizer, reg_strength=0.00, chosen_model=chosen_model)
    myReducedModel.compile()
    history = myReducedModel.model.fit(
        x=X_mod['train'], y=Y['train'],
        validation_data=(X_mod['test'], Y['test']),
        batch_size=128, 
        epochs=optimizer.maxEpochs,
        shuffle=True,
        verbose=2,  # 0=low, 2=high
        callbacks=optimizer.get_all_callbacks()
    )
    
    # Return the results
    return (test_id, history, myReducedModel)

class ModelTrainer:

    def __init__(
                self, 
                chosen_model,
                all_input_features,
                test_config,
                model_inputs_file,
                use_multiprocessing = True,
                ):
        
        self.use_multiprocessing = use_multiprocessing
        self.all_input_features = all_input_features
        self.test_config = test_config
        self.chosen_model = chosen_model
        self.model_inputs_file = model_inputs_file

        # for reproducibility
        np.random.seed(0)
        tf.random.set_seed(0)

        importlib.reload(model)

    def optimize_model_wrapper(self, args):
        return optimize_model(*args)

    def run(self):

        if self.use_multiprocessing:
            with mp.Pool() as pool:
                results = list(
                    tqdm(
                        pool.imap(self.optimize_model_wrapper, [(self.test_config, test_id, self.chosen_model,
                                                                self.all_input_features, self.model_inputs_file)
                                                               for test_id in range(len(self.test_config))]),
                        total=len(self.test_config),
                    )
                )
                pool.close()
                pool.join()

        else:   # Single Process
            results = []
            for test_id in tqdm(range(len(self.test_config))):
                result = optimize_model(self.test_config, test_id, self.chosen_model, 
                                        self.all_input_features, self.model_inputs_file)
                results.append(result)

        # create a dict out of the results
        test_ids, histories, returnedModels = zip(*results)
        train_histories = dict(sorted(({test_id: history for test_id, history in zip(test_ids, histories)}.items())))
        myModels = dict(sorted({test_id: returnedModel for test_id, returnedModel in zip(test_ids, returnedModels)}.items()))                               

        return train_histories, myModels
    
