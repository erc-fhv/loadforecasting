
import json
from datetime import datetime
import torch
import pickle
import pytz
import scripts.Simulation_config as config
import scripts.Model
from collections import defaultdict
from itertools import islice
from pprint import pprint
import numpy as np
import plotly.express as px
import pandas as pd

# Persist dicts with complex keys.
# The dict keys are converted from multi-class into json format.
#
class Serialize:

    # Use torch to save a dictionary with training results to disc.
    # Especially useful for torch models.
    #
    @staticmethod
    def store_results_with_torch(all_trained_models):      
        
        # Squeeze the dict keys
        all_trained_models = Serialize.get_serialized_dicts(all_trained_models, isModel = True)

        # Save the total model with torch.save
        timestamp = Serialize.get_act_timestamp()
        torch.save(all_trained_models, f'scripts/outputs/all_trained_models{timestamp}.pth')
        torch.save(all_trained_models, f'scripts/outputs/all_trained_models.pth')

    # Use pickle to save a dictionary with training results to disc.
    #
    @staticmethod
    def store_results_with_pickle(all_train_histories):
        
        # Squeeze the dict keys
        all_train_histories = Serialize.get_serialized_dicts(all_train_histories, isModel = False)
        
        # Store the variables in a persistent files with the timestamp
        timestamp = Serialize.get_act_timestamp()
        with open(f"scripts/outputs/all_train_histories{timestamp}.pkl", 'wb') as f:
            pickle.dump(all_train_histories, f)
        with open(f"scripts/outputs/all_train_histories.pkl", 'wb') as f:
            pickle.dump(all_train_histories, f)

    # Serialize the given dicts.
    #
    @staticmethod
    def get_serialized_dicts(dict_with_komplex_keys, isModel):
        serialized_models = {}
        for key, data in dict_with_komplex_keys.items():
            serialized_key = Serialize.serialize_complex_key(key)
            if isModel:
                # Not the whole models shall be saved, but only its parameters.
                model = data
                serialized_models[serialized_key] = model.state_dict()
            else:
                serialized_models[serialized_key] = data

        return serialized_models
    
    # Change the complex key of the given train history dictionary to a simple json key.
    #
    @staticmethod
    def serialize_complex_key(complex_key):

        # Serialize the config object by iterating over the fields and handling each type appropriately
        config_serialized = {field: Serialize.serialize_value(getattr(complex_key[2], field)) for field in complex_key[2]._fields}

        # Convert the whole key to a json
        string_key = json.dumps({
            "model_type": complex_key[0],
            "load_profile": complex_key[1],
            "config": config_serialized
        })
        return string_key

    # Helper function for the serialization. Return the value of the named tuple field.
    #
    @staticmethod
    def serialize_value(value):
        if isinstance(value, (bool, int, str, tuple)):  # Basic types
            return value
        elif isinstance(value, type):
            return value.__dict__  # If the object is a class, etc.
        else:
            assert False, "Unimmplemented variable occurs in the config."

    # Return a current timestamp.
    #
    @staticmethod
    def get_act_timestamp(tz='Europe/Vienna'):
        return datetime.now(pytz.timezone(tz)).strftime("_%Y%m%d_%H%M")

# Get the trained models and train history from disc.
#
class Deserialize:

    # Get the training histories from disc and deserialize/unpack them.
    #
    @staticmethod
    def get_training_histories(path):
        
        with open(path, 'rb') as file:
            train_histories_serialized = pickle.load(file)
        
        train_histories = {}        
        for serialized_key, history in train_histories_serialized.items():
            deserialize_key = Deserialize.deserialize_key(serialized_key)            
            train_histories[deserialize_key] = history
        
        return train_histories
    
    # Get the trained models from disc and deserialize/unpack them.
    #
    @staticmethod
    def get_trained_model(path_to_trained_parameters, model_type, test_profile, chosenConfig, num_of_features):
        
        serialized_dict = torch.load(path_to_trained_parameters)
        
        for serialized_key, state_dict in serialized_dict.items():
            deserialize_key = Deserialize.deserialize_key(serialized_key)
            if model_type == deserialize_key[0] and \
                test_profile == deserialize_key[1] and \
                chosenConfig == deserialize_key[2]:
                model = scripts.Model.Model(model_type=deserialize_key[0], 
                                            model_size=deserialize_key[2].modelSize,
                                            num_of_features=num_of_features,
                                            )
                model.my_model.load_state_dict(state_dict)
                return model
        
        assert False, "Model not found!"

    # Convert a dict to a named tuple
    #
    @staticmethod
    def dict_to_named_tuple(**kwargs):
        return config.Config_of_one_run(**kwargs)

    # If the named tuple includes lists, convert it to a tuples.
    #
    @staticmethod
    def convert_lists_to_tuples(config_namedtuple):
        config_dict = config_namedtuple._asdict()
        for field, value in config_dict.items():
            if isinstance(value, list):
                config_dict[field] = tuple(value)
        return type(config_namedtuple)(**config_dict)

    # Helper function to deserialize keys and recreate the config namedtuple
    #
    @staticmethod
    def deserialize_key(serialized_key):

        # Rebuild named tuple from the dictionary
        key_dict = json.loads(serialized_key)
        config_rebuilt_named_tuple = Deserialize.dict_to_named_tuple(**key_dict['config'])
        config_rebuilt = Deserialize.convert_lists_to_tuples(config_rebuilt_named_tuple)

        return (
            key_dict['model_type'],
            key_dict['load_profile'],
            config_rebuilt
        )

# Evaluate the stored training results 
#
class Evaluate_Models:

    # Get all training results as nested dicts
    #
    @staticmethod
    def get_training_results(path_to_train_histories, 
                             skip_first_n_configs=None, 
                             skip_last_n_configs=None,                              
                             ):
        
        all_train_histories = Deserialize.get_training_histories(path_to_train_histories)

        # Create a nested dictionary of the results
        result_per_config = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(dict))))
        for key, value in all_train_histories.items():
            model_type = key[0]
            load_profile = key[1]
            sim_config = key[2]
            results = value
            result_per_config[sim_config][model_type][load_profile]['loss'] = float(results['loss'][-1])
            result_per_config[sim_config][model_type][load_profile]['test_loss'] = float(results['test_loss'][-1])
            result_per_config[sim_config][model_type][load_profile]['test_sMAPE'] = float(results['test_sMAPE'][-1])

        # Optionally: Skip given configs
        result_per_config = dict(islice(result_per_config.items(), skip_first_n_configs, skip_last_n_configs))

        return result_per_config

    # Get the best models per energy community (i.e. the "winners")
    #
    @staticmethod
    def get_winner_models(path_to_train_histories):
        
        result_per_config = Evaluate_Models.get_training_results(path_to_train_histories)
              
        # Calculate the best model ("winner") for each profile
        winner_per_config = {}
        winner_per_model = {}
        for config, result_per_model in result_per_config.items():
                
            # Initialize the used dict values with 0
            winner_per_config[config] = {model_type: 0 for model_type in result_per_model.keys()}
            for model_type in result_per_model:
                winner_per_model.setdefault(model_type, 0)
                
            # Loop through each load_profile within the model data
            for load_profile in next(iter(result_per_model.values())).keys():
                
                # Determine the model type with the best performance for the current load profile
                best_model = min(result_per_model, key = lambda model: result_per_model[model][load_profile]['test_sMAPE'])
                
                # Increment the count for the winning model type
                winner_per_config[config][best_model] += 1
                winner_per_model[best_model] += 1

        # Summarize the wins per model
        #
        print(f'Total Winner per model:\n  Models:', end=" ")
        counts = ""
        for model, count in winner_per_model.items():
            print(f'{model}', end=" ")
            counts += f'& {count} '
        print(f'{  counts}')
        
        return winner_per_config, winner_per_model

    # Calculate and print results for each config and model_type
    #
    @staticmethod
    def print_results(path_to_train_histories, print_style = 'latex'):

        result_per_config = Evaluate_Models.get_training_results(path_to_train_histories)
        df = pd.DataFrame()

        for config_id, (config, result_per_model) in enumerate(result_per_config.items()):
            
            pprint(f'Configuration: {config}')            
            latex_string = ''
            
            for model_type, result_per_profile in result_per_model.items():
                
                # Get al list of all losses of the current config and modeltype
                train_losses, test_MAE, test_sMAPE  = [], [], []
                for load_profile, results in result_per_profile.items():
                    train_losses.append(results['loss'])
                    test_MAE.append(results['test_loss'])
                    test_sMAPE.append(results['test_sMAPE'])        
                assert len(result_per_profile) == config.nrOfComunities
                
                decimal_points_MAE = 4
                decimal_points_sMAPE = 2
                mean_test_MAE = f'{np.mean(test_MAE):.{decimal_points_MAE}f}'
                mean_test_sMAPE = f'{np.mean(test_sMAPE):.{decimal_points_sMAPE}f}'
                std_dev_test_MAE = f'{np.std(test_MAE):.{decimal_points_MAE}f}'
                std_dev_test_sMAPE = f'{np.std(test_sMAPE):.{decimal_points_sMAPE}f}'
                mean_train_MAE = f'{np.mean(train_losses):.{decimal_points_MAE}f}'

                if print_style == 'latex':
                    latex_string += f' & {mean_test_sMAPE} ({std_dev_test_sMAPE})'
                elif print_style == 'pandas_df':
                    df[config_id, model_type] = mean_test_sMAPE
                    df[config_id, model_type + '_std_dev'] = std_dev_test_sMAPE
                elif print_style == 'shell':
                    # Print the results of the current config and modeltype
                    print(f'    Model: {model_type}')
                    print(f'      Mean Test MAE: {mean_test_MAE}')
                    print(f'      Mean Test MAPE: {mean_test_sMAPE}')
                    print(f'      Standard Deviation Test MAE: {std_dev_test_MAE}')
                    print(f'      Standard Deviation Test MAPE: {std_dev_test_sMAPE}')
                    print(f'      Mean Train MAE: {mean_train_MAE}\n')
                else:
                    assert "Please choose correct 'print_style' argument."

            if print_style == 'latex':
                print(f'Latex Summary for this Configuration: {latex_string}')
        
        return df

    # Calculate and print results for each config and model_type
    #
    @staticmethod
    def plot_training_losses_over_epochs(path_to_train_histories, 
                                         plot_only_single_config = False,
                                         plotted_config = None):
        
        all_train_histories = Deserialize.get_training_histories(path_to_train_histories)

        # Target config(s) to plot
        if plot_only_single_config:
            print(f"Plotted Config:")
            pprint(f"{plotted_config}")
        filtered_train_histories = dict(
            (key, value) for key, value in all_train_histories.items() 
            if plot_only_single_config == False or key[2] == plotted_config
        )

        # Create a combined list of loss values and corresponding run names
        combined_loss_data = []
        for run_id, (run_config, history) in enumerate(filtered_train_histories.items()):
            # Define the labels of the following graph
            model_type, load_profile, act_config = run_config
            label = (model_type, act_config.aggregation_Count, act_config.modelSize)

            # Add each epoch's loss for the current run's history
            for epoch, loss_value in enumerate(history['loss']):
                combined_loss_data.append((f"{label}_{run_id}", epoch + 1, loss_value))

        # Create a DataFrame from the combined loss data
        df = pd.DataFrame(combined_loss_data, columns=['Run_History', 'Epoch', 'Loss'])

        # Use plotly express to plot the line graph
        fig = px.line(df, x='Epoch', y='Loss', color='Run_History', line_group='Run_History',
                    labels={'Loss': 'Training Loss', 'Epoch': 'Epoch'},
                    title='Training Loss Over Epochs for Different Runs and Histories')
        fig.update_yaxes(range=[0, 1])
        fig.update_traces(marker=dict(size=5))
        fig.update_layout(showlegend=False)
        fig.show()


                
        
        