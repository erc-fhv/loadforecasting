
import json
from datetime import datetime
import torch
import pickle
import pytz
import scripts.Simulation_config as config
import scripts.Model

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
        all_trained_models = Serialize.get_serialized_dicts(all_trained_models)

        # Save the total model with torch.save
        timestamp = Serialize.get_act_timestamp()
        torch.save(all_trained_models, f'scripts/outputs/all_trained_models{timestamp}.pth')
        torch.save(all_trained_models, f'scripts/outputs/all_trained_models.pth')

    # Use pickle to save a dictionary with training results to disc.
    #
    @staticmethod
    def store_results_with_pickle(all_train_histories):      
        
        # Squeeze the dict keys
        all_train_histories = Serialize.get_serialized_dicts(all_train_histories)
        
        # Store the variables in a persistent files with the timestamp
        timestamp = Serialize.get_act_timestamp()
        with open(f"scripts/outputs/all_train_histories{timestamp}.pkl", 'wb') as f:
            pickle.dump(all_train_histories, f)
        with open(f"scripts/outputs/all_train_histories.pkl", 'wb') as f:
            pickle.dump(all_train_histories, f)
        
    # Serialize the given dicts.
    #
    @staticmethod
    def get_serialized_dicts(dict_with_komplex_keys):            
        serialized_models = {}
        for key, model in dict_with_komplex_keys.items():
            serialized_key = Serialize.serialize_complex_key(key)
            if isinstance(model, torch.nn.Module):
                # Not the whole torch model shall be saved, but only its parameters.
                serialized_models[serialized_key] = model.state_dict()
            else:                    
                serialized_models[serialized_key] = model

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
    def get_act_timestamp():
        return datetime.now(pytz.timezone('Europe/Vienna')).strftime("_%Y%m%d_%H%M")

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
    def get_trained_models(path):
        
        serialized_dict = torch.load(path)
        
        reloaded_models = {}        
        for serialized_key, state_dict in serialized_dict.items():
            deserialize_key = Deserialize.deserialize_key(serialized_key)
            model = scripts.Model.Model(model_type=deserialize_key[0], 
                                        model_size=deserialize_key[2].modelSize)
            model.my_model.load_state_dict(state_dict)            
            reloaded_models[deserialize_key] = model
        
        return reloaded_models

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

