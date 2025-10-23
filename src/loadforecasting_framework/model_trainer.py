import os
import argparse
import pickle
from datetime import timedelta, date

import pandas as pd
import holidays
import pytz
from demandlib import bdew
import torch

from loadforecasting_framework import simulation_config
from loadforecasting_framework.model_adapter import ModelAdapter
from loadforecasting_framework import utils
from loadforecasting_framework import import_weather_data
from loadforecasting_models import KNN, Persistence, xLSTM, LSTM, Transformer, Perfect, Normalizer


class ModelTrainer:
    """
    Class to train and evaluate load forecasting models.
    """

    def __init__(self):
        self.test_set_size_days = 131    # Size of the testset is fixed to 131 days ~ 4 month

    def run(self, my_configs):

        # Run every single config
        all_train_histories, all_trained_models = {}, {}
        for act_sim_config_index, act_sim_config in enumerate(my_configs):

            # Fetch and prepare all needed data
            loadprofiles = self.preprocess_data(my_configs, act_sim_config_index)

            # Train and test the given models
            act_sim_config = my_configs[act_sim_config_index]
            results = []
            for model_type in act_sim_config.usedModels:
                for load_profile in loadprofiles:
                    result = self.optimize_model(model_type, load_profile, my_configs,
                        act_sim_config_index)
                    results.append(result)

            # Store the results into dicts
            model_types, load_profiles, sim_configs, histories, returned_models = zip(*results)
            for i, _ in enumerate(model_types):
                result_key = (model_types[i], load_profiles[i], sim_configs[i])
                all_train_histories[result_key] = histories[i]
                all_trained_models[result_key] = returned_models[i]

        # Persist all results
        utils.Serialize.store_results_with_pickle(all_train_histories)
        utils.Serialize.store_results_with_torch(all_trained_models)

        return

    # Do Model training and evaluation
    #
    def optimize_model(self, model_type, load_profile, configs, act_sim_config_index):
        """
        Train and evaluate the given model_type on the given load_profile.
        """

        print(f"\nProcessing model {model_type} with load profile {load_profile} and sim_config \
            {act_sim_config_index+1}/{len(configs)}.", flush=True)

        # Load a new powerprofile
        with open(load_profile, 'rb') as f:
            (x, y, model_adapter) = pickle.load(f)

        # Train and evaluate the model
        sim_config = configs[act_sim_config_index]
        num_of_features = x['train'].shape[2]

        # Initialize, train and evaluate the given model
        if model_type is KNN:
            my_model = model_type(k=40, weights = 'distance', model_adapter=model_adapter)
            history = my_model.train_model(x['train'], y['train'])
            history = my_model.evaluate(x['test'], y['test'], results=history, de_normalize=True)
        elif model_type is Persistence:
            my_model = model_type(lagged_load_feature=11, model_adapter=model_adapter)
            history = my_model.train_model()
            history = my_model.evaluate(x['test'], y['test'], results=history, de_normalize=True)
        elif model_type is Perfect:
            my_model = model_type(model_adapter=model_adapter)
            history = my_model.train_model()
            history = my_model.evaluate(y['test'], results=history, de_normalize=True)
        elif model_type in (xLSTM, LSTM, Transformer):
            my_model = model_type(sim_config.modelSize, num_of_features, model_adapter=model_adapter)
            history = my_model.train_model(x['train'], y['train'], pretrain_now=False,
                finetune_now=sim_config.doTransferLearning, epochs=sim_config.epochs)
            history = my_model.evaluate(x['test'], y['test'], results=history, de_normalize=True)
        else:
            assert False, f"Unimplemented model_type given: {model_type}"

        ret_tuple = (model_type, load_profile, sim_config, history, my_model)
        return ret_tuple

    def preprocess_data(self, configs, act_sim_config_index):
        
        sim_config = configs[act_sim_config_index]
        if sim_config.epochs <= 5:
            print(f"WARNING: Only {sim_config.epochs} epochs chosen. Please check, if this really \
                fits your needs.")
        print(f"\n\nDo Data Preprocessing for run config={sim_config}.", flush=True)
        
        loadProfiles, weatherData, public_holidays_timestamps = self.load_data(sim_config)
        
        # Bring the power profiles to the model shape of (nr_of_batches, timesteps, features)
        #
        loadProfiles_filenames = []
        for i, powerProfile in enumerate(loadProfiles[:sim_config.nrOfComunities]):

            # Preprocess data to get X and Y for the model
            normalizer = Normalizer()
            modelAdapter = ModelAdapter(public_holidays_timestamps,
                                            trainHistory = sim_config.trainingHistory,
                                            testSize = sim_config.testSize,
                                            devSize = sim_config.devSize,
                                            trainFuture = sim_config.trainingFuture,
                                            normalizer = normalizer,
                                            )
            X, Y = modelAdapter.transformData(powerProfile, weatherData)

            output_path = os.path.join(os.path.dirname(__file__), 'outputs', 'file_' + str(i) + '.pkl')
            with open(output_path, 'wb') as file:
                pickle.dump((X, Y, normalizer), file)
            loadProfiles_filenames.append(output_path)

        # Load the BDEW standard load profiles for the desired datetime range
        standard_loadprofiles = []
        startDate = loadProfiles[0].index[0].to_pydatetime().replace(tzinfo=None)
        endDate = loadProfiles[0].index[-1].to_pydatetime().replace(tzinfo=None)
        for year in range(startDate.year, endDate.year + 1):
            load_profile = bdew.ElecSlp(year, holidays=public_holidays_timestamps)\
                .get_profile({"h0": 1000})
            standard_loadprofiles.append(load_profile)
        all_standard_loadprofiles = pd.concat(standard_loadprofiles)['h0']
        all_standard_loadprofiles = all_standard_loadprofiles[(all_standard_loadprofiles.index >= startDate)
                                                                & (all_standard_loadprofiles.index <= endDate)]
        all_standard_loadprofiles = all_standard_loadprofiles.tz_localize("UTC")
        
        # Preprocess data to get X and Y for the model
        normalizer = Normalizer()
        modelAdapter = ModelAdapter(public_holidays_timestamps,
                                        trainHistory = sim_config.trainingHistory,
                                        testSize = sim_config.testSize,
                                        devSize = sim_config.devSize, 
                                        trainFuture = sim_config.trainingFuture,
                                        normalizer = normalizer,
                                        )
        X, Y = modelAdapter.transformData(all_standard_loadprofiles, weatherData=None)
        pretraining_filename = os.path.join(os.path.dirname(__file__), 'outputs', 'standard_loadprofile.pkl')
        with open(pretraining_filename, 'wb') as file:
            pickle.dump((X, Y, normalizer), file)

        # If required, do pretraining
        if sim_config.doPretraining:

            for model_type in sim_config.usedModels:
                if issubclass(model_type, torch.nn.Module):
                    print(f"\nPretraining {model_type} model and and sim_config \
                        {act_sim_config_index+1}/{len(configs)}.", flush=True)
                    num_of_features = X['all'].shape[2]
                    my_model = model_type(sim_config.modelSize, num_of_features)                    
                    my_model.train_model(X['all'], Y['all'], pretrain_now=True,
                        finetune_now=False, epochs=sim_config.epochs)
                else:
                    print(f"\nNo pretraining possible for baseline model {model_type}. Sim_config \
                        {act_sim_config_index+1}/{len(configs)}.", flush=True)

        return loadProfiles_filenames

    def load_data(self, sim_config):

        # Readout the power profiles, bring them to the format needed by the model and store those 
        # profiles
        #
        file_path = os.path.join(os.path.dirname(__file__), sim_config.aggregation_Count[1])
        loadProfiles = pd.read_pickle(file_path)
        loadProfiles = loadProfiles[:sim_config.nrOfComunities]

        # Readout the weather data
        #
        startDate = loadProfiles[0].index[0].to_pydatetime().replace(tzinfo=None)
        endDate = loadProfiles[0].index[-1].to_pydatetime().replace(tzinfo=None)
        weather_measurements = import_weather_data.WeatherMeasurements()
        weatherData = weather_measurements.get_data(
                    startDate = startDate,
                    endDate = endDate,
                    lat = 51.5085,      # Location:
                    lon = -0.1257,      # London Heathrow,
                    alt = 25,           # Weatherstation
                    sample_periode = 'hourly',
                    tz = 'UTC',
                    )
        weatherData = weatherData.loc[:, (weatherData != 0).any(axis=0)]    # remove empty columns

        # Load the public holiday calendar
        #
        public_holidays_dict = holidays.CountryHoliday('GB', prov='ENG', years=range(startDate.year, endDate.year + 1))

        # Add Christmas holidays from Dec 24 to Dec 31 for each year
        for year in range(startDate.year, endDate.year + 1):
            for day in range(8):  # 9 days from Dec 24 to Dec 31 inclusive
                public_holidays_dict[date(year, 12, 24) + timedelta(days=day)] = "Christmas Holidays"

        public_holidays_timestamps = [pd.Timestamp(date, tzinfo=pytz.utc) for date in public_holidays_dict.keys()]

        return loadProfiles, weatherData, public_holidays_timestamps

if __name__ == "__main__":

    # Parse the optional cmd-line-arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=['default', 'ci'], default='default')
    mode = parser.parse_args().mode

    # Load the simulations configs
    if mode == 'default':
        configs = simulation_config.configs
    else:
        configs = simulation_config.configs_for_the_ci

    # Run the simulation
    ModelTrainer().run(configs)
