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
from loadforecasting_framework.simulation_config import ConfigOfOneRun
from loadforecasting_framework.data_preprocessor import DataPreprocessor
from loadforecasting_framework import utils
from loadforecasting_framework import import_weather_data
from loadforecasting_models import Knn, Persistence, xLstm, Lstm, Transformer, Perfect, Normalizer

class ModelTrainer:
    """
    Class to train and evaluate load forecasting models.
    """

    def __init__(self):
        self.test_set_size_days = 131    # Size of the testset is fixed to 131 days ~ 4 month

    def run(self, my_configs):
        """Run every single simulation given in my_configs."""

        all_train_histories, all_trained_models = {}, {}
        for act_sim_config_index, act_sim_config in enumerate(my_configs):

            # Fetch and prepare all needed data
            loadprofiles = self.preprocess_data(my_configs, act_sim_config_index)

            # Train and test the given models
            act_sim_config = my_configs[act_sim_config_index]
            results = []
            for model_type in act_sim_config.used_models:
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
    def optimize_model(self, model_type, load_profile, my_configs, act_sim_config_index):
        """
        Train and evaluate the given model_type on the given load_profile.
        """
        print(f"\nProcessing model {model_type} with load profile {load_profile} and sim_config \
            {act_sim_config_index+1}/{len(my_configs)}.", flush=True)

        # Load a new powerprofile
        with open(load_profile, 'rb') as f:
            (x, y, normalizer) = pickle.load(f)

        # Create the given model and train and evaluate it
        sim_config = my_configs[act_sim_config_index]
        my_model, history = ModelTrainer.create_model(
            model_type,
            normalizer,
            num_of_features = x['train'].shape[2],
            sim_config = sim_config,
            do_training = True,
            x_train = x['train'],
            y_train = y['train'],
            pretrain_now = False,
            finetune_now = sim_config.do_transfer_learning,
            do_evaluation = True,
            x_test = x['test'],
            y_test = y['test'],
            )

        ret_tuple = (model_type, load_profile, sim_config, history, my_model)
        return ret_tuple

    @staticmethod
    def create_model(
        model_type: str,
        normalizer: Normalizer,
        num_of_features: int,
        sim_config: ConfigOfOneRun | None = None,
        do_training: bool = False,
        x_train: torch.Tensor | None = None,
        y_train: torch.Tensor | None = None,
        pretrain_now: bool = False,
        finetune_now: bool = True,
        do_evaluation: bool = False,
        x_test: torch.Tensor | None = None,
        y_test: torch.Tensor | None = None,
        ) -> tuple:
        """
        Create an instance of the given model_type and (optionally) train and evaluate it.
        """

        history = {}

        # Check the inputs
        #
        if x_train is None:
            x_train = torch.Tensor([])
        if y_train is None:
            y_train = torch.Tensor([])
        if x_test is None:
            x_test = torch.Tensor([])
        if y_test is None:
            y_test = torch.Tensor([])
        if do_training:
            assert x_train.numel() > 0, "x_train must be given, if do_training is True!"
            assert y_train.numel() > 0, "y_train must be given, if do_training is True!"
        if do_evaluation:
            assert x_test.numel() > 0, "x_test must be given, if do_evaluation is True!"
            assert y_test.numel() > 0, "y_test must be given, if do_evaluation is True!"

        if model_type == 'Knn':
            my_model = Knn(k=40, weights = 'distance', normalizer=normalizer)
            if do_training:
                history = my_model.train_model(x_train, y_train)
            if do_evaluation:
                history = my_model.evaluate(x_test, y_test, results=history, de_normalize=True)

        elif model_type == 'Persistence':
            my_model = Persistence(lagged_load_feature=11, normalizer=normalizer)
            if do_training:
                history = my_model.train_model()
            if do_evaluation:
                history = my_model.evaluate(x_test, y_test, results=history, de_normalize=True)

        elif model_type == 'Perfect':
            my_model = Perfect(normalizer=normalizer)
            if do_training:
                history = my_model.train_model()
            if do_evaluation:
                history = my_model.evaluate(y_test, results=history, de_normalize=True)

        else:   # Machine Learning Models
            if model_type == 'xLstm':
                my_class = xLstm
            elif model_type == 'Lstm':
                my_class = Lstm
            elif model_type == 'Transformer':
                my_class = Transformer
            else:
                assert False, f"Unimplemented model_type given: {model_type}"

            if sim_config is None:
                raise ValueError("sim_config must be given for Machine Learning Models!")
            model_size = sim_config.model_size
            my_model = my_class(model_size, num_of_features, normalizer=normalizer)

            if do_training:
                history = my_model.train_model(x_train, y_train, pretrain_now=pretrain_now,
                    finetune_now=finetune_now, epochs=sim_config.epochs)
            if do_evaluation:
                history = my_model.evaluate(x_test, y_test, results=history, de_normalize=True)

        return my_model, history

    def preprocess_data(self, my_configs, act_sim_config_index):
        """
        Preprocess the data for the given simulation configuration.
        """

        sim_config = my_configs[act_sim_config_index]
        if sim_config.epochs <= 5:
            print(f"WARNING: Only {sim_config.epochs} epochs chosen. Please check, if this really \
                fits your needs.")
        print(f"\n\nDo Data Preprocessing for run config={sim_config}.", flush=True)

        load_profiles, weather_data, public_holidays_timestamps = self.load_data(sim_config)

        # Bring the power profiles to the model shape of (nr_of_batches, timesteps, features)
        #
        load_profiles_filenames = []
        for i, power_profile in enumerate(load_profiles[:sim_config.nr_of_communities]):

            # Preprocess data to get x and y for the model
            normalizer = Normalizer()
            model_preprocessor = DataPreprocessor(
                normalizer = normalizer,
                data_split = sim_config.data_split,
                )
            x, y = model_preprocessor.transform_data(
                power_profile,
                weather_data,
                public_holidays_timestamps
                )

            output_path = os.path.join(os.path.dirname(__file__), 'outputs',
                'file_' + str(i) + '.pkl')
            with open(output_path, 'wb') as file:
                pickle.dump((x, y, normalizer), file)
            load_profiles_filenames.append(output_path)

        # Load the BDEW standard load profiles for the desired datetime range
        standard_loadprofiles = []
        start_date = load_profiles[0].index[0].to_pydatetime().replace(tzinfo=None)
        end_date = load_profiles[0].index[-1].to_pydatetime().replace(tzinfo=None)
        for year in range(start_date.year, end_date.year + 1):
            load_profile = bdew.ElecSlp(year, holidays=public_holidays_timestamps)\
                .get_profile({"h0": 1000})
            standard_loadprofiles.append(load_profile)
        all_standard_loadprofiles = pd.concat(standard_loadprofiles)['h0']
        all_standard_loadprofiles = all_standard_loadprofiles[(all_standard_loadprofiles.index
            >= start_date) & (all_standard_loadprofiles.index <= end_date)]
        all_standard_loadprofiles = all_standard_loadprofiles.tz_localize("UTC")

        # Preprocess data to get x and y for the model
        normalizer = Normalizer()
        model_preprocessor = DataPreprocessor(
            normalizer = normalizer,
            data_split = sim_config.data_split,
            num_of_weather_features=weather_data.shape[1],
            )
        x, y = model_preprocessor.transform_data(
            power_profile=all_standard_loadprofiles,
            weather_data=None,
            public_holidays=public_holidays_timestamps,
            )
        pretraining_filename = os.path.join(os.path.dirname(__file__), 'outputs',
            'standard_loadprofile.pkl')
        with open(pretraining_filename, 'wb') as file:
            pickle.dump((x, y, normalizer), file)

        # If needed, do pretraining
        if sim_config.do_transfer_learning:

            for model_type in sim_config.used_models:
                if model_type in ('xLstm', 'Lstm', 'Transformer'):

                    # Pretraining possible for Machine Learning Models
                    print(f"\nPretraining {model_type} model and and sim_config \
                        {act_sim_config_index+1}/{len(my_configs)}.", flush=True)
                    ModelTrainer.create_model(
                        model_type,
                        normalizer,
                        num_of_features = x['all'].shape[2],
                        sim_config = my_configs[act_sim_config_index],
                        do_training = True,
                        x_train = x['all'],
                        y_train = y['all'],
                        pretrain_now = True,
                        finetune_now = False,
                        )

                else:
                    print(f"\nNo pretraining possible for baseline model {model_type}. Sim_config \
                        {act_sim_config_index+1}/{len(my_configs)}.", flush=True)

        return load_profiles_filenames

    def load_data(self, sim_config):
        """
        Load all needed data: power profiles, weather data and public holidays.
        """

        # Readout the power profiles, bring them to the format needed by the model and store those
        # profiles
        #
        file_path = os.path.join(os.path.dirname(__file__), sim_config.aggregation_count[1])
        load_profiles = pd.read_pickle(file_path)
        load_profiles = load_profiles[:sim_config.nr_of_communities]

        # Readout the weather data
        #
        start_date = load_profiles[0].index[0].to_pydatetime().replace(tzinfo=None)
        end_date = load_profiles[0].index[-1].to_pydatetime().replace(tzinfo=None)
        weather_measurements = import_weather_data.WeatherMeasurements()

        weather_data = weather_measurements.get_data(
                    startDate = start_date,
                    endDate = end_date,
                    lat = 51.5085,      # Location:
                    lon = -0.1257,      # London Heathrow,
                    alt = 25,           # Weatherstation
                    sample_periode = 'hourly',
                    tz = 'UTC',
                    )

        weather_data = weather_data.loc[:, (weather_data != 0).any(axis=0)] # remove empty columns

        public_holidays_timestamps = self.load_holidays(start_date, end_date)

        return load_profiles, weather_data, public_holidays_timestamps

    def load_holidays(self, start_date, end_date) -> list:
        """Load the public holiday calendar and return as list."""

        public_holidays = holidays.CountryHoliday('GB', prov='ENG',
            years=range(start_date.year, end_date.year + 1))

        # Add Christmas holidays from Dec 24 to Dec 31 for each year
        for year in range(start_date.year, end_date.year + 1):
            for day in range(8):  # 9 days from Dec 24 to Dec 31 inclusive
                public_holidays[date(year, 12, 24) + timedelta(days=day)] = "Christmas Holidays"

        # Convert from dict to list
        public_holidays_timestamps = \
            [pd.Timestamp(date, tzinfo=pytz.utc) for date in public_holidays.keys()]

        return public_holidays_timestamps


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
