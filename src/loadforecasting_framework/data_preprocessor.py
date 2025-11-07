"""Data Preprocessor for Load Forecasting Models"""

import datetime
import pandas as pd
import numpy as np
import torch
from loadforecasting_models import Normalizer
from .interfaces import DataSplitType

class DataPreprocessor:
    """Brings the given data into the data format needed by the model"""

    def __init__(self,
        normalizer: Normalizer | None = None,
        data_split: DataSplitType | None = None,
        add_lagged_profiles: tuple = (7, 14, 21),
        num_of_weather_features: int | None = None,
        first_prediction_clocktime: datetime.time = datetime.time(0, 0),
        prediction_horizon: pd.Timedelta = pd.Timedelta(days=0, hours=23, minutes=0),
        prediction_rate: pd.Timedelta = pd.Timedelta(days=1),
        ) -> None:
        """
        Constructor of the DataPreprocessor class.
        Args:
            normalizer (Normalizer | None): 
                Normalizer to be used for input data and target value. 
                If None, no normalization is applied.
            data_split (DataSplitType | None): 
                Data split to be used for train, dev and test sets.
                If None, no data split is applied.
            add_lagged_profiles (tuple): 
                Days ago of lagged profiles to be added as input features.
                Default are the profiles from 7, 14 and 21 days ago.
            num_of_weather_features (int | None): 
                Number of weather features in the input data.
                If weather data are given in the transform_data method, this parameter
                is calculated automatically.
            first_prediction_clocktime (datetime.time): 
                Clock time of the first prediction.
                Default is 00:00 (midnight).
            prediction_horizon (pd.Timedelta): 
                Prediction horizon of the model.
                Default is 23 hours (i.e. one day ahead prediction with hourly resolution).
            prediction_rate (pd.Timedelta): 
                Time between two consecutive predictions.
                Default is 1 day.
        """

        # Set default parameters
        if data_split is None:
            data_split = DataSplitType(train_set_1=0, dev_set=0, test_set=0, train_set_2=0, pad=0)

        # Store the parameters
        self.prediction_rate = prediction_rate
        self.prediction_horizon = prediction_horizon
        self.first_prediction_clocktime = first_prediction_clocktime
        self.add_lagged_profiles = add_lagged_profiles
        self.data_split = data_split
        self.normalizer = normalizer
        self.num_of_weather_features = num_of_weather_features

        # Initialize internal variables
        self._first_prediction_date = None
        self._sampling_time = None
        self._last_available_datetime = None
        self._total_set_size = None
        self._padding_start = None
        self._train_set_2_start = None
        self._test_set_start = None
        self._dev_set_start = None
        self._train_set_1_start = None

    def transform_data(self,
        power_profiles: pd.DataFrame,
        weather_data: pd.DataFrame | None = None,
        public_holidays: list | None = None,
        ) -> tuple[np.ndarray, np.ndarray]:
        """Brings the given data into the data format needed by the model"""

        if public_holidays is None:
            public_holidays = []

        # Set the first and last available timestamps
        self.set_prediction_timerange(power_profiles)

        # Convert the power timeseries to a nd-array with format (batches, timesteps, outputs)
        y_all = self.formatting_y(power_profiles)

        # Convert the input features to a nd-array with format (batches, timesteps, features)
        x_all = self.formatting_x(power_profiles, weather_data, public_holidays)

        # Split up the data into train, dev, test and modeldata
        x_all, y_all = self.split_up_data(x_all, y_all)

        # Normalize all input data and target value
        x_all, y_all = self.normalize_all(x_all, y_all)

        # Convert from ndarray to torch tensor
        x_all, y_all = self.convert_to_torch_tensor(x_all, y_all)

        return x_all, y_all

    def set_prediction_timerange(
        self,
        power_profiles: pd.DataFrame,
        ) -> None:
        """
        Set the first and last available timestamps and the sampling time.
        """

        # Calculate the first possible prediction timestamp
        nr_of_lagged_days = len(self.add_lagged_profiles)
        first_timestamp = power_profiles.index[0] + pd.Timedelta(days=7*nr_of_lagged_days)

        # Calculate and store the sampling time
        if not isinstance(power_profiles.index, pd.DatetimeIndex):
            raise TypeError("power_profiles.index must be a DatetimeIndex")
        self._sampling_time = power_profiles.index.to_series().diff().median()

        # Choose a prediction datetime, which is on the same day as the 'first_timestamp'.
        target_timestamp = pd.Timestamp.combine(first_timestamp.date(),
            self.first_prediction_clocktime).tz_localize(first_timestamp.tzinfo)

        # Check if the calculated timestamp is before or after the target time
        if target_timestamp < first_timestamp:
            self._first_prediction_date = target_timestamp + pd.Timedelta(days=1)
        else:
            self._first_prediction_date = target_timestamp

        # Additionally set the last available timestep
        self._last_available_datetime = power_profiles.index[-1]

    def formatting_x(
        self,
        power_profiles: pd.DataFrame,
        weather_data: pd.DataFrame | None,
        public_holidays: list,
    ) -> np.ndarray:
        """
        Convert the input data to the model format.
        For more informations regarding the shape see model design for this project.
        """

        # Initialize and test internal variables
        if not isinstance(self._first_prediction_date, pd.Timestamp):
            raise TypeError("self._first_prediction_date was not set properly.")
        if not isinstance(self._last_available_datetime, pd.Timestamp):
            raise TypeError("self._last_available_datetime was not set properly.")
        if not isinstance(self._sampling_time, pd.Timedelta):
            raise TypeError("self._sampling_time was not set properly.")
        act_prediction_date = self._first_prediction_date

        # Calculate/define the number of features of x
        nr_of_features = 11     # 7 weekday one-hot + 2 hour-of-day + 2 day-of-year cyclical
        if self.add_lagged_profiles:
            nr_of_features += len(self.add_lagged_profiles)
        if weather_data is None:
            # Weather data is not available
            if not isinstance(self.num_of_weather_features, int):
                raise TypeError("self.num_of_weather_features was not set in the constructor.")
            self.num_of_weather_features = self.num_of_weather_features
        else:
            # Weather data is available
            n = weather_data.shape[1]
            if (self.num_of_weather_features is not None) and (n != self.num_of_weather_features):
                raise ValueError("num_of_weather_features was set incorrect in the constructor.")
            self.num_of_weather_features = n
        nr_of_features += self.num_of_weather_features

        # Calculate/define the shape of x
        seq_end_time = self._first_prediction_date + self.prediction_horizon
        nr_of_timesteps = len(pd.date_range(start=self._first_prediction_date, end=seq_end_time,
            freq=self._sampling_time))
        x_all = np.zeros(shape=(0, nr_of_timesteps, nr_of_features))

        while act_prediction_date + self.prediction_horizon <= self._last_available_datetime:

            # Create a new batch (= one prediction)
            new_batch = np.zeros(shape=(1, nr_of_timesteps, nr_of_features))

            # Define the current time range
            end_of_prediction = act_prediction_date + self.prediction_horizon
            total_input_range = pd.Series(pd.date_range(start=act_prediction_date,
                end=end_of_prediction, freq=self._sampling_time)).dt

            # Get the current weekday indices [0 ... 6] of all nr_of_timesteps.
            # The shape of the following variable is (nr_of_timesteps, 1).
            weekday_numbers = total_input_range.dayofweek.values.astype(int)

            # Identify public holidays and replace that day with Sunday
            public_holiday_indices = total_input_range.floor("D").isin(public_holidays).to_numpy()
            weekday_numbers[public_holiday_indices] = 6

            # Create a one-hot encoding array with shape (nr_of_timesteps, 7).
            one_hot_encoding = np.eye(7)[weekday_numbers]
            index = one_hot_encoding.shape[1]
            new_batch[0, :, :index] = one_hot_encoding

            # Convert clock_time to cyclical features
            hour_sin = np.sin(2 * np.pi * total_input_range.hour / 24.0)
            hour_cos = np.cos(2 * np.pi * total_input_range.hour / 24.0)
            new_batch[0, :, index]  = hour_sin
            index += 1
            new_batch[0, :, index]  = hour_cos
            index += 1

            # Convert day-of-year to cyclical features
            day_of_year_sin = np.sin(2 * np.pi * total_input_range.day_of_year / 366)
            day_of_year_cos = np.cos(2 * np.pi * total_input_range.day_of_year / 366)
            new_batch[0, :, index]  = day_of_year_sin
            index += 1
            new_batch[0, :, index]  = day_of_year_cos
            index += 1

            # Optionally add lagged profiles
            if len(self.add_lagged_profiles) > 0:
                # Add the given lagged profiles, i.e. days ago.
                for day in self.add_lagged_profiles:
                    start = act_prediction_date - pd.Timedelta(days=day)
                    end = start + self.prediction_horizon
                    lagged_power = power_profiles.loc[start:end]
                    new_batch[0, :, index]  = np.array(lagged_power.values)
                    index += 1

            # If available: Add weather (past or forecasted) to the model input
            if weather_data is not None:
                weather_data_slice = weather_data.loc[ \
                    act_prediction_date-self.prediction_horizon:act_prediction_date]
                weather_seq_len = weather_data_slice.shape[0]
                for feature in weather_data_slice.columns:
                    new_batch[:, :weather_seq_len, index]  = weather_data_slice[feature][:]
                    index += 1
            else:
                # No weather data is available: Set all weather features to zero 
                # (e.g. during pre-training)
                new_batch[0, :, index:self.num_of_weather_features]  = 0.0
                index += self.num_of_weather_features

            # Add the current prediction and step to the next one
            x_all = np.concatenate((x_all, new_batch), axis=0)
            act_prediction_date += self.prediction_rate

        return x_all

    def formatting_y(
        self,
        df: pd.DataFrame,
        ) ->  np.ndarray:
        """
        Convert the given power profiles to the model format.
        For more informations regarding the shape see model design for this project.
        """

        # Initialize and test internal variables
        if not isinstance(self._first_prediction_date, pd.Timestamp):
            raise TypeError("self._first_prediction_date was not set properly.")
        if not isinstance(self._last_available_datetime, pd.Timestamp):
            raise TypeError("self._last_available_datetime was not set properly.")
        if not isinstance(self._sampling_time, pd.Timedelta):
            raise TypeError("self._sampling_time was not set properly.")
        batch_id = 0
        act_prediction_date = self._first_prediction_date

        # Calculate/define the shape of y
        seq_end_time = self._first_prediction_date + self.prediction_horizon
        nr_of_timesteps = len(pd.date_range(start=self._first_prediction_date, end=seq_end_time,
            freq=self._sampling_time))
        y_all = np.zeros(shape=(0, nr_of_timesteps, 1))

        while act_prediction_date + self.prediction_horizon <= self._last_available_datetime:

            # Add a new batch to the y array
            new_values = np.zeros(shape=(1, nr_of_timesteps, 1))
            y_all = np.concatenate((y_all, new_values), axis=0)

            # Get values within the specified time range
            end_prediction_horizon = act_prediction_date + self.prediction_horizon
            demandprofile_slice = df.loc[act_prediction_date:end_prediction_horizon]

            # Set all target power values
            y_all[batch_id, :, 0] = demandprofile_slice

            # Go to the next prediction (= batch)
            act_prediction_date += self.prediction_rate
            batch_id += 1

        return y_all

    def convert_to_torch_tensor(self, x_all, y_all):
        """Convert from nd-array to torch tensor"""

        for dataset in x_all:
            x_all[dataset] = torch.tensor(x_all[dataset])

        for dataset in y_all:
            y_all[dataset] = torch.tensor(y_all[dataset])

        return x_all, y_all

    def normalize_all(self, x_all, y_all):
        """Normalize all input data and target value, if a normalizer was given."""

        if self.normalizer is not None:
            x_all['train'] = self.normalizer.normalize_x(x_all['train'], training=True)
            y_all['train'] = self.normalizer.normalize_y(y_all['train'], training=True)
            x_all['dev'] = self.normalizer.normalize_x(x_all['dev'], training=False)
            y_all['dev'] = self.normalizer.normalize_y(y_all['dev'], training=False)
            x_all['test'] = self.normalizer.normalize_x(x_all['test'], training=False)
            y_all['test'] = self.normalizer.normalize_y(y_all['test'], training=False)
            x_all['all'] = self.normalizer.normalize_x(x_all['all'], training=False)
            y_all['all'] = self.normalizer.normalize_y(y_all['all'], training=False)

        return x_all, y_all

    def split_up_data(self, x_all, y_all):
        """Split up the data into train-, dev- and test-set"""

        # Do train-dev-test data split
        #
        # --------------> time axis
        #
        # |--------------------------------------------------------------------------------------- |
        # | un-used | train_set_1_size | dev_set_size | test_set_size | train_set_2_size | padding |
        # |---------|------------------|--------------|---------------|------------------|-------- |
        # |         |  x['train']      | x['dev']     |  x['test']    |  x['train']      |         |
        # |         |  y['train']      | y['dev']     |  y['test']    |  y['train']      |         |
        # |--------------------------------------------------------------------------------------- |
        # |                   x['all'] (entire timeseries)                                         |
        # |                   y['all'] (entire timeseries)                                         |
        # |--------------------------------------------------------------------------------------- |
        x, y = {}, {}
        self._total_set_size = x_all.shape[0]
        self._padding_start = self._total_set_size - self.data_split.pad
        self._train_set_2_start = self._padding_start - self.data_split.train_set_2
        self._test_set_start = self._train_set_2_start - self.data_split.test_set
        self._dev_set_start = self._test_set_start - self.data_split.dev_set
        if self.data_split.train_set_1 != -1:
            self._train_set_1_start = self._dev_set_start - self.data_split.train_set_1
        else:
            self._train_set_1_start = None # Set train length to max

        x['train'] = np.concatenate([
            x_all[self._train_set_1_start:self._dev_set_start],
            x_all[self._train_set_2_start:self._padding_start]
            ])
        x['dev'] = x_all[self._dev_set_start:self._test_set_start]
        x['test'] = x_all[self._test_set_start:self._train_set_2_start]
        x['all'] = x_all[:]

        y['train'] = np.concatenate([
            y_all[self._train_set_1_start:self._dev_set_start],
            y_all[self._train_set_2_start:self._padding_start]
            ])
        y['dev'] = y_all[self._dev_set_start:self._test_set_start]
        y['test'] = y_all[self._test_set_start:self._train_set_2_start]
        y['all'] = y_all[:]

        return x, y

    def get_total_index(self, dataset_type, local_index):
        """
        Return the total index in all data that corresponds to the given
        dataset_tye and index.
        """

        if dataset_type == 'train':
            if local_index < self.data_split.train_set_1:
                total_index = self._train_set_1_start + local_index
            elif local_index < self.data_split.train_set_1 + self.data_split.train_set_2:
                total_index = local_index - self.data_split.train_set_1 + self._train_set_2_start
            else:
                assert False, "Unexpected 'local_index' parameter received."
        elif dataset_type == 'dev':
            total_index = self._dev_set_start + local_index
        elif dataset_type == 'test':
            total_index = self._test_set_start + local_index
        else:
            assert False, "Unexpected 'dataset_type' parameter received."

        return total_index

    def get_start_date_from_index(self, dataset_type, local_index):
        """
        Return the prediction date that corresponds to the given
        dataset_tye and local_index.
        """

        if dataset_type == 'all':
            total_index = local_index
        else:
            # dataset_type != 'all'
            total_index = self.get_total_index(dataset_type, local_index)

        return self._first_prediction_date + total_index * self.prediction_rate

    def get_dataset_type_from_index(self, total_index):
        """Return the dataset-type (train, test, ...) from the given total_index"""

        if total_index >= self._total_set_size:
            dataset_type = 'unknown (error)'
        elif total_index >= self._train_set_2_start:
            dataset_type = 'train'
        elif total_index >= self._test_set_start:
            dataset_type = 'test'
        elif total_index >= self._dev_set_start:
            dataset_type = 'dev'
        elif total_index >= self._train_set_1_start:
            dataset_type = 'train'
        else:
            dataset_type = 'un-used'

        return dataset_type

if __name__ == '__main__':
    pass
