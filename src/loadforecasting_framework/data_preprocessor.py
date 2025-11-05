import datetime
import pandas as pd
import numpy as np
import torch

class DataPreprocessor:
    """Brings the given data into the data format needed by the model"""

    def __init__(self,
                 public_holidays,
                 train_history,
                 test_size,
                 dev_size,
                 train_future,
                 normalizer,
                 add_lagged_power=True,
                 shuffle_data=False,
                 seed=None,
                 sampling_time = pd.Timedelta(hours=1, minutes=0),
                 prediction_rate = pd.Timedelta(days=1),
                 prediction_horizon = pd.Timedelta(days=0, hours=23, minutes=0),
                 ):

        self.prediction_rate = prediction_rate
        self.prediction_horizon = prediction_horizon
        self.sampling_time = sampling_time
        self.public_holidays = public_holidays
        self.add_lagged_power = add_lagged_power
        self.shuffle_data = shuffle_data
        self.train_history = train_history
        self.test_size = test_size
        self.dev_size = dev_size
        self.train_future = train_future
        self.normalizer = normalizer
        self.nr_of_lagged_days = 3

        # Optionally: Fix the random-seed for reproducibility
        if seed is not None:
            np.random.seed(seed)

    def transform_data(self,
                      power_profiles,
                      weather_data,
                      first_prediction_clocktime = datetime.time(0, 0),
                      ):
        """Brings the given data into the data format needed by the model"""

        # Downsample the profiles (e.g. to a frequency of 1/1h)
        power_profiles = power_profiles.resample(self.sampling_time).mean()

        # Get the first and last available timestamps
        self.first_prediction_date = self.get_first_prediction_timestamp(power_profiles, 
            first_prediction_clocktime)
        self.last_available_datetime = power_profiles.index[-1]

        # Convert the power timeseries to a nd-array with format (batches, timesteps, outputs)
        y_all = self.formatting_y(power_profiles)

        # Convert the input features to a nd-array with format (batches, timesteps, features)
        x_all = self.formatting_x(weather_data, power_profiles)

        # Split up the data into train, dev, test and modeldata
        x_all, y_all = self.split_up_data(x_all, y_all)

        # Normalize all input data and target value
        x_all, y_all = self.normalize_all(x_all, y_all)

        # Convert from ndarray to torch tensor
        x_all, y_all = self.convert_to_torch_tensor(x_all, y_all)

        return x_all, y_all

    def get_first_prediction_timestamp(self, power_profiles, first_prediction_clocktime):

        # Calculate the first possible prediction timestamp
        first_timestamp = power_profiles.index[0] + pd.Timedelta(days=7*(self.nr_of_lagged_days))

        # Choose a prediction datetime, which is on the same day as the 'first_timestamp'.
        target_timestamp = pd.Timestamp.combine(first_timestamp.date(),
            first_prediction_clocktime).tz_localize(first_timestamp.tzinfo)

        # Check if the calculated timestamp is before or after the target time
        if target_timestamp < first_timestamp:
            first_prediction_timestamp = target_timestamp + pd.Timedelta(days=1)
        else:
            first_prediction_timestamp = target_timestamp

        return first_prediction_timestamp

    def formatting_x(self, weather_data, power_profiles=None):
        """
        Convert the input data to the model format.
        For more informations regarding the shape see model design for this project.
        """

        batch_id = 0
        next_prediction_date = self.first_prediction_date

        # Calculate/define the number of features of x
        nr_of_features = 11
        if self.add_lagged_power:
            nr_of_features += 3
        if weather_data is None:
            num_of_weather_features = 6 # Default weather features
        else:
            num_of_weather_features = weather_data.shape[1]
        nr_of_features += num_of_weather_features

        seq_start_time = self.first_prediction_date
        seq_end_time = self.first_prediction_date + self.prediction_horizon
        nr_of_timesteps = len(pd.date_range(start=seq_start_time,
            end=seq_end_time, freq=self.sampling_time))
        x_all = np.zeros(shape=(0, nr_of_timesteps, nr_of_features))

        while next_prediction_date + self.prediction_horizon <= self.last_available_datetime:

            # Add a new batch to the x array
            new_values = np.zeros(shape=(1, nr_of_timesteps, nr_of_features))
            x_all = np.concatenate((x_all, new_values), axis=0)

            # Define the current time range
            end_of_prediction = next_prediction_date + self.prediction_horizon
            total_input_range = pd.date_range(start=next_prediction_date, end=end_of_prediction,
                freq=self.sampling_time)

            # Get the current weekday indices [0 ... 6] of all nr_of_timesteps.
            # The shape of the following variable is (nr_of_timesteps, 1).
            weekday_numbers = total_input_range.weekday.values

            # Identify public holidays and replace that day with Sunday
            public_holiday_indices = total_input_range.floor("D").isin(self.public_holidays)
            weekday_numbers[public_holiday_indices] = 6

            # Create a one-hot encoding array with shape (nr_of_timesteps, 7).
            one_hot_encoding = np.eye(7)[weekday_numbers]
            index = 7
            x_all[batch_id, :, :index] = one_hot_encoding

            # Convert clock_time to cyclical features
            hour_sin = np.sin(2 * np.pi * total_input_range.hour / 24.0)
            hour_cos = np.cos(2 * np.pi * total_input_range.hour / 24.0)
            x_all[batch_id, :, index]  = hour_sin
            index += 1
            x_all[batch_id, :, index]  = hour_cos
            index += 1

            # Convert day-of-year to cyclical features
            day_of_year_sin = np.sin(2 * np.pi * total_input_range.day_of_year / 366)
            day_of_year_cos = np.cos(2 * np.pi * total_input_range.day_of_year / 366)
            x_all[batch_id, :, index]  = day_of_year_sin
            index += 1
            x_all[batch_id, :, index]  = day_of_year_cos
            index += 1

            # Optionally add lagged profiles
            if self.add_lagged_power:
                # Add exactly the day one, two and three weeks ago.
                for day in range(1, 1 + self.nr_of_lagged_days):                    
                    start = next_prediction_date - pd.Timedelta(days=day*7)
                    end = start + self.prediction_horizon
                    lagged_power = power_profiles.loc[start:end]
                    x_all[batch_id, :, index]  = np.array(lagged_power.values)
                    index += 1

            # If available: Add past weather measurmenents to the model input
            if weather_data is not None:
                weather_data_slice = weather_data.loc[ \
                    next_prediction_date-self.prediction_horizon:next_prediction_date]
                weather_seq_len = weather_data_slice.shape[0]
                for feature in weather_data_slice.columns:
                    x_all[batch_id, :weather_seq_len, index]  = weather_data_slice[feature][:]
                    index += 1
            else:
                x_all[batch_id, :, index:num_of_weather_features]  = 0.0
                index += num_of_weather_features

            # Go to the next prediction (= batch)
            next_prediction_date += self.prediction_rate
            batch_id += 1

        return x_all

    def formatting_y(self, df):
        """
        Convert the given power profiles to the model format.
        For more informations regarding the shape see model design for this project.
        """

        batch_id = 0
        next_prediction_date = self.first_prediction_date

        # Calculate/define the shape of y
        seq_end_time = self.first_prediction_date + self.prediction_horizon
        nr_of_timesteps = len(pd.date_range(start=self.first_prediction_date, end=seq_end_time,
            freq=self.sampling_time))
        y_all = np.zeros(shape=(0, nr_of_timesteps, 1))

        while next_prediction_date + self.prediction_horizon <= self.last_available_datetime:

            # Add a new batch to the y array
            new_values = np.zeros(shape=(1, nr_of_timesteps, 1))
            y_all = np.concatenate((y_all, new_values), axis=0)

            # Get values within the specified time range
            end_prediction_horizon = next_prediction_date + self.prediction_horizon
            demandprofile_slice = df.loc[next_prediction_date:end_prediction_horizon]

            # Set all target power values
            y_all[batch_id, :, 0] = demandprofile_slice

            # Go to the next prediction (= batch)
            next_prediction_date += self.prediction_rate
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
        """Normalize all input data and target value"""

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

        # Optionally shuffle all indices
        total_samples = x_all.shape[0]
        self.shuffeled_indices = np.arange(total_samples)
        if self.shuffle_data:
            np.random.shuffle(self.shuffeled_indices)

        # Do train-dev-test data split
        #
        # --------------> time axis
        #
        #  -------------------------------------------------------------------
        # | un-used | train_history  |    test      |  train_future  |   dev    |
        # |-------------------------|--------------|---------------|----------|
        # |         |  x['train']   |  x['test']   |  x['train']   | x['dev'] |
        # |         |  y['train']   |  y['test']   |  y['train']   | y['dev'] |
        #  -------------------------------------------------------------------
        # |                   x['all'] (entire timeseries)                    |
        # |                   y['all'] (entire timeseries)                    |
        #  -------------------------------------------------------------------        
        x, y = {}, {}
        self.total_set_size = x_all.shape[0]
        self.dev_set_start = self.total_set_size - self.dev_size
        self.trainFuture_start = self.dev_set_start - self.train_future
        self.test_set_start = self.trainFuture_start - self.test_size
        if self.train_history != -1:
            self.train_set_start = self.test_set_start - self.train_history
        else:
            self.train_set_start = None # Set train length to max

        x['dev'] = x_all[self.shuffeled_indices[self.dev_set_start:]]
        x['test'] = x_all[self.shuffeled_indices[self.test_set_start:self.trainFuture_start]]
        x['train'] = np.concatenate([
                        x_all[self.shuffeled_indices[self.train_set_start:self.test_set_start]],
                        x_all[self.shuffeled_indices[self.trainFuture_start:self.dev_set_start]]
                    ])
        x['all'] = x_all[:]

        y['dev'] = y_all[self.shuffeled_indices[self.dev_set_start:]]
        y['test'] = y_all[self.shuffeled_indices[self.test_set_start:self.trainFuture_start]]
        y['train'] = np.concatenate([
                        y_all[self.shuffeled_indices[self.train_set_start:self.test_set_start]],
                        y_all[self.shuffeled_indices[self.trainFuture_start:self.dev_set_start]]
                    ])
        y['all'] = y_all[:]

        return x, y

    # Return the unshuffled index in all data that corresponds to the given
    # dataset_tye and index.
    #
    def get_unshuffeled_index(self, dataset_type, index):

        # Shuffled data
        if dataset_type == 'train':
            if index < self.train_history:
                unshuffled_index = self.shuffeled_indices[index + self.train_set_start]
            elif index < self.train_history + self.train_future:
                unshuffled_index = self.shuffeled_indices[index - self.train_history \
                    + self.trainFuture_start]
            else:
                assert False, "Unexpected 'index' parameter received."
        elif dataset_type == 'dev':
            unshuffled_index = self.shuffeled_indices[index + self.dev_set_start]
        elif dataset_type == 'test':
            unshuffled_index = self.shuffeled_indices[index + self.test_set_start]
        else:
            assert False, "Unexpected 'dataset_type' parameter received."

        return unshuffled_index

    def get_start_date_from_index(self, dataset_type, index):
        """
        Return the prediction date that corresponds to the given
        dataset_tye and index.
        """

        if dataset_type != 'all':
            index = self.get_unshuffeled_index(dataset_type, index)

        return self.first_prediction_date + index * self.prediction_rate

    def get_dataset_type_from_index(self, unshuffeled_index):
        """Return the dataset-type (train, test, ...) from the given unshuffeled index"""

        shuffled_index = np.where(self.shuffeled_indices == unshuffeled_index)[0][0]

        if shuffled_index >= self.total_set_size:
            dataset_type = 'unknown (error)'
        elif shuffled_index >= self.dev_set_start:
            dataset_type = 'dev'
        elif shuffled_index >= self.trainFuture_start:
            dataset_type = 'train'
        elif shuffled_index >= self.test_set_start:
            dataset_type = 'test'
        elif shuffled_index >= self.train_set_start:
            dataset_type = 'train'
        else:
            dataset_type = 'un-used'

        return dataset_type

if __name__ == '__main__':
    pass
