import datetime
import unittest
import pandas as pd
import numpy as np

from loadforecasting_framework import DataPreprocessor, DataSplitType
from loadforecasting_models import Normalizer

class DummyNormalizer(Normalizer):
    """Dummy Normalizer for testing"""
    def normalize_x(self, x, training=False):
        return x
    def normalize_y(self, y, training=False):
        return y

class TestDataPreprocessor(unittest.TestCase):

    def setUp(self):
        # Dummy power data: 30 days of hourly data
        date_rng = pd.date_range(start='2025-01-01', end='2025-01-30 23:00', freq='H')
        self.power_data = pd.DataFrame(np.random.rand(len(date_rng), 1), index=date_rng,
            columns=['load'])

        # Dummy weather data: same length, 3 features
        self.weather_data = pd.DataFrame(np.random.rand(len(date_rng), 3), index=date_rng,
            columns=['w1', 'w2', 'w3'])

        # Dummy DataSplit
        self.data_split = DataSplitType(train_set_1=10, dev_set=5, test_set=5, train_set_2=5, pad=5)

        # Create DataPreprocessor instance
        self.preprocessor = DataPreprocessor(
            normalizer=DummyNormalizer(),
            data_split=self.data_split,
            add_lagged_profiles=(7,),
            num_of_weather_features=3,
            first_prediction_clocktime=datetime.time(0,0),
            prediction_horizon=pd.Timedelta(hours=23),
            prediction_rate=pd.Timedelta(days=1)
        )

    def test_transform_data_shapes(self):
        x, y = self.preprocessor.transform_data(self.power_data, self.weather_data)

        # Test if x and y are dictionaries with expected keys
        for dataset in ['train', 'dev', 'test', 'all']:
            self.assertIn(dataset, x)
            self.assertIn(dataset, y)

        # Test shape consistency
        for dataset in ['train', 'dev', 'test', 'all']:
            self.assertEqual(x[dataset].shape[0], y[dataset].shape[0])  # batch size
            self.assertEqual(y[dataset].shape[2], 1)  # y output dimension
            self.assertTrue(x[dataset].shape[2] > 0)  # x features > 0

    def test_first_prediction_date_set(self):
        # Before transform_data, _first_prediction_date should be None
        self.assertIsNone(self.preprocessor._first_prediction_date)

        # After transform_data, it should be a pd.Timestamp
        self.preprocessor.transform_data(self.power_data, self.weather_data)
        self.assertIsInstance(self.preprocessor._first_prediction_date, pd.Timestamp)

if __name__ == '__main__':
    unittest.main()
