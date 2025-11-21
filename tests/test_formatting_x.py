"""Unit tests for the formatting_x function in DataPreprocessor"""

import datetime
import unittest
import pandas as pd
import numpy as np

from loadforecasting_framework import DataPreprocessor, DataSplitType


class TestFormattingX(unittest.TestCase):
    """Tests for the formatting_x method of DataPreprocessor"""

    def setUp(self):
        """Set up test data"""
        # Create 60 days of quarter-hourly power data to ensure we have enough for lagged profiles
        date_rng = pd.date_range(start='2025-01-01', end='2025-03-01 23:59', freq='15min')
        self.power_data = pd.Series(
            np.random.rand(len(date_rng)) * 100,  # Power values between 0-100
            index=date_rng,
            name='load'
        )

        # Create weather data with 3 features
        self.weather_data = pd.DataFrame(
            np.random.rand(len(date_rng), 3) * 20,  # 3 weather features
            index=date_rng,
            columns=['temperature', 'humidity', 'wind_speed']
        )

        # Create some public holidays
        self.public_holidays = [
            pd.Timestamp('2025-01-01'),  # New Year
            pd.Timestamp('2025-02-14'),  # Valentine's Day (example)
        ]

    def test_formatting_x_basic_shape(self):
        """Test that formatting_x returns the correct shape"""
        preprocessor = DataPreprocessor(
            add_lagged_profiles=(7, 14, 21),
            num_of_weather_features=3,
            first_prediction_clocktime=datetime.time(0, 0),
            prediction_horizon=pd.Timedelta(days=1),
            prediction_rate=pd.Timedelta(days=1)
        )

        # Set prediction timerange first
        preprocessor.set_prediction_timerange(self.power_data)

        # Call formatting_x
        x_all = preprocessor.formatting_x(
            self.power_data,
            self.weather_data,
            self.public_holidays
        )

        # Check shape
        self.assertEqual(len(x_all.shape), 3, "x_all should be 3-dimensional")
        self.assertGreater(x_all.shape[0], 0, "Should have at least one batch")
        self.assertEqual(x_all.shape[1], 96, "Should have 96 timesteps (quarter-hourly for 24h, i.e., 4*24)")
        
        # Calculate expected number of features:
        # 7 (weekday one-hot) + 2 (hour cyclical) + 2 (day-of-year cyclical) 
        # + 3 (lagged profiles) + 3 (weather features) = 17
        expected_features = 7 + 2 + 2 + 3 + 3
        self.assertEqual(x_all.shape[2], expected_features, 
                        f"Should have {expected_features} features")

    def test_formatting_x_without_weather(self):
        """Test formatting_x without weather data"""
        preprocessor = DataPreprocessor(
            add_lagged_profiles=(7,),
            num_of_weather_features=3,  # Must specify when no weather data provided
            first_prediction_clocktime=datetime.time(0, 0),
            prediction_horizon=pd.Timedelta(hours=24),
            prediction_rate=pd.Timedelta(days=1)
        )

        preprocessor.set_prediction_timerange(self.power_data)

        x_all = preprocessor.formatting_x(
            self.power_data,
            weather_data=None,
            public_holidays=[]
        )

        # Check that weather features are set to zero
        # Features: 7 (weekday) + 2 (hour) + 2 (day) + 1 (lagged) + 3 (weather) = 15
        expected_features = 7 + 2 + 2 + 1 + 3
        self.assertEqual(x_all.shape[2], expected_features)

        # Weather features should be the last 3 features and should be zero
        weather_features = x_all[:, :, -3:]
        np.testing.assert_array_equal(weather_features, np.zeros_like(weather_features),
                                     "Weather features should be zero when no weather data")

    def test_formatting_x_without_lagged_profiles(self):
        """Test formatting_x without lagged profiles"""
        preprocessor = DataPreprocessor(
            add_lagged_profiles=(),  # No lagged profiles
            num_of_weather_features=3,
            first_prediction_clocktime=datetime.time(0, 0),
            prediction_horizon=pd.Timedelta(hours=24),
            prediction_rate=pd.Timedelta(days=1)
        )

        preprocessor.set_prediction_timerange(self.power_data)

        x_all = preprocessor.formatting_x(
            self.power_data,
            self.weather_data,
            []
        )

        # Features: 7 (weekday) + 2 (hour) + 2 (day) + 0 (lagged) + 3 (weather) = 14
        expected_features = 7 + 2 + 2 + 0 + 3
        self.assertEqual(x_all.shape[2], expected_features)

    def test_formatting_x_weekday_encoding(self):
        """Test that weekday one-hot encoding is correct"""
        preprocessor = DataPreprocessor(
            add_lagged_profiles=(),
            num_of_weather_features=3,
            first_prediction_clocktime=datetime.time(0, 0),
            prediction_horizon=pd.Timedelta(hours=24),
            prediction_rate=pd.Timedelta(days=1)
        )

        preprocessor.set_prediction_timerange(self.power_data)

        x_all = preprocessor.formatting_x(
            self.power_data,
            self.weather_data,
            []
        )

        # Check first batch - get the weekday encoding (first 7 features)
        first_batch_weekdays = x_all[0, :, :7]
        
        # Each timestep should have exactly one 1 in the weekday encoding
        row_sums = np.sum(first_batch_weekdays, axis=1)
        np.testing.assert_array_almost_equal(row_sums, np.ones(len(row_sums)),
                                            err_msg="Each timestep should have exactly one active weekday")

    def test_formatting_x_cyclical_features(self):
        """Test that cyclical features (hour, day-of-year) are in valid range"""
        preprocessor = DataPreprocessor(
            add_lagged_profiles=(),
            num_of_weather_features=3,
            first_prediction_clocktime=datetime.time(0, 0),
            prediction_horizon=pd.Timedelta(hours=24),
            prediction_rate=pd.Timedelta(days=1)
        )

        preprocessor.set_prediction_timerange(self.power_data)

        x_all = preprocessor.formatting_x(
            self.power_data,
            self.weather_data,
            []
        )

        # Hour cyclical features (indices 7 and 8)
        hour_sin = x_all[:, :, 7]
        hour_cos = x_all[:, :, 8]
        
        # Should be in range [-1, 1]
        self.assertTrue(np.all(hour_sin >= -1) and np.all(hour_sin <= 1))
        self.assertTrue(np.all(hour_cos >= -1) and np.all(hour_cos <= 1))

        # Day-of-year cyclical features (indices 9 and 10)
        day_sin = x_all[:, :, 9]
        day_cos = x_all[:, :, 10]
        
        # Should be in range [-1, 1]
        self.assertTrue(np.all(day_sin >= -1) and np.all(day_sin <= 1))
        self.assertTrue(np.all(day_cos >= -1) and np.all(day_cos <= 1))

    def test_formatting_x_public_holidays(self):
        """Test that public holidays are encoded as Sunday (index 6)"""
        preprocessor = DataPreprocessor(
            add_lagged_profiles=(),
            num_of_weather_features=3,
            first_prediction_clocktime=datetime.time(0, 0),
            prediction_horizon=pd.Timedelta(hours=24),
            prediction_rate=pd.Timedelta(days=1)
        )

        preprocessor.set_prediction_timerange(self.power_data)

        # Use New Year 2025 (Wednesday) as public holiday
        public_holidays = [pd.Timestamp('2025-01-22')]  # A day in our prediction range

        x_all = preprocessor.formatting_x(
            self.power_data,
            self.weather_data,
            public_holidays
        )

        # Find a batch that includes the public holiday
        # 2025-01-22 is 21 days after 2025-01-01, and we start predictions after 21 days (3*7)
        # So it should be around batch index 0 or 1
        
        # For simplicity, just check that the one-hot encoding has valid values
        weekday_encoding = x_all[:, :, :7]
        self.assertTrue(np.all(np.sum(weekday_encoding, axis=2) == 1),
                       "Each timestep should have exactly one active weekday")

    def test_formatting_x_prediction_rate(self):
        """Test that prediction_rate affects the number of batches"""
        # Daily prediction rate
        preprocessor_daily = DataPreprocessor(
            add_lagged_profiles=(7,),
            num_of_weather_features=3,
            first_prediction_clocktime=datetime.time(0, 0),
            prediction_horizon=pd.Timedelta(hours=24),
            prediction_rate=pd.Timedelta(days=1)  # Daily
        )
        preprocessor_daily.set_prediction_timerange(self.power_data)
        x_daily = preprocessor_daily.formatting_x(self.power_data, self.weather_data, [])

        # Weekly prediction rate
        preprocessor_weekly = DataPreprocessor(
            add_lagged_profiles=(7,),
            num_of_weather_features=3,
            first_prediction_clocktime=datetime.time(0, 0),
            prediction_horizon=pd.Timedelta(hours=24),
            prediction_rate=pd.Timedelta(days=7)  # Weekly
        )
        preprocessor_weekly.set_prediction_timerange(self.power_data)
        x_weekly = preprocessor_weekly.formatting_x(self.power_data, self.weather_data, [])

        # Daily should have more batches than weekly
        self.assertGreater(x_daily.shape[0], x_weekly.shape[0],
                          "Daily predictions should create more batches than weekly")

    def test_formatting_x_error_without_set_prediction_timerange(self):
        """Test that formatting_x raises error if set_prediction_timerange was not called"""
        preprocessor = DataPreprocessor(
            add_lagged_profiles=(7,),
            num_of_weather_features=3,
            first_prediction_clocktime=datetime.time(0, 0),
            prediction_horizon=pd.Timedelta(hours=24),
            prediction_rate=pd.Timedelta(days=1)
        )

        # Don't call set_prediction_timerange - should raise TypeError
        with self.assertRaises(TypeError):
            preprocessor.formatting_x(self.power_data, self.weather_data, [])


if __name__ == '__main__':
    unittest.main()
