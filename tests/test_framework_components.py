"""
This module contains unit tests for the data preprocessing and model training components
of the load forecasting framework.
"""

import unittest
import pandas as pd
import numpy as np

from loadforecasting_models import Normalizer, Transformer
from loadforecasting_framework import DataPreprocessor, ModelTrainer, DataSplitType


class TestDataPipeline(unittest.TestCase):
    """Tests for the full data preprocessing and model training pipeline."""

    def setUp(self):
        """Set up synthetic data and components for testing."""

        # Generate synthetic test data
        self.timestamps = pd.date_range(start='2023-01-01', end='2024-01-01', freq='15min')

        self.df_load = pd.DataFrame({
            'timestamp': self.timestamps,
            'load': np.random.rand(len(self.timestamps)) * 1000
        }).set_index('timestamp')

        self.weather_data = pd.DataFrame({
            'date': self.timestamps,
            'precipitation': np.random.rand(len(self.timestamps)),
            'cloud_cover': np.random.rand(len(self.timestamps)) * 100,
            'global_tilted_irradiance': np.random.rand(len(self.timestamps)) * 1000,
        }).set_index('date')

        # Setup processing components
        self.normalizer = Normalizer()
        self.model_trainer = ModelTrainer()

        self.public_holidays = self.model_trainer.load_holidays(
            start_date=self.df_load.index.min(),
            end_date=self.df_load.index.max(),
            country='AT',
            subdiv='Vorarlberg'
        )

        self.data_preprocessor = DataPreprocessor(
            normalizer=self.normalizer,
            add_lagged_profiles=(7, 14, 21),
            data_split=DataSplitType(
                test_set=90,    # 90 days for testing
                train_set_1=-1, # use the rest for training
                dev_set=0,
                train_set_2=0,
                pad=0
                ),
                add_calendar_year_feature=False,
        )

    def test_full_pipeline(self):
        """Test the full data preprocessing and model training pipeline."""

        # Transform data
        x, y = self.data_preprocessor.transform_data(
            power_profile=self.df_load['load'],
            weather_data=self.weather_data,
            public_holidays=self.public_holidays
        )

        # Ensure the split exists
        self.assertIn("train", x)
        self.assertIn("test", x)

        # Instantiate model
        model = Transformer("5k", normalizer=self.normalizer)

        # Train model
        history = model.train_model(
            x_train=x["train"],
            y_train=y["train"],
            epochs=5,   # shortened for test runtime
            verbose=0,
        )

        # Ensure training produced a history
        self.assertIsNotNone(history)

        # Evaluate model
        test_loss = model.evaluate(
            x_test=x['test'],
            y_test=y["test"],
            de_normalize=True,
            loss_relative_to="mean"
        )

        # Loss should be numeric
        self.assertIsInstance(test_loss, dict)


if __name__ == '__main__':
    unittest.main()
