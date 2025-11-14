"""Unit tests for the Normalizer class"""

import unittest
import numpy as np
import torch

from loadforecasting_models.normalizer import Normalizer


class TestNormalizer(unittest.TestCase):
    """Tests for the Normalizer class"""

    def test_normalize_x_constant_features(self):
        """Test that normalize_x handles constant features correctly"""
        normalizer = Normalizer()
        
        # Create input data with constant features
        # Shape: (batches, timesteps, features)
        # Feature 0: constant non-zero (value 5.0)
        # Feature 1: constant zero (value 0.0)
        # Feature 2: varying feature
        x = np.array([
            [[5.0, 0.0, 1.0],
             [5.0, 0.0, 2.0],
             [5.0, 0.0, 3.0]],
            [[5.0, 0.0, 4.0],
             [5.0, 0.0, 5.0],
             [5.0, 0.0, 6.0]]
        ])
        
        # Normalize with training=True
        x_normalized = normalizer.normalize_x(x, training=True)
        
        # Check that normalization doesn't produce NaN or Inf
        self.assertFalse(np.isnan(x_normalized).any(), 
                        "Normalized data should not contain NaN")
        self.assertFalse(np.isinf(x_normalized).any(), 
                        "Normalized data should not contain Inf")
        
        # Check that constant non-zero feature (feature 0) is normalized
        # It should be zero-centered since it's constant
        expected_feature_0 = np.zeros_like(x_normalized[:, :, 0])
        np.testing.assert_array_almost_equal(
            x_normalized[:, :, 0], 
            expected_feature_0,
            err_msg="Constant non-zero feature should be normalized to zero"
        )
        
        # Check that constant zero feature (feature 1) is also zero after normalization
        expected_feature_1 = np.zeros_like(x_normalized[:, :, 1])
        np.testing.assert_array_almost_equal(
            x_normalized[:, :, 1],
            expected_feature_1,
            err_msg="Constant zero feature should remain zero after normalization"
        )
        
        # Check that std_x was set correctly
        # Feature 0: std should be set to max value (5.0) since it's constant
        # Feature 1: std should be set to 1.0 since it's a null feature
        # Feature 2: std should be calculated normally
        self.assertAlmostEqual(normalizer.std_x[0], 5.0, places=5,
                              msg="Constant non-zero feature std should be max value")
        self.assertAlmostEqual(normalizer.std_x[1], 1.0, places=5,
                              msg="Constant zero feature std should be 1.0")
        self.assertGreater(normalizer.std_x[2], 0,
                          msg="Varying feature std should be positive")
        
        # Check that mean_x was set correctly
        self.assertAlmostEqual(normalizer.mean_x[0], 5.0, places=5,
                              msg="Constant non-zero feature mean should be 5.0")
        self.assertAlmostEqual(normalizer.mean_x[1], 0.0, places=5,
                              msg="Constant zero feature mean should be 0.0")


if __name__ == '__main__':
    unittest.main()
