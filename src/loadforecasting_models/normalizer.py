import numpy as np
   

class Normalizer():
    """Simple normalizer class for input and output normalization and denormalization."""

    def __init__(self):
        self.mean_x = 0
        self.std_x = 0
        self.mean_y = 0
        self.std_y = 0

    def normalize(self, x, y, training=True):
        """Normalize both input and output data of the model."""

        x_normalized = self.normalize_x(x, training)
        y_normalized = self.normalize_y(y, training)

        return x_normalized, y_normalized

    def de_normalize(self, x, y):
        """De-Normalize both input and output data of the model."""

        x_de_normalized = self.de_normalize_x(x)
        y_de_normalized = self.de_normalize_y(y)

        return x_de_normalized, y_de_normalized

    def normalize_x(self, x, training=True):
        """Z-Normalize the input data of the model."""

        if training:
            # Estimate the mean and standard deviation of the data during training
            self.mean_x = np.mean(x, axis=(0, 1))
            self.std_x = np.std(x, axis=(0, 1))

            if np.isclose(self.std_x, 0).any():
                # Avoid a division by zero (which can occur for constant features)
                self.std_x = np.where(np.isclose(self.std_x, 0), 1e-8, self.std_x)

        x_normalized = (x - self.mean_x) / self.std_x

        return x_normalized

    def normalize_y(self, y, training=True):
        """Z-Normalize the output data of the model."""

        if training:
            # Estimate the mean and standard deviation of the data during training
            self.mean_y = np.mean(y, axis=(0, 1))
            self.std_y = np.std(y)

        if np.isclose(self.std_y, 0):
            assert False, "Normalization leads to division by zero."

        y_normalized = (y - self.mean_y) / self.std_y

        return y_normalized

    def de_normalize_y(self, y):
        """Undo normalization"""

        y_denormalized = (y * self.std_y) + self.mean_y

        return y_denormalized

    def de_normalize_x(self, x):
        """Undo z-normalization."""

        x_denormalized = (x * self.std_x) + self.mean_x

        return x_denormalized

