
## Overview

This Python package provides state-of-the-art models for short-term load forecasting (STLF), designed for both academic research and real-world energy applications.

The models and evaluation framework are described in the following publication:

> Moosbrugger et al. (2025). *Load Forecasting for Households and Energy Communities: Are Deep Learning Models Worth the Effort?*  
> [arXiv:2501.05000](https://arxiv.org/abs/2501.05000)

For more details and the full project source code, visit the [GitHub repository](https://github.com/erc-fhv/loadforecasting).

## Quick Start

Install the package:

```bash
pip install loadforecasting_models
```

You can easily integrate and train our forecasting machine learning models in your Python workflow:

```python
from loadforecasting_models import Knn, Lstm, Transformer, xLstm, Persistence, Normalizer
import torch

# ------------------------------------------------------------------------------
# Prepare training data using dummy (replace with your own) data
# ------------------------------------------------------------------------------
# Generate Random Data
x = torch.randn(365, 24, 10)   # Shape: (batches, seq_len, features)
y = torch.randn(365, 24, 1)    # Shape: (batches, seq_len, 1)

# Do train/test split
x_train = x[:330 ,: ,:]
y_train = y[:330 ,: ,:]
x_test = x[330: ,: ,:]
y_test = y[330: ,: ,:]

# Normalize data
normalizer = Normalizer()
x_train = normalizer.normalize_x(x_train, training=True)
y_train = normalizer.normalize_y(y_train, training=True)
x_test = normalizer.normalize_x(x_test, training=False)
y_test = normalizer.normalize_y(y_test, training=False)

# ------------------------------------------------------------------------------
# Train the model
# ------------------------------------------------------------------------------
myModel = Transformer(model_size='5k', normalizer=normalizer)
myModel.train_model(x_train, y_train, epochs=100, verbose=1)

# ------------------------------------------------------------------------------
# Make predictions
# ------------------------------------------------------------------------------
y_pred = myModel.predict(x_test)
y_pred = normalizer.de_normalize_y(y_pred)

print('\nOutput shape:', y_pred.shape)    # Should output 'torch.Size([35, 24, 1])'
```

Using a non-machine learning model is very similar, e.g. for a KNN model:

```python
from loadforecasting_models import Knn, Lstm, Transformer, xLstm, Persistence, Normalizer
import torch

# Same setup as above
# ...
myModel = Knn(k=40, weights='distance', normalizer=normalizer)
myModel.train_model(x_train, y_train)
# ...
```

Using automatic machine-learning hyperparameter tuning. This automatically optimizes 
hyperparameters like nr-of-epochs, batch-sizes, model-size, and learning-rate:

```python
from loadforecasting_models import Knn, Lstm, Transformer, xLstm, Persistence, Normalizer
import torch

# Same setup as above
# ...
myModel = Transformer()
myModel.train_model_auto(x_train, y_train)
# ...
```

## Currently Available Model Types:

-  'Transformer'

-  'Lstm'

-  'xLstm'

-  'Knn'

-  'Persistence'

## Citation

If you use this package in your work, please cite the following paper:

```
@article{moosbrugger2025load,
  title={Load Forecasting for Households and Energy Communities: Are Deep Learning Models Worth the Effort?},
  author={Moosbrugger, Lukas and Seiler, Valentin and Wohlgenannt, Philipp and Hegenbart, Sebastian and Ristov, Sashko and Eder, Elias and Kepplinger, Peter},
  journal={arXiv preprint},
  year={2025},
  doi={10.48550/arXiv.2501.05000}
}
```

## License

This project is open-source and available under the terms of the MIT License.

