
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

You can easily integrate and train our forecasting models in your Python workflow. Here's an example using the Transformer-based sequence-to-sequence model:

```python
from loadforecasting_models import Knn, Lstm, Transformer, xLstm, Persistence, Normalizer
import torch 

# Please define as needed
#
features = 10
seq_len = 24
batches_train = 365
batches_test = 90

# Train the sequence-to-sequence model
#
normalizer = Normalizer()
x_train = torch.randn(batches_train, seq_len, features)  # Your train features of shape (batch_len, sequence_len, features)
x_train = normalizer.normalize_x(x_train, training=True)  # Normalize x
y_train = torch.randn(batches_train, seq_len, 1)  # Your train target of shape (batch_len, sequence_len, 1)
y_train = normalizer.normalize_y(y_train, training=True)  # Normalize y
myModel = Transformer(model_size='5k', num_of_features=features, normalizer=normalizer)   # Alternative Models: 'LSTM', 'xLSTM', 'KNN'
myModel.train_model(x_train, y_train, epochs=100, verbose=0)


# Predict
#
x_test = torch.randn(batches_test, seq_len, features)  # Your test features of shape (batch_len, sequence_len, features)
x_test = normalizer.normalize_x(x_test, training=False)
y_pred = myModel.predict(x_test)
y_pred = normalizer.normalize_y(y_pred, training=False)
print('\nOutput Shape = ', y_pred.shape)

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

