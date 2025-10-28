
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
from loadforecasting_models import Transformer
import torch

# Generate dummy training data
X_train = torch.randn(365, 24, 10)  # shape: (batch_size, sequence_length, num_features)
Y_train = torch.randn(365, 24, 1)   # shape: (batch_size, sequence_length, 1)

# Normalize data
my_normalizer = Normalizer()
X_train, Y_train = my_normalizer(X_train, Y_train, training=True)

# Initialize and train the model
model = Transformer(model_size='5k', num_of_features=X_train.shape[2])
model.train_model(X_train, Y_train, pretrain_now=False, finetune_now=False, epochs=100, verbose=0)

# Generate predictions
X_test = torch.randn(90, 24, 10)
Y_pred = model.predict(X_test)

print(f"Prediction output shape: {Y_pred.shape}")
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

