# Framework for Short-Term Load Forecasting

## Summary

This repository supports the development of short-term load forecasting (STLF) applications by providing a flexible framework and a variety of models. It includes both advanced deep learning architectures such as LSTMs, xLSTMs, and Transformers, as well as simpler baseline models like KNN and persistence prediction.

More information about the models and the framework can be found in the following paper: 

>  *Load Forecasting for Households and Energy Communities: Are Deep Learning Models Worth the Effort?* [https://arxiv.org/abs/2501.05000](https://arxiv.org/abs/2501.05000)


## Folder Structure

The repository is organized as follows:

| Folder           | Folder or File                                  | Description                                                                                  |
|------------------|-------------------------------------------------|----------------------------------------------------------------------------------------------|
| `data/`          |                                                 |                                                                                              |
|                  | `london_housholds_preprocessing.ipynb`          | Fetching and preprocessing of the London smartmeter dataset.                                 |
|                  | `london_loadprofiles_1households_each.pkl`      | Preprocessed 20 load profiles of single households.                                          |
|                  | `london_loadprofiles_2households_each.pkl`      | Preprocessed 20 load profiles of virtual energy communities with 2 households each.          |
|                  | `london_loadprofiles_10households_each.pkl`     | Preprocessed 20 load profiles of virtual energy communities with 10 households each.         |
|                  | `london_loadprofiles_50households_each.pkl`     | Preprocessed 20 load profiles of virtual energy communities with 50 households each.         |
|                  | `london_loadprofiles_100households_each.pkl`    | Preprocessed 20 load profiles of virtual energy communities with 100 households each.        |
|                  | `weather_data.py`                               | Script to fetch Meteostat weather data.                                                      |
| `envs/`          |                                                 |                                                                                              |
|                  | `env_linux.yml`                                 | Use this Conda environment file to reproduce the results of our paper on Linux.                                                  |
|                  | `env_from_nxai.yml`                               | Conda env file from xLSTM developers (reference)                                                 |
| `models/`        |                                                 |                                                                                              |
|                  | `Model.py`                                      | Model wrapper, owns exactly one model per instance.                                          |
|                  | `*.py`                                          | Implementations of the single deep learning/baseline models.                                 |
| `scripts/`       |                                                 |                                                                                              |
|                  | `case_study/`                                   | Example optimization of an energy community.                                                 |
|                  | `outputs/`                                      | Auto-generated figures, results, and profiles.                                               |
|                  | `ModelAdapter.py`                               | Brings the data into the model format.                                                       |
|                  | `ModelTrainer.py`                               | Big train and test loop accord to the config.                                                |
|                  | `Paper_Illustration.ipynb`                      | Create Figures and Tables after the testrun.                                                 |
|                  | `Simulation_config.py`                          | Config for a full automated simulation run.                                                  |
|                  | `Utils.py`                                      | Helper functions for the simulation run.                                                     |
|                  | `Model_Evaluation.ipynb`                        | Plotly Visualization of all model in- and outputs.                                           |


## Code Structure

The main parts of the evaluation framework are connected as follows:

```

+-----------------------------+           +------------------------------+
| Data                        |           | ModelAdapter                 |
|-----------------------------|           |------------------------------|
| # Weather, load, standard-  |           | + transformData()            |
|   load, and holidays.       +-----------+ # Preprocesses the data      |
+-----------------------------+           +------------+-----------------+
                                                       |
                                                       |
+-----------------------------+           +------------+-----------------+
| Simulation_config           |           | ModelTrainer                 |
|-----------------------------|           |------------------------------|
| configs: list               |           | + run()                      |
| # Parameterize the run      +-----------+ # Trains all models          |
| # loop.                     |           | # accord to the config.      |
+-----------------------------+           +------------+-----------------+
                                                       |
                                                       |
                                          +------------+-----------------+
                                          | Model                        |
                                          |------------------------------|
                                          | my_model: (xLSTM to KNN)     |
                                          | + train_model()              |
                                          | + evaluate()                 |
                                          +------------+-----------------+
                                                       |            
                                                       |                 
       +-----------------+-------------+---------------+-----------------+
       |                 |             |               |                 |
       |                 |             |               |                 |
+------+------+ +--------+----+ +------+------+ +------+------+ +--------+----+
| KNN         | | Persistence | | xLSTM       | | LSTM        | | Transformer |
|             | |             | |             | |             | |             |
|-------------| |-------------| |-------------| |-------------| |-------------|
| + forward() | | + forward() | | + forward() | | + forward() | | + forward() |
+-------------+ +-------------+ +-------------+ +-------------+ +-------------+

```

## Steps to Reproduce the Complete Paper

1. **Install the conda enviroment** on a linux system:
    ```bash
    conda env create --name load_forecasting --file=envs/env_linux.yml
    conda activate load_forecasting
    ```

2. **Train the models** using `ModelTrainer`:
    ```bash
    python scripts/ModelTrainer.py
    ```

3. To **evaluate the results,** just run `scripts/model_evaluate.ipynb` or `Paper_Illustration.ipynb`.


## Steps to Reuse Our Model Implementations

1. **Install** our model implementations as **package**:
    ```bash
    pip install git+https://github.com/erc-fhv/loadforecasting.git
    ```

2. **Use our implementation** in Python:
    ```python
    from models import Model
    import torch 


    # Train the sequence-to-sequence model
    #

    X_train = torch.randn(365, 24, 10)  # Your train features of shape (batch_len, sequence_len, features)
    Y_train = torch.randn(365, 24, 1)  # Your train target of shape (batch_len, sequence_len, 1)
    myModel = Model('Transformer', model_size='5k', num_of_features=X_train.shape[2])   # Alternative Models: 'LSTM', 'xLSTM', 'KNN'
    myModel.train_model(X_train, Y_train, pretrain_now=False, finetune_now=False, epochs=100, verbose=0)


    # Make predictions
    #

    X_test = torch.randn(90, 24, 10)  # Your test features of shape (batch_len, sequence_len, features)
    Y_pred = myModel.predict(X_test)
    print('\nOutput Shape = ', Y_pred.shape)

    ```


## Citation

If you use this codebase, or find our work valuable please cite the following paper:

```
@article{
  title={Load Forecasting for Households and Energy Communities: Are Deep Learning Models Worth the Effort?},
  author={Moosbrugger, Lukas and Seiler, Valentin and Wohlgenannt, Philipp and Hegenbart, Sebastian and Ristov, Sashko and Kepplinger, Peter},
  journal={Preprint submitted to Energy and AI},
  year={2025},
  doi={10.48550/arXiv.2501.05000}
}
```

