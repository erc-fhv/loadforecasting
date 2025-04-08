# Framework for Short-Term Load Forecasting

## Summary

This repository supports the development of short-term load forecasting (STLF) applications by providing a flexible framework and a variety of models. It includes both advanced deep learning architectures such as LSTMs, xLSTMs, and Transformers, as well as simpler baseline models like KNN and persistence prediction.

More information about the models and the framework can be found in the following paper: 

>  *Load Forecasting for Households and Energy Communities: Are Deep Learning Models Worth the Effort?* [https://arxiv.org/abs/2501.05000](https://arxiv.org/abs/2501.05000) *Submitted to Elsevier Energy and AI, December 2024*


## Folder Structure

The repository is organized as follows:

| Folder           | Folder or File                      | Description                                        |
|------------------|-------------------------------------|----------------------------------------------------|
| `data/`          |                                     |                                                    |
|                  | `london_housholds_preprocessing.ipynb`   | Preprocessing of smartmeter dataset                |
|                  | `london_loadprofiles_*.pkl`         | 20 virtual communities with varying households     |
|                  | `weather_data.py`                   | Script to fetch Meteostat weather data             |
| `envs/`          |                                     |                                                    |
|                  | `env_from_nxai.yml`                 | Conda env file from xLSTM developers (reference)   |
|                  | `env_linux.yml`                     | Provided Conda environment file                    |
| `models/`        |                                     |                                                    |
|                  | `Model.py`                          | Model wrapper, owns exactly one model.   |
|                  | `*.py`                              | Other deep learning/baseline model scripts         |
| `scripts/`       |                                     |                                                    |
|                  | `case_study/`                       | Example optimization of an energy community        |
|                  | `outputs/`                          | Auto-generated figures, results, and profiles      |
|                  | `ModelAdapter.py`                   | Brings the data into the model format              |
|                  | `ModelTrainer.py`                   | Big train and test loop accord to the config       |
|                  | `Paper_Illustration.ipynb`          | Create Figures and Tables after the testrun        |
|                  | `Simulation_config.py`              | Config for a full automated simulation run         |
|                  | `Utils.py`                          | Helper functions for the simulation run            |
|                  | `Visualization.py`                  | Plotly Visualization of all model in- and outputs  |


## Code Structure

The main parts of the evaluation framework are connected as follows:

```

+-----------------------------+           +------------------------------+
| Data                        |           | ModelAdapter                 |
|-----------------------------|           |------------------------------|
| # Weather, load, standard-  |           | + transformData()            |
|   load, and holidays.       |---------->| # Preprocesses the data      |
+-----------------------------+           +------------------------------+
                                                       |
                                                       |
+-----------------------------+           +------------v-----------------+
| Simulation_config           |           | ModelTrainer                 |
|-----------------------------|           |------------------------------|
| configs: list               |           | + run()                      |
| # Parameterize the run      |---------> | # Trains all models          |
| # loop.                     |           | # accord to the config.      |
+-----------------------------+           +------------------------------+
                                                       |
                                                       |
                                          +------------v-----------------+
                                          | Model                        |
                                          |------------------------------|
                                          | my_model: (xLSTM to KNN)     |
                                          | + train_model()              |
                                          | + evaluate()                 |
                                          +------------------------------+
+-------------+                                        |
| LSTM        <--------+                               |
|             |        |                               |                 
|-------------|        |                               |                 
| + forward() |        |                               |                 
+-------------+        |                               |                 
                       |                               |                 
+-------------+        |                               |                 
| Transformer <--------+----------+---------------+----+------------+
|             |        |          |               |                 |
|-------------|        |          |               |                 |
| + forward() |        |          |               |                 |
+-------------+        |          |               |                 |
                       |          |               |                 |
+-------------+        |   +------v------+ +------v------+ +--------v----+
| xLSTM       <--------+   | Persistence | | Synthetic   | | KNN         |
|             |            | Prediction  | | Load        | |             |
|-------------|            |-------------| |-------------| |-------------|
| + forward() |            | + forward() | | + forward() | | + forward() |
+-------------+            +-------------+ +-------------+ +-------------+

```

<!-- ## Components: todo! -->

## How to Use

1. **Install the conda enviroment** on a linux system:
    ```
    conda env create --name load_forecasting --file=envs/env_linux.yml
    conda activate load_forecasting
    ```

2. **Train the model** using `ModelTrainer`:
    ```python
    python scripts/ModelTrainer.py
    ```

3. **Evaluate the results** e.g. within `scripts/model_evaluate.ipynb` or within `Paper_Illustration.ipynb`.

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

