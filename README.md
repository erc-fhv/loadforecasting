# Shortterm Loadforecasting with xLSTM

## Summary

This work is an extension of [this](https://arxiv.org/abs/2407.08434) IEEE RTSI 2024 Conference Paper.
It uses transfer learning from standard load profiles together with state of the art models like xLSTMs and Transformers and compares them to very simple models.

## Software Design

The basic code design is as follows.
```
+-----------------------------+           +------------------------------+
| config.py                   |           | ModelTrainer                 |
|-----------------------------|           |------------------------------|
| configs: list               |           | + run()                      |
| # Parameterize the run      |---------> | # Trains all models          |           
| # loop.                     |           | # and persists the results.  |
+-----------------------------+           +------------------------------+
                                                      |
                                                      |
                                                      |
                                          +-----------v------------------+
                                          | Model                        |
                                          |------------------------------|
                                          | my_model: (xLSTM to KNN)     |
                                          | + train_model()              |
                                          | + evaluate()                 |
                                          +------------|-----------------+
                                                       |
       +---------------+---------------+---------------+-----------------+
       |               |               |               |                 | 
+------v------+ +------v------+ +------v------+ +------v------+ +--------v----+ 
| xLSTM       | | LSTM        | | Transformer | | Persistence | | KNN         |
|             | |             | |             | | Prediction  | |             |
|-------------| |-------------| |-------------| |-------------| |-------------|
| + forward() | | + forward() | | + forward() | | + forward() | | + forward() |
|             | |             | |             | |             | |             |
+-------------+ +-------------+ +-------------+ +-------------+ +-------------+
```

<!-- ## Components: todo! -->

## How to Use

1. **Install the conda enviroment** on a linux system:
    ```
    conda env create --name load_forecsting --file=envs/env_linux.yml
    conda activate load_forecasting
    ```

2. **Train the model** using `ModelTrainer`:
    ```python
    python scripts/ModelTrainer.py
    ```

3. **Evaluate the results** e.g. within `scripts/model_evaluate.ipynb`

## Citation

If you use this codebase, or find our work valuable please cite the following paper:

```
@article{
  title={Transfer learning for short-term prediction of electrical loads for domestic households and energy communities: a deep learning approach using xLSTM},
  author={Lukas Moosbrugger, Valentin Seiler, Sebastian Hegenbart, Philipp Wohlgenannt, Sashko Ristov, Peter Kepplinger},
  journal={arXiv preprint},
  year={2024}
}
```

