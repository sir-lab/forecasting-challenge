# Huawei Cloud Time Series Forecasting Challenge

This readme describes the instructions for the Huawei Time Series Forecasting Challenge for the Edinburgh EPSRC CDT in Machine Learning Systems.

## Setup

To set up the environment, clone the repository and run:

```bash
conda env create -f environment.yml
```

To activate the environment, use:

```bash
conda activate forecasting
```

## Downloading the Data

Download the dataset from [this link](https://sir-dataset.obs.cn-east-3.myhuaweicloud.com/datasets/cold_start_dataset/benchmarking_preview/data.7z), unzip it, and place `data.csv` in the `data` folder.

This is a cleaned version of the data used in our 2025 data release. Should you find it helpful to train on more data, you can download from our [2023 and 2025 data releases](https://github.com/sir-lab/data-release).


## Quick start

See the [quick start notebook](https://github.com/sir-lab/forecasting-challenge/blob/main/src/quickstart.ipynb) for an introduction to the data format and how to load and plot the data.


## Forecasting

### Darts forecasting code

We have written an [example script](https://github.com/sir-lab/forecasting-challenge/blob/main/src/darts_experiments.py) of how to quickly do time series forecasting using the [Darts framework](https://unit8co.github.io/darts/index.html). Darts is easy to use and has GPU support. It contains many state-of-the-art and baseline models.

Run the code using the command below to quickly train an FFT model for each time series.

```bash
python python src/darts_experiments.py --model_type fft
```

We have included a few example models in the script that run quickly on CPU, but it is easy to extend to use more models and utilize a GPU. 

You are by no means required to use any of the example code! It is just there to get you started.

## PyTorch forecasting code

For a PyTorch implementation, see the [linear forecasting repository](https://github.com/sir-lab/linear-forecasting/tree/main).


## Evaluating your results

Your results will be evaluated by SIR Lab on a held-out test dataset of 7 days. Due to the multivariate nature of our dataset, the overall forecasting performance will be evaluated using the average rank of each group by MAE. The Darts example script splits the training dataset of 31 days into 24 days of training and 7 days of testing just to illustrate the format you should output your results in. 

For more details on how this will be computed, see the [results notebook](https://github.com/sir-lab/forecasting-challenge/blob/main/src/evaluate_results.ipynb).

### Submission format

Your submissions for evaluation in this challenge should be presented as described below. For more details, refer to the example predictions file shown in the [results notebook](https://github.com/sir-lab/forecasting-challenge/blob/main/src/evaluate_results.ipynb).

You should submit 3 files for 3 different forecast horizons (1 day, 3 days, 7 days).
1. `df_predictions_1d.csv` - 1 day (time from 2678400 to 2764740, inclusive) with df.shape (1440, 241)
2. `df_predictions_3d.csv` - 3 days (time from 2678400 to 2937540, inclusive) with df.shape (4320, 241)
3. `df_predictions_7d.csv` - 7 days (time from 2678400 to 3283140, inclusive) with df.shape (10080, 241)

It is vital that you submit the files in the correct format.
