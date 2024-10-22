import argparse
import os
from datetime import datetime
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from darts import TimeSeries
from darts.models import LinearRegressionModel, NHiTSModel, FFT
from tqdm import tqdm
import json

def parse_args():
    parser = argparse.ArgumentParser(description="Train time series models and evaluate performance.")
    parser.add_argument('--data_path', type=str, default='data/data.csv', help='Path to the dataset CSV file')
    parser.add_argument('--train_days', type=int, default=24, help='Number of days to use for training')
    parser.add_argument('--model_type', type=str, default='linear', help='Type of model to use (e.g., linear)')
    parser.add_argument('--context_length', type=int, default=1440, help='Context length for the model')
    parser.add_argument('--target_length', type=int, default=10, help='Target length for the model')
    parser.add_argument('--plot', type=bool, default=True, help='Whether to plot predictions or not')
    return parser.parse_args()

def plot_predictions(train, test, prediction, ts_name, output_dir):
    """
    Plots the training, test, and prediction data for a time series and saves the plot as a PNG.

    Parameters:
    - train (TimeSeries): The training data as a Darts TimeSeries object.
    - test (TimeSeries): The test data as a Darts TimeSeries object.
    - prediction (TimeSeries): The prediction data as a Darts TimeSeries object.
    - ts_name (str): The name of the time series, used as a title and filename.
    - output_dir (str): The directory where the plot image will be saved.

    Returns:
    - None
    """
    fig, ax = plt.subplots(1, 1, figsize=(15, 2))
    train.plot(label='train', ax=ax)
    test.plot(label='test', ax=ax, color='dodgerblue')
    prediction.plot(label="prediction", color='red', ax=ax)
    plt.legend(loc='upper left')
    title_str = f'{ts_name}'
    plt.title(title_str)
    save_str = f'{output_dir}/{ts_name}.png'
    plt.tight_layout()
    plt.savefig(save_str)
    plt.close()

def main():
    args = parse_args()

    # Set up paths and other parameters
    datetime_str = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    output_dir = f'results/{datetime_str}/'
    os.makedirs(output_dir, exist_ok=True)

    args_dict = vars(args)
    with open(f'{output_dir}/config.json', 'w') as f:
        json.dump(args_dict, f, indent=4)
    print(f'Saved configuration to {output_dir}config.json')

    # Load and preprocess data
    df = pd.read_csv(args.data_path)
    dummy_start_date = datetime(2024, 11, 1, 0, 0, 0)
    df['time'] = pd.to_timedelta(df['time'], unit='s') + dummy_start_date

    # Split train and test
    df_train = df.iloc[:args.train_days * 1440]
    df_test = df.iloc[args.train_days * 1440:]
    test_time_col = (np.arange(len(df_test['time'])) + len(df_train))*60

    print(f'Train data from {df_train.time.min()} to {df_train.time.max()}')
    print(f'Test data from {df_test.time.min()} to {df_test.time.max()}')

    colnames = [col for col in df.columns if col != 'time']
    ts_names = []
    predictions_list = []

    print(f'Training {len(colnames)} {args.model_type} models')
    for ts_name in tqdm(colnames):
        train = TimeSeries.from_dataframe(df_train, "time", ts_name)
        test = TimeSeries.from_dataframe(df_test, "time", ts_name)

        # Select model
        if args.model_type=='linear':
            # This is a linear model with weights from every sample in the context window to every sample in the horizon window.
            # Linear model documentation: https://unit8co.github.io/darts/generated_api/darts.models.forecasting.linear_regression_model.html
            model = LinearRegressionModel(
                lags=args.context_length, 
                output_chunk_length=args.target_length
            )
        elif args.model_type=='nhits':
            # This is a deep neural network designed for time series. It is (relatively) quick to train.
            # N-HiTS model documentation: https://unit8co.github.io/darts/generated_api/darts.models.forecasting.linear_regression_model.html 
            model = NHiTSModel(
                args.context_length, 
                args.target_length, 
                n_epochs=5, 
                layer_widths=128
            )
        elif args.model_type=='fft':
            # This is an FFT model. It is simple and quick to train. 
            # FFT model documentation: https://unit8co.github.io/darts/generated_api/darts.models.forecasting.fft.html 
            model = FFT(
                nr_freqs_to_keep=20,
                trend= "poly",
                trend_poly_degree=2
            )
        else:
            print(f'{args.model_type} has not been implemented yet.')
            raise NotImplementedError

        # Fit the model. For some models in Darts, you can optionally supply a validation set 'val_series' to
        # enable early stopping to prevent overfitting, which is described at the link below.
        # https://unit8co.github.io/darts/generated_api/darts.models.forecasting.nhits.html#darts.models.forecasting.nhits.NHiTSModel.fit
        model.fit(train)

        # Make a forecast of a given length.
        prediction = model.predict(len(test))

        ts_names.append(ts_name)
        predictions_list.append(prediction.values())

        # Plot
        if args.plot:
            plot_predictions(train, test, prediction, ts_name, output_dir)

    # Save results to DataFrames and CSV
    df_predictions = pd.DataFrame(np.concatenate(predictions_list, axis=1), columns=ts_names)
    df_predictions.insert(0, 'time', test_time_col)
    df_predictions.to_csv(f'{output_dir}/df_predictions.csv', index=False)
    print(f'Saved df_predictions to {output_dir}')

if __name__ == "__main__":
    main()
