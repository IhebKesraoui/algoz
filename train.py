#!/usr/bin/env python3

import logging
from pathlib import Path
from datetime import datetime
import shutil

import pandas as pd
from skorch.helper import predefined_split

import nanoz as nz
from nanoz.utils import timing, version
from nanoz.io import Config
from nanoz.data_preparation import UnfoldDataset
from nanoz.modeling import Algorithm

__author__ = "Matthieu Nugue"
__license__ = "Apache License 2.0"
__version__ = version()
__maintainer__ = "Matthieu Nugue"
__email__ = "matthieu.nugue@nanoz-group.com"
__status__ = "Development"

pd.options.mode.chained_assignment = None  # Disable warning SettingWithCopyWarning, default='warn'


# TODO: fix that for merging ML and DL
# Logging by scikit-learn is done by the print(), to have it saved into an external file,
# you should redirect sys.stdout to a file
# sys.stdout = open(Path(dir_to_save, 'train.log'),"a")

"""
# TODO: dataloader and sampler ?
dataloader = DataLoader(mydataset, batch_size=4,
                            shuffle=False, num_workers=0)
# TODO: Slicedataset for gridsearch ?
X_sl = SliceDataset(mydataset, idx=0)
print('X_sl')
print(X_sl)
y_sl = SliceDataset(mydataset, idx=1)
print('y_sl')
print(y_sl)
net_regr.fit(X_sl, y_sl)"""


@timing
def initialization():
    start_time = datetime.now()

    # Load config files
    configs = Config(Path('config', 'io', 'desktop_train.json'))
    # TODO: check integrity of .json: error for missing key and warning for not know key

    # Results directories
    save_path = Path(configs.io['output_path'], start_time.strftime("%Y%m%d%H%M%S") + '_' + configs.io['algorithm'])
    save_path_img = Path(save_path, 'img')
    save_path_img.mkdir(parents=True, exist_ok=True)
    save_path_data = Path(save_path_img, 'data')
    save_path_data.mkdir(parents=True, exist_ok=True)

    # File logger
    flogger = nz.io.configure_logger(Path(save_path, 'train.log'))

    # Copy config files into the results directories
    configs.copy_config_file(save_path)
    shutil.copyfile('VERSION', Path(save_path, 'VERSION'))

    # Log information
    logging.info('Algoz version {0}'.format(__version__))
    logging.info('Execution started at {0}'.format(start_time.strftime("%Y-%m-%d %H:%M:%S")))
    logging.info('Results directory created at: {0}'.format(save_path))
    configs.device = configs.get_device()
    configs.log_info()

    return configs, save_path, save_path_img, save_path_data


@timing
def prepare_data(configs, save_path_data):
    dataset_train = UnfoldDataset(
        configs.data_train['data_paths'],
        configs.data_train['chips'],
        configs.data_train['gases'],
        configs.module['windows'],
        configs.data_train['minibatch_step'],
        configs.data_train['data_skip'],
        device=configs.device,
        minibatch_scaler=configs.data_train['scaler'],
        thresholds=configs.data_train['thresholds'],  # TODO: transform
        artefacts_correction=configs.data_train['artefacts_correction']
    )
    dataset_train.plot_density(Path(save_path_data, 'Train'))
    dataset_train.shuffle()
    dataset_test = UnfoldDataset(
        configs.data_test['data_paths'],
        configs.data_test['chips'],
        configs.data_test['gases'],
        configs.module['windows'],
        configs.data_test['minibatch_step'],
        configs.data_test['data_skip'],
        device=configs.device,
        minibatch_scaler=configs.data_test['scaler'],
        thresholds=configs.data_test['thresholds'],  # TODO: transform
        artefacts_correction=configs.data_test['artefacts_correction']
    )
    dataset_test.plot_density(Path(save_path_data, 'Test'))
    dataset_train.plot_comparison_density(Path(save_path_data, 'Comparison'), dataset_test)
    return dataset_train, dataset_test


@timing
def train_model(configs, save_path, dataset_train, dataset_test):
    # Prepare the algorithm
    algorithm = Algorithm(configs.io['algorithm'], configs.hyperparams, configs.module, configs.device)
    algorithm.algo.train_split = predefined_split(dataset_test)
    model = algorithm.algo
    model.callbacks[0].dirname = save_path  # TODO: do it inside Algorithm

    # Train the model
    model.fit(dataset_train, y=None)
    return model


@timing
def save_results(configs, save_path, save_path_img, dataset_train, dataset_test, model):
    # Results directories for performance evaluation
    save_path_perf = Path(save_path, 'perf')
    save_path_perf.mkdir(parents=True, exist_ok=True)

    # Save history  # TODO: saving class
    model_history = model.history.to_list()
    df_history = pd.DataFrame(model_history)
    df_history = df_history.drop(['batches'], axis=1)

    loss_plot = nz.visualization.NanozFig()
    loss_plot.plot_one_y_axis(
        {'train_loss': df_history['train_loss'], 'valid_loss': df_history['valid_loss']},
        x=df_history['epoch'],
        title='Loss during training',
        y_label='MSELoss',
        x_label='epoch',
        save_path=Path(save_path_img, 'loss.png')
    )
    df_history.to_csv(Path(save_path, 'history.csv'))
    del df_history, model_history

    # Loop on train and test subsets
    mn = configs.io['metrics_name']
    bv = configs.io['bin_values']
    df_perf = pd.DataFrame(columns=mn)
    for dataset, config, subset in zip(
            [dataset_train, dataset_test], [configs.data_train, configs.data_test], ['Train', 'Test']):

        # Unshuffle dataset
        dataset.unshuffle()

        # Prediction on training dataset
        ground_truth = dataset.ground_truth
        prediction = model.predict(dataset)
        logging.debug('Prediction shape of {0} dataset: {1}'.format(subset.lower(), prediction.shape))

        # Save predictions # TODO: saving class
        df_pred = pd.DataFrame()
        for id_gas, gas in enumerate(config['gases']):
            gas_pred = gas + '_pred'
            df_pred[gas] = ground_truth[:, id_gas]
            df_pred[gas_pred] = prediction[:, id_gas]

            # Plot ground truth and prediction over the time
            prediction_plot = nz.visualization.NanozFig()
            prediction_plot.plot_one_y_axis(
                {gas: df_pred[gas], gas_pred: df_pred[gas_pred]},
                title='Ground truth and prediction of ' + gas + ' on ' + subset.lower() + ' subset',
                y_label='Concentration (ppm)',
                x_label='time (1pt = 100 ms)',
                save_path=Path(save_path_img, 'prediction_' + subset.lower() + '_' + gas + '.png')
            )

            # Plot ground truth vs. prediction
            gt_pred_plot = nz.visualization.NanozFig()
            gt_pred_plot.scatter_one_y_axis(
                {gas: df_pred[gas_pred]},
                x=df_pred[gas],
                title='Ground truth vs. prediction of ' + gas + ' on ' + subset.lower() + ' subset',
                y_label='Prediction - Concentration (ppm)',
                x_label='Ground truth - Concentration (ppm)',
                save_path=Path(save_path_img, 'gt_vs_pred_' + subset.lower() + '_' + gas + '.png')
            )
        df_pred.to_csv(Path(save_path, 'prediction_' + subset.lower() + '.csv'))

        # Calculate performance metrics
        for gas in config['gases']:
            gas_pred = gas + '_pred'
            gas_name = gas + '_' + subset.lower()
            df_gas_perf = nz.evaluation.compute_metrics(df_pred[gas], df_pred[gas_pred], mn, gas_name)
            df_perf = pd.concat([df_perf, df_gas_perf], axis=0)

            df_gas_bin = nz.data_preparation.bin_df_values(df_pred[[gas, gas_pred]], gas, bv)
            df_gas_bin_perf = nz.evaluation.compute_metrics_on_bins(df_gas_bin[gas], df_gas_bin[gas_pred], mn)

            # Plot regression performances by concentrations
            conc_perf_plot = nz.visualization.NanozFig()
            conc_perf_plot.plot_double_y_axis(
                {'MAE': df_gas_bin_perf['MAE'].values, 'RMSE': df_gas_bin_perf['RMSE'].values},
                {'Maximum Error': df_gas_bin_perf['MAX'].values},
                x=df_gas_bin_perf.index.values.mid,
                title='Regression performances by concentrations of ' + gas + ' on ' + subset.lower() + ' subset',
                y_label=['MAE, RMSE (ppm)', 'Maximum Error (ppm)'],
                x_label='Concentration (ppm)',
                x_scale='log',
                x_ticks=df_gas_bin_perf.index.values.right,
                y_lim=0,
                y_lim_right=0,
                save_path=Path(save_path_img, 'bin_perf_' + subset.lower() + '_' + gas + '.png')
            )

            # Plot regression performances by concentrations in percent
            conc_perf_percent_plot = nz.visualization.NanozFig()
            conc_perf_percent_plot.plot_double_y_axis(
                {'MAE': df_gas_bin_perf['MAPE'].values, 'RMSE': df_gas_bin_perf['RMSE%'].values},
                {'Maximum Error': df_gas_bin_perf['MAX%'].values},
                x=df_gas_bin_perf.index.values.mid,
                title='Regression performances by concentrations of ' + gas + ' on ' + subset.lower() + ' subset',
                y_label=['MAE, RMSE (%)', 'Maximum Error (%)'],
                x_label='Concentration (ppm)',
                x_scale='log',
                x_ticks=df_gas_bin_perf.index.values.right,
                y_lim=[0, 200],
                y_lim_right=[0, 500],
                save_path=Path(save_path_img, 'bin_perf_percent_' + subset.lower() + '_' + gas + '.png')
            )
            df_gas_bin_perf.to_csv(Path(save_path_perf, 'perf_bin_' + subset.lower() + '_' + gas + '.csv'))

    df_perf.to_csv(Path(save_path_perf, 'performances.csv'))


@timing
def main():
    # TODO: doc

    configs, save_path, save_path_img, save_path_data = initialization()

    dataset_train, dataset_test = prepare_data(configs, save_path_data)

    model = train_model(configs, save_path, dataset_train, dataset_test)

    save_results(configs, save_path, save_path_img, dataset_train, dataset_test, model)


if __name__ == "__main__":
    main()
