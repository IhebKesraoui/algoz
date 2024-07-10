#!/usr/bin/env python3

import logging
from pathlib import Path
from datetime import datetime
import shutil

import pandas as pd
import torch

import nanoz as nz
from nanoz.config import ALGOZ_PATH
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


@timing
def initialization(json_filepath, device):
    start_time = datetime.now()

    # Load config files
    configs = Config(json_filepath)

    # Results directories
    train_path = Path(configs.model_path)
    inference_path = Path(train_path, 'inference')
    save_path = Path(inference_path, configs.folder_name)
    save_path_img = Path(save_path, 'img')
    save_path_img.mkdir(parents=True, exist_ok=True)

    # File logger
    flogger = nz.io.configure_logger(Path(save_path, 'test.log'))

    # Copy config files into the results directories
    configs.copy_config_file(save_path)
    shutil.copyfile('VERSION', Path(save_path, 'VERSION'))

    # Log information
    logging.info('Algoz version {0}'.format(__version__))
    logging.info('Execution started at {0}'.format(start_time.strftime("%Y-%m-%d %H:%M:%S")))
    logging.info('Results directory created at: {0}'.format(save_path))
    configs.device = device
    configs.log_info()

    return configs, save_path, save_path_img


@timing
def prepare_data(configs):
    dataset_test = UnfoldDataset(
        configs.data_test['data_paths'],
        configs.data_test['chips'],
        configs.data_test['gases'],
        configs.module['windows'],
        configs.data_test['minibatch_step'],
        device=configs.device,
        minibatch_scaler=configs.data_test['scaler'],
        thresholds=configs.data_test['thresholds'],  # TODO: transform
        artefacts_correction=configs.data_test['artefacts_correction']
    )
    return dataset_test


@timing
def get_model(configs):
    algorithm = Algorithm(configs.algo, configs.hyperparams, configs.module)
    model = algorithm.algo
    model.initialize()
    model.load_params(f_params=Path(configs.io['model_path'], configs.io['model_name']))
    model.module.eval()
    return model


@timing
def save_results(configs, save_path, save_path_img, dataset_test, model):
    # Results directories for performance evaluation
    save_path_perf = Path(save_path, 'perf')
    save_path_perf.mkdir(parents=True, exist_ok=True)

    # Prediction on test dataset
    ground_truth = dataset_test.ground_truth
    with torch.no_grad():
        prediction = model.predict(dataset_test)
    logging.debug('Prediction shape of test dataset: {0}'.format(prediction.shape))

    # Save predictions # TODO: saving class
    df_pred = pd.DataFrame()
    for id_gas, gas in enumerate(configs.data_test['gases']):
        gas_pred = gas + '_pred'
        df_pred[gas] = ground_truth[:, id_gas]
        df_pred[gas_pred] = prediction[:, id_gas]

        # Plot ground truth and prediction over the time
        prediction_plot = nz.visualization.NanozFig()
        prediction_plot.plot_one_y_axis(
            {gas: df_pred[gas], gas_pred: df_pred[gas_pred]},
            title='Ground truth and prediction of ' + gas + ' on ' + configs.folder_name,
            y_label='Concentration (ppm)',
            x_label='time (1pt = 100 ms)',
            save_path=Path(save_path_img, 'prediction_' + gas + '.png')
        )

        # Plot ground truth vs. prediction
        gt_pred_plot = nz.visualization.NanozFig()
        gt_pred_plot.scatter_one_y_axis(
            {gas: df_pred[gas_pred]},
            x=df_pred[gas],
            title='Ground truth vs. prediction of ' + gas + ' on ' + configs.folder_name,
            y_label='Prediction - Concentration (ppm)',
            x_label='Ground truth - Concentration (ppm)',
            save_path=Path(save_path_img, 'gt_vs_pred_' + gas + '.png')
        )
    df_pred.to_csv(Path(save_path, 'prediction.csv'))

    # Calculate performance metrics
    mn = configs.io['metrics_name']
    bv = configs.io['bin_values']
    df_perf = pd.DataFrame(columns=mn)
    for gas in configs.data_test['gases']:
        gas_pred = gas + '_pred'
        df_gas_perf = nz.evaluation.compute_metrics(df_pred[gas], df_pred[gas_pred], mn, gas)
        df_perf = pd.concat([df_perf, df_gas_perf], axis=0)

        df_gas_bin = nz.data_preparation.bin_df_values(df_pred[[gas, gas_pred]], gas, bv)
        df_gas_bin_perf = nz.evaluation.compute_metrics_on_bins(df_gas_bin[gas], df_gas_bin[gas_pred], mn)

        # Plot regression performances by concentrations
        conc_perf_plot = nz.visualization.NanozFig()
        conc_perf_plot.plot_double_y_axis(
            {'MAE': df_gas_bin_perf['MAE'].values, 'RMSE': df_gas_bin_perf['RMSE'].values},
            {'Maximum Error': df_gas_bin_perf['MAX'].values},
            x=df_gas_bin_perf.index.values.mid,
            title='Regression performances by concentrations of ' + gas + ' on ' + configs.folder_name,
            y_label=['MAE, RMSE (ppm)', 'Maximum Error (ppm)'],
            x_label='Concentration (ppm)',
            x_scale='log',
            x_ticks=df_gas_bin_perf.index.values.right,
            y_lim=0,
            y_lim_right=0,
            save_path=Path(save_path_img, 'bin_perf_' + gas + '.png')
        )

        # Plot regression performances by concentrations in percent
        conc_perf_percent_plot = nz.visualization.NanozFig()
        conc_perf_percent_plot.plot_double_y_axis(
            {'MAE': df_gas_bin_perf['MAPE'].values, 'RMSE': df_gas_bin_perf['RMSE%'].values},
            {'Maximum Error': df_gas_bin_perf['MAX%'].values},
            x=df_gas_bin_perf.index.values.mid,
            title='Regression performances by concentrations of ' + gas + ' on ' + configs.folder_name,
            y_label=['MAE, RMSE (%)', 'Maximum Error (%)'],
            x_label='Concentration (ppm)',
            x_scale='log',
            x_ticks=df_gas_bin_perf.index.values.right,
            y_lim=[0, 200],
            y_lim_right=[0, 500],
            save_path=Path(save_path_img, 'bin_perf_percent_' + gas + '.png')
        )
        df_gas_bin_perf.to_csv(Path(save_path_perf, 'perf_bin_' + gas + '.csv'))

    df_perf.to_csv(Path(save_path_perf, 'performances.csv'))


@timing
def main():
    # TODO: doc

    multi_inferences_path = Path(ALGOZ_PATH, 'config', 'io', 'multi_inferences')

    for json_path in multi_inferences_path.glob('desktop_*.json'):
        configs, save_path, save_path_img = initialization(json_path, 'cuda:1')
    
        dataset_test = prepare_data(configs)
    
        model = get_model(configs)
    
        save_results(configs, save_path, save_path_img, dataset_test, model)


if __name__ == "__main__":
    main()
