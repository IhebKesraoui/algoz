#!/usr/bin/env python3

import logging
from pathlib import Path
from datetime import datetime
from argparse import ArgumentParser

from torch import no_grad

from nanoz.utils import timing, version
from nanoz.nzio import ConfigFactory, configure_logger
from nanoz.data_preparation import DatasetFactory
from nanoz.modeling import AvailableAlgorithm, AlgorithmFactory
from nanoz.evaluation import EvaluatorFactory
from nanoz.visualization import save_data_distribution, DashboardFactory


__author__ = "Matthieu Nugue"
__license__ = "Apache License 2.0"
__version__ = version()
__maintainer__ = "Matthieu Nugue"
__email__ = "matthieu.nugue@nanoz-group.com"
__status__ = "Development"


def parse_args():
    parser = ArgumentParser(prog="Algoz",
                            description="Tools to train, test, predict, resume training and optimize hyperparameters.")
    parser.add_argument("--mode", type=str, required=True,
                        help="mode of execution: train, inference, resume_training or hyperparameter_optimization")
    parser.add_argument("--config_path", type=str, required=True,
                        help="path to the json-file for algoz configuration")
    return vars(parser.parse_args())


@timing
def initialization(mode, io_path):
    start_time = datetime.now()

    # Load config files
    configs = ConfigFactory.create_config(mode, io_path=io_path)
    print(configs)
    # Results directories
    save_paths = {"output": Path(configs.io["output_path"], start_time.strftime("%Y%m%d%H%M%S"))}
    save_paths["output"].mkdir(parents=True, exist_ok=True)

    # File logger
    _ = configure_logger(Path(save_paths["output"], mode+".log"))

    # Copy config files into the results directories
    configs.copy_config_files(save_paths["output"])

    # Assign device
    configs.device_assignment()
    # Log information
    logging.info('Algoz version {0}'.format(__version__))
    logging.info('Execution started at {0}'.format(start_time.strftime("%Y-%m-%d %H:%M:%S")))
    logging.info('Results directory created at: {0}'.format(save_paths["output"]))
    configs.log_info()

    return configs, save_paths


@timing
def get_datasets(configs, save_paths):
    # Create validation or test dataset
    datasets = {"test": DatasetFactory.create_dataset(configs.algo,
                                                      config_data=configs.parameters["config_data_test"],
                                                      device=configs.device)}

    if configs.mode == "train" or configs.mode == "resume" or configs.mode == "hyperparameter":
        # Create train dataset
        datasets["train"] = DatasetFactory.create_dataset(configs.algo,
                                                          config_data=configs.parameters["config_data_train"],
                                                          device=configs.device)
        datasets["train"].shuffle()

    # Save dataset(s) distribution
    save_data_distribution(datasets, Path(save_paths["output"], "data_distribution.html"))
    return datasets


@timing
def get_algorithm(mode, configs, datasets, save_paths):
    return AlgorithmFactory.create_algorithm(mode, config=configs, datasets=datasets, save_paths=save_paths)


@timing
def save_results(configs, algorithm, datasets, save_paths):
    # Get algorithm type
    algo_type = AvailableAlgorithm.get_type(configs.algo)
    if algo_type not in ["regression", "classification"]:
        logging.warning(f"No performance evaluation for this algorithm type: {algo_type}.")
        return

    # Create results directory
    save_paths["results"] = Path(save_paths["output"], "results")
    save_paths["results"].mkdir(parents=True, exist_ok=True)

    # Prepare model for inference  TODO: in Algorithm ?
    algorithm.model.module.eval()

    predictions = {}
    probabilities = {}
    for name, dataset in datasets.items():
        dataset.unshuffle()

        with no_grad():
            predictions[name] = algorithm.model.predict(dataset)
            probabilities[name] = algorithm.model.predict_proba(dataset)
        logging.debug(f'Prediction shape of {name} dataset: {predictions[name].shape}')
        logging.debug(f'Proba shape of {name} dataset: {probabilities[name].shape}')

        # Get intervals
        intervals = None
        if algo_type == "regression":
            try:  # regression with performance intervals
                intervals = configs.io["performance_intervals"]
            except KeyError:
                pass
        elif algo_type == "classification":
            intervals = dataset.classes

        # Compute performance metrics
        evaluator = EvaluatorFactory.create_evaluator(configs.algo,
                                                      ground_truth=dataset.ground_truth,
                                                      prediction=predictions[name],
                                                      probabilities=probabilities[name],
                                                      intervals=intervals,
                                                      targets_name=dataset.config_data["gases"])

        # Save csv-files
        if configs.mode == "inference":
            evaluator.save_prediction(Path(save_paths["results"], "prediction").with_suffix(".csv"))
        evaluator.target_performances.to_csv(Path(save_paths["results"], name + "_performances").with_suffix(".csv"))

        if algo_type == "regression" and intervals:
            for target, df_perf in evaluator.intervals_performances.items():
                df_perf.to_csv(
                    Path(save_paths["results"], name + "_" + target + "_intervals_performances").with_suffix(".csv")
                )
        elif algo_type == "classification" and intervals:
            evaluator.save_confusion_matrix(Path(save_paths["results"], name + "_cm").with_suffix(".csv"))
            evaluator.intervals_performances.to_csv(
                Path(save_paths["results"], name + "_intervals_performances").with_suffix(".csv")
            )

        # Save results as a dashboard
        dashboard = DashboardFactory.create_dashboard(configs.algo, name=name, evaluator=evaluator)
        dashboard.save_dashboards(save_paths["results"], name)


@timing
def main():
    args = parse_args()
    mode = args["mode"]
    io_path = Path(args["config_path"])

    configs, save_paths = initialization(mode, io_path)
    print(configs)
    datasets = get_datasets(configs, save_paths)
    print(configs.parameters)
    algorithm = get_algorithm(mode, configs, datasets, save_paths)

    save_results(configs, algorithm, datasets, save_paths)


if __name__ == "__main__":
    main()
