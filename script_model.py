#!/usr/bin/env python3

import logging
from pathlib import Path
from datetime import datetime
from argparse import ArgumentParser

from nanoz.utils import timing, version
from nanoz.nzio import ScriptConfig, configure_logger
from nanoz.modeling import InferenceAlgorithm


__author__ = "Matthieu Nugue"
__license__ = "Apache License 2.0"
__version__ = version()
__maintainer__ = "Matthieu Nugue"
__email__ = "matthieu.nugue@nanoz-group.com"
__status__ = "Development"


def parse_args():
    parser = ArgumentParser(prog="Script model",
                            description="Script model for inference")
    parser.add_argument("--config_path", type=str, required=True,
                        help="path to the json-file for script model configuration")
    return vars(parser.parse_args())


@timing
def initialization(io_path):
    start_time = datetime.now()

    # Load config files
    configs = ScriptConfig(io_path=io_path)

    # Results directories
    save_paths = {"output": Path(configs.io["output_path"], configs.paths["model"].stem)}
    save_paths["output"].mkdir(parents=True, exist_ok=True)

    # File logger
    _ = configure_logger(Path(save_paths["output"], start_time.strftime("%Y%m%d%H%M%S")+".log"))

    # Copy config files into the results directories
    configs.copy_config_files(save_paths["output"])

    # Log information
    logging.info('Algoz version {0}'.format(__version__))
    logging.info('Execution started at {0}'.format(start_time.strftime("%Y-%m-%d %H:%M:%S")))
    logging.info('Results directory created at: {0}'.format(save_paths["output"]))
    configs.log_info()

    return configs, save_paths


@timing
def get_algorithm(configs, save_paths):
    return InferenceAlgorithm(config=configs, save_paths=save_paths)


@timing
def main():
    args = parse_args()
    io_path = Path(args["config_path"])

    configs, save_paths = initialization(io_path)

    algorithm = get_algorithm(configs, save_paths)
    algorithm.save_model(save_paths["output"], mode="scripted")


if __name__ == "__main__":
    main()
