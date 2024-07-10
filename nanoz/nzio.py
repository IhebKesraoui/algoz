#!/usr/bin/env python3

"""This file is renamed from io.py to nzio.py to avoid PyCharm brekpoint issue."""

from pathlib import Path
import logging
import shutil
import json
from dataclasses import dataclass

from torch import cuda

from nanoz.config import ALGOZ_PATH
from nanoz.utils import assign_free_gpus


@dataclass
class ConsoleLogger:
    """
    Data class to configure the console logger.

    Attributes
    ----------
    level : int
        Logging level.
    format : str
        Format of the record.
    date_format : str
        Format of the date.
    """
    level: int = logging.INFO
    format: str = "[%(levelname)s] %(message)s"
    date_format: str = "%m/%d/%Y %I:%M:%S %p"


@dataclass
class FileLogger:
    """
    Data class to configure the file logger.

    Attributes
    ----------
    level : int
        Logging level.
    format : str
        Format of the record.
    date_format : str
        Format of the date.
    mode : str
        Access mode of the file.
    encoding : str
        Character encoding of the file.
    """
    level: int = logging.DEBUG
    format: str = "[%(asctime)s] [%(levelname)s] [%(pathname)s:%(lineno)d] %(message)s"
    date_format: str = "%m/%d/%Y %I:%M:%S %p"
    mode: str = "a"
    encoding: str = "utf-8"


def configure_logger(file_name=None, logging_level=None):
    """
    Configures a logger object based on the specified file name and logging level.

    Parameters
    ----------
    file_name : str or Path, optional
        The name of the file to log to. If not provided, logging will be done to the console instead.
    logging_level : int, optional
        The logging level to use. If not provided, the default level specified in the configuration will be used.

    Returns
    -------
    logger : logging.Logger
        The configured logger object.
    """
    if isinstance(file_name, str) or isinstance(file_name, Path):
        config = FileLogger()
        handler = logging.FileHandler(filename=file_name, mode=config.mode, encoding=config.encoding)
    else:
        config = ConsoleLogger()
        handler = logging.StreamHandler()

    level = logging_level if logging_level else config.level
    handler.setLevel(level)
    handler.setFormatter(logging.Formatter(config.format, datefmt=config.date_format))
    logger = logging.getLogger()
    logger.setLevel(level)
    logger.addHandler(handler)

    return logger


def json_to_dict(json_path):
    """
    Load a JSON file at the specified path and return its contents as a Python dictionary.

    Parameters
    ----------
    json_path : str or Path
        The path to the JSON file to be loaded.

    Returns
    -------
    dict
        A dictionary representing the contents of the loaded JSON file.

    Raises
    ------
    FileNotFoundError
        If the specified file does not exist or cannot be found.
    JSONDecodeError
        If the loaded JSON data is malformed and cannot be decoded.

    Notes
    -----
    This function assumes that the JSON file contains a valid JSON object.
    If the file contains an array or other non-object data, the resulting dictionary will be incomplete or incorrect.
    """
    with open(json_path) as json_file:
        parameters_dict = json.load(json_file)
    logging.debug(f"Json file founded at {json_path} and converted into dict.")
    return parameters_dict


def get_last_model_path(model_dir):
    """
    Returns the path to the last model in the specified directory.

    Parameters
    ----------
    model_dir : str or Path
        The path to the directory containing the model files.

    Returns
    -------
    str or Path
        The path to the last model in the specified directory.
    """
    model_files = list(Path(model_dir).glob("model*.pt"))
    if len(model_files) > 0:
        return max(model_files, key=lambda x: int(x.stem.split("_")[1]))
    else:
        return None


class ConfigFactory:
    """
    A factory for creating configuration objects.

    This factory provides a static method `create_config` which creates
    a configuration object based on the specified mode.

    Methods
    -------
    create_config(mode, **kwargs)
        Creates a configuration object based on the specified mode.
    """
    @staticmethod
    def create_config(mode, **kwargs):
        """
        Creates a configuration object based on the specified mode.

        Parameters
        ----------
        mode : str
            The mode for which to create the configuration object.
        **kwargs : dict
            Keyword arguments to pass to the configuration object.

        Returns
        -------
        config : TrainConfig or InferenceConfig or PredictConfig or ResumeConfig or HyperparameterConfig
            A configuration object for the specified mode.

        Raises
        ------
        ValueError
            If an invalid mode is specified.
        """
        if mode == "train":
            logging.debug(f"Creating TrainConfig with {kwargs}")  # TODO: output folder not created at this moment
            return TrainConfig(**kwargs)
        elif mode == "inference":
            logging.debug(f"Creating InferenceConfig with {kwargs}")  # TODO: output folder not created at this moment
            return InferenceConfig(**kwargs)
        elif mode == "predict":
            logging.debug(f"Creating PredictConfig with {kwargs}")  # TODO: output folder not created at this moment
            return PredictConfig(**kwargs)
        elif mode == "resume":
            logging.debug(f"Creating ResumeConfig with {kwargs}")  # TODO: output folder not created at this moment
            return ResumeConfig(**kwargs)
        elif mode == "hyperparameter":
            logging.debug(f"Creating HyperparameterConfig with {kwargs}")  # TODO: output folder not created at this moment
            return HyperparameterConfig(**kwargs)
        else:
            raise ValueError(f"Invalid mode: {mode}")


class Config:
    """
    A class representing a configuration for a machine learning algorithm.

    Attributes
    ----------
    mode : str or None
        The mode of operation for the algorithm. e.g. 'train', 'test', etc.
    paths : dict
        A dictionary containing paths to various files used by the algorithm.
    parameters : dict
        A dictionary containing parameter values for the algorithm.
    io : dict
        A dictionary containing I/O configuration information.
    configs : dict
        A dictionary containing paths to configuration files used by the algorithm.

    Methods
    -------
    copy_config_files(output_path)
        Copies the configuration files to the specified output path.
    log_info()
        Logs information about the configuration to the console.
    """
    def __init__(self, **kwargs):
        """
        Initializes a new instance of the Config class.

        Parameters
        ----------
        **kwargs : dict
            Keyword arguments containing configuration information for the algorithm.
        """
        self.mode = None
        self.device = None
        self.paths = {}
        self.parameters = {}
        self.paths["io"] = Path(ALGOZ_PATH, kwargs.get("io_path"))
        self.io = json_to_dict(self.paths["io"])
        self.configs = {key: Path(ALGOZ_PATH, value) for key, value in self.io.items() if "config_" in key}
        for key, value in self.configs.items():
            self.paths[key] = value
            self.parameters[key] = json_to_dict(value)

    def copy_config_files(self, output_path):
        """
        Copies the configuration files to the specified output path.

        Parameters
        ----------
        output_path : str or Path
            The path to the output directory where the configuration files should be copied.
        """
        io_path = Path(output_path, "io.json")
        shutil.copyfile(self.paths["io"], io_path)
        logging.debug(f"{self.paths['io']} copied into {io_path}.")
        for config_path in self.configs.values():
            out_path = Path(output_path, config_path.name)
            shutil.copyfile(config_path, out_path)
            logging.debug(f"{config_path} copied into {out_path}.")

    def device_assignment(self):
        """
        Determines the computation device to use. e.g. cpu, cuda, cuda:0, 2
        """
        device = "cpu"
        if cuda.is_available() and "device" in self.parameters["config_hyperparameters"]:
            if "cuda" in self.parameters["config_hyperparameters"]["device"]:
                device = "cuda:" + assign_free_gpus(max_gpus=1, wait=True, sleep_time=100, ban_process=["python"])
            else:
                device = self.parameters["config_hyperparameters"]["device"]
        self.device = device
        logging.debug(f"Device assigned: {self.device}")

    def log_info(self):
        """
        Logs information about the configuration to the console.
        """
        logging.info(f"Mode: {self.mode}")


class TrainConfig(Config):
    """
    A configuration object for training an algorithm.

    This class is a subclass of the Config class and inherits its properties and methods.
    It adds properties specific to training an algorithm, such as the algorithm used and the device
    on which to run the training.

    Parameters:
        **kwargs: Additional keyword arguments to be passed to the parent constructor.

    Attributes:
        mode : str
            The mode of the configuration object, set to "train".
        algo : str
            The name of the algorithm used for training.
        device : str
            The device on which to run the training.

    Methods:
        copy_config_files(output_path)
            Copies the configuration files to the specified output path.
        log_info()
            Logs information about the training configuration.
    """
    def __init__(self, **kwargs):
        """
        Initializes a TrainConfig object.

        Parameters:
            **kwargs: Additional keyword arguments to be passed to the parent constructor.
        """
        
        super().__init__(**kwargs)
        self.mode = "train"
        self.algo = self.io["algorithm"]
        print(self.io["module"])
        


        self.module = self.io["module"] if "module" in self.io else None
        logging.debug(f"Train config created with {self.__dict__}")
        print(self.mode)
        print(self.parameters)
        print(self.module)
        print(self.configs)
        print(self.io)
        print(self.device)


    def log_info(self):
        """
        Logs information about the training configuration.
        """
        logging.info(f"Mode: {self.mode}")
        logging.info(f"Training dataset: {Path(self.io['config_data_train']).stem}")
        logging.info(f"Validation dataset: {Path(self.io['config_data_test']).stem}")
        logging.info(f"Algorithm: {self.algo}")
        if self.module:
            logging.info(f"Module: {self.module}")
        if "cuda:" in self.device:
            gpu_id = self.device.split("cuda:")[-1].split(",")
            for gpu in gpu_id:
                logging.info(f"Device: cuda:{gpu} ({cuda.get_device_name(int(gpu))})")
        else:
            logging.info(f"Device: {self.device}")


class InferenceConfig(Config):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.mode = "inference"
        if Path(self.io["train_path"]).is_dir():
            self.io["output_path"] = Path(self.io["train_path"], "inference")
        else:
            raise ValueError(f"Invalid train_path: {self.io['train_path']}")

        # Get model path
        if "model_path" in self.io:
            self.paths["model"] = Path(self.io["model"])
        else:
            self.paths["model"] = get_last_model_path(self.io["train_path"])
            checkpoint_path = Path(self.io["train_path"], "checkpoint")
            if self.paths["model"] is None:
                if checkpoint_path.is_dir():
                    logging.debug(f"No model found in {self.io['train_path']}, checking in {checkpoint_path}")
                    self.paths["model"] = get_last_model_path(checkpoint_path)
        if self.paths["model"] is None:
            raise ValueError(f'No key "model_path" found in the configuration file '
                             f'and no model found in {self.io["train_path"]}')

        # Get io config of the training
        self.paths["io_train"] = Path(self.io["train_path"], "io.json")
        self.parameters["io_train"] = json_to_dict(self.paths["io_train"])
        self.algo = self.parameters["io_train"]["algorithm"]
        self.module = self.parameters["io_train"]["module"] if "module" in self.parameters["io_train"] else None

        config_hyperparameters_name = Path(self.parameters["io_train"]["config_hyperparameters"]).name
        self.paths["config_hyperparameters"] = Path(self.io["train_path"], config_hyperparameters_name)
        self.parameters["config_hyperparameters"] = json_to_dict(self.paths["config_hyperparameters"])

        config_module_name = Path(self.parameters["io_train"]["config_module"]).name
        self.paths["config_module"] = Path(self.io["train_path"], config_module_name)
        self.parameters["config_module"] = json_to_dict(self.paths["config_module"])

    def log_info(self):
        """
        Logs information about the inference configuration.
        """
        logging.info(f"Mode: {self.mode}")
        logging.info(f"Test dataset: {Path(self.io['config_data_test']).stem}")
        logging.info(f"Model: {self.paths['model']}")
        logging.info(f"Algorithm: {self.algo}")
        if self.module:
            logging.info(f"Module: {self.module}")
        if "cuda:" in self.device:
            gpu_id = self.device.split("cuda:")[-1].split(",")
            for gpu in gpu_id:
                logging.info(f"Device: cuda:{gpu} ({cuda.get_device_name(int(gpu))})")
        else:
            logging.info(f"Device: {self.device}")


class PredictConfig(Config):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.mode = "predict"
        # TODO


class ResumeConfig(Config):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.mode = "resume"
        # TODO


class HyperparameterConfig(Config):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.mode = "hyperparameter"
        # TODO


class ScriptConfig(Config):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        if Path(self.io["model_path"]).is_file():
            self.paths["model"] = Path(self.io["model_path"])
        else:
            raise ValueError(f"Invalid model_path: {self.io['model_path']}")

        self.device = "cpu"

        # Search config file of the model
        for i in range(3):
            self.paths["train_path"] = Path(self.paths["model"].parents[i])
            if Path(self.paths["train_path"], "io.json").is_file():
                self.paths["io_train"] = Path(self.paths["train_path"], "io.json")
                break
        if "io_train" not in self.paths:
            raise ValueError(f"No io.json file found in the parent directories of {self.paths['model']}")

        self.io["output_path"] = Path(self.paths["train_path"], "scripted_models")

        # Get io config of the training
        self.parameters["io_train"] = json_to_dict(self.paths["io_train"])
        self.algo = self.parameters["io_train"]["algorithm"]
        self.module = self.parameters["io_train"]["module"] if "module" in self.parameters["io_train"] else None

        config_hyperparameters_name = Path(self.parameters["io_train"]["config_hyperparameters"]).name
        self.paths["config_hyperparameters"] = Path(self.paths["train_path"], config_hyperparameters_name)
        self.parameters["config_hyperparameters"] = json_to_dict(self.paths["config_hyperparameters"])

        config_module_name = Path(self.parameters["io_train"]["config_module"]).name
        self.paths["config_module"] = Path(self.paths["train_path"], config_module_name)
        self.parameters["config_module"] = json_to_dict(self.paths["config_module"])

    def log_info(self):
        """
        Logs information about the script model configuration.
        """
        logging.info(f"Model: {self.paths['model']}")
        logging.info(f"Algorithm: {self.algo}")
        if self.module:
            logging.info(f"Module: {self.module}")
