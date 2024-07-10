#!/usr/bin/env python3

import logging
from pathlib import Path
from inspect import isclass

import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor
from sklearn.linear_model import SGDRegressor
from sklearn.svm import SVR
from torch import no_grad
from torch.jit import script
from torch.optim import SGD, Adam
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from skorch import NeuralNetRegressor, NeuralNetClassifier
from skorch.dataset import Dataset, ValidSplit
from skorch.helper import predefined_split
from skorch.callbacks import (Callback, Checkpoint, TrainEndCheckpoint, LoadInitState, PrintLog, EpochScoring,
                              TensorBoard)

from nanoz.modules.NN import MLP, CNN1D, CNN2D, RNN, CRNN1D, CRNN2D
from nanoz.modules.AE import (AutoEncoderNet, AutoEncoderRegressorNet, AutoEncoderClassifierNet,
                              AutoEncoder, AutoEncoderRegressor, AutoEncoderClassifier)
from nanoz.modules.losses import available_loss
from nanoz.nzio import get_last_model_path
from nanoz.utils import KeyErrorMessage, copy_doc

pd.options.plotting.backend = "plotly"


# TODO: merge with available_algorithm method from Algorithm
class AvailableAlgorithm:
    regression = ["ETR", "HGBR", "SGDR", "SVR", "NNR", "AERN"]
    classification = ["NNC", "AECN"]
    autoencoder = ["AEN"]

    @classmethod
    def get_type(cls, algorithm):
        """
        Return the type of the algorithm from the algorithm name.

        Parameters
        ----------
        algorithm : str
            Name of the algorithm.

        Returns
        -------
        str
            Type of the algorithm.
        """
        if algorithm in cls.regression:
            return "regression"
        elif algorithm in cls.classification:
            return "classification"
        elif algorithm in cls.autoencoder:
            return "autoencoder"
        else:
            raise ValueError(f"Invalid algorithm: {algorithm}")


class AlgorithmFactory:
    @staticmethod
    def create_algorithm(mode, **kwargs):
        if mode == "train":
            logging.debug(f"Creating TrainAlgorithm with {kwargs}")
            return TrainAlgorithm(**kwargs)
        elif mode == "inference":
            logging.debug(f"Creating InferenceAlgorithm with {kwargs}")
            return InferenceAlgorithm(**kwargs)
        elif mode == "predict":
            logging.debug(f"Creating PredictAlgorithm with {kwargs}")
            return PredictAlgorithm(**kwargs)
        elif mode == "resume":
            logging.debug(f"Creating ResumeAlgorithm with {kwargs}")
            return ResumeAlgorithm(**kwargs)
        elif mode == "hyperparameter":
            return HyperparameterAlgorithm(**kwargs)
        else:
            raise ValueError(f"Invalid mode: {mode}")


class Algorithm:
    def __init__(self, **kwargs):
        self.mode = None
        self.save_paths = {} if kwargs.get("save_paths") is None else kwargs.get("save_paths")
        self.config = kwargs.get("config")
        print (self.config)
        self.type = AvailableAlgorithm.get_type(self.config.algo)
        self.device = self.config.device  # TODO: /!\ inference, resume, hyperparameters
        self.predefined_split = None
        self.hyperparameters = self._callable_hyperparameters(self.config.parameters["config_hyperparameters"])
        print(self.hyperparameters)
        if self.config.module and "module" in self.hyperparameters:
            self.hyperparameters["module"] = self._callable_module()
        self.model = self._create_model(self.config.algo)

    @property
    def available_algorithms(self):  # TODO: merge with AvailableAlgorithm
        """
        Available algorithms to train a machine learning model:
            "ETR": ExtraTreesRegressor (from sklearn)
            "HGBR": HistGradientBoostingRegressor (from sklearn)
            "SGDR": Linear model fitted by minimizing a regularized empirical loss with SGD (from sklearn)
            "SVR": Epsilon-Support Vector Regression (from sklearn)
            "NNR": Neural Network Regressor (from skorch)
            "NNC": Neural Network Classifier (from skorch)
            "AEN": AutoEncoder Net
            "AERN": AutoEncoder Regressor Net
            "AECN": AutoEncoder Classifier Net
        """
        available_algorithms = {
            "ETR": ExtraTreesRegressor,
            "HGBR": HistGradientBoostingRegressor,
            "SGDR": SGDRegressor,
            "SVR": SVR,
            "NNR": NeuralNetRegressor,
            "NNC": NeuralNetClassifier,
            "AEN": AutoEncoderNet,
            "AERN": AutoEncoderRegressorNet,
            "AECN": AutoEncoderClassifierNet
        }
        logging.debug(f'Available algorithms: {list(available_algorithms)}.')
        return available_algorithms

    @property
    def available_modules(self):
        """
        Available torch modules to train a deep learning model:
            "MLP": Multi-Layer Perceptron
            "CNN1D": 1D-Convolutional Neural Network
            "CNN2D": 2D-Convolutional Neural Network
            "RNN": Recurrent Neural Network
            "CRNN1D": 1D-Convolutional Neural Network with Recurrent Neural Network
            "CRNN2D": 2D-Convolutional Neural Network with Recurrent Neural Network
            "AE": AutoEncoder
            "AER": AutoEncoder Regressor
            "AEC": AutoEncoder Classifier
        """
        available_modules = {
            "MLP": MLP,
            "CNN1D": CNN1D,
            "CNN2D": CNN2D,
            "RNN": RNN,
            "CRNN1D": CRNN1D,
            "CRNN2D": CRNN2D,
            "AE": AutoEncoder,
            "AER": AutoEncoderRegressor,
            "AEC": AutoEncoderClassifier
        }
        logging.debug(f'Available module: {list(available_modules)}.')
        return available_modules

    @property
    @copy_doc(available_loss)
    def available_hyperparameters(self):
        """
        Available callable hyperparameters to train a machine learning model:
            "SGD": implements stochastic gradient descent optimizer (from torch)
            "Adam": implements Adam optimizer (from torch)
            "DataLoader": combines a dataset and a sampler, and provides an iterable over the given dataset (from torch)
            "Dataset": general dataset wrapper that can be used in conjunction with PyTorch (from skorch)
            "ValidSplit": class that performs the internal train/valid split on a dataset (from skorch)
            "predefined_split": uses ``dataset`` for validation in :class:`.NeuralNet` (from skorch)
            "Checkpoint": save the model during training if the given metric improved (from skorch)
            "PrintLog": print useful information from the model's history as a table (from skorch)
            "EpochScoring": callback that performs generic scoring on predictions (from skorch)
            "LogBestEpoch": callback that logs the best epoch's metrics
            "ComparisonTensorBoard": callback that logs the epoch's metrics in a TensorBoard file and compares them

            [COPYDOC]
        """
        available_hyperparameters = {
            "SGD": SGD,
            "Adam": Adam,
            "DataLoader": DataLoader,
            "Dataset": Dataset,
            "ValidSplit": ValidSplit,
            "predefined_split": predefined_split,
            "Checkpoint": Checkpoint,
            "TrainEndCheckpoint": TrainEndCheckpoint,
            "LoadInitState": LoadInitState,
            "PrintLog": PrintLog,
            "EpochScoring": EpochScoring,
            "LogBestEpoch": LogBestEpoch,
            "ComparisonTensorBoard": ComparisonTensorBoard,
        }
        available_hyperparameters.update(available_loss())
        logging.debug(f'Available hyperparameters: {available_hyperparameters.keys()}.')
        return available_hyperparameters

    @copy_doc(available_hyperparameters)
    def _callable_hyperparameters(self, hyperparameters):
        """
        Process and validate callable hyperparameters.

        This method processes and validates the callable hyperparameters provided as a dictionary. It checks if the
        values match the available hyperparameters and replace them with the corresponding callable objects or lists of
        callable objects based on the available hyperparameters.
        [COPYDOC]

        Parameters
        ----------
        hyperparameters : dict
            The dictionary of hyperparameters to process.

        Returns
        -------
        dict
            The dictionary of hyperparameters with callable objects or lists of callable objects.

        Raises
        ------
        KeyErrorMessage
            If a key in a nested dictionary is not found in the available hyperparameters.
        """
        avail_hp = self.available_hyperparameters
        print(avail_hp)
        print("aada\n",hyperparameters)
        for key, value in hyperparameters.items():
            if isinstance(value, dict):  # this dictionary only contains nested dictionary(ies)
                if key == "callbacks":
                    callbacks_list = []
                    for k, v in value.items():
                        if k == "Checkpoint":
                            v["dirname"] = Path(self.save_paths["output"], v["dirname"])
                        elif k == "ComparisonTensorBoard":
                            v["logdir"] = Path(self.save_paths["output"], v["logdir"])
                        try:
                            callbacks_list.append(avail_hp[k](**v))
                        except KeyError as ke:
                            raise KeyErrorMessage(f"\n\t{ke}\n"
                                                  f"\tKey {k} not found in available hyperparameters.\n"
                                                  f"\tAvailable hyperparameters: "
                                                  f"{self.available_hyperparameters.keys()}")
                    hyperparameters[key] = callbacks_list
                    logging.debug(f"Hyperparameter {key} set to {hyperparameters[key]}.")
                elif key == "train_split" and list(value.keys())[0] == "predefined_split":
                    self.predefined_split = value["predefined_split"]["dataset"]
                    hyperparameters[key] = None
                    logging.debug(f"Parameters of predefined_split: {self.predefined_split}")
                else:  # contains only one nested dictionary with the key in available_hyperparameters
                    try:
                        hyperparameters[key] = avail_hp[list(value.keys())[0]](**value[list(value.keys())[0]])
                        logging.debug(f"Hyperparameter {key} set to {hyperparameters[key]}.")
                    except KeyError as ke:
                        raise KeyErrorMessage(f"\n\t{ke}\n"
                                              f"\tKey {list(value.keys())[0]} not found in available hyperparameters.\n"
                                              f"\tAvailable hyperparameters: {self.available_hyperparameters.keys()}")
            elif not isinstance(value, list):
                print(value)
                if value in avail_hp:
                    hyperparameters[key] = avail_hp[value]
                    logging.debug(f"Hyperparameter {key} set to {hyperparameters[key]}.")
        return hyperparameters

    @copy_doc(available_modules)
    def _callable_module(self):
        """
        Get the callable module based on the configuration.

        This method returns a callable module object based on the specified module in the configuration. The module is
        initialized with the parameters provided in the configuration.
        [COPYDOC]

        Returns
        -------
        object
            The callable module object.

        Raises
        ------
        KeyErrorMessage
            If the specified module is not found in the available modules.
        """
        try:
            print(self.config.parameters["config_module"])
            module = self.available_modules[self.config.module](**self.config.parameters["config_module"])
            
            logging.debug(f"Module {self.config.module} set to {module} with "
                          f"{self.config.parameters['config_module']} parameters.")
            return module
        except KeyError as ke:
            raise KeyErrorMessage(f"\n\t{ke}\n"
                                  f"\tModule {self.config.module} not found in available modules.\n"
                                  f"\tAvailable modules: {self.available_modules}")

    # @copy_doc(available_algorithms)
    def _create_model(self, algorithm_name):
        try:
            model = self.available_algorithms[algorithm_name](**self.hyperparameters)
            model.device = self.device
            logging.debug(f"Algorithm {algorithm_name} set to {model} with {self.hyperparameters} hyperparameters.")
        except KeyError as ke:
            raise KeyErrorMessage(f"\n\t{ke}\n"
                                  f"\tAlgorithm {algorithm_name} not found in available algorithms.\n"
                                  f"\tAvailable algorithms: {self.available_algorithms}")
        return model

    def load_model(self, model_path):
        self.model.initialize()
        self.model.load_params(f_params=model_path)
        logging.debug(f"Model loaded from {model_path}.")

    def save_model(self, path, mode):
        if mode == "scripted":
            model = script(self.model.module)
            model_name = "model_scripted.pt"
        else:  # normal save
            model = self.model.module
            model_name = "model.pt"
        logging.debug(f"Model:\n{model}")
        model.save(Path(path, model_name))
        logging.info(f"Model saved to {Path(path, model_name)}.")

    # def load_checkpoint(self, chekpoint_path):  # TODO: try LoadInitState for resume from checkpoint
    #     model_path = Path(chekpoint_path, self._get_last_model_name(chekpoint_path.glob("*.pt")))
    #     optimizer_path = Path(chekpoint_path, self._get_last_model_name(chekpoint_path.glob("*.pt")))
    #     self.model.initialize()
    #     self.model.load_params(f_params=model_path)
    #     self.model.load_params(
    #         f_params='model.pkl', f_optimizer='opt.pkl', f_history='history.json')
    #     # logging.debug(f"Model loaded from {model_path}.")

    def _make_light_history(self, save_path=None):
        df_history = pd.DataFrame(self.model.history.to_list())
        if "batches" in list(df_history):
            df_history = df_history.drop(["batches"], axis=1)
        if save_path is not None:
            df_history.to_csv(save_path, index=False)
        return df_history

    def _plot_loss(self, df, save_path):
        criterion = self.model.criterion
        loss = criterion.__name__ if isclass(criterion) else criterion.__class__.__name__
        fig = df.plot(x='epoch', y=['train_loss', 'valid_loss'], title="Train Loss vs. Validation Loss",
                      labels={'value': loss, 'epoch': 'Epoch'}, template="plotly_dark", markers=True)
        fig.update_traces(line=dict(width=2))
        fig.write_html(save_path)


class TrainAlgorithm(Algorithm):
    def __init__(self, **kwargs):
        self.save_paths = kwargs.get("save_paths")
        super().__init__(**kwargs)
        self.mode = "train"
        self.datasets = kwargs.get("datasets")
        print(self.datasets)
        self.train_model(self.datasets)
        self.light_history = self._make_light_history(save_path=Path(self.save_paths["output"], "history.csv"))
        self._plot_loss(self.light_history, Path(self.save_paths["output"], "loss.html"))
        checkpoint_callback = [isinstance(element, Checkpoint) for element in self.model.callbacks]
        if any(checkpoint_callback):
            idx_checkpoint = checkpoint_callback.index(True)
            last_model = get_last_model_path(self.model.callbacks[idx_checkpoint].dirname)
            self.load_model(last_model)
            logging.debug(f"Model loaded from {last_model} for validation.")

    def train_model(self, datasets):
        if self.predefined_split is not None:
            self.model.train_split = predefined_split(datasets[self.predefined_split])
            logging.debug(f"train_split set to predefined_split ({self.model.train_split}) "
                          f"with {self.predefined_split} dataset.")

        self.model.module.train()
        self.model.fit(datasets["train"], y=None)
        return self.model


class InferenceAlgorithm(Algorithm):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.mode = "inference"
        self.load_model(self.config.paths["model"])
        # TODO: add self.model.trim_for_prediction() with skorch version >= 0.12.0
        # TODO: don't use callbacks for inference (create folder in the output folder not needed)


class PredictAlgorithm(Algorithm):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.mode = "predict"
        self.load_model(self.config.paths["model"])
        # # TODO
        # self.model.module.eval()
        # with no_grad():
        #     # TODO
        #     pass


class ResumeAlgorithm(Algorithm):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.mode = "train"
        # TODO


class HyperparameterAlgorithm(Algorithm):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.mode = "train"
        # TODO


class LogBestEpoch(Callback):
    """
    Callback for logging the best epoch at the end of training.

    This callback logs the best epoch and its corresponding metrics from the training history. It identifies the best
    epoch based on the validation loss. If available, it also logs the best checkpoint epoch.

    Parameters
    ----------
    **kwargs
        Additional keyword arguments.

    Methods
    -------
    on_train_end(net, **kwargs)
        Callback method called at the end of training to log the best epoch.
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def on_train_end(self, net, **kwargs):
        """
        Callback method called at the end of training to log the best epoch.

        This method retrieves the best epoch based on the validation loss from the training history and logs the best
        epoch and its corresponding metrics. If available, it also logs the best checkpoint epoch.

        Parameters
        ----------
        net : skorch.NeuralNet
            The neural network instance.
        **kwargs
            Additional keyword arguments.
        """
        history_keys = tuple(k for k in net.history[0] if k != "batches")  # without batches
        best_epoch_id = [e[0] for e in net.history[:, ("epoch", "valid_loss_best")] if e[1]][-1] - 1  # epoch start at 1
        best_epoch = {k: v for k, v in zip(history_keys, net.history[best_epoch_id, history_keys])}
        print("")  # Empty line between training log and best epoch log
        if "event_cp" in net.history[0]:
            best_checkpoint = [e[0] for e in net.history[:, ("epoch", "event_cp")] if e[1]][-1]
            logging.info(f"Best checkpoint: epoch {best_checkpoint}")
        logging.info(f"Best epoch: {best_epoch}\n")


class ComparisonTensorBoard(TensorBoard):
    def __init__(self, logdir, comparison_list):
        super().__init__(SummaryWriter(logdir))
        self.comparison_list = comparison_list

    def on_epoch_end(self, net, **kwargs):
        """Automatically log values from the last history step."""
        hist = net.history[-1]
        comparison_dict = {}
        for key in self.comparison_list:
            comparison_dict[key] = [f"Loss/{k}" for k in hist.keys() if k.endswith(f"_{key.lower()}")]
        layout = {"Comparison": {key.capitalize(): ["Multiline", val] for key, val in comparison_dict.items()}}
        self.writer.add_custom_scalars(layout)
        super().on_epoch_end(net, **kwargs)  # call super last
