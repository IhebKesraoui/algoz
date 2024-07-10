#!/usr/bin/env python3

import logging

import numpy as np
import pandas as pd
from scipy import stats
from sklearn import metrics
from sklearn.preprocessing import OneHotEncoder

from nanoz.modeling import AvailableAlgorithm


class EvaluatorFactory:
    @staticmethod
    def create_evaluator(algo, **kwargs):
        algo_type = AvailableAlgorithm.get_type(algo)
        if algo_type == "regression":
            logging.debug(f"Creating RegressionEvaluator with {kwargs}")
            return RegressionEvaluator(**kwargs)
        elif algo_type == "classification":
            logging.debug(f"Creating ClassificationEvaluator with {kwargs}")
            return ClassificationEvaluator(**kwargs)
        else:
            raise ValueError(f"Invalid algorithm: {algo}")


class Evaluator:
    def __init__(self, **kwargs):
        self.ground_truth = np.array(kwargs.get("ground_truth"))
        self.prediction = np.array(kwargs.get("prediction"))
        self.probabilities = np.array(kwargs.get("probabilities"))
        self.intervals = kwargs.get("intervals", None)
        self.targets_name = kwargs.get("targets_name", ["Target gas"])

        self.metrics = {}
        self.metrics_per_intervals = {}
        self.target_performances = pd.DataFrame()
        self.performances = {}
        self.intervals_performances = {}
        for target in self.targets_name:
            self.performances[target] = pd.DataFrame()
            self.intervals_performances[target] = pd.DataFrame()

    @staticmethod
    def compute_performances(ground_truth, prediction, callable_metrics, index_name='', labels=None):
        df = pd.DataFrame(columns=list(callable_metrics))
        if len(ground_truth) > 0 and len(prediction) > 0:
            for name, metric in callable_metrics.items():
                df.loc[index_name, name] = metric(ground_truth, prediction, labels=labels)
        else:
            for name, _ in callable_metrics.items():
                df.loc[index_name, name] = np.nan
        return df

    def save_prediction(self, path):
        if self.ground_truth.ndim == 1:
            self.ground_truth = self.ground_truth[:, np.newaxis]
        if self.prediction.ndim == 1:
            self.prediction = self.prediction[:, np.newaxis]
        df = pd.DataFrame(data=np.concatenate((self.ground_truth, self.prediction), axis=1),
                          columns=self.targets_name + [t + "_pred" for t in self.targets_name])
        df.to_csv(path)


class RegressionEvaluator(Evaluator):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.metrics = RegressionMetrics("full").metrics
        self.metrics_per_intervals = RegressionMetrics("interval").metrics

        for index, target in enumerate(self.targets_name):
            if self.intervals:
                self.intervals_performances[target] = self.compute_intervals_performances(self.ground_truth[:, index],
                                                                                          self.prediction[:, index])
                lower_bound_mask = self.ground_truth[:, index] >= self.intervals[0][0]
                upper_bound_mask = self.ground_truth[:, index] <= self.intervals[-1][1]
                mask = lower_bound_mask & upper_bound_mask
                ground_truth = self.ground_truth[:, index][mask]
                prediction = self.prediction[:, index][mask]
            else:
                ground_truth = self.ground_truth[:, index]
                prediction = self.prediction[:, index]
            self.performances[target] = self.compute_performances(ground_truth, prediction,
                                                                  self.metrics, index_name=target)
            self.target_performances = pd.concat([self.target_performances, self.performances[target]], axis=0)

    def compute_intervals_performances(self, ground_truth, prediction):  # Ordinal intervals
        df = pd.DataFrame(columns=list(self.metrics_per_intervals))
        for interval in self.intervals:
            idx = np.where((ground_truth > interval[0]) & (ground_truth <= interval[1]))[0]
            df_interval = self.compute_performances(ground_truth[idx], prediction[idx],
                                                    self.metrics_per_intervals, index_name=str(interval))
            df = pd.concat([df, df_interval], axis=0)
        return df


class ClassificationEvaluator(Evaluator):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.n_classes = len(self.intervals)
        self.metrics = self._get_global_classification_metrics()
        self.metrics_per_intervals = ClassificationMetrics("classe").metrics

        self.confusion_matrix = metrics.confusion_matrix(self.ground_truth, self.prediction,
                                                         labels=range(self.n_classes))
        self.normalized_cm = metrics.confusion_matrix(self.ground_truth, self.prediction,
                                                      labels=range(self.n_classes), normalize="true")
        self.fpr, self.tpr, self.roc_th, self.roc_auc = self._compute_roc_metrics()
        self.precision, self.recall, self.pr_th, self.average_precision = self._compute_pr_metrics()

        self.performances = self.compute_performances(self.ground_truth, self.prediction,
                                                      self.metrics, index_name=self.targets_name[0])  # 0 -> one target
        self.performances["Macro AUC"] = self.roc_auc["macro"]
        self.performances["Micro AUC"] = self.roc_auc["micro"]
        self.performances["Macro Average Precision"] = self.average_precision["macro"]
        self.performances["Micro Average Precision"] = self.average_precision["micro"]

        self.target_performances = self.performances.copy()
        self.intervals_performances = self.compute_intervals_performances()
        for i in range(self.n_classes):
            self.intervals_performances.loc[str(self.intervals[i]), "AUC"] = self.roc_auc[i]
            self.intervals_performances.loc[str(self.intervals[i]), "Average Precision"] = self.average_precision[i]

    def _get_global_classification_metrics(self):
        if self.n_classes > 2:
            callable_metrics = ClassificationMetrics("macro", "micro-IoU").metrics
        else:
            callable_metrics = ClassificationMetrics("binary").metrics
        return callable_metrics

    def compute_intervals_performances(self):  # Ordinal intervals
        df_performance = self.compute_performances(self.ground_truth, self.prediction,
                                                   self.metrics_per_intervals, labels=range(self.n_classes))
        df = df_performance.explode(list(self.metrics_per_intervals))
        df.index = [str(interval) for interval in self.intervals]
        return df

    @staticmethod
    def _reduce_list(the_list, desired_length=1000):
        if len(the_list) <= desired_length:
            return the_list
        else:
            indices = [int(i * (len(the_list) - 1) / (desired_length - 1)) for i in range(desired_length)]
            reduced_list = [the_list[i] for i in indices]
            return reduced_list

    @staticmethod
    def _mean_dictionary(dictionary):
        keys = list(dictionary)
        means = dictionary[keys[0]].copy()
        for key in keys[1:]:
            means = [sum(val) for val in zip(means, dictionary[key])]
        return [val / len(keys) for val in means]

    def _compute_roc_metrics(self):
        fpr = {}
        tpr = {}
        th = {}
        roc_auc = {}

        classes = range(self.n_classes)
        encoder = OneHotEncoder(categories=[classes], drop=None, sparse=False, dtype=int)
        onehot = encoder.fit_transform([[val] for val in self.ground_truth])

        for i in classes:
            fpr[i], tpr[i], th[i] = metrics.roc_curve(onehot[:, i], self.probabilities[:, i])
            fpr[i] = self._reduce_list(fpr[i])
            tpr[i] = self._reduce_list(tpr[i])
            th[i] = self._reduce_list(th[i])
            roc_auc[i] = metrics.auc(fpr[i], tpr[i])

        fpr["macro"] = self._mean_dictionary(fpr)
        tpr["macro"] = self._mean_dictionary(tpr)
        roc_auc["macro"] = metrics.auc(fpr["macro"], tpr["macro"])

        fpr["micro"], tpr["micro"], th["micro"] = metrics.roc_curve(onehot.ravel(), self.probabilities.ravel())
        fpr["micro"] = self._reduce_list(fpr["micro"])
        tpr["micro"] = self._reduce_list(tpr["micro"])
        th["micro"] = self._reduce_list(th["micro"])
        roc_auc["micro"] = metrics.auc(fpr["micro"], tpr["micro"])

        return fpr, tpr, th, roc_auc

    def _compute_pr_metrics(self):
        precision = {}
        recall = {}
        th = {}
        average_precision = {}

        classes = range(self.n_classes)
        encoder = OneHotEncoder(categories=[classes], drop=None, sparse=False, dtype=int)
        onehot = encoder.fit_transform([[val] for val in self.ground_truth])

        for i in classes:
            precision[i], recall[i], th[i] = metrics.precision_recall_curve(onehot[:, i], self.probabilities[:, i])
            precision[i] = self._reduce_list(precision[i])
            recall[i] = self._reduce_list(recall[i])
            th[i] = self._reduce_list(th[i])
            average_precision[i] = metrics.average_precision_score(onehot[:, i], self.probabilities[:, i])

        precision["macro"] = self._mean_dictionary(precision)
        recall["macro"] = self._mean_dictionary(recall)
        average_precision["macro"] = metrics.average_precision_score(onehot, self.probabilities, average="macro")

        precision["micro"], recall["micro"], th["micro"] = metrics.precision_recall_curve(onehot.ravel(),
                                                                                          self.probabilities.ravel())
        precision["micro"] = self._reduce_list(precision["micro"])
        recall["micro"] = self._reduce_list(recall["micro"])
        th["micro"] = self._reduce_list(th["micro"])
        average_precision["micro"] = metrics.average_precision_score(onehot, self.probabilities, average="micro")

        return precision, recall, th, average_precision

    def save_confusion_matrix(self, path):
        np.savetxt(path, self.confusion_matrix, delimiter=",", fmt="%d")


class RegressionMetrics:
    def __init__(self, *args):
        self.metrics = self._get_callable_metrics(*args)

    @staticmethod
    def _get_callable_metrics(*args):
        metrics_dict = {}
        for arg in args:
            if arg == "full":
                new_metrics = {
                    "MAE": mae,
                    "MAPE": mape,
                    "MSE": mse,
                    "RMSE": rmse,
                    "R2": r2,
                    "Max error": max_error
                }
            elif arg == "interval":
                new_metrics = {
                    "MAE": mae,
                    "MAPE": mape,
                    "MSE": mse,
                    "RMSE": rmse,
                    "Max error": max_error
                }
            else:
                new_metrics = {}
            metrics_dict = {**metrics_dict, **new_metrics}
        return metrics_dict


class ClassificationMetrics:
    def __init__(self, *args):
        self.metrics = self._get_callable_metrics(*args)

    @staticmethod
    def _get_callable_metrics(*args):
        metrics_dict = {}
        for arg in args:
            if arg == "classe":
                new_metrics = {
                    "Accuracy": classe_accuracy,
                    "Precision": classe_precision,
                    "Recall": classe_recall,
                    "F1": classe_f1,
                    "IoU": classe_iou,
                }
            elif arg == "binary":
                new_metrics = {
                    "Accuracy": accuracy_score,
                    "Binary Precision": binary_precision,
                    "Binary Recall": binary_recall,
                    "Binary F1": binary_f1,
                    "Binary IoU": binary_iou,
                }
            elif arg == "micro":
                new_metrics = {
                    "Accuracy": accuracy_score,
                    "µ Precision": micro_precision,
                    "µ Recall": micro_recall,
                    "µ F1": micro_f1,
                    "µ IoU": micro_iou,
                }
            elif arg == "micro-IoU":
                new_metrics = {
                    "µ IoU": micro_iou,
                }
            elif arg == "macro":
                new_metrics = {
                    "Accuracy": accuracy_score,
                    "Macro Precision": macro_precision,
                    "Macro Recall": macro_recall,
                    "Macro F1": macro_f1,
                    "Macro IoU": macro_iou,
                }
            else:
                new_metrics = {}
            metrics_dict = {**metrics_dict, **new_metrics}
        return metrics_dict


def mae(ground_truth, prediction, **kwargs):
    return metrics.mean_absolute_error(ground_truth, prediction)


def mape(ground_truth, prediction, **kwargs):
    """
    TODO: docstring

    Parameters
    ----------
    ground_truth
    prediction
    kwargs

    Returns
    -------

    Notes
    -----
    Don't use metrics.mean_absolute_percentage_error(ground_truth, prediction)
    """
    np.seterr(divide='ignore', invalid='ignore')  # Ignore divide by 0 warning
    ape = 100 * np.ma.masked_invalid(np.divide(np.abs(ground_truth - prediction), ground_truth))
    np.seterr(divide='warn', invalid='warn')  # Pay attention of divide by 0 warning
    return np.mean(ape)


def mse(ground_truth, prediction, **kwargs):
    return metrics.mean_squared_error(ground_truth, prediction)


def r2(ground_truth, prediction, **kwargs):
    return metrics.r2_score(ground_truth, prediction)


def max_error(ground_truth, prediction, **kwargs):
    return metrics.max_error(ground_truth, prediction)


def rmse(ground_truth, prediction, **kwargs):
    return metrics.mean_squared_error(ground_truth, prediction, squared=False)


def accuracy_score(ground_truth, prediction, **kwargs):
    return metrics.accuracy_score(ground_truth, prediction)


def classe_accuracy(ground_truth, prediction, **kwargs):
    labels = kwargs.get("labels", None)
    matrix = metrics.confusion_matrix(ground_truth, prediction, labels=labels)
    matrix.diagonal() / matrix.sum(axis=1)
    return matrix.diagonal() / matrix.sum(axis=1)


def classe_precision(ground_truth, prediction, **kwargs):
    labels = kwargs.get("labels", None)
    return metrics.precision_score(ground_truth, prediction, average=None, labels=labels)


def classe_recall(ground_truth, prediction, **kwargs):
    labels = kwargs.get("labels", None)
    return metrics.recall_score(ground_truth, prediction, average=None, labels=labels)


def classe_f1(ground_truth, prediction, **kwargs):
    labels = kwargs.get("labels", None)
    return metrics.f1_score(ground_truth, prediction, average=None, labels=labels)


def classe_iou(ground_truth, prediction, **kwargs):
    labels = kwargs.get("labels", None)
    return metrics.jaccard_score(ground_truth, prediction, average=None, labels=labels)


def binary_precision(ground_truth, prediction, **kwargs):
    return metrics.precision_score(ground_truth, prediction, average='binary')


def binary_recall(ground_truth, prediction, **kwargs):
    return metrics.recall_score(ground_truth, prediction, average='binary')


def binary_f1(ground_truth, prediction, **kwargs):
    return metrics.f1_score(ground_truth, prediction, average='binary')


def binary_iou(ground_truth, prediction, **kwargs):
    return metrics.jaccard_score(ground_truth, prediction, average='binary')


def micro_precision(ground_truth, prediction, **kwargs):
    return metrics.precision_score(ground_truth, prediction, average='micro')


def micro_recall(ground_truth, prediction, **kwargs):
    return metrics.recall_score(ground_truth, prediction, average='micro')


def micro_f1(ground_truth, prediction, **kwargs):
    return metrics.f1_score(ground_truth, prediction, average='micro')


def micro_iou(ground_truth, prediction, **kwargs):
    return metrics.jaccard_score(ground_truth, prediction, average='micro')


def macro_precision(ground_truth, prediction, **kwargs):
    return metrics.precision_score(ground_truth, prediction, average='macro')


def macro_recall(ground_truth, prediction, **kwargs):
    return metrics.recall_score(ground_truth, prediction, average='macro')


def macro_f1(ground_truth, prediction, **kwargs):
    return metrics.f1_score(ground_truth, prediction, average='macro')


def macro_iou(ground_truth, prediction, **kwargs):
    return metrics.jaccard_score(ground_truth, prediction, average='macro')


def gaussian_kde(data, step=1000):
    x_pts = np.linspace(min(data), max(data), step)
    gkde = stats.gaussian_kde(data)
    estimated_pdf = gkde.evaluate(x_pts)
    return x_pts, estimated_pdf
