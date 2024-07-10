#!/usr/bin/env python3

import logging

import torch
from torch import nn


def available_loss():
    """
    Available loss functions to train a machine learning model:
        "MAELoss": measures the mean absolute error (L1 norm) (from torch)
        "MSELoss": measures the mean squared error (squared L2 norm) (from torch)
        "CrossEntropyLoss": computes the cross entropy loss between input logits and target (from torch)
        "StandardAELoss": computes the standard autoencoder loss
        "SparseAELoss": computes the sparse autoencoder loss
        "SumOfLoss": computes the sum of losses
    """
    loss_functions = {
        "MAELoss": nn.L1Loss,
        "MSELoss": nn.MSELoss,
        "CrossEntropyLoss": nn.CrossEntropyLoss,
        "StandardAELoss": StandardAELoss,
        "SparseAELoss": SparseAELoss,
        "SumOfLoss": SumOfLoss
    }
    logging.debug(f'Available loss: {loss_functions.keys()}.')
    return loss_functions


class SumOfLoss(nn.MSELoss):
    def __init__(self, criterions, weights=None, network_outputs=None):
        super(SumOfLoss, self).__init__()
        self.criterions = criterions
        self.weights = weights if weights is not None else [1] * len(criterions)
        self.network_outputs = network_outputs if network_outputs is not None else [1] * len(criterions)
        self.losses = self._initialize_loss()

    def _initialize_loss(self):
        list_of_loss = []
        avail_loss = available_loss()
        for criterion in self.criterions:
            if isinstance(criterion, str):
                list_of_loss.append(avail_loss[criterion]())
            elif isinstance(criterion, dict):
                loss_name = list(criterion.keys())[0]
                list_of_loss.append(avail_loss[loss_name](**criterion[loss_name]))
        return list_of_loss

    def forward(self, y_pred, y_true):
        i_pred = len(y_pred) - 1
        i_true = len(y_true) - 1
        sum_of_loss = 0
        for loss, weight, net_out in zip(self.losses, self.weights, self.network_outputs):
            pred = y_pred[i_pred-net_out+1:i_pred+1] if net_out > 1 else y_pred[i_pred]
            sum_of_loss += weight * loss(pred, y_true[i_true])
            i_pred -= net_out
            i_true -= 1
        return sum_of_loss


class StandardAELoss(nn.MSELoss):
    def __init__(self):
        super(StandardAELoss, self).__init__()

    def forward(self, y_pred, y_true):
        decoded, encoded = y_pred
        return super().forward(decoded, y_true)


class SparseAELoss(nn.MSELoss):
    def __init__(self, regularization_param=1e-3):
        super(SparseAELoss, self).__init__()
        self.regularization_param = regularization_param

    def forward(self, y_pred, y_true):
        decoded, encoded = y_pred
        loss_reconstruction = super().forward(decoded, y_true)
        loss_l1 = self.regularization_param * torch.abs(encoded).sum()
        return loss_reconstruction + loss_l1
