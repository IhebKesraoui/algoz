#!/usr/bin/env python3

from torch import nn
from skorch import NeuralNetRegressor, NeuralNetClassifier

from nanoz.modules.NN import MLP, CNN1D, RNN, TConv1DBlock


class AutoEncoderNet(NeuralNetRegressor):
    def get_loss(self, y_pred, y_true, *args, **kwargs):
        return super().get_loss(y_pred, y_true, *args, **kwargs)


class AutoEncoderRegressorNet(NeuralNetRegressor):
    def get_loss(self, y_pred, y_true, *args, **kwargs):
        x = kwargs.get("X")
        y_true = (y_true, x)
        return super().get_loss(y_pred, y_true, *args, **kwargs)


class AutoEncoderClassifierNet(NeuralNetClassifier):
    def get_loss(self, y_pred, y_true, *args, **kwargs):
        x = kwargs.get("X")
        y_true = (y_true, x)
        return super().get_loss(y_pred, y_true, *args, **kwargs)


class AutoEncoder(nn.Module):
    def __init__(self, architecture, parameters):
        super(AutoEncoder, self).__init__()
        self.encoder, self.decoder = None, None
        if architecture.lower() == "mlp":
            self.encoder = get_mlp_encoder(**parameters)
            self.decoder = get_mlp_decoder(**parameters)
        elif architecture.lower() == "cnn1d":
            self.encoder = get_cnn1d_encoder(**parameters)
            self.decoder = get_cnn1d_decoder(**parameters)
        elif architecture.lower() == "rnn":
            # TODO: The current implementation of the RNN encoder give he output tensor and not the hidden state needed
            #  for the RNN autoencoder
            # self.encoder = get_rnn_encoder(**parameters)
            # self.decoder = get_rnn_decoder(input_dim, embedding_dim, **parameters)
            raise NotImplementedError("RNN autoencoder is not implemented yet.")
        else:
            raise ValueError(f"Unknown architecture type ({architecture}) for AutoEncoder.")

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded, encoded


class AutoEncoderRegressor(nn.Module):
    def __init__(self, **kwargs):
        super(AutoEncoderRegressor, self).__init__()
        architectures = kwargs.get("architectures")
        parameters = kwargs.get("parameters")

        self.autoencoder = AutoEncoder(architectures[0], parameters[0])

        regressor_architecture = architectures[1]
        if regressor_architecture.lower() == "mlp":
            self.regressor = MLP(**parameters[1])
        elif regressor_architecture.lower() == "cnn1d":
            self.regressor = CNN1D(**parameters[1])
        elif regressor_architecture.lower() == "rnn":
            self.regressor = RNN(**parameters[1])
        else:
            raise ValueError(f"Unknown regressor type ({regressor_architecture}) for AutoEncoderRegressor.")

    def forward(self, x):
        decoded, encoded = self.autoencoder(x)
        predicted = self.regressor(encoded)
        return predicted, decoded, encoded


class AutoEncoderClassifier(nn.Module):
    def __init__(self, **kwargs):
        super(AutoEncoderClassifier, self).__init__()
        architectures = kwargs.get("architectures")
        parameters = kwargs.get("parameters")

        self.autoencoder = AutoEncoder(architectures[0], parameters[0])

        classifier_architecture = architectures[1]
        if classifier_architecture.lower() == "mlp":
            self.regressor = MLP(**parameters[1])
        elif classifier_architecture.lower() == "cnn1d":
            self.regressor = CNN1D(**parameters[1])
        elif classifier_architecture.lower() == "rnn":
            self.regressor = RNN(**parameters[1])
        else:
            raise ValueError(f"Unknown regressor type ({classifier_architecture}) for AutoEncoderClassifier.")

    def forward(self, x):
        decoded, encoded = self.autoencoder(x)
        predicted = self.regressor(encoded)
        return predicted, decoded, encoded


def get_mlp_encoder(**parameters):
    return MLP(parameters.get("input_dim"),
               parameters.get("hidden_dims", []),
               parameters.get("embedding_dim"),
               False,
               normalization=parameters.get("normalization", None),
               pp_normalization=parameters.get("pp_normalization", None))


def get_mlp_decoder(**parameters):
    input_dim = parameters.get("input_dim")
    mlp_decoder = MLP(parameters.get("embedding_dim"),
                      list(reversed(parameters.get("hidden_dims", []))),
                      input_dim,
                      False,
                      normalization=parameters.get("normalization", None))
    if isinstance(input_dim, list):
        return nn.Sequential(mlp_decoder, nn.Unflatten(1, tuple(input_dim)))
    else:
        return mlp_decoder


def get_cnn1d_encoder(**parameters):
    return CNN1D(parameters.get("input_dim"),
                 parameters.get("conv_map", [1]),
                 parameters.get("conv_kernel", 3),
                 parameters.get("pool_kernel", 2),
                 parameters.get("mlp_hidden_dims", []),
                 parameters.get("embedding_dim"),
                 False,
                 conv_norm=parameters.get("conv_norm", None),
                 pp_conv_norm=parameters.get("pp_conv_norm", None),
                 mlp_norm=parameters.get("mlp_norm", None),
                 pp_mlp_norm=None)


def get_cnn1d_decoder(**parameters):
    input_dim = parameters.get("input_dim")

    mlp_dim = (input_dim[0] * parameters.get("conv_map", [1])[-1] * input_dim[1] //
               (parameters.get("pool_kernel", 2) ** len(parameters.get("conv_map", [1]))))
    mlp_parameters = parameters.copy()
    mlp_parameters.update({"input_dim": mlp_dim})
    mlp_parameters.update({k[len("mlp_"):]: v for k, v in parameters.items() if "mlp_" in k})
    mlp_decoder = get_mlp_decoder(**mlp_parameters)

    deconv_dim = [input_dim[0] * parameters.get("conv_map", [1])[-1],
                  input_dim[1] // parameters.get("pool_kernel", 2) ** len(parameters.get("conv_map", [1]))]
    deconv_decoder = TConv1DBlock(deconv_dim,
                                  list(reversed(parameters.get("conv_map", [1]))),
                                  parameters.get("conv_kernel", 3),
                                  parameters.get("pool_kernel", 2),
                                  parameters.get("conv_norm", None))
    return nn.Sequential(mlp_decoder, nn.Unflatten(1, tuple(deconv_dim)), deconv_decoder)


def get_rnn_encoder(**parameters):
    return RNN(parameters.get("input_dim"),
               parameters.get("rnn_type"),
               parameters.get("rnn_map", [1]),
               parameters.get("bidirectional", False),
               parameters.get("embedding_dim"),
               False,
               rnn_norm=parameters.get("rnn_norm", None),
               pp_rnn_norm=parameters.get("pp_rnn_norm", None),
               mlp_norm=parameters.get("mlp_norm", None),
               pp_mlp_norm=None)
