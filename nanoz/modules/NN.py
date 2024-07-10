#!/usr/bin/env python3

import math

from torch import nn, permute


def normalization_1d(normalization, dim):
    if normalization == "batch":
        return nn.BatchNorm1d(dim, eps=1e-18)
    elif normalization == "instance":  # Work only for batched multivariate 1d signals (3D-tensors)
        return nn.InstanceNorm1d(dim, eps=1e-18)
    elif normalization == "layer":
        return nn.LayerNorm(dim, eps=1e-18)


def normalization_2d(normalization, dim):
    if normalization == "batch":
        return nn.BatchNorm2d(dim, eps=1e-18)
    elif normalization == "instance":
        return nn.InstanceNorm2d(dim, eps=1e-18)
    elif normalization == "layer":
        return nn.LayerNorm(dim, eps=1e-18)


class BaseLayersArchitecture(nn.Module):
    def __init__(self, layers):
        super(BaseLayersArchitecture, self).__init__()
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


class MLP(BaseLayersArchitecture):
    def __init__(self, input_dim, hidden_dims, output_dim, use_softmax,
                 normalization=None, pp_normalization=None):
        if normalization == "instance" or pp_normalization == "instance":
            raise RuntimeError(f'Instance normalization is not supported for MLPs, use "layer" normalization instead.')

        layers = []
        if isinstance(input_dim, list):
            input_dim = math.prod(input_dim)
        layers.append(nn.Flatten())
        if pp_normalization:
            layers.append(normalization_1d(pp_normalization, input_dim))

        for h_dim in hidden_dims:
            layers.append(nn.Linear(input_dim, h_dim))
            layers.append(nn.ReLU())
            if normalization:
                layers.append(normalization_1d(normalization, h_dim))
            input_dim = h_dim

        out_dim = output_dim if isinstance(output_dim, list) else [output_dim]
        layers.append(nn.Linear(input_dim, math.prod(out_dim)))
        if not hidden_dims:
            layers.append(nn.ReLU())
        if use_softmax:
            layers.append(nn.Softmax(dim=1))
        super(MLP, self).__init__(layers)


class Conv1DBlock(BaseLayersArchitecture):
    def __init__(self, input_dim, conv_map, conv_kernel, pool_kernel,
                 normalization=None, pp_normalization=None):
        layers = []
        last_dim = input_dim[0]
        if pp_normalization:
            norm_dim = last_dim if pp_normalization != "layer" else input_dim
            layers.append(normalization_1d(pp_normalization, norm_dim))
        for ite, conv in enumerate(conv_map):
            conv = input_dim[0] * conv
            layers.append(nn.Conv1d(last_dim, conv,
                                    kernel_size=conv_kernel, padding=(conv_kernel-1) // 2,  groups=input_dim[0]))
            layers.append(nn.ReLU())
            if normalization:
                if normalization != "layer":
                    norm_dim = conv
                else:
                    norm_dim = [conv] + [d // (pool_kernel ** ite) for d in input_dim[1:]]
                layers.append(normalization_1d(normalization, norm_dim))
            layers.append(nn.MaxPool1d(pool_kernel, stride=pool_kernel))
            last_dim = conv
        super(Conv1DBlock, self).__init__(layers)


class TConv1DBlock(BaseLayersArchitecture):
    def __init__(self, input_dim, conv_map, conv_kernel, pool_kernel,
                 normalization=None, pp_normalization=None):
        layers = []
        last_dim = input_dim[0]
        output_dim = input_dim[0] // conv_map[0]
        conv_map.append(1)
        conv_map.pop(0)
        if pp_normalization:
            norm_dim = last_dim if pp_normalization != "layer" else input_dim
            layers.append(normalization_1d(pp_normalization, norm_dim))
        for ite, conv in enumerate(conv_map):
            layers.append(nn.Upsample(scale_factor=pool_kernel, mode="nearest"))
            conv = output_dim * conv
            layers.append(nn.ConvTranspose1d(last_dim, conv,
                                             kernel_size=conv_kernel, padding=(conv_kernel-1) // 2,  groups=output_dim))
            layers.append(nn.ReLU())
            if normalization:
                if normalization != "layer":
                    norm_dim = conv
                else:
                    norm_dim = [conv] + [d // (pool_kernel ** ite) for d in input_dim[1:]]
                layers.append(normalization_1d(normalization, norm_dim))
            last_dim = conv
        super(TConv1DBlock, self).__init__(layers)


class Conv2DBlock(BaseLayersArchitecture):
    def __init__(self, input_dim, conv_map, conv_kernel, pool_kernel,
                 normalization=None, pp_normalization=None):
        layers = []
        if len(input_dim) == 2:  # Convert 2D input to 3D with adding a channel dimension
            layers.append(InsertDim(1))
            input_dim = [1] + input_dim
        last_dim = input_dim[0]
        if pp_normalization:
            norm_dim = input_dim[0] if pp_normalization != "layer" else input_dim
            layers.append(normalization_2d(pp_normalization, norm_dim))
        for ite, conv in enumerate(conv_map):
            conv = input_dim[0] * conv
            layers.append(nn.Conv2d(last_dim, conv, kernel_size=conv_kernel, padding=(conv_kernel-1) // 2))
            layers.append(nn.ReLU())
            if normalization:
                if normalization != "layer":
                    norm_dim = conv
                else:
                    norm_dim = [conv] + [d // (pool_kernel ** ite) for d in input_dim[1:]]
                layers.append(normalization_2d(normalization, norm_dim))
            layers.append(nn.MaxPool2d(pool_kernel, stride=pool_kernel))
            last_dim = conv
        super(Conv2DBlock, self).__init__(layers)


class RNNBlock(BaseLayersArchitecture):
    def __init__(self, rnn_type, input_dim, rnn_map, bidirectional,
                 normalization=None, pp_normalization=None):
        last_dim = input_dim[0]
        layers = [nn.Flatten(2, -1)]
        if pp_normalization:
            norm_dim = input_dim[0] if pp_normalization != "layer" else input_dim
            layers.append(normalization_1d(pp_normalization, norm_dim))
        layers.append(SwitchDim((0, 2, 1)))
        for rnn in rnn_map:
            rnn = input_dim[0] * rnn
            if rnn_type == "gru":
                layers.append(nn.GRU(last_dim, rnn, batch_first=True, bidirectional=bidirectional))
            elif rnn_type == "lstm":
                layers.append(nn.LSTM(last_dim, rnn, batch_first=True, bidirectional=bidirectional))
            else:
                raise ValueError(f"Unknown RNN type: {rnn_type}")
            layers.append(GetOutputFromRNN())
            last_dim = rnn * 2 if bidirectional else rnn
            if normalization:
                norm_dim = last_dim if normalization != "layer" else [last_dim, input_dim[1]]
                layers.append(SwitchDim((0, 2, 1)))
                layers.append(normalization_1d(normalization, norm_dim))
                layers.append(SwitchDim((0, 2, 1)))
        super(RNNBlock, self).__init__(layers)


class CNN1D(BaseLayersArchitecture):
    def __init__(self, input_dim, conv_map, conv_kernel, pool_kernel,
                 mlp_hidden_dims, output_dim, use_softmax,
                 conv_norm=None, pp_conv_norm=None, mlp_norm=None, pp_mlp_norm=None):
        layers = []
        if isinstance(input_dim, int):
            input_dim = [input_dim]
        if len(input_dim) == 1:
            input_dim = [1, input_dim[0]]
            layers.append(InsertDim(1))
        mlp_dim = [input_dim[0] * conv_map[-1], input_dim[1] // (pool_kernel ** len(conv_map))]
        layers.append(Conv1DBlock(input_dim, conv_map, conv_kernel, pool_kernel, conv_norm, pp_conv_norm))
        layers.append(MLP(mlp_dim, mlp_hidden_dims, output_dim, use_softmax, mlp_norm, pp_mlp_norm))
        super(CNN1D, self).__init__(layers)


class CNN2D(BaseLayersArchitecture):
    def __init__(self, input_dim, conv_map, conv_kernel, pool_kernel,
                 mlp_hidden_dims, output_dim, use_softmax,
                 conv_norm=None, pp_conv_norm=None, mlp_norm=None, pp_mlp_norm=None):
        if len(input_dim) == 2:
            mlp_dim = conv_map[-1] * input_dim[1] // (pool_kernel ** len(conv_map))
        else:
            mlp_dim = (input_dim[0] * conv_map[-1] *
                       math.prod([i // (pool_kernel ** len(conv_map)) for i in input_dim[1:]]))
        layers = [
            Conv2DBlock(input_dim, conv_map, conv_kernel, pool_kernel, conv_norm, pp_conv_norm),
            MLP(mlp_dim, mlp_hidden_dims, output_dim, use_softmax, mlp_norm, pp_mlp_norm)
        ]
        super(CNN2D, self).__init__(layers)


class RNN(BaseLayersArchitecture):
    def __init__(self, input_dim, rnn_type, rnn_map, bidirectional, output_dim, use_softmax,
                 rnn_norm=None, pp_rnn_norm=None, mlp_norm=None, pp_mlp_norm=None):
        layers = []
        if isinstance(input_dim, int):
            input_dim = [input_dim]
        if len(input_dim) == 1:
            input_dim = [1, input_dim[0]]
            layers.append(InsertDim(1))
        mlp_dim = [input_dim[0] * rnn_map[-1], input_dim[1]]
        if bidirectional:
            mlp_dim[0] = mlp_dim[0] * 2
        layers.append(RNNBlock(rnn_type, input_dim, rnn_map, bidirectional, rnn_norm, pp_rnn_norm))
        layers.append(MLP(mlp_dim, [], output_dim, use_softmax, mlp_norm, pp_mlp_norm))
        super(RNN, self).__init__(layers)


class CRNN1D(BaseLayersArchitecture):
    def __init__(self, input_dim, conv_map, conv_kernel, pool_kernel,
                 rnn_type, rnn_map, bidirectional, output_dim, use_softmax,
                 conv_norm=None, pp_conv_norm=None, rnn_norm=None, pp_rnn_norm=None, mlp_norm=None, pp_mlp_norm=None):
        rnn_input_dim = [
            input_dim[0] * conv_map[-1],
            input_dim[1] // (pool_kernel ** len(conv_map))
        ]
        layers = [
            Conv1DBlock(input_dim, conv_map, conv_kernel, pool_kernel, conv_norm, pp_conv_norm),
            RNN(rnn_input_dim, rnn_type, rnn_map, bidirectional, output_dim, use_softmax,
                rnn_norm, pp_rnn_norm, mlp_norm, pp_mlp_norm)
        ]
        super(CRNN1D, self).__init__(layers)


class CRNN2D(BaseLayersArchitecture):
    def __init__(self, input_dim, conv_map, conv_kernel, pool_kernel,
                 rnn_type, rnn_map, bidirectional, output_dim, use_softmax,
                 conv_norm=None, pp_conv_norm=None, rnn_norm=None, pp_rnn_norm=None, mlp_norm=None, pp_mlp_norm=None):
        ext_input_dim = [1] + input_dim if len(input_dim) == 2 else input_dim
        rnn_input_dim = [
            ext_input_dim[0] * conv_map[-1],
            ext_input_dim[1] * ext_input_dim[2] // (pool_kernel ** (2 * len(conv_map)))
        ]
        layers = [
            Conv2DBlock(input_dim, conv_map, conv_kernel, pool_kernel, conv_norm, pp_conv_norm),
            RNN(rnn_input_dim, rnn_type, rnn_map, bidirectional, output_dim, use_softmax,
                rnn_norm, pp_rnn_norm, mlp_norm, pp_mlp_norm)
        ]
        super(CRNN2D, self).__init__(layers)


class PrintShape(nn.Module):
    def __init__(self, text=""):
        super(PrintShape, self).__init__()
        self.text = text

    def forward(self, x):
        if isinstance(x, tuple):
            print(f"({self.text}) {[x[i].shape for i in range(len(x))]}")
        else:
            print(f"({self.text}) {x.shape}")
        return x


class InsertDim(nn.Module):
    def __init__(self, pos):
        super(InsertDim, self).__init__()
        self.pos = pos

    def forward(self, x):
        x = x.unsqueeze(self.pos)
        return x


class SwitchDim(nn.Module):
    def __init__(self, dims):
        super(SwitchDim, self).__init__()
        self.dims = tuple(dims)

    def forward(self, x):
        x = permute(x, self.dims)
        return x


class GetOutputFromRNN(nn.Module):
    def __init__(self):
        super(GetOutputFromRNN, self).__init__()

    @staticmethod
    def forward(x):
        tensor, _ = x  # get only the output tensor, not the hidden state and the cell state (in the case of RNN)
        return tensor


class GetLastOutputFromRNN(nn.Module):
    def __init__(self):
        super(GetLastOutputFromRNN, self).__init__()

    @staticmethod
    def forward(x):
        tensor, _ = x  # get only the output tensor, not the hidden state and the cell state (in the case of RNN)
        output = tensor[:, -1, :]  # get the last prediction from RNN
        return output
