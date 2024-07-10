#!/usr/bin/env python3

import logging

import numpy as np
import librosa
import torch


class Transpose(object):
    """
    Transpose a tensor. The dimensions to transpose are given as arguments.
    Default is to transpose the first two dimensions

    Notes
    -----
    Dataset's convention use the following dimensions:
    - 2D tensor : (features, channels)
    - 3D tensor : (features, channels, height)
    - 4D tensor : (features, channels, height, width)

    PyTorch's convention in the module torch.nn is to use the following dimensions:
    - 2D tensor : (batch_size, features)
    - 3D tensor : (batch_size, channels, features)
    - 4D tensor : (batch_size, channels, height, width)
    """
    def __init__(self, **kwargs):
        self.dim0 = kwargs.get("dim0", 0)
        self.dim1 = kwargs.get("dim1", 1)

    def __call__(self, tensor):
        return torch.transpose(tensor, self.dim0, self.dim1)


class Scaler(object):
    def __init__(self, **kwargs):
        self.axis = kwargs.get("axis", None)

    def __call__(self, tensor):
        return self.fit_transform(tensor)

    def fit_transform(self, tensor):
        raise NotImplementedError


class MinMaxScaler(Scaler):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.feature_range = kwargs.get("feature_range", (0, 1))
        logging.debug(f"MinMaxScaler initialized with {kwargs}")

    def fit_transform(self, tensor):
        tensor_min = tensor.min(self.axis, keepdim=True)[0] if self.axis else tensor.min()
        tensor_max = tensor.max(self.axis, keepdim=True)[0] if self.axis else tensor.max()
        tensor_std = (tensor - tensor_min) / (tensor_max - tensor_min)
        return tensor_std * (self.feature_range[1] - self.feature_range[0]) + self.feature_range[0]


class RobustScaler(Scaler):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.quantile_range = kwargs.get("quantile_range", (0.25, 0.75))
        logging.debug(f"RobustScaler initialized with {kwargs}")

    def fit_transform(self, tensor):
        lower_quantile, upper_quantile = self.quantile_range
        lower_bound = torch.quantile(tensor, lower_quantile, dim=self.axis, keepdim=True)
        upper_bound = torch.quantile(tensor, upper_quantile, dim=self.axis, keepdim=True)
        iqr = upper_bound - lower_bound
        iqr[iqr == 0] = 1.0  # Avoid division by zero

        return (tensor - lower_bound) / iqr


class StandardScaler(Scaler):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.epsilon = 1e-9  # Avoid division by zero
        logging.debug(f"StandardScaler initialized with {kwargs} and epsilon={self.epsilon}")

    def fit_transform(self, tensor):
        mean = tensor.mean(self.axis, keepdim=True) if self.axis else tensor.mean()
        std = tensor.std(self.axis, keepdim=True) if self.axis else tensor.std()
        return (tensor - mean) / (std + self.epsilon)


class BaselineCorrection(object):
    def __init__(self, **kwargs):
        self.i_ref = kwargs.get("i_ref", [0] * 4)
        self.i_0 = kwargs.get("i_0", [0] * 4)
        self.device = kwargs.get("device", "cpu")

        self.delta_i = torch.tensor(self.i_ref) - torch.tensor(self.i_0)
        self.delta_i = self.delta_i.reshape(-1, 1).to(self.device)
        logging.debug(f"BaselineCorrection initialized with {kwargs}")

    def __call__(self, tensor):
        return tensor + self.delta_i.repeat((1, tensor.shape[-1]))


class SensibilityCorrection(object):

    math_func = {
        "linear": lambda x, a, b: a * x + b,
        "log": lambda x, a, b, c: a * torch.log(b * x) + c,
        "exp": lambda x, a, b, c: a * torch.exp(b * x) + c,
        "power": lambda x, a, b, c: a * torch.pow(x, b) + c
    }

    def __init__(self, **kwargs):
        self.device = kwargs.get("device", "cpu")
        self.law = kwargs.get("law", [lambda: "Invalid"])
        self.param = kwargs.get("param", [])
        self.t_idx = kwargs.get("t_idx", 0)

        logging.debug(f"SensibilityCorrection initialized with {kwargs}")

    def __call__(self, tensor):
        tensor = tensor.to(self.device)
        lifetime = tensor[self.t_idx].unsqueeze(0)
        currents = torch.cat((tensor[:self.t_idx], tensor[self.t_idx + 1:]), dim=0)
        fitted_currents = []
        for law, param in zip(self.law, self.param):
            fitted_currents.append(self.math_func[law](lifetime, **param))
        residual = currents - torch.cat(fitted_currents, dim=0).to(self.device)
        return residual.to(self.device)


class MelSpectrogram(object):
    def __init__(self, **kwargs):
        self.sample_rate = kwargs.get("sample_rate", 10)
        self.n_fft = kwargs.get("n_fft", 50)
        self.hop_length = kwargs.get("hop_length", 10)
        self.n_mels = kwargs.get("n_mels", 32)
        self.power = kwargs.get("power", 2.0)
        self.device = kwargs.get("device", "cpu")

        self.mel_filter = torch.tensor(
            librosa.filters.mel(sr=self.sample_rate, n_fft=self.n_fft, n_mels=self.n_mels),
            dtype=torch.float32,
            device=self.device
        )
        logging.debug(f"MelSpectrogram initialized with {kwargs}")

    def __call__(self, tensor):
        return self.transform(tensor)

    def transform(self, tensor):
        """
        Input shape of the tensor : [200, 4]
        Compute stft on one dimension (x4), shape : [200] -> [26, 16, 2]
        Compute Forbenius norm (x4), shape : [26, 16, 2] -> [26, 16]
        Apply mel filter (x4), shape : [26, 16] -> [32, 16]
        Stack the 4 tensors, shape : [32, 16] -> [32, 16, 4]

        into CNN2D_regression.json : "input_dim": [4, 16, 32]
        """
        mel_spectrogram = []
        for i in range(tensor.shape[1]):
            stft = torch.stft(tensor[:, i], n_fft=self.n_fft, hop_length=self.hop_length, center=False)
            stft = torch.norm(stft, p=self.power, dim=-1)
            mel_spectrogram.append(self.mel_filter @ stft)
        tensor = torch.stack(mel_spectrogram, dim=-1)
        return tensor


class MelSpectrogramByOpenAI(object):
    def __init__(self, **kwargs):
        self.n_fft = kwargs.get("n_fft", 50)
        self.hop_length = kwargs.get("hop_length", 10)
        self.return_complex = kwargs.get("return_complex", True)
        self.device = kwargs.get("device", "cpu")
        self.window = torch.hann_window(self.n_fft).to(self.device)
        self.mel_filter_path = kwargs.get("mel_filter_path", "assets/mel_filter.npz")
        self.filter = self.load_mel_filter()
        self.filter = self.filter.to(self.device)
        logging.debug(f"MelSpectrogramByOpenAI initialized with {kwargs}")

    def __call__(self, tensor):
        return self.transform(tensor)

    def transform(self, tensor):
        """
        Input shape of the tensor : [200, 4]
        Compute stft on one dimension (x4), shape : [200] -> [26, 21]
        Compute magnitudes (x4), shape : [26, 21] -> [26, 20]
        Apply mel filter (x4), shape : [26, 20] -> [32, 20]
        Stack the 4 tensors, shape : [32, 20] -> [32, 20, 4]

        into CNN2D_regression.json : "input_dim": [4, 20, 32]
        """
        tensor = tensor.to(device=self.device)
        mel_spectrogram = []
        for i in range(tensor.shape[1]):
            stft = torch.stft(tensor[:, i], self.n_fft, self.hop_length,
                              window=self.window, return_complex=self.return_complex)
            magnitudes = stft[..., :-1].abs() ** 2
            mel_spec = self.filter @ magnitudes
            log_spec = torch.clamp(mel_spec, min=1e-10).log10()
            log_spec = torch.maximum(log_spec, log_spec.max() - 8.0)
            # log_spec = (log_spec + 4.0) / 4.0  # Not useful for us
            mel_spectrogram.append(log_spec)
        tensor = torch.stack(mel_spectrogram, dim=-1)
        return tensor

    def load_mel_filter(self):
        with np.load(self.mel_filter_path, allow_pickle=False) as f:
            return torch.from_numpy(f[f"mel_32"])

