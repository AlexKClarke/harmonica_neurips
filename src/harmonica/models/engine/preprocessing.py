"""Functions to preprocess the signal ready for blind source separation"""

from typing import cast, Tuple

import numpy as np
from numpy.typing import NDArray
import torch
from scipy.signal import butter, filtfilt


def high_pass_filter(signal: torch.Tensor, f_samp: float, cut_off: float):
    """Filter signal channel by channel"""

    keep = torch.zeros(1).type_as(signal)
    signal_np = signal.cpu().numpy()

    b, a = cast(Tuple[NDArray[np.float64], NDArray[np.float64]], butter(2, 2 * cut_off / f_samp, btype="highpass", output="ba"))
    for channel in range(signal_np.shape[1]):
        signal_np[:, channel] = filtfilt(b, a, signal_np[:, channel])

    return torch.from_numpy(signal_np).type_as(keep)


def low_pass_filter(signal: torch.Tensor, f_samp: float, cut_off: float):
    """Filter signal channel by channel"""

    keep = torch.zeros(1).type_as(signal)
    signal_np = signal.cpu().numpy()

    b, a = cast(Tuple[NDArray[np.float64], NDArray[np.float64]], butter(2, 2 * cut_off / f_samp, btype="lowpass", output="ba"))
    for channel in range(signal_np.shape[1]):
        signal_np[:, channel] = filtfilt(b, a, signal_np[:, channel])

    return torch.from_numpy(signal_np).type_as(keep)


def extend(signal: torch.Tensor, factor: int) -> torch.Tensor:
    """Extends each sample with factor past values"""

    assert signal.ndim == 2, "Input must be two-dimensional"

    # Pad end with zeros to stop torch.roll moving end samples to start
    signal = torch.concat(
        [torch.zeros([factor, signal.shape[1]]).type_as(signal), signal]
    )

    # Perform extension and return with rolled values removed
    signal = torch.concat([signal.roll(shift, 0) for shift in range(factor)], 1)
    return signal[factor:]


def whiten(x: torch.Tensor, method: str = "pca") -> torch.Tensor:
    """
    Performs whitening on input of shape (samples, channels)

    Whitening transform can be calculated by selecting the below options
    for the method argument:

    "chol": Cholesky method
    "zca": ZCA method on covariance matrix
    "pca": PCA method on covariance matrix
    "zca_cor": ZCA method on correlation matrix
    "pca_cor": PCA method on correlation matrix

    """

    # Inconsistent behaviour on cuda - switch to cpu
    keep = torch.zeros(1).type_as(x)
    x = x.to(device="cpu", dtype=torch.float32)

    x = x.t()
    x -= x.mean(1, keepdim=True)
    cov = x.cov()

    w: torch.Tensor
    if method in ["zca", "pca", "chol"]:
        u, s, _ = torch.linalg.svd(cov)
        if method == "chol":
            s_inv = (s + 1e-10).reciprocal().diag()
            cov_inv = u.matmul(s_inv).matmul(u.t())
            w = torch.linalg.cholesky(cov_inv).t()
        elif method == "zca":
            s_inv_sqrt = torch.sqrt(s + 1e-10).reciprocal().diag()
            w = u.matmul(s_inv_sqrt).matmul(u.t())
        else:  # method == "pca"
            s_inv_sqrt = torch.sqrt(s + 1e-10).reciprocal().diag()
            w = torch.matmul(s_inv_sqrt, u.t())
    elif method in ["zca_cor", "pca_cor"]:
        v_inv_sqrt = cov.diag().sqrt().reciprocal().diag()
        corr = v_inv_sqrt.matmul(cov).matmul(v_inv_sqrt)
        u, s, _ = torch.linalg.svd(corr)
        s_inv_sqrt = torch.sqrt(s + 1e-10).reciprocal().diag()
        if method == "zca_cor":
            w = u.matmul(s_inv_sqrt).matmul(u.t()).matmul(v_inv_sqrt)
        else:  # method == "pca_cor"
            w = s_inv_sqrt.matmul(u.t()).matmul(v_inv_sqrt)
    else:
        raise Exception("Specified method not in list.")

    return torch.matmul(w, x).t().type_as(keep)
