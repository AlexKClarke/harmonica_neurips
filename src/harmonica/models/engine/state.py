"""Dataclass structure for configuration and model state containers"""

from typing import List, Optional

import torch
import torch.optim as optim

from .timestamping import peak_finder
from .nets import ConditionalAffine, ConditionalGivensMatrix, ConditionalLowRank
from .configs import Config


class State:

    def __init__(self, signal: torch.Tensor, config: Config, auxiliary: Optional[torch.Tensor] = None):
        """Initialize state with an already-preprocessed signal.
        
        Args:
            signal: Pre-processed signal tensor (filtered, extended, whitened)
            config: Configuration object
            auxiliary: Optional external auxiliary signal of shape (T, D). If None,
                      positional encodings are generated.
        """
        self.config = config

        # Prepare auxiliary
        if auxiliary is not None:
            if auxiliary.dim() == 1:
                auxiliary = auxiliary.unsqueeze(1)
            auxiliary = auxiliary.float().to(signal.device)
        else:
            auxiliary = self.get_auxiliary(signal.shape[0])
            auxiliary = auxiliary.float().to(signal.device)

        # Get activity
        activity = self.estimate_activity(signal)

        # Store the full signal and auxiliary
        self.signal = signal
        self.auxiliary = auxiliary
        self.activity = activity

        # Initialize libraries
        self.all_timestamp_lib: List[torch.Tensor] = []

    def get_auxiliary(self, signal_length: int) -> torch.Tensor:
        t = torch.arange(signal_length, dtype=torch.float32) / self.config.sampling_frequency
        periods = torch.tensor(self.config.positional_encoding_periods_s)
        return torch.sin(2 * torch.pi * t.unsqueeze(1) / periods)

    def init_all(self):
        """Initialize all variables"""
        self.init_exponent()
        self.init_separation_weights()
        self.init_compensation_weights()
        self.init_optimisers()

    def estimate_activity(self, signal: torch.Tensor) -> torch.Tensor:
        """Absolute sum across channels for separ vector initialisation"""

        abs_signal = signal.abs().sum(1)
        peaks = peak_finder(
            abs_signal,
            self.config.act_min_peak_separation,
            float(self.config.act_min_peak_height * abs_signal.std() + abs_signal.mean()),
        )
        peaks = peaks[abs_signal[peaks].argsort(descending=True)]

        # Top 10% likely noise or superposition so remove
        return peaks[int(peaks.shape[0] * 0.1) :]

    def init_exponent(self):
        """Exponent of the ICA loss"""
        self.exponent_tensor = self.config.exponent * torch.ones(1).type_as(self.signal)

    def init_separation_weights(self):
        """Initialize the gradient descent ICA separation vector"""
        if self.activity.shape[0] > 0:
            sample = self.signal[self.activity[0]]
            self.activity = self.activity[1:]
        else:
            sample = torch.randn_like(self.signal[0])

        weights = torch.divide(sample, sample.abs().sum())
        self.separation_weights = torch.nn.Parameter(weights)

    def init_compensation_weights(self):
        """Initialize the variables associated with dynamic compensation"""
        # Select model class based on config
        model_classes = {
            "affine": ConditionalAffine,
            "givens": ConditionalGivensMatrix,
            "lowrank": ConditionalLowRank,
        }
        model_class = model_classes[self.config.compensation_model]

        extra_kwargs = {}
        if self.config.compensation_model == "lowrank":
            extra_kwargs["rank"] = self.config.compensation_rank

        self.compensation_weights = model_class(
            input_shape=[self.auxiliary.shape[1]],
            output_shape=[self.signal.shape[1]],
            out_chans_per_layer=list(self.config.compensation_hidden_layers),
            dropout=self.config.dropout,
            n_layers=self.config.compensation_n_layers,
            **extra_kwargs,
        ).to(self.signal.device, self.signal.dtype)

        # Initialize params cache to zeros (identity transform)
        self.compensation_weights.init_cache(
            self.signal.shape[0], self.signal.device, self.signal.dtype
        )

    def update_compensation_cache(self):
        """Update compensation params cache from current auxiliary."""
        self.compensation_weights.update_cache(self.auxiliary)

    def init_optimisers(self):
        """Initialize the alternating backprop optimisers"""
        self.separation_optimiser = optim.RMSprop(
            params=[self.separation_weights],
            lr=self.config.separation_lr,
        )
        self.compensation_optimiser = optim.RMSprop(
            self.compensation_weights.parameters(), lr=self.config.compensation_lr
        )

    def remove_activity(self, timestamps: torch.Tensor):
        """Remove timestamps from activity indices"""
        mask = ~torch.isin(self.activity, timestamps)
        self.activity = self.activity[mask]

