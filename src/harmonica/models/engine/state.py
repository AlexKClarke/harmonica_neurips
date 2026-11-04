"""State: the whitened signal, its context signal, and the trainable weights."""

from typing import Optional

import torch
import torch.optim as optim

from .nets import ConditionalAffine, ConditionalLowRank
from .configs import Config


class State:
    """Whitened signal ``(T, C)``, context ``(T, n_ctx)``, and the trainable
    separation vector, compensator and their optimisers."""

    def __init__(self, signal: torch.Tensor, config: Config, auxiliary: Optional[torch.Tensor] = None):
        """Initialize state with an already-preprocessed signal.

        Args:
            signal: Whitened signal, ``(T, C)``.
            config: Configuration object.
            auxiliary: Optional context signal of shape ``(T,)`` or
                ``(T, n_ctx)`` (e.g. the joint angle). If None, sinusoidal
                positional encodings of time are generated.
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

        self.signal = signal
        self.auxiliary = auxiliary

    def get_auxiliary(self, signal_length: int) -> torch.Tensor:
        t = torch.arange(signal_length, dtype=torch.float32) / self.config.sampling_frequency
        periods = torch.tensor(self.config.positional_encoding_periods_s)
        return torch.sin(2 * torch.pi * t.unsqueeze(1) / periods)

    def init_exponent(self):
        """Exponent of the ICA loss"""
        self.exponent_tensor = self.config.exponent * torch.ones(1).type_as(self.signal)

    def init_compensation_weights(self):
        """Initialize the variables associated with dynamic compensation"""
        # Select model class based on config
        model_classes = {
            "affine": ConditionalAffine,
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
