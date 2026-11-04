"""DecompositionModule: the per-unit training step and neural peel-off."""

from typing import Optional, Tuple, Union

import torch

from .timestamping import signal_to_timestamps, timestamps_to_windows


class DecompositionModule(torch.nn.Module):
    """Separation vector + compensator training and peel-off on a :class:`State`.

    The caller sets ``config`` and ``state`` before use (see
    :func:`harmonica.baselines.signal_space.run_signal_space`).
    """

    def calculate_source(
        self, signal: torch.Tensor, compensated: bool = False
    ) -> torch.Tensor:
        """Apply separation vectors to signal to get sources"""

        if compensated:
            comp_signal = self.state.compensation_weights.forward(
                signal, self.state.auxiliary, use_cache=True
            )
            source = torch.matmul(comp_signal, self.state.separation_weights)
        else:
            source = torch.matmul(signal, self.state.separation_weights)

        return source

    def calculate_timestamps(
        self,
        calculate_silhouette: bool = False,
        source: Optional[torch.Tensor] = None,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """Calulate timestamps from fitted GMM on whole source.

        Args:
            calculate_silhouette: Whether to return silhouette score.
            source: Optional precomputed normalized source to avoid recomputation.
        """
        with torch.no_grad():
            self.eval()
            if source is None:
                source = self.calculate_source(self.state.signal, compensated=True)
                source = (source - source.mean()) / source.std()
            return signal_to_timestamps(
                source,
                min_peak_separation=self.config.min_peak_separation,
                min_peak_height=self.config.min_peak_height,
                min_peaks=self.config.min_peaks_in_source,
                calculate_silhouette=calculate_silhouette,
            )

    def sample_spikes_and_nulls(
        self,
        source: Optional[torch.Tensor] = None,
        reuse_cached: bool = False,
    ):
        """Sample spike and null vectors/aux for compensation training.

        Args:
            source: Optional precomputed normalized source to avoid recomputation.
            reuse_cached: If True and cached timestamps exist, skip peak finding
                and reuse cached spike/null indices. Fresh signal vectors are still
                read at those indices.
        """
        with torch.no_grad():
            if reuse_cached and hasattr(self, "_cached_spike_ts") and self._cached_spike_ts is not None:
                spike_timestamps = self._cached_spike_ts
                isi = self._cached_isi
            else:
                result = self.calculate_timestamps(source=source)
                assert isinstance(result, torch.Tensor)
                spike_timestamps = result

                if spike_timestamps.shape[0] < int(1 / self.config.dropout) + 2:
                    self._cached_spike_ts = None
                    return None

                isi = int(spike_timestamps.diff().float().quantile(0.5).item())
                self._cached_spike_ts = spike_timestamps
                self._cached_isi = isi

            # Get null indices via rejection sampling (fresh each call)
            signal_length = self.state.signal.shape[0]
            num_needed = self.config.num_null_samples
            oversample = int(num_needed * 1.5) + spike_timestamps.shape[0]
            candidates = torch.randint(
                0, signal_length, (oversample,), device=self.state.signal.device
            )
            is_spike = torch.isin(candidates, spike_timestamps)
            null_timestamps = candidates[~is_spike][:num_needed]

            # Subsample spikes (fresh shuffle each call)
            num_spikes = int(spike_timestamps.shape[0] * (1 - self.config.spike_dropout))
            spike_perm = torch.randperm(
                spike_timestamps.shape[0], device=spike_timestamps.device
            )
            spike_timestamps_sub = spike_timestamps[spike_perm[:num_spikes]]

            return (
                self.state.signal[spike_timestamps_sub],
                self.state.auxiliary[spike_timestamps_sub],
                self.state.signal[null_timestamps],
                self.state.auxiliary[null_timestamps],
                isi,
            )

    def joint_step(self, use_compensation: bool):
        """Joint optimization of separation and compensation.

        Args:
            use_compensation: Whether to run compensation this step.

        Returns:
            ISI estimate for patience tracking
        """
        self.train()
        self.state.separation_optimiser.zero_grad()
        self.state.compensation_optimiser.zero_grad()

        # 1) Uncompensated source for ICA loss (only updates separation weights)
        source = self.calculate_source(self.state.signal, compensated=False)

        # 2) Separation loss (ICA)
        sep_source = (source - source.mean()) / source.std()
        sep_source = sep_source.clamp(max=self.config.clamp_max)
        sep_nonlin = sep_source.sign() * (self.config.nonlin_scale * sep_source).abs().pow(
            self.state.exponent_tensor
        )
        L_sep = -sep_nonlin.mean()

        L_sep.backward()
        torch.nn.utils.clip_grad_norm_([self.state.separation_weights], 10.0)
        self.state.separation_optimiser.step()

        # 3) Compensation loss (spike / null contrast) — separate backward
        isi = self.state.signal.shape[0]

        if use_compensation:
            self._comp_step_count = getattr(self, "_comp_step_count", 0) + 1
            resample = self._comp_step_count % self.config.compensation_resample_every == 1 or \
                self.config.compensation_resample_every == 1

            if resample:
                with torch.no_grad():
                    comp_source = self.calculate_source(self.state.signal, compensated=True)
                    comp_source = (comp_source - comp_source.mean()) / comp_source.std()
            else:
                comp_source = None

            samples = self.sample_spikes_and_nulls(
                source=comp_source, reuse_cached=not resample
            )

            if samples is not None:
                spike_vectors, spike_aux, null_vectors, null_aux, isi = samples

                with torch.no_grad():
                    w = self.state.separation_weights.detach().clone()

                comp_spike_vectors = self.state.compensation_weights.forward(
                    spike_vectors, spike_aux, use_cache=False
                )
                comp_null_vectors = self.state.compensation_weights.forward(
                    null_vectors, null_aux, use_cache=False
                )

                spike_source = torch.matmul(comp_spike_vectors, w)
                null_source = torch.matmul(comp_null_vectors, w)

                spike_std = spike_source.std()
                null_std = null_source.std()
                uncomp_null_var = torch.matmul(null_vectors, w).var()

                L_comp = (
                    spike_std.square() + torch.maximum(null_std.square(), uncomp_null_var)
                ).sqrt()

                L_comp.backward()
                torch.nn.utils.clip_grad_norm_(self.state.compensation_weights.parameters(), max_norm=0.1)
                self.state.compensation_optimiser.step()

                self.state.update_compensation_cache()

        return isi

    def peel_off(self, timestamps: torch.Tensor):
        """Neural peel-off: subtract the compensated-frame STA, mapped back
        through the inverse compensator, at every timestamp."""
        # First get the spike triggered windows using cached params
        comp_signal = self.state.compensation_weights.forward(
            self.state.signal, self.state.auxiliary, use_cache=True
        )
        windows = timestamps_to_windows(
            comp_signal, timestamps, self.config.peel_off_window_size
        )

        sta = windows.mean(0)

        # Finally peel off a compensated version of the STA from the signal
        hw = int(self.config.peel_off_window_size / 2)
        odd = int(self.config.peel_off_window_size % 2)
        for s in timestamps:
            start = s - hw
            end = s + hw + odd

            # Use inverse transform with sliced cached params
            local_sta = self.state.compensation_weights.inverse(
                sta, self.state.auxiliary, use_cache=True, indices=slice(start, end)
            )
            self.state.signal[start:end] -= local_sta
