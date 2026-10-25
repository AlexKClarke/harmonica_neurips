"""The main function for running a decomposition"""

from typing import Optional, List, Tuple, Union

import torch

from .state import State
from .configs import Config
from .logger import Logger
from .timestamping import (
    signal_to_timestamps,
    find_quality_metric,
    timestamps_to_windows,
)
from .preprocessing import (
    low_pass_filter,
    high_pass_filter,
    extend,
    whiten,
)
from .plotting import plot_sources


class DecompositionModule(torch.nn.Module):
    """
    Class implementing decomposition
    """

    def __init__(self):
        super().__init__()
        self._plot_fig = None
        self._plot_axes = None

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

    @staticmethod
    def baseline_drift(source: torch.Tensor, n_segments: int = 20) -> torch.Tensor:
        """Variance of segment means — measures low-frequency baseline drift."""
        chunks = source.tensor_split(n_segments)
        segment_means = torch.stack([c.mean() for c in chunks])
        return segment_means.var()

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
        """Peel off to prevent repeats"""
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

    def run_decomposition(self):
        """Runs a complete source independent component analysis on the entire signal"""

        # Initialize all components
        self.state.init_all()

        self._comp_step_count = 0
        self._cached_spike_ts = None

        # Run decomposition steps
        patience = 0
        best_isi = self.state.signal.shape[0]
        isi = self.state.signal.shape[0]
        early_end = False
        plot = False
        silhouette = torch.tensor(0.0)
        num_peaks = 0
        ref_drift = 1.0

        for steps in range(self.config.max_decomp_steps):

            use_compensation = steps >= self.config.compensation_delay

            # Check silhouette at compensation delay boundary and record baseline drift
            if steps == self.config.compensation_delay:
                _, silhouette = self.calculate_timestamps(True)
                if silhouette < self.config.min_sil_for_compensation:
                    early_end = True
                    if self.config.save_results or self.config.verbose_mode:
                        print(f"    Step {steps:4d}: Low silhouette ({silhouette:.3f}), ending early", flush=True)
                with torch.no_grad():
                    source_ref = self.calculate_source(self.state.signal, compensated=False)
                    source_ref = (source_ref - source_ref.mean()) / source_ref.std()
                    ref_drift = max(self.baseline_drift(source_ref).item(), 1.0)

            # Check compensation stability periodically
            if use_compensation and not early_end and steps > self.config.compensation_delay:
                if self.config.log_every_n_steps > 0 and steps % self.config.log_every_n_steps == 0:
                    with torch.no_grad():
                        comp_source = self.calculate_source(self.state.signal, compensated=True)
                        comp_source = (comp_source - comp_source.mean()) / comp_source.std()
                        current_drift = self.baseline_drift(comp_source)
                    if current_drift > ref_drift * self.config.max_compensation_drift:
                        early_end = True
                        if self.config.save_results or self.config.verbose_mode:
                            print(
                                f"    Step {steps:4d}: Compensation drift {current_drift:.3f} > "
                                f"{self.config.max_compensation_drift:.1f}x reference {ref_drift:.3f}, stopping",
                                flush=True,
                            )

            # Joint optimization step
            isi = self.joint_step(use_compensation and not early_end)

            # Analyze ISI for early stopping
            if not early_end:
                if best_isi > isi:
                    patience = 0
                    best_isi = isi
                else:
                    patience += 1
                    if patience > self.config.max_decomp_patience:
                        early_end = True

            # Log progress periodically
            should_log = (
                (self.config.save_results or self.config.verbose_mode)
                and self.config.log_every_n_steps > 0
                and (steps + 1) % self.config.log_every_n_steps == 0
            )
            if should_log:
                phase = "sep+comp" if steps >= self.config.compensation_delay else "sep"
                # Calculate current silhouette for logging
                timestamps_log, silhouette_log = self.calculate_timestamps(True)
                num_peaks = timestamps_log.shape[0]
                print(
                    f"    Step {steps + 1:4d}/{self.config.max_decomp_steps} | "
                    f"Phase: {phase:8s} | "
                    f"ISI: {int(isi):6d} | "
                    f"Peaks: {num_peaks:4d} | "
                    f"Sil: {silhouette_log:.3f} | "
                    f"Patience: {patience:3d}/{self.config.max_decomp_patience}",
                    flush=True,
                )

            # Check if we should plot
            plot = False
            if self.config.source_plot_per_n_steps:
                if steps % self.config.source_plot_per_n_steps == 0:
                    plot = True
            elif self.config.source_plot_at_end:
                if steps == (self.config.max_decomp_steps - 1) or early_end:
                    plot = True

            # Get source and timestamps if needed
            timestamps: torch.Tensor = torch.tensor([])
            silhouette = torch.tensor(0.0)
            if plot or early_end or steps == (self.config.max_decomp_steps - 1):
                timestamps, silhouette = self.calculate_timestamps(True)

            # Plot if needed
            if plot:
                with torch.no_grad():
                    self.eval()
                    source = self.calculate_source(self.state.signal, compensated=True)
                    self._plot_fig, self._plot_axes = plot_sources(
                        source.subtract(source.mean()).divide(source.std()),
                        [timestamps],
                        self.state.exponent_tensor,
                        silhouette.unsqueeze(0),
                        self._plot_fig,
                        self._plot_axes,
                        save_dir=self._save_dir,
                    )

            if early_end:
                break

        # Get final timestamps and silhouette
        timestamps, silhouette = self.calculate_timestamps(True)

        # Check if source meets quality criteria
        if silhouette < self.config.acceptance_silhouette:
            message = "reject low silhouette source"
            success = False
        elif timestamps.shape[0] < self.config.min_peaks_in_source:
            message = "reject source below min peak number"
            success = False
        else:
            # Check if source is a repeat
            if len(self.state.all_timestamp_lib) > 0:
                max_accuracy = torch.stack(
                    [
                        find_quality_metric(
                            timestamps,
                            prev_timestamps,
                            "accuracy",
                            self.config.accuracy_tolerance,
                            self.config.extension_factor,
                        )
                        for prev_timestamps in self.state.all_timestamp_lib
                    ]
                ).max()

                # End run if repeat with optional peel off
                if max_accuracy > self.config.acceptance_max_accuracy:
                    if max_accuracy > 90:
                        self.peel_off(timestamps)
                    success = False
                    message = "reject repeat source"
                else:
                    success = True
                    message = "new unit found"
            else:
                success = True
                message = "new unit found"

        # If source is good, add to libraries
        if success:
            if timestamps.shape[0] > 1:
                self.peel_off(timestamps)
                self.state.remove_activity(timestamps)
            
            self.state.all_timestamp_lib.append(timestamps.sort()[0])

        return success, message

    def run(
        self,
        signal: torch.Tensor,
        config: Config,
        auxiliary: Optional[torch.Tensor] = None,
    ) -> List[torch.Tensor]:
        """Sets optimiser and runs

        Args:
            signal: Input signal tensor
            config: Configuration object
            auxiliary: Optional external auxiliary signal of shape (T, D). If None,
                      positional encodings are generated.
        """

        # Store config
        self.config = config

        # Preprocess signal
        device = signal.device
        signal = signal.cpu().float()
        if self.config.low_pass_cutoff is not None:
            assert self.config.sampling_frequency is not None, \
                "Sampling frequency must be set in config if filtering."
            signal = low_pass_filter(
                signal, self.config.sampling_frequency, self.config.low_pass_cutoff
            )
        if self.config.high_pass_cutoff is not None:
            assert self.config.sampling_frequency is not None, \
                "Sampling frequency must be set in config if filtering."
            signal = high_pass_filter(
                signal, self.config.sampling_frequency, self.config.high_pass_cutoff
            )
        signal = extend(signal, self.config.extension_factor)
        signal = whiten(signal, self.config.whitening_method)
        signal[: self.config.extension_factor, :] = 0
        signal[-self.config.extension_factor :, :] = 0
        signal = signal.to(device)

        # Create state with preprocessed signal
        self.state = State(signal, self.config, auxiliary)

        # Setup logger
        logger = Logger(self.config)
        logger.setup()
        self._save_dir = logger.run_dir

        # Start the projection pursuit iterations
        patience = 0
        for iteration in range(self.config.max_iterations):

            # First run decomp
            success, message = self.run_decomposition()

            # Increment patience if failure
            if success:
                patience = 0
            else:
                patience += 1

            # Log iteration
            logger.log_iteration(
                iteration=iteration,
                success=success,
                message=message,
                num_units=len(self.state.all_timestamp_lib),
                patience=patience,
                max_iterations=self.config.max_iterations,
                iteration_patience=self.config.iteration_patience,
            )

            # Finish finding sources if patience is broken
            if patience == self.config.iteration_patience:
                logger.log_early_stopping()
                break

        # Save results and cleanup
        logger.save_results(self.state.all_timestamp_lib)
        logger.close()

        return self.state.all_timestamp_lib
