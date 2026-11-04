"""Engine configuration: the Config base and the SurfaceConfig preset."""

from typing import Optional, Sequence

from pydantic import BaseModel, computed_field


class Config(BaseModel):

    # LOGGING CONFIGS
    checkpoint_dir: str = ".logging/harmonica"
    run_name: Optional[str] = None  # If None, uses timestamp
    save_results: bool = True

    # DATA CLASS CONFIGS

    # Preprocessing parameters
    sampling_frequency: float
    low_pass_cutoff: Optional[float] = None
    high_pass_cutoff: Optional[float] = None
    extension_factor: int = 15
    whitening_method: str = "zca"
    positional_encoding_periods_s: Sequence[float] = (0.1, 0.2, 0.5, 1.0, 2.0)

    # Activity index parameters (ms)
    act_min_peak_separation_ms: float = 15.0
    act_min_peak_height: float = 1.0

    # Separation parameters / hyperparameters
    separation_lr: float = 0.0001  # *

    # Compensation parameters / hyperparameters
    compensation_lr: float = 0.0001  # *
    compensation_model: str = "lowrank"  # "lowrank" or "affine"
    compensation_n_layers: int = 1  # Number of compensation layers
    compensation_hidden_layers: Sequence[int] = (32, 32)
    dropout: float = 0.3
    num_null_samples: int = 1000
    spike_dropout: float = 0.5
    compensation_rank: int = 3  # Rank for lowrank model

    # RUN CONFIGS
    # ICA parameters / hyperparameters
    clamp_max: float = 30.0  # Clamp normalized source to prevent outlier-dominated gradients
    exponent: float = 6.0  # *
    topk_lim: int = 30
    nonlin_scale: float = 100
    nonlin_scale_decrement: float = 1.0
    max_decomp_patience: int = 100
    max_decomp_steps: int = 1000  # *
    compensation_delay: int = 100  # *
    min_sil_for_compensation: float = 0.5
    max_compensation_drift: float = 3.0  # Stop compensation if baseline drift exceeds this multiple of pre-compensation drift
    compensation_resample_every: int = 5  # Re-find spike/null peaks every N comp steps (1 = every step)

    # Source timestamping parameters (ms)
    min_peak_separation_ms: float = 25.0
    min_peak_height: float = 5
    k_window_ms: Optional[float] = None

    # Main run parameters
    max_iterations: int = 1000

    # Projection pursuit controls
    iteration_patience: int = 50
    acceptance_silhouette: float = 0.9
    min_peaks_in_source: int = 10
    acceptance_max_accuracy: float = 30
    accuracy_tolerance_ms: float = 2.5
    accuracy_max_shift_ms: float = 10.0
    peel_off_window_ms: float = 2.5
    num_peel_off_neighbours: int = 10

    # Plotting and verbosity
    verbose_mode: bool = False
    log_every_n_steps: int = 100  # Print progress every N decomposition steps
    source_plot_per_n_steps: Optional[int] = None
    source_plot_at_end: bool = False

    # Computed properties: ms -> samples

    @computed_field
    @property
    def act_min_peak_separation(self) -> int:
        return round(self.act_min_peak_separation_ms / 1000 * self.sampling_frequency)

    @computed_field
    @property
    def min_peak_separation(self) -> int:
        return round(self.min_peak_separation_ms / 1000 * self.sampling_frequency)

    @computed_field
    @property
    def k_window_size(self) -> Optional[int]:
        if self.k_window_ms is None:
            return None
        return round(self.k_window_ms / 1000 * self.sampling_frequency)

    @computed_field
    @property
    def accuracy_tolerance(self) -> int:
        return round(self.accuracy_tolerance_ms / 1000 * self.sampling_frequency)

    @computed_field
    @property
    def accuracy_max_shift(self) -> int:
        return round(self.accuracy_max_shift_ms / 1000 * self.sampling_frequency)

    @computed_field
    @property
    def peel_off_window_size(self) -> int:
        return round(self.peel_off_window_ms / 1000 * self.sampling_frequency)


class SurfaceConfig(Config):

    # Preprocessing parameters
    sampling_frequency: float = 2048
    low_pass_cutoff: Optional[float] = 1000
    high_pass_cutoff: Optional[float] = 10
    extension_factor: int = 100

    # Activity index parameters
    act_min_peak_height: float = 3

    # Separation parameters / hyperparameters
    separation_lr: float = 0.00001

    # Compensation parameters / hyperparameters
    compensation_hidden_layers: Sequence[int] = (64, 64, 64)
    dropout: float = 0.2

    # ICA parameters / hyperparameters
    exponent: float = 3.0
    nonlin_scale: float = 1
    max_decomp_patience: int = 1000
    compensation_delay: int = 500

    # Source timestamping parameters
    min_peak_height: float = 3

    # Main run parameters
    max_iterations: int = 100

    # Projection pursuit controls
    iteration_patience: int = 20
    acceptance_silhouette: float = 0.85
    min_peaks_in_source: int = 20
    accuracy_max_shift_ms: float = 7.5
    num_peel_off_neighbours: int = 30
