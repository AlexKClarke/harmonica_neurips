"""Logger class for handling I/O operations during decomposition"""

import os
import json
from typing import List, Optional, TextIO
from datetime import datetime

import torch

from .configs import Config


class Logger:
    """Handles all I/O operations for decomposition runs."""

    def __init__(self, config: Config):
        self.config = config
        self.run_dir: Optional[str] = None
        self.log_file: Optional[TextIO] = None

    def setup(self) -> None:
        """Create run directory and initialize log file."""
        if not self.config.save_results:
            return

        if self.config.run_name is None:
            run_name = f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        else:
            run_name = self.config.run_name
        self.run_dir = os.path.join(self.config.checkpoint_dir, run_name)
        os.makedirs(self.run_dir, exist_ok=True)
        print(f"Run directory: {self.run_dir}", flush=True)

        # Save config as JSON
        config_dict = self.config.model_dump()
        config_path = os.path.join(self.run_dir, "config.json")
        with open(config_path, "w") as f:
            json.dump(config_dict, f, indent=2)

        # Setup CSV log file
        log_file_path = os.path.join(self.run_dir, "iteration_log.csv")
        self.log_file = open(log_file_path, "w")
        self.log_file.write("iteration,success,message,num_units_found,patience\n")

    def log_iteration(
        self,
        iteration: int,
        success: bool,
        message: str,
        num_units: int,
        patience: int,
        max_iterations: int,
        iteration_patience: int,
    ) -> None:
        """Log iteration progress to console and CSV."""
        # Print progress
        if self.config.save_results or self.config.verbose_mode:
            status = "✓" if success else "✗"
            print(
                f"Iter {iteration + 1:4d}/{max_iterations} | "
                f"{status} {message:<30} | "
                f"Units: {num_units:3d} | "
                f"Patience: {patience:2d}/{iteration_patience}",
                flush=True,
            )

        # Log to CSV
        if self.log_file is not None:
            self.log_file.write(
                f"{iteration},{success},{message},{num_units},{patience}\n"
            )
            self.log_file.flush()

    def log_early_stopping(self) -> None:
        """Log early stopping message."""
        if self.config.save_results or self.config.verbose_mode:
            print("Early stopping: patience exhausted", flush=True)

    def save_results(self, timestamps: List[torch.Tensor]) -> None:
        """Save final results to disk."""
        if self.run_dir is not None:
            results_path = os.path.join(self.run_dir, "results.pt")
            torch.save(timestamps, results_path)
            print(f"Saved {len(timestamps)} units to {results_path}", flush=True)

    def close(self) -> None:
        """Close log file."""
        if self.log_file is not None:
            self.log_file.close()
            self.log_file = None

