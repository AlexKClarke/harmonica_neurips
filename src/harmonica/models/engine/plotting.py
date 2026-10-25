"""Matplotlib functions for plotting decomposition outputs"""

from typing import List, Optional, Tuple

import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.axes import Axes


def plot_sources(
    sources: torch.Tensor,
    timestamps: Optional[List[torch.Tensor]] = None,
    exponents: Optional[torch.Tensor] = None,
    fitness: Optional[torch.Tensor] = None,
    fig: Optional[Figure] = None,
    ax_mat: Optional[List[Axes]] = None,
    save_dir: Optional[str] = None,
) -> Tuple[Figure, List[Axes]]:
    """Sources of shape [time, sources]
    
    Returns (fig, ax_mat) for reuse in subsequent calls.
    """

    # Check sources dimensions
    if len(sources.shape) == 1:
        sources = sources.unsqueeze(1)
    elif len(sources.shape) > 2:
        raise ValueError(
            "sources must be shape [time, source] or [time] for a single source"
        )
    assert (
        sources.shape[0] > sources.shape[1]
    ), "time must be longer than source in sources input, is it transposed?"

    # Convert inputs to something plt can understand
    sources_np = sources.detach().cpu().numpy()
    timestamps_np = None
    if timestamps is not None:
        timestamps_np = [t.detach().cpu().numpy() for t in timestamps]
    exponents_np = None
    if exponents is not None:
        exponents_np = exponents.detach().cpu().numpy()
    fitness_np = None
    if fitness is not None:
        fitness_np = fitness.detach().cpu().numpy()

    # Create figure if it doesn't exist
    if fig is None or ax_mat is None:
        # Build out the plt axis objects
        num_rows = (sources_np.shape[1] // 2) + (1 if sources_np.shape[1] % 2 != 0 else 0)
        fig, axes = plt.subplots(
            num_rows, 1 if (sources_np.shape[1] == 1) else 2, figsize=(16, 8)
        )
        
        # Get screen size and adjust figure size
        mng = plt.get_current_fig_manager()
        if mng is not None and hasattr(mng, 'window'):
            window = getattr(mng, 'window', None)
            if window is not None:
                try:
                    screen_width = window.winfo_screenwidth()
                    screen_height = window.winfo_screenheight()
                    # Use 50% of screen size
                    fig_width = screen_width/200
                    fig_height = screen_height/200
                    fig.set_size_inches(fig_width, fig_height)
                    
                    # Center the window with fixed position
                    x = int((screen_width - fig_width*100) // 2)
                    y = int((screen_height - fig_height*100) // 2)
                    # Force window to specific position
                    window.geometry(f"+{x}+{y}")
                    # Ensure window is raised and focused
                    window.lift()
                    window.focus_force()
                except Exception as e:
                    print(f"Warning: Could not set window position: {e}")
        
        if sources_np.shape[1] != 1:
            ax_mat = list(axes.flatten())
        else:
            ax_mat = [axes] if not isinstance(axes, np.ndarray) else list(axes)
        
        # Enable interactive mode
        plt.ion()
    
    # Clear existing plots
    for ax in ax_mat:
        ax.clear()

    # Add data to plots source by source
    for source_idx in range(sources_np.shape[1]):
        # Plot the source
        ax_mat[source_idx].plot(sources_np[:, source_idx])

        # If silhouettes and timestamps entered then highlight "best" timestamps
        if fitness_np is not None:
            colour = "or" if source_idx == np.argmax(fitness_np) else "ok"
        else:
            colour = "ok"

        # If timestamps entered then can mark these on the sources
        if timestamps_np is not None:
            ax_mat[source_idx].plot(
                timestamps_np[source_idx],
                sources_np[timestamps_np[source_idx], source_idx],
                colour,
            )

        # Finally add additional text if exponents and/or fitness entered
        exp, sil = "", ""
        if exponents_np is not None:
            exp = "Exponent: " + str(np.round(exponents_np[source_idx], 2))
        if fitness_np is not None:
            sil = "Fitness: " + str(np.round(fitness_np[source_idx], 2))
        ax_mat[source_idx].set_title(exp + " " + sil, fontsize=22)

    plt.draw()
    plt.pause(0.001)

    if save_dir is not None:
        import time as _time
        fig.savefig(f"{save_dir}/source_{_time.time_ns()}.png", dpi=100, bbox_inches="tight")

    return fig, ax_mat
