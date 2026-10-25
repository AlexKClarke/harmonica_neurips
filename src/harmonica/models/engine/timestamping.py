"""Functions needed to convert time series into spike timestamps and
assess the quality of these timestamps"""

from typing import Tuple, Union

import torch


def silhouette_measure(
    height: torch.Tensor, assignment: torch.Tensor, centroids: torch.Tensor
):
    """Calculates a two-class pseudo-silhouette based on the distance of each
    sample to its assigned and nearest non-assigned cetroids

    input and centroids must be 1D tensors
    """

    # Get distances to centroids
    distances = (
        height.unsqueeze(1).tile([1, centroids.shape[0]]) - centroids.unsqueeze(0)
    ).abs()

    # Find nearest and second nearest centroid for each sample
    in_class_assignment = torch.nn.functional.one_hot(
        assignment, centroids.shape[0]
    ).type_as(distances)
    out_class_assignment = torch.nn.functional.one_hot(
        (distances + 1e5 * in_class_assignment).argmin(1), centroids.shape[0]
    ).type_as(distances)

    # Get distances to in class and out class centroids
    in_class_distance = (distances * in_class_assignment).sum(1)
    out_class_distance = (distances * out_class_assignment).sum(1)

    # Finally use these means to calculate the silhouette score
    maximal_value = torch.maximum(in_class_distance, out_class_distance)
    return ((out_class_distance - in_class_distance) / maximal_value).mean()


def peak_finder(
    input: torch.Tensor,
    min_peak_separation: int = 30,
    min_peak_height: float = 0.0,
) -> torch.Tensor:
    """Simple max pool based peak finder for a 1d tensor input"""

    arg_max_indices = torch.nn.functional.max_pool1d(
        input.view(1, 1, -1),
        min_peak_separation * 2,
        stride=1,
        padding=int(min_peak_separation),
        return_indices=True,
    )[1].squeeze()

    unique = arg_max_indices.unique()
    locations = unique[arg_max_indices[unique] == unique]

    return locations[input[locations] > min_peak_height]


def soft_k_means(
    input: torch.Tensor,
    k: int = 2,
    beta: float = 1.0,
    tol: float = 0.1,
    max_iterations: int = 10,
) -> torch.Tensor:
    """A soft allocation k means applied to a 1D input"""

    tiled_input = input.unsqueeze(1).tile([1, k])
    topk = min(5, input.shape[0])
    centroids = torch.linspace(
        float(input.topk(topk, largest=False)[0].mean().detach().cpu()),
        float(input.topk(topk)[0].mean().detach().cpu()),
        k,
    ).type_as(input)

    for _ in range(max_iterations):
        prev_centroids = centroids

        distances = torch.abs(tiled_input - centroids.unsqueeze(0))
        exp_mse = distances.square().multiply(-beta).exp()
        weighting = exp_mse.divide(exp_mse.sum(1, keepdim=True) + 1e-10)
        centroids = weighting.multiply(tiled_input).sum(0) / (weighting.sum(0))

        if (prev_centroids - centroids).abs().sum() < tol:
            break

    centroid_order = centroids.argsort()
    return centroids[centroid_order]


def signal_to_timestamps(
    source: torch.Tensor,
    calculate_silhouette: bool = False,
    min_peak_separation: int = 30,
    min_peak_height: float = 3,
    min_peaks: int = 1,
) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
    """
    Two class k means/median peak selection that returns the cluster with
    the highest peaks and the silhouette score

    Source must be a 1D torch tensor
    """

    # Find all peaks in source
    assert source.isfinite().all(), "Source contains NaNs/Infs"

    # Sample peaks from the source and get their heights
    all_peaks = peak_finder(source, min_peak_separation // 2, min_peak_height=-1)
    peak_heights = source[all_peaks]
    peak_heights = peak_heights.subtract(peak_heights.mean()).divide(
        peak_heights.std()
    )

    # Fit a soft k
    centroids = soft_k_means(peak_heights)

    # Assign spikes
    distances = torch.abs(
        peak_heights.unsqueeze(1).tile([1, centroids.shape[0]])
        - centroids.unsqueeze(0)
    )
    spikes = all_peaks[distances.argmin(1) == centroids.argmax()]

    # Next get set of peaks that meet requirements
    threshold_peaks = peak_finder(source, min_peak_separation, min_peak_height)

    # Finally return selected peaks that are in requirements
    threshold_mask = torch.isin(spikes, threshold_peaks)
    timestamps = spikes[threshold_mask]

    if calculate_silhouette:
        is_spike = torch.isin(all_peaks, timestamps)
        spike_heights, null_heights = (
            peak_heights[is_spike],
            peak_heights[~is_spike],
        )
        match_num = min(spike_heights.shape[0], null_heights.shape[0])
        spike_subselect = spike_heights.sort()[0][:match_num]
        null_subselect = null_heights.sort(descending=True)[0][:match_num]
        combined_heights = torch.concat([spike_subselect, null_subselect])
        combined_labels = torch.concat(
            [
                centroids.argmax()
                * torch.ones(match_num, device=centroids.device, dtype=torch.int64),
                centroids.argmin()
                * torch.ones(match_num, device=centroids.device, dtype=torch.int64),
            ]
        )
        silhouette = silhouette_measure(combined_heights, combined_labels, centroids)
        
        if timestamps.shape[0] < min_peaks:
            silhouette = torch.zeros_like(silhouette)
        return timestamps, silhouette
    else:
        return timestamps


def timestamps_to_windows(
    signal: torch.Tensor, timestamps: torch.Tensor, window: int = 1
) -> torch.Tensor:
    """Given timestamps, calculate a spike triggered average of
    shape [window, channels] on signal of shape [time, channels]"""

    assert len(signal.shape) == 2, "signal must be of shape [time, channels]"

    hw = int(window / 2)
    odd = int(window % 2)

    signal_windows = []
    for s in timestamps:
        start, end = int(s - hw), int(s + hw + odd)
        if start < 0:
            pad = torch.zeros(
                [-start] + list(signal.shape[1:]),
                dtype=signal.dtype,
                device=signal.device,
            )
            w = torch.concat([pad, signal[0:end]])
        elif end > signal.shape[0]:
            pad = torch.zeros(
                [end - signal.shape[0]] + list(signal.shape[1:]),
                dtype=signal.dtype,
                device=signal.device,
            )
            w = torch.concat([signal[start : signal.shape[0]], pad])
        else:
            w = signal[start:end]

        signal_windows.append(w)
    return torch.stack(signal_windows)


def find_quality_metric(
    timestamps_1: torch.Tensor,
    timestamps_2: torch.Tensor,
    mode: str = "accuracy",
    tolerance: int = 5,
    max_shift: int = 100,
) -> torch.Tensor:
    """Matrix operation based implementation of quality metric finding
    between two timestamp sets. If using precision or recall,
    timestamps_1 is the prediction and timestamps_2 the ground truth"""

    assert tolerance >= 0, "tolerance must be non-negative"
    assert max_shift >= 0, "max_shift must be non-negative"

    # Make sure timestamps on same device
    timestamps_2 = timestamps_2.type_as(timestamps_1)

    # Set tolerance aranges
    tol_range = torch.arange(-tolerance, tolerance).type_as(timestamps_1)

    # Create shifted versions of timestamp_1 with tolerance bands
    expand_1 = timestamps_1.unsqueeze(0)
    expand_1 = expand_1 - tol_range.unsqueeze(1)
    expand_1 = expand_1.unsqueeze(-1).tile([1, 1, timestamps_2.shape[0]])

    # Tile up timestamp_2
    expand_2 = (
        timestamps_2.unsqueeze(0)
        .unsqueeze(0)
        .tile([expand_1.shape[0], expand_1.shape[1], 1])
    )

    # Iterate through shift amounts
    matches = []
    for shift in range(-max_shift, max_shift):
        matches.append((expand_1 + shift == expand_2).any(2).any(0).sum())
    matches = torch.stack(matches).max()

    if mode == "accuracy":
        return 100 * matches.divide(
            timestamps_1.shape[0] + timestamps_2.shape[0] - matches
        )
    elif mode == "precision":
        return 100 * matches.divide(timestamps_1.shape[0])
    elif mode == "recall":
        return 100 * matches.divide(timestamps_2.shape[0])
    else:
        raise ValueError("Specified quality metric not available.")
