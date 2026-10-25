"""Time-Contrastive Learning (TCL) — nonlinear-ICA baseline.

``DeepTCL``, ``segment_signal``, energy features, and
``deep_tcl_decompose`` (the textbook pipeline).

TCL (Hyvarinen & Morioka, NeurIPS 2016) is one of the two canonical
*identifiable* nonlinear-ICA methods. The pipeline is:

1. Train an MLP classifier to predict which time-segment (or angle bin)
   each whitened frame came from.
2. Extract the penultimate-layer features (the learned nonlinear
   representation).
3. Whiten those features and run linear ICA on them.
4. Detect spikes on the resulting ICA sources.

The whitening / ICA front-ends come from ``harmonica.ica``, so they are
identical to the rest of the paper's code path.

Nothing here is on the main refinement path — it backs the
nonlinear-ICA comparison in the supplementary material.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from scipy.ndimage import uniform_filter1d
from torch.utils.data import DataLoader, TensorDataset

from ..ica.algorithm import run_ica_loop
from ..ica.whitening import whitening

_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ---------------------------------------------------------------------------
# Segment labeling
# ---------------------------------------------------------------------------


def segment_signal(n_samples: int, n_segments: int, mode: str = "equal") -> np.ndarray:
    """Assign each sample to an equal-length time-segment label.

    Args:
        n_samples: Number of samples in the signal.
        n_segments: Number of contiguous segments to split into.
        mode: Kept for API parity; only ``'equal'`` is implemented.

    Returns:
        ``(n_samples,)`` integer segment labels in ``[0, n_segments)``.
    """
    labels = np.zeros(n_samples, dtype=int)
    seg_len = n_samples // n_segments
    for i in range(n_segments):
        start = i * seg_len
        end = n_samples if i == n_segments - 1 else (i + 1) * seg_len
        labels[start:end] = i
    return labels


def angle_segment_labels(angle: np.ndarray, n_segments: int = 24) -> np.ndarray:
    """Quantile-bin a continuous joint-angle trajectory into segment labels.

    This is the fair "TCL-θ" auxiliary variable: instead of contiguous
    equal-time segments, samples are grouped by which angle bin they fall
    in. Quantile edges keep the bins balanced (equal sample counts), which
    avoids the class imbalance that equal-width bins would create on a
    sinusoidal angle profile.

    Args:
        angle: ``(n_samples,)`` continuous angle trajectory.
        n_segments: Target number of angle bins (labels may be fewer if
            the angle takes very few distinct values).

    Returns:
        ``(n_samples,)`` integer labels remapped to a contiguous
        ``[0, K)`` range with ``K <= n_segments``.
    """
    angle = np.asarray(angle, dtype=float)
    edges = np.quantile(angle, np.linspace(0.0, 1.0, n_segments + 1))
    # Digitize on interior edges; clip to valid bin range.
    raw = np.clip(np.digitize(angle, edges[1:-1]), 0, n_segments - 1)
    uniq = np.unique(raw)
    remap = {u: i for i, u in enumerate(uniq)}
    return np.array([remap[v] for v in raw], dtype=int)


# ---------------------------------------------------------------------------
# Energy features (used by input_mode='energy' / 'both')
# ---------------------------------------------------------------------------


def compute_energy(sig: np.ndarray, ws: int = 256) -> np.ndarray:
    """Sliding-window mean-squared energy per channel."""
    return uniform_filter1d(sig ** 2, size=ws, axis=1, mode="nearest")


def compute_multi_scale_energy(
    sig: np.ndarray,
    window_sizes: tuple[int, ...] = (64, 256, 1024),
    normalize: bool = True,
) -> np.ndarray:
    """Multi-scale energy features: ``(n_channels * n_scales, n_samples)``."""
    energies = []
    for ws in window_sizes:
        e = compute_energy(sig, ws)
        if normalize:
            em = e.mean(axis=1, keepdims=True)
            es = e.std(axis=1, keepdims=True) + 1e-10
            e = (e - em) / es
        energies.append(e)
    return np.vstack(energies)


# ---------------------------------------------------------------------------
# PyTorch GPU-accelerated neural-network TCL classifier
# ---------------------------------------------------------------------------


class DeepTCL:
    """GPU-accelerated MLP time-contrastive classifier.

    Trains ``input -> [Linear-BatchNorm-ReLU-Dropout] x L -> Linear`` to
    predict segment labels, then exposes the penultimate-layer activations
    as learned nonlinear features (``get_features``).
    """

    def __init__(
        self,
        n_segments: int = 16,
        hidden_dims: list[int] | None = None,
        learning_rate: float = 1e-3,
        n_epochs: int = 200,
        batch_size: int = 2048,
        weight_decay: float = 1e-4,
        seed: int = 42,
        device: str | None = None,
        verbose: bool = False,
    ) -> None:
        self.n_segments = n_segments
        self.hidden_dims = hidden_dims or [256, 128]
        self.lr = learning_rate
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.weight_decay = weight_decay
        self.seed = seed
        self.device = device or _DEVICE
        self.verbose = verbose
        self.model: nn.Sequential | None = None
        self.loss_history: list[float] = []

    def _build_model(self, n_features: int) -> nn.Sequential:
        torch.manual_seed(self.seed)
        layers: list[nn.Module] = []
        in_dim = n_features
        for h_dim in self.hidden_dims:
            layers.extend([
                nn.Linear(in_dim, h_dim),
                nn.BatchNorm1d(h_dim),
                nn.ReLU(),
                nn.Dropout(0.1),
            ])
            in_dim = h_dim
        layers.append(nn.Linear(in_dim, self.n_segments))
        return nn.Sequential(*layers)

    def fit(self, X: np.ndarray, labels: np.ndarray) -> "DeepTCL":
        """Train the classifier.

        Args:
            X: ``(n_features, n_samples)`` input features.
            labels: ``(n_samples,)`` integer segment labels.
        """
        n_features, n_samples = X.shape
        self.model = self._build_model(n_features).to(self.device)

        optimizer = optim.AdamW(
            self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self.n_epochs, eta_min=self.lr * 0.01
        )
        criterion = nn.CrossEntropyLoss()

        X_t = torch.tensor(X.T, dtype=torch.float32, device=self.device)
        y_t = torch.tensor(labels, dtype=torch.long, device=self.device)
        dataset = TensorDataset(X_t, y_t)
        # drop_last avoids a size-1 final batch crashing BatchNorm1d for
        # sample counts not divisible by batch_size (original exp31 relied
        # on ch070's 20480 being an exact multiple of 2048).
        loader = DataLoader(
            dataset, batch_size=self.batch_size, shuffle=True, drop_last=True
        )

        self.loss_history = []
        self.model.train()
        for epoch in range(self.n_epochs):
            epoch_loss = 0.0
            n_batches = 0
            for X_batch, y_batch in loader:
                optimizer.zero_grad()
                logits = self.model(X_batch)
                loss = criterion(logits, y_batch)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
                n_batches += 1
            scheduler.step()
            avg_loss = epoch_loss / max(n_batches, 1)
            self.loss_history.append(avg_loss)
            if self.verbose and (epoch + 1) % 50 == 0:
                acc = self.classification_accuracy(X, labels)
                print(
                    f"    [TCL] epoch {epoch + 1}/{self.n_epochs}: "
                    f"loss={avg_loss:.4f} acc={acc:.3f}",
                    flush=True,
                )
        return self

    def get_features(self, X: np.ndarray) -> np.ndarray:
        """Extract penultimate-layer features, ``(hidden_dims[-1], n_samples)``."""
        assert self.model is not None
        self.model.eval()
        X_t = torch.tensor(X.T, dtype=torch.float32, device=self.device)
        with torch.no_grad():
            features = X_t
            for layer in list(self.model.children())[:-1]:
                features = layer(features)
        return features.cpu().numpy().T

    def classification_accuracy(self, X: np.ndarray, labels: np.ndarray) -> float:
        assert self.model is not None
        self.model.eval()
        X_t = torch.tensor(X.T, dtype=torch.float32, device=self.device)
        with torch.no_grad():
            logits = self.model(X_t)
            preds = logits.argmax(dim=1).cpu().numpy()
        return float(np.mean(preds == labels))


# ---------------------------------------------------------------------------
# Proper TCL decomposition (exp31 pipeline, harmonica front-ends)
# ---------------------------------------------------------------------------


def deep_tcl_decompose(
    white_sig: np.ndarray,
    fs: float,
    n_segments: int = 16,
    input_mode: str = "raw",
    hidden_dims: list[int] | None = None,
    n_epochs: int = 200,
    labels: np.ndarray | None = None,
    sil_th: float = 0.6,
    ica_seed: int = 1909,
    ica_n_iter: int = 100,
    verbose: bool = False,
) -> tuple[np.ndarray, dict[int, np.ndarray], np.ndarray, np.ndarray, float, int]:
    """Textbook proper TCL: classify segments -> features -> whiten -> ICA.

    Faithful adaptation of ``exp31.deep_tcl_decompose`` with the whitening
    and ICA calls routed through ``harmonica.ica`` so the
    front-ends match the rest of the code path.

    Args:
        white_sig: ``(n_features, n_samples)`` whitened extended EMG.
        fs: Sampling frequency, in Hz.
        n_segments: Number of time segments when ``labels`` is None.
        input_mode: ``'raw'`` (whitened signal, learns energy via ReLU —
            the exp31 default), ``'energy'`` (multi-scale energy), or
            ``'both'``.
        hidden_dims: MLP hidden sizes (default ``[256, 128]``).
        n_epochs: Classifier training epochs.
        labels: Optional precomputed per-sample segment labels. Pass
            angle-binned labels here for the fair "TCL-θ" variant; falls
            back to equal-time :func:`segment_signal` when None.
        sil_th: SIL acceptance threshold for the ICA-on-features stage.
            Deliberately permissive (0.6) so candidate sources are not
            over-filtered — we want a trustworthy count of what TCL can and
            cannot recover.
        ica_seed: ICA RNG seed.
        ica_n_iter: Max ICA sources to extract from the features.
        verbose: Print classifier training progress.

    Returns:
        ``(sources, spikes, sil, mu_filters, acc, feat_dim)``.
    """
    hidden_dims = hidden_dims or [256, 128]

    if input_mode == "raw":
        X = white_sig
    elif input_mode == "energy":
        X = compute_multi_scale_energy(white_sig)
    elif input_mode == "both":
        X = np.vstack([white_sig, compute_multi_scale_energy(white_sig)])
    else:
        raise ValueError(f"Unknown input_mode: {input_mode!r}")

    if labels is None:
        labels = segment_signal(X.shape[1], n_segments)
    else:
        labels = np.asarray(labels, dtype=int)
        n_segments = int(labels.max()) + 1

    model = DeepTCL(
        n_segments=n_segments,
        hidden_dims=hidden_dims,
        learning_rate=1e-3,
        n_epochs=n_epochs,
        batch_size=2048,
        seed=42,
        verbose=verbose,
    )
    model.fit(X, labels)
    acc = model.classification_accuracy(X, labels)

    features = model.get_features(X)  # (hidden_dims[-1], T)
    feat_dim = features.shape[0]
    if verbose:
        print(
            f"  [TCL] input_mode={input_mode} n_seg={n_segments} "
            f"acc={acc:.3f} feat_dim={feat_dim}",
            flush=True,
        )

    # Whiten the learned features, then run linear ICA on them.
    features = features - np.mean(features, axis=1, keepdims=True)
    feat_white, _ = whitening(features, method="ZCA")

    sources, spikes, sil, mu_filters = run_ica_loop(
        feat_white, fs, n_iter=ica_n_iter, sil_th=sil_th, seed=ica_seed,
    )
    return sources, spikes, sil, mu_filters, acc, feat_dim
