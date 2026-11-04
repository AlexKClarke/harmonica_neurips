"""Time-contrastive learning (TCL), an identifiable nonlinear-ICA baseline.

TCL (Hyvarinen & Morioka, NeurIPS 2016) in its standard pipeline:

1. Train an MLP classifier to predict which segment (an equal-time segment,
   or a joint-angle bin) each whitened frame came from.
2. Take the penultimate-layer activations as the learned nonlinear features.
3. Whiten the features and run linear ICA on them.

Whitening and ICA come from :mod:`harmonica.ica`, the same front end as the
rest of the code. Used by :func:`harmonica.baselines.run_tcl`.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from ..ica.algorithm import run_ica_loop
from ..ica.whitening import whitening

_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ---------------------------------------------------------------------------
# Segment labels
# ---------------------------------------------------------------------------


def segment_signal(n_samples: int, n_segments: int) -> np.ndarray:
    """Assign each sample to one of ``n_segments`` contiguous equal-time segments.

    Args:
        n_samples: Number of samples in the signal.
        n_segments: Number of segments; the last one takes the remainder.

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

    The "TCL-θ" auxiliary variable: samples are grouped by angle bin rather
    than by time. Quantile edges give equal-count bins, which avoids the class
    imbalance that equal-width bins create on a sinusoidal angle profile.

    Args:
        angle: ``(n_samples,)`` continuous angle trajectory.
        n_segments: Target number of angle bins (fewer if the angle takes
            very few distinct values).

    Returns:
        ``(n_samples,)`` integer labels in a contiguous ``[0, K)`` range,
        ``K <= n_segments``.
    """
    angle = np.asarray(angle, dtype=float)
    edges = np.quantile(angle, np.linspace(0.0, 1.0, n_segments + 1))
    raw = np.clip(np.digitize(angle, edges[1:-1]), 0, n_segments - 1)
    uniq = np.unique(raw)
    remap = {u: i for i, u in enumerate(uniq)}
    return np.array([remap[v] for v in raw], dtype=int)


# ---------------------------------------------------------------------------
# Segment classifier
# ---------------------------------------------------------------------------


class DeepTCL:
    """MLP segment classifier whose penultimate layer gives the TCL features.

    The network is ``input -> [Linear-BatchNorm-ReLU-Dropout] x L -> Linear``,
    trained with AdamW and a cosine learning-rate schedule. ``torch`` is
    seeded with ``seed`` when the network is built.
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

        With ``verbose=True`` the loss and training accuracy are printed every
        50 epochs. The accuracy check puts the network in eval mode and
        training does not switch it back, so from epoch 51 on BatchNorm uses
        its running statistics and dropout is off. A verbose fit therefore
        trains a different network from a quiet one.

        Args:
            X: ``(n_features, n_samples)`` input features.
            labels: ``(n_samples,)`` integer segment labels.

        Returns:
            ``self``.
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
        # drop_last: a size-1 final batch would crash BatchNorm1d.
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
        """Penultimate-layer features, ``(hidden_dims[-1], n_samples)``."""
        assert self.model is not None
        self.model.eval()
        X_t = torch.tensor(X.T, dtype=torch.float32, device=self.device)
        with torch.no_grad():
            features = X_t
            for layer in list(self.model.children())[:-1]:
                features = layer(features)
        return features.cpu().numpy().T

    def classification_accuracy(self, X: np.ndarray, labels: np.ndarray) -> float:
        """Fraction of samples assigned to their own segment.

        Leaves the network in eval mode.
        """
        assert self.model is not None
        self.model.eval()
        X_t = torch.tensor(X.T, dtype=torch.float32, device=self.device)
        with torch.no_grad():
            logits = self.model(X_t)
            preds = logits.argmax(dim=1).cpu().numpy()
        return float(np.mean(preds == labels))


# ---------------------------------------------------------------------------
# TCL decomposition
# ---------------------------------------------------------------------------


def deep_tcl_decompose(
    white_sig: np.ndarray,
    fs: float,
    n_segments: int = 16,
    hidden_dims: list[int] | None = None,
    n_epochs: int = 200,
    labels: np.ndarray | None = None,
    sil_th: float = 0.6,
    ica_seed: int = 1909,
    ica_n_iter: int = 100,
    verbose: bool = False,
) -> tuple[np.ndarray, float]:
    """TCL: classify segments, take the features, whiten them, run ICA.

    Args:
        white_sig: ``(n_features, n_samples)`` whitened extended EMG.
        fs: Sampling frequency, in Hz.
        n_segments: Number of equal-time segments when ``labels`` is None.
        hidden_dims: MLP hidden sizes (default ``[256, 128]``).
        n_epochs: Classifier training epochs.
        labels: Optional per-sample segment labels (angle bins for TCL-θ);
            equal-time :func:`segment_signal` labels when None.
        sil_th: SIL acceptance threshold of the ICA on the features. The
            permissive 0.6 keeps a large candidate set.
        ica_seed: ICA RNG seed.
        ica_n_iter: Maximum number of ICA sources extracted from the features.
        verbose: Print training progress (this also changes the training;
            see :meth:`DeepTCL.fit`).

    Returns:
        ``(sources, accuracy)``: the ``(n_sources, n_samples)`` ICA sources
        accepted at ``sil_th``, and the classifier's training accuracy.
    """
    hidden_dims = hidden_dims or [256, 128]
    X = white_sig

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
    if verbose:
        print(
            f"  [TCL] n_seg={n_segments} acc={acc:.3f} feat_dim={features.shape[0]}",
            flush=True,
        )

    # Whiten the learned features, then run linear ICA on them.
    features = features - np.mean(features, axis=1, keepdims=True)
    feat_white, _ = whitening(features, method="ZCA")

    sources, _spikes, _sil, _filters = run_ica_loop(
        feat_white, fs, n_iter=ica_n_iter, sil_th=sil_th, seed=ica_seed,
    )
    return sources, acc
