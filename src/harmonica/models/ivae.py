"""Identifiable VAE (iVAE) — nonlinear-ICA baseline.

After Khemakhem et al., AISTATS 2020. ``est_spike_times``,
``remove_duplicates`` and ``remove_bad_sources`` come from
``harmonica.ica`` so the detection / cleanup primitives match the
rest of the paper's code path.

iVAE is the second of the two canonical *identifiable* nonlinear-ICA
methods. Conditioning the latent prior ``p(z|u)`` on an observed auxiliary
variable ``u`` (here the joint angle) makes the recovered sources
identifiable up to permutation and component-wise transform. The encoder
``q(z|x,u)`` plays the role of the demixing function; each latent channel
is spike-detected after training.

Nothing here is on the main refinement path — it backs the
nonlinear-ICA comparison in the supplementary material.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from ..ica.detect import est_spike_times
from ..ica.utils import remove_bad_sources, remove_duplicates

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------


class iVAE(nn.Module):
    """Identifiable Variational Autoencoder for source separation.

    Encoder ``q(z|x,u)`` : MLP on ``concat(x, onehot(u))`` -> (mean, logvar).
    Decoder ``p(x|z)``    : linear (default) or shallow MLP.
    Prior   ``p(z|u)``    : ``N(0, exp(lambda(u)))`` with ``lambda = MLP(u)``.
    """

    def __init__(
        self,
        data_dim: int,
        latent_dim: int,
        n_aux: int,
        encoder_hidden: tuple[int, ...] = (256, 128),
        prior_hidden: tuple[int, ...] = (64,),
        decoder_type: str = "linear",
        decoder_hidden: tuple[int, ...] = (128,),
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.data_dim = data_dim
        self.latent_dim = latent_dim
        self.n_aux = n_aux

        enc_layers: list[nn.Module] = []
        in_dim = data_dim + n_aux
        for h in encoder_hidden:
            enc_layers.extend([
                nn.Linear(in_dim, h),
                nn.BatchNorm1d(h),
                nn.LeakyReLU(0.1),
                nn.Dropout(dropout),
            ])
            in_dim = h
        self.encoder_body = nn.Sequential(*enc_layers)
        self.enc_mean = nn.Linear(in_dim, latent_dim)
        self.enc_logvar = nn.Linear(in_dim, latent_dim)

        prior_layers: list[nn.Module] = []
        p_in = n_aux
        for h in prior_hidden:
            prior_layers.extend([nn.Linear(p_in, h), nn.LeakyReLU(0.1)])
            p_in = h
        prior_layers.append(nn.Linear(p_in, latent_dim))
        self.prior_logvar = nn.Sequential(*prior_layers)

        if decoder_type == "linear":
            self.decoder: nn.Module = nn.Linear(latent_dim, data_dim)
        else:
            dec_layers: list[nn.Module] = []
            d_in = latent_dim
            for h in decoder_hidden:
                dec_layers.extend([
                    nn.Linear(d_in, h),
                    nn.LeakyReLU(0.1),
                    nn.Dropout(dropout),
                ])
                d_in = h
            dec_layers.append(nn.Linear(d_in, data_dim))
            self.decoder = nn.Sequential(*dec_layers)

        self.decoder_logvar = nn.Parameter(torch.zeros(1))

    def encode(self, x: torch.Tensor, u_onehot: torch.Tensor):
        h = torch.cat([x, u_onehot], dim=1)
        h = self.encoder_body(h)
        return self.enc_mean(h), self.enc_logvar(h)

    def reparameterize(self, mean: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mean + std * eps

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)

    def prior_params(self, u_onehot: torch.Tensor):
        logvar = self.prior_logvar(u_onehot)
        mean = torch.zeros_like(logvar)
        return mean, logvar

    def forward(self, x: torch.Tensor, u_onehot: torch.Tensor):
        enc_mean, enc_logvar = self.encode(x, u_onehot)
        z = self.reparameterize(enc_mean, enc_logvar)
        x_recon = self.decode(z)
        prior_mean, prior_logvar = self.prior_params(u_onehot)
        return x_recon, z, enc_mean, enc_logvar, prior_mean, prior_logvar

    def get_sources(self, x: torch.Tensor, u_onehot: torch.Tensor) -> torch.Tensor:
        self.eval()
        with torch.no_grad():
            mean, _ = self.encode(x, u_onehot)
        return mean


def ivae_loss(
    x_recon, x, enc_mean, enc_logvar, prior_mean, prior_logvar,
    decoder_logvar, beta: float = 1.0, sparsity_weight: float = 0.0,
):
    """iVAE ELBO with optional super-Gaussianity (kurtosis) regulariser."""
    recon_var = torch.exp(decoder_logvar)
    recon_loss = 0.5 * (decoder_logvar + (x - x_recon) ** 2 / recon_var).sum(dim=1)

    kl = 0.5 * (
        prior_logvar - enc_logvar - 1
        + torch.exp(enc_logvar - prior_logvar)
        + (enc_mean - prior_mean) ** 2 / torch.exp(prior_logvar)
    ).sum(dim=1)

    loss = (recon_loss + beta * kl).mean()

    if sparsity_weight > 0:
        z_centered = enc_mean - enc_mean.mean(dim=0, keepdim=True)
        var = (z_centered ** 2).mean(dim=0)
        kurt = ((z_centered ** 4).mean(dim=0) / (var ** 2 + 1e-8)) - 3.0
        sparsity_loss = -kurt.mean()
        loss = loss + sparsity_weight * sparsity_loss

    return loss, recon_loss.mean().item(), kl.mean().item()


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------


def train_ivae(
    model: iVAE,
    X: np.ndarray,
    u_labels: np.ndarray,
    n_epochs: int = 200,
    lr: float = 1e-3,
    batch_size: int = 512,
    beta_start: float = 0.0,
    beta_end: float = 1.0,
    beta_warmup_frac: float = 0.5,
    sparsity_weight: float = 0.0,
    device: str | None = None,
    verbose: bool = True,
):
    """Train iVAE with KL annealing.

    Args:
        model: :class:`iVAE` instance.
        X: ``(n_features, n_samples)`` whitened signal.
        u_labels: ``(n_samples,)`` integer auxiliary-variable labels.
        n_epochs: Training epochs.
        lr: Learning rate.
        batch_size: Minibatch size.
        beta_start / beta_end / beta_warmup_frac: KL-annealing schedule.
        sparsity_weight: Weight on the kurtosis term (0 = off).
        device: Torch device.
        verbose: Print progress every 50 epochs.

    Returns:
        ``(model, history)``.
    """
    if device is None:
        device = next(model.parameters()).device

    n_features, n_samples = X.shape
    n_aux = model.n_aux

    X_t = torch.tensor(X.T, dtype=torch.float32)
    u_onehot = torch.zeros(n_samples, n_aux, dtype=torch.float32)
    u_onehot[torch.arange(n_samples), u_labels] = 1.0

    dataset = TensorDataset(X_t, u_onehot)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)

    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs)

    warmup_epochs = int(n_epochs * beta_warmup_frac)
    history: dict[str, list[float]] = {"loss": [], "recon": [], "kl": [], "beta": []}

    for epoch in range(n_epochs):
        model.train()
        if epoch < warmup_epochs:
            beta = beta_start + (beta_end - beta_start) * epoch / max(warmup_epochs, 1)
        else:
            beta = beta_end

        epoch_loss = epoch_recon = epoch_kl = 0.0
        n_batches = 0
        for X_batch, u_batch in loader:
            X_batch = X_batch.to(device)
            u_batch = u_batch.to(device)

            x_recon, z, enc_mean, enc_logvar, prior_mean, prior_logvar = model(
                X_batch, u_batch
            )
            loss, recon, kl = ivae_loss(
                x_recon, X_batch, enc_mean, enc_logvar,
                prior_mean, prior_logvar, model.decoder_logvar,
                beta=beta, sparsity_weight=sparsity_weight,
            )
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()

            epoch_loss += loss.item()
            epoch_recon += recon
            epoch_kl += kl
            n_batches += 1
        scheduler.step()

        history["loss"].append(epoch_loss / max(n_batches, 1))
        history["recon"].append(epoch_recon / max(n_batches, 1))
        history["kl"].append(epoch_kl / max(n_batches, 1))
        history["beta"].append(beta)

        if verbose and (epoch + 1) % 50 == 0:
            print(
                f"    [iVAE] epoch {epoch + 1}/{n_epochs}: "
                f"loss={history['loss'][-1]:.4f} recon={history['recon'][-1]:.4f} "
                f"kl={history['kl'][-1]:.4f} beta={beta:.3f}",
                flush=True,
            )

    return model, history


# ---------------------------------------------------------------------------
# Source extraction and spike detection
# ---------------------------------------------------------------------------


def extract_sources(
    model: iVAE, X: np.ndarray, u_labels: np.ndarray, device: str | None = None
) -> np.ndarray:
    """Extract encoder-mean source signals, ``(latent_dim, n_samples)``."""
    if device is None:
        device = next(model.parameters()).device

    model.eval()
    n_features, n_samples = X.shape
    n_aux = model.n_aux

    X_t = torch.tensor(X.T, dtype=torch.float32, device=device)
    u_onehot = torch.zeros(n_samples, n_aux, dtype=torch.float32, device=device)
    u_onehot[
        torch.arange(n_samples), torch.tensor(u_labels, device=device)
    ] = 1.0

    with torch.no_grad():
        chunk_size = 8192
        means = []
        for i in range(0, n_samples, chunk_size):
            end = min(i + chunk_size, n_samples)
            m, _ = model.encode(X_t[i:end], u_onehot[i:end])
            means.append(m.cpu())
        sources = torch.cat(means, dim=0).numpy().T
    return sources


def ivae_decompose(
    X: np.ndarray,
    fs: float,
    u_labels: np.ndarray,
    n_sources: int = 5,
    n_epochs: int = 200,
    lr: float = 1e-3,
    encoder_hidden: tuple[int, ...] = (256, 128),
    prior_hidden: tuple[int, ...] = (64,),
    decoder_type: str = "linear",
    decoder_hidden: tuple[int, ...] = (128,),
    sil_th: float = 0.9,
    batch_size: int = 512,
    sparsity_weight: float = 0.0,
    device: str | None = None,
    verbose: bool = True,
):
    """Full iVAE pipeline: train -> extract -> detect -> dedup/filter.

    Args:
        X: ``(n_features, n_samples)`` whitened extended signal.
        fs: Sampling frequency, in Hz.
        u_labels: ``(n_samples,)`` auxiliary-variable labels (angle bins).
        n_sources: Latent dimensionality (set ~ number of GT MUs).
        n_epochs: Training epochs.
        lr: Learning rate.
        encoder_hidden / prior_hidden / decoder_type / decoder_hidden:
            Architecture knobs.
        sil_th: SIL acceptance threshold for :func:`remove_bad_sources`.
        batch_size: Minibatch size.
        sparsity_weight: Kurtosis regulariser weight.
        device: Torch device.
        verbose: Print progress.

    Returns:
        ``(sources_clean, spikes_clean, sil_clean, model, history)``.
    """
    if device is None:
        device = DEVICE

    n_features, n_samples = X.shape
    n_aux = int(u_labels.max()) + 1
    if verbose:
        print(
            f"  [iVAE] {n_features}d -> {n_sources} sources, "
            f"{n_aux} aux categories, {n_epochs} epochs",
            flush=True,
        )

    model = iVAE(
        data_dim=n_features, latent_dim=n_sources, n_aux=n_aux,
        encoder_hidden=encoder_hidden, prior_hidden=prior_hidden,
        decoder_type=decoder_type, decoder_hidden=decoder_hidden,
    ).to(device)

    model, history = train_ivae(
        model, X, u_labels, n_epochs=n_epochs, lr=lr,
        batch_size=batch_size, sparsity_weight=sparsity_weight,
        device=device, verbose=verbose,
    )

    sources = extract_sources(model, X, u_labels, device=device)

    all_spikes: dict[int, np.ndarray] = {}
    all_sil = np.zeros(n_sources)
    for i in range(n_sources):
        try:
            sp, sil_val = est_spike_times(sources[i], fs)
            all_spikes[i] = sp
            all_sil[i] = sil_val
        except Exception:
            all_spikes[i] = np.array([], dtype=int)
            all_sil[i] = 0.0

    mu_filters = np.eye(n_features, n_sources)

    sources_clean, spikes_clean, sil_clean, _ = remove_duplicates(
        sources, all_spikes, all_sil, mu_filters, fs,
        max_shift=0.1, tol=0.001, threshold=0.3,
    )
    sources_clean, spikes_clean, sil_clean, _ = remove_bad_sources(
        sources_clean, spikes_clean, sil_clean,
        mu_filters[:, : sources_clean.shape[0]],
        threshold=sil_th, min_num_spikes=10,
    )

    if verbose:
        n_found = sources_clean.shape[0]
        print(f"  [iVAE] accepted {n_found} sources (sil_th={sil_th})", flush=True)

    return sources_clean, spikes_clean, sil_clean, model, history


# ---------------------------------------------------------------------------
# Auxiliary-variable label builders
# ---------------------------------------------------------------------------


def make_segment_labels(n_samples: int, n_segments: int = 32) -> np.ndarray:
    """Equal-length time-segment labels."""
    labels = np.zeros(n_samples, dtype=int)
    seg_len = n_samples // n_segments
    for i in range(n_segments):
        start = i * seg_len
        end = n_samples if i == n_segments - 1 else (i + 1) * seg_len
        labels[start:end] = i
    return labels


def bin_angle(angle: np.ndarray, n_bins: int = 24) -> np.ndarray:
    """Quantile-bin a continuous angle trajectory into integer pose indices.

    Equal-count (quantile) bins keep the auxiliary categories balanced on a
    sinusoidal angle profile. Labels are remapped to a contiguous
    ``[0, K)`` range with ``K <= n_bins``.
    """
    angle = np.asarray(angle, dtype=float)
    edges = np.quantile(angle, np.linspace(0.0, 1.0, n_bins + 1))
    raw = np.clip(np.digitize(angle, edges[1:-1]), 0, n_bins - 1)
    uniq = np.unique(raw)
    remap = {u: i for i, u in enumerate(uniq)}
    return np.array([remap[v] for v in raw], dtype=int)


def make_pose_labels(angle_profile: np.ndarray, n_whitened: int) -> np.ndarray:
    """Align integer pose indices to the whitened-signal length.

    Center-crops when the whitened signal is shorter than the raw angle
    trajectory (the convolutive-extension edge trim). Expects
    ``angle_profile`` to already be integer pose indices — bin a continuous
    angle with :func:`bin_angle` first.
    """
    n_raw = len(angle_profile)
    if n_whitened < n_raw:
        offset = (n_raw - n_whitened) // 2
        return angle_profile[offset : offset + n_whitened].astype(int)
    return angle_profile[:n_whitened].astype(int)


def make_combined_labels(
    angle_profile: np.ndarray, n_whitened: int, n_time_segments: int = 8
) -> tuple[np.ndarray, int]:
    """Combine pose and within-pose time segments into a richer aux variable."""
    pose_labels = make_pose_labels(angle_profile, n_whitened)
    n_poses = int(pose_labels.max()) + 1

    combined = pose_labels * n_time_segments
    for p in range(n_poses):
        mask = pose_labels == p
        if mask.sum() == 0:
            continue
        indices = np.where(mask)[0]
        n_p = len(indices)
        seg_len = n_p // n_time_segments
        for s in range(n_time_segments):
            start = s * seg_len
            end = n_p if s == n_time_segments - 1 else (s + 1) * seg_len
            combined[indices[start:end]] += s

    n_total = n_poses * n_time_segments
    return combined.astype(int), n_total
