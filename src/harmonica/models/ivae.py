"""Identifiable VAE (iVAE), an identifiable nonlinear-ICA baseline.

After Khemakhem et al., AISTATS 2020. Conditioning the latent prior
``p(z|u)`` on an observed auxiliary variable ``u`` (here the joint angle)
makes the latent sources identifiable up to permutation and component-wise
transform. The encoder mean ``q(z|x,u)`` plays the role of the demixing
function. Used by :func:`harmonica.baselines.run_ivae`.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

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
# Source extraction
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
    u_labels: np.ndarray,
    n_sources: int = 5,
    n_epochs: int = 200,
    lr: float = 1e-3,
    encoder_hidden: tuple[int, ...] = (256, 128),
    prior_hidden: tuple[int, ...] = (64,),
    decoder_type: str = "linear",
    decoder_hidden: tuple[int, ...] = (128,),
    batch_size: int = 512,
    sparsity_weight: float = 0.0,
    device: str | None = None,
    verbose: bool = True,
):
    """Build and train an iVAE, then return its encoder-mean sources.

    The network is initialised from, and trained with, the global ``torch``
    RNG; seed it beforehand for a repeatable run.

    Args:
        X: ``(n_features, n_samples)`` whitened extended signal.
        u_labels: ``(n_samples,)`` auxiliary-variable labels (angle bins).
        n_sources: Latent dimensionality (set ~ number of GT MUs).
        n_epochs: Training epochs.
        lr: Learning rate.
        encoder_hidden / prior_hidden / decoder_type / decoder_hidden:
            Architecture knobs.
        batch_size: Minibatch size.
        sparsity_weight: Kurtosis regulariser weight.
        device: Torch device.
        verbose: Print progress.

    Returns:
        ``(sources, model, history)``: the ``(n_sources, n_samples)``
        encoder means, the trained model and the training history.
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
    return sources, model, history
