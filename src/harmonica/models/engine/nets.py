"""Compensator networks: context-conditioned invertible maps A(t) of the signal."""

import math
from typing import Sequence, Optional, List
import torch
import torch.nn as nn


class Reshape(nn.Module):
    def __init__(self, output_shape: Sequence[int]):
        super().__init__()
        self.output_shape = output_shape

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.reshape((-1,) + tuple(self.output_shape))


class ConditionalAffine(nn.Module):
    """Conditional affine transform network with optional multi-layer flow.
    
    Maps context to scale (log_weight) and shift (bias) parameters,
    then applies invertible affine transform: x * exp(log_weight) + bias
    
    Args:
        input_shape: Shape of the context input [..., Feature]
        output_shape: Shape of the data to transform (output params are 2x this)
        out_chans_per_layer: Hidden layer sizes for the param network
        use_batch_norm: Whether to use batch norm
        dropout: Dropout probability
        output_activation: Optional activation on params
        use_output_bias: Whether to use bias on output layer
        n_layers: Number of affine layers (default=1)
    """
    def __init__(
        self,
        input_shape: Sequence[int],
        output_shape: Sequence[int],
        out_chans_per_layer: List[int],
        use_batch_norm: bool = True,
        dropout: Optional[float] = None,
        output_activation: Optional[str] = None,
        use_output_bias: bool = True,
        n_layers: int = 1,
    ):
        super().__init__()

        out_chans_per_layer = [c for c in out_chans_per_layer if c]
        self.output_dim = math.prod(output_shape)
        self.n_layers = n_layers

        # Create parameter networks for each layer
        self.blocks = nn.ModuleList()
        for _ in range(n_layers):
            block = self._build_param_network(
                input_shape, out_chans_per_layer, use_batch_norm,
                dropout, output_activation, use_output_bias,
            )
            self.blocks.append(block)

        # Params cache (initialized via init_cache) - stores params as list, one per layer
        self.params: Optional[List[torch.Tensor]] = None

    def _build_param_network(
        self,
        input_shape: Sequence[int],
        out_chans_per_layer: List[int],
        use_batch_norm: bool,
        dropout: Optional[float],
        output_activation: Optional[str],
        use_output_bias: bool,
    ) -> nn.Sequential:
        """Build a single parameter prediction network."""
        layers: List[nn.Module] = []
        in_features = input_shape[-1]
        
        for out_features in out_chans_per_layer:
            if dropout is not None:
                layers.append(nn.Dropout(dropout))
            layer = nn.Linear(in_features, out_features)
            nn.init.xavier_uniform_(layer.weight)
            if layer.bias is not None:
                nn.init.zeros_(layer.bias)
            layers.append(layer)
            layers.append(nn.ReLU())
            if use_batch_norm:
                # Use default initialization (weight=1, bias=0) for stable identity-like behavior
                bn_layer = nn.BatchNorm1d(out_features)
                layers.append(bn_layer)
            in_features = out_features

        # Output 2x the data dimension for log_weight and bias
        flat_output_dim = 2 * self.output_dim
        out_features = in_features
        if len(out_chans_per_layer) > 0:
            out_features = out_chans_per_layer[-1]
        output_layer = nn.Linear(out_features, flat_output_dim, bias=use_output_bias)
        nn.init.zeros_(output_layer.weight)
        if use_output_bias:
            nn.init.zeros_(output_layer.bias)
        layers.append(output_layer)
        layers.append(Reshape((flat_output_dim,)))
        layers.append(
            getattr(torch.nn, output_activation)()
            if output_activation
            else torch.nn.Identity()
        )
        return nn.Sequential(*layers)

    def init_cache(self, signal_length: int, device: torch.device, dtype: torch.dtype):
        """Initialize params cache to zeros (identity transform)."""
        # Cache stores params as list of tensors, one per layer (avoids large contiguous allocation)
        self.params = [
            torch.zeros([signal_length, 2 * self.output_dim], dtype=dtype, device=device)
            for _ in range(self.n_layers)
        ]

    def update_cache(self, context: torch.Tensor):
        """Update params cache from context."""
        was_training = self.training
        self.eval()
        with torch.no_grad():
            self.params = self.get_params(context)
        if was_training:
            self.train()

    def get_params(self, context: torch.Tensor) -> List[torch.Tensor]:
        """Get raw parameters from context for all layers."""
        return [block(context) for block in self.blocks]

    def _apply_single_affine(
        self, x: torch.Tensor, params: torch.Tensor, inverse: bool = False
    ) -> torch.Tensor:
        """Apply a single affine transform."""
        log_weight, bias = params.split(params.shape[-1] // 2, dim=-1)
        if inverse:
            return (x - bias) / log_weight.exp()
        else:
            return x * log_weight.exp() + bias

    def forward(
        self,
        x: torch.Tensor,
        context: torch.Tensor,
        use_cache: bool = False,
        indices: Optional[slice] = None,
    ) -> torch.Tensor:
        """Apply forward transform through all layers.
        
        Args:
            x: Input data tensor
            context: Conditioning context tensor
            use_cache: If True, use cached params instead of computing from context
            indices: Optional slice to index cached params (only used with use_cache=True)
            
        Returns:
            Transformed tensor
        """
        if use_cache:
            assert self.params is not None, "Cache not initialized, call init_cache first"
            all_params = [p if indices is None else p[indices] for p in self.params]
        else:
            all_params = self.get_params(context)

        for i in range(self.n_layers):
            # Apply affine transform
            x = self._apply_single_affine(x, all_params[i], inverse=False)

        return x

    def inverse(
        self,
        z: torch.Tensor,
        context: torch.Tensor,
        use_cache: bool = False,
        indices: Optional[slice] = None,
    ) -> torch.Tensor:
        """Apply inverse transform through all layers (in reverse order).
        
        Args:
            z: Transformed data tensor
            context: Conditioning context tensor
            use_cache: If True, use cached params instead of computing from context
            indices: Optional slice to index cached params (only used with use_cache=True)
            
        Returns:
            Original tensor
        """
        if use_cache:
            assert self.params is not None, "Cache not initialized, call init_cache first"
            all_params = [p if indices is None else p[indices] for p in self.params]
        else:
            all_params = self.get_params(context)

        x = z

        # Apply layers in reverse order
        for i in range(self.n_layers - 1, -1, -1):
            # Apply inverse affine transform
            x = self._apply_single_affine(x, all_params[i], inverse=True)

        return x


class _ParamNetworkMixin:
    """Shared helper to build prediction networks with Xavier init and zero output."""

    @staticmethod
    def _build_prediction_net(
        input_dim: int,
        hidden_dims: List[int],
        output_dim: int,
        dropout: Optional[float],
    ) -> nn.Sequential:
        layers: List[nn.Module] = []
        in_features = input_dim
        for out_features in hidden_dims:
            if dropout is not None:
                layers.append(nn.Dropout(dropout))
            layer = nn.Linear(in_features, out_features)
            nn.init.xavier_uniform_(layer.weight)
            nn.init.zeros_(layer.bias)
            layers.append(layer)
            layers.append(nn.ReLU())
            in_features = out_features

        output_layer = nn.Linear(in_features, output_dim, bias=True)
        nn.init.zeros_(output_layer.weight)
        nn.init.zeros_(output_layer.bias)
        layers.append(output_layer)
        return nn.Sequential(*layers)


class ConditionalLowRank(nn.Module, _ParamNetworkMixin):
    """Conditional low-rank mixing: A(t) = I + U(t) @ V^T.

    Network predicts U(t) of shape [T, C, r] per sample.
    V is a learned shared matrix [C, r].
    Inverse via Woodbury: (I + UV^T)^{-1} = I - U(I + V^TU)^{-1}V^T.
    """

    def __init__(
        self,
        input_shape: Sequence[int],
        output_shape: Sequence[int],
        out_chans_per_layer: List[int],
        rank: int = 4,
        dropout: Optional[float] = None,
        n_layers: int = 1,
        **kwargs,
    ):
        super().__init__()
        self.C = math.prod(output_shape)
        self.rank = rank
        self.n_layers = n_layers

        hidden_dims = [c for c in out_chans_per_layer if c]

        self.V = nn.ParameterList([
            nn.Parameter(torch.randn(self.C, rank) * 0.01)
            for _ in range(n_layers)
        ])

        self.blocks = nn.ModuleList([
            self._build_prediction_net(input_shape[-1], hidden_dims, self.C * rank, dropout)
            for _ in range(n_layers)
        ])

        self.params: Optional[List[torch.Tensor]] = None

    def init_cache(self, signal_length: int, device: torch.device, dtype: torch.dtype):
        self.params = [
            torch.zeros(signal_length, self.C * self.rank, dtype=dtype, device=device)
            for _ in range(self.n_layers)
        ]

    def update_cache(self, context: torch.Tensor):
        was_training = self.training
        self.eval()
        with torch.no_grad():
            self.params = [block(context) for block in self.blocks]
        if was_training:
            self.train()

    def _solve_small(self, A: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """Solve A @ x = b for small batched A, closed-form for r<=3 (MPS-friendly)."""
        if self.rank == 1:
            return b / A[:, 0, 0].unsqueeze(-1)
        elif self.rank == 2:
            a, bc, d = A[:, 0, 0], A[:, 0, 1], A[:, 1, 1]
            cb = A[:, 1, 0]
            det = a * d - bc * cb
            x0 = (d * b[:, 0] - bc * b[:, 1]) / det
            x1 = (a * b[:, 1] - cb * b[:, 0]) / det
            return torch.stack([x0, x1], dim=-1)
        elif self.rank == 3:
            m = A
            det = (m[:, 0, 0] * (m[:, 1, 1] * m[:, 2, 2] - m[:, 1, 2] * m[:, 2, 1])
                   - m[:, 0, 1] * (m[:, 1, 0] * m[:, 2, 2] - m[:, 1, 2] * m[:, 2, 0])
                   + m[:, 0, 2] * (m[:, 1, 0] * m[:, 2, 1] - m[:, 1, 1] * m[:, 2, 0]))
            inv = torch.empty_like(A)
            inv[:, 0, 0] = m[:, 1, 1] * m[:, 2, 2] - m[:, 1, 2] * m[:, 2, 1]
            inv[:, 0, 1] = m[:, 0, 2] * m[:, 2, 1] - m[:, 0, 1] * m[:, 2, 2]
            inv[:, 0, 2] = m[:, 0, 1] * m[:, 1, 2] - m[:, 0, 2] * m[:, 1, 1]
            inv[:, 1, 0] = m[:, 1, 2] * m[:, 2, 0] - m[:, 1, 0] * m[:, 2, 2]
            inv[:, 1, 1] = m[:, 0, 0] * m[:, 2, 2] - m[:, 0, 2] * m[:, 2, 0]
            inv[:, 1, 2] = m[:, 0, 2] * m[:, 1, 0] - m[:, 0, 0] * m[:, 1, 2]
            inv[:, 2, 0] = m[:, 1, 0] * m[:, 2, 1] - m[:, 1, 1] * m[:, 2, 0]
            inv[:, 2, 1] = m[:, 0, 1] * m[:, 2, 0] - m[:, 0, 0] * m[:, 2, 1]
            inv[:, 2, 2] = m[:, 0, 0] * m[:, 1, 1] - m[:, 0, 1] * m[:, 1, 0]
            inv = inv / det.unsqueeze(-1).unsqueeze(-1)
            return torch.einsum("bij,bj->bi", inv, b)
        else:
            rhs = b.unsqueeze(-1)
            try:
                return torch.linalg.solve(A, rhs).squeeze(-1)
            except RuntimeError:
                return torch.linalg.solve(A.cpu(), rhs.cpu()).to(A.device).squeeze(-1)

    def _apply_transform(self, x: torch.Tensor, flat_U: torch.Tensor, V: nn.Parameter, inverse: bool) -> torch.Tensor:
        U = flat_U.view(-1, self.C, self.rank)  # [B, C, r]
        if not inverse:
            Vtx = x @ V  # [B, r]
            return x + (U @ Vtx.unsqueeze(-1)).squeeze(-1)
        else:
            # (I + UV^T)^{-1} x = x - U(I + V^TU)^{-1}V^T x
            VtU = torch.einsum("rc,bcq->brq", V.T, U)  # [B, r, r]
            inner = torch.eye(self.rank, device=x.device, dtype=x.dtype).unsqueeze(0) + VtU
            Vtx = x @ V  # [B, r]
            solved = self._solve_small(inner, Vtx)
            return x - (U @ solved.unsqueeze(-1)).squeeze(-1)

    def forward(self, x, context, use_cache=False, indices=None):
        if use_cache:
            assert self.params is not None
            all_params = [p if indices is None else p[indices] for p in self.params]
        else:
            all_params = [block(context) for block in self.blocks]
        for i in range(self.n_layers):
            x = self._apply_transform(x, all_params[i], self.V[i], inverse=False)
        return x

    def inverse(self, z, context, use_cache=False, indices=None):
        if use_cache:
            assert self.params is not None
            all_params = [p if indices is None else p[indices] for p in self.params]
        else:
            all_params = [block(context) for block in self.blocks]
        x = z
        for i in range(self.n_layers - 1, -1, -1):
            x = self._apply_transform(x, all_params[i], self.V[i], inverse=True)
        return x
