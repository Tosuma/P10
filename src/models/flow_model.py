"""
Normalizing flow model for patch-level anomaly scoring.

The flow is trained on FROZEN MAE encoder features.  During inference:
    anomaly_score = NLL = 0.5 * ||z||² − log|det J|

where z is the flow-transformed feature vector and log|det J| is the
log-determinant of the Jacobian.  Higher NLL = more anomalous.

Architecture:
  - Preferred:  FrEIA GLOWCouplingBlock (if package available)
  - Fallback:   SimpleRealNVP implemented in pure PyTorch

Per-patch (token-level) flow:
  The MAE encoder produces N=64 patch tokens per 128×128 input.
  We score each token independently: input is (B, D), output is (B,) NLL.
  Using score_patches() we process (B, N, D) → (B, N) scores, giving an
  8×8 anomaly map per input patch which stitches into a full-image heatmap.

Reference: Yu et al., "FastFlow: Unsupervised Anomaly Detection and
Localization via 2D Normalizing Flows", arXiv 2021.
"""

from __future__ import annotations

import torch
import torch.nn as nn

# ---------------------------------------------------------------------------
# FrEIA-based flow (preferred)
# ---------------------------------------------------------------------------

try:
    import FrEIA.framework as Ff
    import FrEIA.modules  as Fm
    _FREIA_AVAILABLE = True
except ImportError:
    _FREIA_AVAILABLE = False


def _subnet_fc(dims_in: int, dims_out: int) -> nn.Sequential:
    """MLP subnet used inside each GLOW coupling block."""
    hidden = max(dims_in * 2, 256)
    return nn.Sequential(
        nn.Linear(dims_in, hidden),
        nn.LayerNorm(hidden),
        nn.GELU(),
        nn.Linear(hidden, dims_out),
    )


def _build_freia_flow(feature_dim: int, n_blocks: int, clamp: float) -> nn.Module:
    """Build a FrEIA ReversibleGraphNet with GLOW coupling blocks."""
    nodes = [Ff.InputNode(feature_dim, name="input")]
    for k in range(n_blocks):
        nodes.append(Ff.Node(
            nodes[-1],
            Fm.GLOWCouplingBlock,
            {"subnet_constructor": _subnet_fc, "clamp": clamp},
            name=f"coupling_{k}",
        ))
        nodes.append(Ff.Node(nodes[-1], Fm.PermuteRandom, {}, name=f"perm_{k}"))
    nodes.append(Ff.OutputNode(nodes[-1], name="output"))
    return Ff.ReversibleGraphNet(nodes, verbose=False)


# ---------------------------------------------------------------------------
# Pure-PyTorch fallback: SimpleRealNVP
# ---------------------------------------------------------------------------

class _AffineCouplingBlock(nn.Module):
    """Single affine coupling layer with random fixed permutation."""

    def __init__(self, dim: int, perm: torch.Tensor) -> None:
        super().__init__()
        self.register_buffer("perm", perm)
        d1 = dim // 2
        d2 = dim - d1
        hidden = max(d1 * 2, 128)
        self.net_s = nn.Sequential(
            nn.Linear(d1, hidden), nn.GELU(), nn.Linear(hidden, d2)
        )
        self.net_t = nn.Sequential(
            nn.Linear(d1, hidden), nn.GELU(), nn.Linear(hidden, d2)
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = x[:, self.perm]
        d1 = x.shape[1] // 2
        x1, x2 = x[:, :d1], x[:, d1:]
        s = torch.tanh(self.net_s(x1)) * 2.0   # clamp to ±2
        t = self.net_t(x1)
        y2 = x2 * s.exp() + t
        log_jac = s.sum(dim=-1)
        return torch.cat([x1, y2], dim=-1), log_jac

    def inverse(self, y: torch.Tensor) -> torch.Tensor:
        d1 = y.shape[1] // 2
        y1, y2 = y[:, :d1], y[:, d1:]
        s = torch.tanh(self.net_s(y1)) * 2.0
        t = self.net_t(y1)
        x2 = (y2 - t) * (-s).exp()
        x = torch.cat([y1, x2], dim=-1)
        inv_perm = torch.argsort(self.perm)
        return x[:, inv_perm]


class SimpleRealNVP(nn.Module):
    """Stack of affine coupling layers as a plain normalizing flow."""

    def __init__(self, feature_dim: int, n_blocks: int) -> None:
        super().__init__()
        g = torch.Generator()
        g.manual_seed(0)
        self.blocks = nn.ModuleList([
            _AffineCouplingBlock(
                feature_dim,
                torch.randperm(feature_dim, generator=g),
            )
            for _ in range(n_blocks)
        ])

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        log_jac = torch.zeros(x.shape[0], device=x.device)
        for blk in self.blocks:
            x, lj = blk(x)
            log_jac = log_jac + lj
        return x, log_jac


# ---------------------------------------------------------------------------
# FlowModel (public API)
# ---------------------------------------------------------------------------

class FlowModel(nn.Module):
    """
    Normalizing flow for anomaly scoring on MAE encoder patch features.

    Args:
        feature_dim:       Input feature dimension (MAE embed_dim, default 384).
        n_coupling_blocks: Number of coupling / affine layers (default 8).
        clamp:             GLOW activation clamp (FrEIA only, default 2.0).
    """

    def __init__(
        self,
        feature_dim: int = 384,
        n_coupling_blocks: int = 8,
        clamp: float = 2.0,
    ) -> None:
        super().__init__()
        self.feature_dim = feature_dim

        if _FREIA_AVAILABLE:
            self.flow = _build_freia_flow(feature_dim, n_coupling_blocks, clamp)
            self._use_freia = True
        else:
            self.flow = SimpleRealNVP(feature_dim, n_coupling_blocks)
            self._use_freia = False

    def forward(self, features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Map features through the normalizing flow.

        Args:
            features: (B, feature_dim)
        Returns:
            z:           (B, feature_dim) transformed features
            log_jac_det: (B,) log-determinant of the Jacobian
        """
        if self._use_freia:
            # FrEIA returns (output_list, log_jac_det) where output_list is a
            # list of tensors (one per output node) and log_jac_det is a (B,)
            # tensor — NOT a list.
            z_list, log_jac_det = self.flow(features)
            z = z_list[0]
            if log_jac_det.dim() > 1:
                log_jac_det = log_jac_det.squeeze(-1)
        else:
            z, log_jac_det = self.flow(features)
        return z, log_jac_det

    def anomaly_score(self, features: torch.Tensor) -> torch.Tensor:
        """
        Compute per-sample NLL anomaly score.

        NLL = 0.5 * ||z||² − log|det J|

        Higher values indicate anomalous samples.

        Args:
            features: (B, feature_dim)
        Returns:
            scores: (B,) float32
        """
        z, log_jac_det = self.forward(features)
        nll = 0.5 * (z ** 2).sum(dim=-1) - log_jac_det
        return nll

    def score_patches(self, patch_features: torch.Tensor) -> torch.Tensor:
        """
        Score all N patch tokens in a batch of images.

        Args:
            patch_features: (B, N, D) — N token features per image
        Returns:
            scores: (B, N) float32 NLL per token
        """
        B, N, D = patch_features.shape
        flat = patch_features.reshape(B * N, D)
        scores = self.anomaly_score(flat)
        return scores.reshape(B, N)

    def flow_loss(self, features: torch.Tensor) -> torch.Tensor:
        """
        Training loss: mean NLL over batch.

        Args:
            features: (B, D)
        Returns:
            scalar loss
        """
        return self.anomaly_score(features).mean()
