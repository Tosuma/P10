"""
FastFlow-style normalizing flow for anomaly scoring.

Architecture choice
-------------------
We implement a FastFlow-inspired model (Yu et al. 2021, "FastFlow: Unsupervised
Anomaly Detection and Localization via 2D Normalizing Flows") using FrEIA
(Framework for Easily Invertible Architectures) coupling blocks.

Why a normalizing flow over reconstruction error (autoencoder)?
  - The MAE encoder outputs a rich feature space.  Reconstruction error
    conflates the encoder's inability to represent normal vs. anomalous
    samples — it is not a well-calibrated probability.
  - A normalizing flow models p(z) explicitly, giving us a negative log-
    likelihood score that is a proper probability estimate under the model.
  - NLL scores are more sensitive to subtle distribution shifts (early plant
    stress) than reconstruction MSE.

Why FastFlow over CFlow-AD?
  - FastFlow operates on 2-D feature maps without requiring a conditioning
    network, which simplifies the training pipeline.
  - CFlow-AD requires pyramid pooling and a separate conditioning encoder.
    Since we already have a strong MAE encoder, a simpler flow suffices.

Input features
--------------
We operate on the MAE encoder's patch-level output tokens (strip CLS):
    shape (B, N, D) — N = num_patches, D = encoder embed_dim

For per-patch anomaly scoring, we process each patch token independently.
At inference, we map patch-level NLL scores back to spatial positions to
assemble the per-pixel heatmap.

Flow architecture
-----------------
We use GLOW-style affine coupling layers in FrEIA's RealNVP variant:
  - Alternating split/coupling with learnable scale and translation networks
  - ActNorm layers for stable training (replaces batch norm)
  - 8 coupling blocks (configurable) with 2-layer MLP subnets
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

try:
    import FrEIA.framework as Ff
    import FrEIA.modules as Fm
    FREIA_AVAILABLE = True
except ImportError:
    FREIA_AVAILABLE = False


# ── Subnet factory (used by FrEIA coupling layers) ────────────────────────────

def _make_subnet(in_dim: int, out_dim: int, hidden_dim: int = 512) -> nn.Sequential:
    """
    2-layer MLP subnet used inside each coupling block.

    We use LayerNorm instead of BatchNorm because flow models see varying
    batch sizes and the per-dimension statistics are more stable.
    """
    return nn.Sequential(
        nn.Linear(in_dim, hidden_dim),
        nn.LayerNorm(hidden_dim),
        nn.GELU(),
        nn.Linear(hidden_dim, out_dim),
    )


# ── FrEIA-based FastFlow implementation ──────────────────────────────────────

class FastFlowAnomalyModel(nn.Module):
    """
    Normalizing flow model for patch-level anomaly scoring.

    Operates on 1-D feature vectors (per patch token from the MAE encoder).
    Computes NLL under the learned Gaussian base distribution as anomaly score.

    Parameters
    ----------
    feature_dim : int
        Dimensionality of each patch token (encoder embed_dim).
    num_coupling_blocks : int
        Number of affine coupling blocks.  8 is a good default for D~384–768.
    subnet_hidden_dim : int
        Hidden dimension of coupling block MLPs.
    """

    def __init__(
        self,
        feature_dim: int = 384,
        num_coupling_blocks: int = 8,
        subnet_hidden_dim: int = 512,
    ):
        super().__init__()
        self.feature_dim = feature_dim

        if not FREIA_AVAILABLE:
            raise ImportError(
                "FrEIA is required for FastFlowAnomalyModel. "
                "Install with: pip install FrEIA"
            )

        # Build FrEIA Invertible Network
        # Input node
        nodes = [Ff.InputNode(feature_dim, name="input")]

        for i in range(num_coupling_blocks):
            # Actnorm for stable scale normalisation (replaces BatchNorm in flows)
            nodes.append(
                Ff.Node(
                    nodes[-1],
                    Fm.ActNorm,
                    {},
                    name=f"actnorm_{i}",
                )
            )
            # Affine coupling with permutation (FrEIA AllInOneBlock handles this)
            nodes.append(
                Ff.Node(
                    nodes[-1],
                    Fm.AllInOneBlock,
                    {
                        "subnet_constructor": lambda in_d, out_d: _make_subnet(
                            in_d, out_d, subnet_hidden_dim
                        ),
                        "affine_clamping": 2.0,   # Clamp log-scale for stability
                        "permute_soft": True,      # Learnable soft permutation
                    },
                    name=f"coupling_{i}",
                )
            )

        nodes.append(Ff.OutputNode(nodes[-1], name="output"))
        self.flow = Ff.ReversibleGraphNet(nodes, verbose=False)

    def forward(self, z: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through the flow: feature space → latent space.

        Parameters
        ----------
        z : (B, D)  — patch feature vectors

        Returns
        -------
        z_latent : (B, D)
        log_jac_det : (B,) — log|det J| of the transformation
        """
        z_latent, log_jac_det = self.flow(z)
        return z_latent, log_jac_det

    def nll_score(self, z: torch.Tensor) -> torch.Tensor:
        """
        Compute negative log-likelihood (anomaly score) for each input patch.

        NLL = -log p(z) = 0.5 * ||z_latent||² + 0.5*D*log(2π) - log|det J|

        Higher NLL → more anomalous.

        Parameters
        ----------
        z : (B, D)

        Returns
        -------
        nll : (B,)
        """
        z_latent, log_jac_det = self.flow(z)
        D = z.shape[-1]
        # Gaussian log-likelihood in latent space
        log_pz = -0.5 * (z_latent ** 2).sum(dim=-1) - 0.5 * D * torch.log(
            torch.tensor(2 * torch.pi, device=z.device)
        )
        nll = -(log_pz + log_jac_det)
        return nll


# ── Fallback: simple RealNVP implemented without FrEIA ───────────────────────

class AffineCouplingBlock(nn.Module):
    """Minimal affine coupling layer for the FrEIA-free fallback."""

    def __init__(self, dim: int, hidden_dim: int = 512):
        super().__init__()
        half = dim // 2
        self.net_s = _make_subnet(half, half, hidden_dim)
        self.net_t = _make_subnet(half, half, hidden_dim)

    def forward(
        self, x: torch.Tensor, reverse: bool = False
    ) -> tuple[torch.Tensor, torch.Tensor]:
        x1, x2 = x.chunk(2, dim=-1)
        s = torch.tanh(self.net_s(x1)) * 2.0   # Clamp scale
        t = self.net_t(x1)
        if not reverse:
            y2 = x2 * torch.exp(s) + t
            log_det = s.sum(dim=-1)
        else:
            y2 = (x2 - t) * torch.exp(-s)
            log_det = -s.sum(dim=-1)
        return torch.cat([x1, y2], dim=-1), log_det


class SimpleRealNVP(nn.Module):
    """
    FrEIA-free fallback: RealNVP with alternating coupling masks.

    Used when FrEIA is unavailable.  Less expressive than FrEIA but
    sufficient for 1-D feature distribution modelling.
    """

    def __init__(
        self,
        feature_dim: int = 384,
        num_coupling_blocks: int = 8,
        subnet_hidden_dim: int = 512,
    ):
        super().__init__()
        self.feature_dim = feature_dim
        self.blocks = nn.ModuleList([
            AffineCouplingBlock(feature_dim, subnet_hidden_dim)
            for _ in range(num_coupling_blocks)
        ])
        # Learnable per-dimension scale (ActNorm equivalent)
        self.log_scale = nn.Parameter(torch.zeros(feature_dim))
        self.bias = nn.Parameter(torch.zeros(feature_dim))

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        log_det_total = torch.zeros(x.shape[0], device=x.device)

        # ActNorm
        x = (x - self.bias) * torch.exp(-self.log_scale)
        log_det_total -= self.log_scale.sum()

        for i, blk in enumerate(self.blocks):
            # Alternate flip to ensure all dimensions are coupled
            if i % 2 == 1:
                x = x.flip(-1)
            x, ld = blk(x)
            if i % 2 == 1:
                x = x.flip(-1)
            log_det_total = log_det_total + ld

        return x, log_det_total

    def nll_score(self, z: torch.Tensor) -> torch.Tensor:
        z_latent, log_jac_det = self.forward(z)
        D = z.shape[-1]
        log_pz = -0.5 * (z_latent ** 2).sum(dim=-1) - 0.5 * D * torch.log(
            torch.tensor(2 * torch.pi, device=z.device)
        )
        return -(log_pz + log_jac_det)


def build_flow_model(
    feature_dim: int,
    num_coupling_blocks: int = 8,
    subnet_hidden_dim: int = 512,
) -> nn.Module:
    """
    Factory: returns FastFlowAnomalyModel if FrEIA is available,
    else falls back to SimpleRealNVP with a warning.
    """
    if FREIA_AVAILABLE:
        return FastFlowAnomalyModel(feature_dim, num_coupling_blocks, subnet_hidden_dim)
    else:
        import warnings
        warnings.warn(
            "FrEIA not found — using SimpleRealNVP fallback.  "
            "Install FrEIA for full FastFlow functionality.",
            ImportWarning,
        )
        return SimpleRealNVP(feature_dim, num_coupling_blocks, subnet_hidden_dim)
