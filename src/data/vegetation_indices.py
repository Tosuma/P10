"""
Vegetation index computation.

Each index is computed from specific spectral bands and appended as additional
input channels to the multi-spectral tensor.  Including these as explicit
channels rather than letting the network discover them serves two purposes:
  1. It injects domain knowledge (agronomic literature) directly into the
     representation, which speeds convergence and improves interpretability.
  2. The anomaly detector can leverage well-understood plant stress signatures
     (e.g. NDVI drop, NDRE shift) that may be subtle in raw reflectance.

Band naming convention (matching band_loader.py DEFAULT_MS_SUFFIXES):
    channel 0: G   (green,     ~550 nm)
    channel 1: R   (red,       ~660 nm)
    channel 2: RE  (red-edge,  ~717 nm)
    channel 3: NIR (near-IR,   ~840 nm)

If RGB bands are prepended (include_rgb=True in band_loader), the indices are:
    channel 0: R_rgb
    channel 1: G_rgb
    channel 2: B_rgb
    channel 3: G_ms
    channel 4: R_ms
    channel 5: RE
    channel 6: NIR

Pass `rgb_offset=3` (default) to account for this.
"""

from __future__ import annotations

import numpy as np


_EPS = 1e-8  # Avoid division by zero in ratio indices


def ndvi(nir: np.ndarray, red: np.ndarray) -> np.ndarray:
    """
    Normalized Difference Vegetation Index.

    NDVI = (NIR - R) / (NIR + R)

    Range: [-1, 1].  Healthy vegetation typically > 0.4.
    Stress, bare soil, or water yields lower values.
    """
    return (nir - red) / (nir + red + _EPS)


def ndre(nir: np.ndarray, red_edge: np.ndarray) -> np.ndarray:
    """
    Normalized Difference Red-Edge Index.

    NDRE = (NIR - RE) / (NIR + RE)

    More sensitive to chlorophyll content and early stress than NDVI because
    the red-edge band sits at the steep slope of the plant reflectance curve.
    Preferred for detecting subtle stress before it is visible in NDVI.
    """
    return (nir - red_edge) / (nir + red_edge + _EPS)


def savi(nir: np.ndarray, red: np.ndarray, L: float = 0.5) -> np.ndarray:
    """
    Soil-Adjusted Vegetation Index.

    SAVI = ((NIR - R) / (NIR + R + L)) * (1 + L)

    L=0.5 is the standard choice for intermediate vegetation cover, which is
    appropriate for agricultural fields where bare soil between crop rows is
    often visible in drone imagery.
    """
    return ((nir - red) / (nir + red + L + _EPS)) * (1.0 + L)


def evi(
    nir: np.ndarray,
    red: np.ndarray,
    blue: np.ndarray,
    G: float = 2.5,
    C1: float = 6.0,
    C2: float = 7.5,
    L: float = 1.0,
) -> np.ndarray:
    """
    Enhanced Vegetation Index.

    EVI = G * (NIR - R) / (NIR + C1*R - C2*B + L)

    Uses a blue band to correct for aerosol scattering.  More accurate than
    NDVI over dense canopies (not saturating at high LAI) and in areas of
    high atmospheric variability between flight days.

    Default coefficients are MODIS standard (Liu & Huete 1995).
    """
    return G * (nir - red) / (nir + C1 * red - C2 * blue + L + _EPS)


def compute_vegetation_indices(
    image: np.ndarray,
    rgb_offset: int = 3,
    ms_band_order: tuple[int, int, int, int] = (0, 1, 2, 3),
    has_blue: bool = True,
) -> np.ndarray:
    """
    Compute NDVI, NDRE, SAVI, EVI and concatenate to `image` as extra channels.

    Parameters
    ----------
    image : np.ndarray  (H, W, C)
        Multi-channel image tensor.  If rgb_offset > 0 the first `rgb_offset`
        channels are RGB; spectral bands follow.
    rgb_offset : int
        Number of leading RGB channels (0 if no RGB prepended).
    ms_band_order : tuple of 4 ints
        (G_idx, R_idx, RE_idx, NIR_idx) — indices *within the MS portion*
        (i.e. after slicing image[..., rgb_offset:]).
    has_blue : bool
        If True, use image[..., 2] (B in RGB) for EVI; otherwise skip EVI.

    Returns
    -------
    np.ndarray  (H, W, C + 4) or (H, W, C + 3) if not has_blue
    """
    ms = image[..., rgb_offset:]          # (H, W, num_ms_bands)
    g_idx, r_idx, re_idx, nir_idx = ms_band_order

    G_band   = ms[..., g_idx]
    R_band   = ms[..., r_idx]
    RE_band  = ms[..., re_idx]
    NIR_band = ms[..., nir_idx]

    ndvi_ch  = ndvi(NIR_band, R_band)[..., np.newaxis]
    ndre_ch  = ndre(NIR_band, RE_band)[..., np.newaxis]
    savi_ch  = savi(NIR_band, R_band)[..., np.newaxis]

    extra_channels = [ndvi_ch, ndre_ch, savi_ch]

    if has_blue and rgb_offset >= 3:
        B_band = image[..., 2]  # Blue from RGB
        evi_ch = evi(NIR_band, R_band, B_band)[..., np.newaxis]
        extra_channels.append(evi_ch)

    indices_array = np.concatenate(extra_channels, axis=-1)  # (H, W, 3 or 4)

    # Clip to a sane range: all standard vegetation indices are bounded but
    # numerical noise near zero denominators can produce extreme values.
    indices_array = np.clip(indices_array, -1.5, 1.5).astype(np.float32)

    return np.concatenate([image, indices_array], axis=-1)
