# SPDX-License-Identifier: MIT
# Copyright (c) 2025 <Hugin J. Zachariasen, Magnus H. Jensen, Martin C. B. Nielsen, Tobias S. Madsen>.

import torch
from torchmetrics.functional.regression import mean_squared_error
from torchmetrics.functional.image import (
    peak_signal_noise_ratio,
    structural_similarity_index_measure,
    spectral_angle_mapper,
)


class MetricCalculator:
    """
    Compute metrics for a *single* prediction-ground-truth pair.

    Metrics (cube-level):
      - MRAE, MSE, RMSE         on the full cube (pred vs gt)
      - PSNR, SSIM, SAM         on the full cube (pred vs gt)

    Metrics (band-level, per spectrum):
      - MRAE_{BAND}, MSE_{BAND}, RMSE_{BAND}
      - PSNR_{BAND}, SSIM_{BAND}
      - SAM_{BAND}  (note: SAM is typically multi-spectral; per-band SAM becomes a scalar-angle)

    NDVI/NDRE (per-image means, no cross-use):
      - NDVI_PRED, NDVI_GT
      - NDRE_PRED, NDRE_GT

    Expected input shapes per call (pred, gt):
      - (C, H, W)    (e.g. [bands, H, W])
      - (H, W, C)
      - (B, C, H, W) (if you pass a batch; metrics are averaged over all)

    Band order is assumed to be: ["G", "R", "RE", "NIR"] (C=4) unless overridden.
    """

    def __init__(
        self,
        data_range: float = 1.0,
        nir_index: int = 3,
        red_index: int = 1,
        rededge_index: int = 2,
        band_names: list[str] | None = None,
        eps: float = 1e-8,
        device: str | None = None,
    ):
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)

        self.data_range = data_range
        self.nir_index = nir_index
        self.red_index = red_index
        self.rededge_index = rededge_index
        self.eps = eps

        self.band_names = band_names if band_names is not None else ["G", "R", "RE", "NIR"]

    # ---------- public API ---------- #
    @torch.no_grad()
    def compute(self, pred, gt) -> dict[str, float]:
        """
        Compute all metrics for a single (pred, gt) pair.

        Returns a dict of Python floats.
        """
        pred_t, gt_t = self._prepare_tensors(pred, gt)  # -> (B, C, H, W)

        # --- cube-level error metrics ---
        mrae_val = self._mrae(pred_t, gt_t)

        mse_val = mean_squared_error(pred_t, gt_t)
        rmse_val = torch.sqrt(mse_val)

        psnr_val = peak_signal_noise_ratio(pred_t, gt_t, data_range=self.data_range)
        ssim_val = structural_similarity_index_measure(pred_t, gt_t, data_range=self.data_range)
        sam_val = spectral_angle_mapper(pred_t, gt_t)

        # --- band-level metrics ---
        per_band = self._per_band_metrics(pred_t, gt_t)

        # --- NDVI / NDRE per-image scores (no cross-use) ---
        ndvi_pred_map, ndvi_gt_map = self._compute_ndvi(pred_t, gt_t)
        ndre_pred_map, ndre_gt_map = self._compute_ndre(pred_t, gt_t)

        # mean over batch + spatial dimensions
        ndvi_pred_mean = ndvi_pred_map.mean()
        ndvi_gt_mean = ndvi_gt_map.mean()

        ndre_pred_mean = ndre_pred_map.mean()
        ndre_gt_mean = ndre_gt_map.mean()

        # Convert all to plain floats on CPU
        out: dict[str, float] = {
            "MRAE": float(mrae_val.cpu()),
            "MSE": float(mse_val.cpu()),
            "RMSE": float(rmse_val.cpu()),
            "PSNR": float(psnr_val.cpu()),
            "SSIM": float(ssim_val.cpu()),
            "SAM": float(sam_val.cpu()),
            "NDVI_PRED": float(ndvi_pred_mean.cpu()),
            "NDVI_GT": float(ndvi_gt_mean.cpu()),
            "NDRE_PRED": float(ndre_pred_mean.cpu()),
            "NDRE_GT": float(ndre_gt_mean.cpu()),
        }

        # Merge band-level outputs
        out.update(per_band)
        return out

    # ---------- internal helpers ---------- #
    def _prepare_tensors(self, pred, gt):
        # Accept numpy or torch
        if not isinstance(pred, torch.Tensor):
            pred = torch.as_tensor(pred)
        if not isinstance(gt, torch.Tensor):
            gt = torch.as_tensor(gt)

        pred = pred.to(self.device, dtype=torch.float32)
        gt = gt.to(self.device, dtype=torch.float32)

        # Handle (H, W, C) -> (C, H, W)
        if pred.ndim == 3 and pred.shape[-1] == gt.shape[-1] and pred.shape[-1] <= 16:
            pred = pred.permute(2, 0, 1)
            gt = gt.permute(2, 0, 1)

        # If 3D, treat as single sample: (C, H, W) -> (1, C, H, W)
        if pred.ndim == 3:
            pred = pred.unsqueeze(0)
            gt = gt.unsqueeze(0)
        elif pred.ndim != 4:
            raise ValueError(
                f"Expected pred/gt to be 3D or 4D (C,H,W) or (B,C,H,W), "
                f"got {pred.shape=}, {gt.shape=}"
            )

        if pred.shape != gt.shape:
            raise ValueError(f"Shape mismatch: pred {pred.shape}, gt {gt.shape}")

        # Validate band indices + band names
        b, c, h, w = pred.shape
        if self.band_names is None:
            self.band_names = [f"C{i}" for i in range(c)]
        if len(self.band_names) != c:
            raise ValueError(
                f"band_names has length {len(self.band_names)} but input has {c} channels. "
                f"Got band_names={self.band_names}."
            )

        for idx, name in [
            (self.nir_index, "nir_index"),
            (self.red_index, "red_index"),
            (self.rededge_index, "rededge_index"),
        ]:
            if not (0 <= idx < c):
                raise ValueError(f"{name}={idx} is out of range for cube with {c} channels")

        return pred, gt

    def _mrae(self, pred: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
        """Mean Relative Absolute Error = mean(|pred - gt| / (|gt| + eps))."""
        assert pred.shape == gt.shape
        error = torch.abs(pred - gt) / torch.abs(gt + self.eps)
        error = error.clamp(0.0, 1.0)
        return torch.mean(error.reshape(-1))

    def _sam_scalar(self, pred_1c: torch.Tensor, gt_1c: torch.Tensor) -> torch.Tensor:
        """
        "Per-band SAM" for a single channel (B,1,H,W) as a scalar-angle:
          angle = arccos( (p*g) / (|p|*|g| + eps) )
        Returns mean angle over all pixels/batch.
        """
        # shapes: (B,1,H,W)
        p = pred_1c.reshape(-1)
        g = gt_1c.reshape(-1)

        denom = (torch.abs(p) * torch.abs(g) + self.eps)
        cos = (p * g) / denom
        cos = cos.clamp(-1.0, 1.0)
        ang = torch.arccos(cos)
        return ang.mean()

    def _per_band_metrics(self, pred: torch.Tensor, gt: torch.Tensor) -> dict[str, float]:
        """
        Compute MRAE/MSE/RMSE/PSNR/SSIM (and scalar SAM) per band.
        pred, gt: (B, C, H, W)
        """
        out: dict[str, float] = {}
        for c_idx, band in enumerate(self.band_names):
            p = pred[:, c_idx : c_idx + 1, :, :]  # (B,1,H,W)
            g = gt[:, c_idx : c_idx + 1, :, :]

            mrae_val = self._mrae(p, g)
            mse_val = mean_squared_error(p, g)
            rmse_val = torch.sqrt(mse_val)

            psnr_val = peak_signal_noise_ratio(p, g, data_range=self.data_range)
            ssim_val = structural_similarity_index_measure(p, g, data_range=self.data_range)

            sam_val = self._sam_scalar(p, g)

            out[f"MRAE_{band}"] = float(mrae_val.cpu())
            out[f"MSE_{band}"] = float(mse_val.cpu())
            out[f"RMSE_{band}"] = float(rmse_val.cpu())
            out[f"PSNR_{band}"] = float(psnr_val.cpu())
            out[f"SSIM_{band}"] = float(ssim_val.cpu())
            out[f"SAM_{band}"] = float(sam_val.cpu())

        return out

    def _compute_ndvi(self, pred: torch.Tensor, gt: torch.Tensor):
        """
        pred, gt: (B, C, H, W)
        Returns:
          ndvi_pred, ndvi_gt: (B, H, W) each
        """
        nir_p = pred[:, self.nir_index, :, :]
        red_p = pred[:, self.red_index, :, :]
        nir_g = gt[:, self.nir_index, :, :]
        red_g = gt[:, self.red_index, :, :]

        ndvi_pred = (nir_p - red_p) / (nir_p + red_p + self.eps)
        ndvi_gt = (nir_g - red_g) / (nir_g + red_g + self.eps)

        return ndvi_pred, ndvi_gt

    def _compute_ndre(self, pred: torch.Tensor, gt: torch.Tensor):
        """
        pred, gt: (B, C, H, W)
        Returns:
          ndre_pred, ndre_gt: (B, H, W) each
        """
        nir_p = pred[:, self.nir_index, :, :]
        re_p = pred[:, self.rededge_index, :, :]
        nir_g = gt[:, self.nir_index, :, :]
        re_g = gt[:, self.rededge_index, :, :]

        ndre_pred = (nir_p - re_p) / (nir_p + re_p + self.eps)
        ndre_gt = (nir_g - re_g) / (nir_g + re_g + self.eps)

        return ndre_pred, ndre_gt
