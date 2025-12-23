"""Ljung–Box whiteness test utilities."""
from __future__ import annotations

from typing import Optional

import torch
from statsmodels.stats.diagnostic import acorr_ljungbox


def ljung_box_pvalues(innovations: torch.Tensor, mask: Optional[torch.Tensor] = None, lag: int = 20) -> torch.Tensor:
    """Compute Ljung–Box p-values per batch and dimension."""
    batch, time, dim = innovations.shape
    pvalues = torch.zeros(batch, dim, device=innovations.device)
    for b in range(batch):
        for d in range(dim):
            series_tensor = innovations[b, :, d]
            series = series_tensor.detach().cpu().numpy()
            if mask is not None:
                valid_mask = (~mask[b]).cpu().numpy()
            else:
                valid_mask = torch.ones_like(series_tensor, dtype=torch.bool).cpu().numpy()
            finite_mask = ~torch.isnan(series_tensor).cpu().numpy()
            combined = valid_mask & finite_mask
            series = series[combined]
            if len(series) <= lag + 1:
                pvalues[b, d] = 0.0
                continue
            result = acorr_ljungbox(series, lags=[lag], return_df=True)
            pvalues[b, d] = float(result["lb_pvalue"].iloc[0])
    return pvalues


__all__ = ["ljung_box_pvalues"]
