from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
import torch
from loguru import logger

from spatialprot_data.utils.dataset.transforms import FilterFactory


class BaseStandardizer(ABC):
    """Base interface for stats-backed multiplex standardizers."""

    def __init__(
        self,
        *,
        name: str,
        modality_dir: str | Path,
        channels_per_image: pd.DataFrame,
        disable_quantile_mask: bool = False,
        filter_factory: FilterFactory | None = None,
        quantile_level: str = "image",
        stats_level: str = "global",
        verbose: bool = True,
    ) -> None:
        self.name = name
        self.modality_dir = Path(modality_dir)
        self.channels_per_image = channels_per_image
        self.disable_quantile_mask = disable_quantile_mask
        self.filter_factory = filter_factory
        self.quantile_level = quantile_level
        self.stats_level = stats_level
        self.verbose = verbose
        self.valid_mask_df: pd.DataFrame | None = None
        self._mask_cache: dict[tuple, np.ndarray] = {}

    @abstractmethod
    def apply(
        self,
        x: np.ndarray | torch.Tensor,
        tissue_id: str,
        measured_mask: np.ndarray,
        selected_mask: np.ndarray,
    ) -> Tuple[torch.Tensor, np.ndarray]:
        raise NotImplementedError

    def _ensure_tensor(self, x: np.ndarray | torch.Tensor) -> torch.Tensor:
        if isinstance(x, np.ndarray):
            return torch.from_numpy(x).float()
        return x.float()

    def _apply_optional_filters(self, x: torch.Tensor) -> torch.Tensor:
        if self.filter_factory is not None:
            x = self.filter_factory.apply_filters(x)
        return x

    def _table_row(self, table: pd.DataFrame, tissue_id: str) -> np.ndarray:
        if tissue_id in table.index:
            row = table.loc[tissue_id]
            if isinstance(row, pd.DataFrame):
                row = row.iloc[0]
        elif len(table) == 1:
            row = table.iloc[0]
        else:
            raise KeyError(
                f"Tissue id {tissue_id} not found in table with index {table.index[:5]!r}"
            )
        return row.to_numpy()

    def _masked_values(
        self,
        table: pd.DataFrame,
        tissue_id: str,
        measured_mask: np.ndarray,
        refined_mask: np.ndarray,
        *,
        dtype: np.dtype = np.float32,
    ) -> np.ndarray:
        row = self._table_row(table, tissue_id)
        combined = measured_mask.copy()
        combined[measured_mask] = refined_mask
        return row[combined].astype(dtype, copy=False)

    def _refine_selected_mask(
        self,
        tissue_id: str,
        measured_mask: np.ndarray,
        selected_mask: np.ndarray,
    ) -> np.ndarray:
        if self.disable_quantile_mask or self.valid_mask_df is None:
            return selected_mask

        cache_key = (tissue_id, measured_mask.tobytes(), selected_mask.tobytes())
        if cache_key in self._mask_cache:
            return self._mask_cache[cache_key]

        valid_row = self._table_row(self.valid_mask_df, tissue_id).astype(bool, copy=False)
        result = selected_mask & valid_row[measured_mask]
        self._mask_cache[cache_key] = result
        return result

    def _refine_and_slice(
        self,
        x_t: torch.Tensor,
        tissue_id: str,
        measured_mask: np.ndarray,
        selected_mask: np.ndarray,
    ) -> Tuple[torch.Tensor, np.ndarray]:
        refined = self._refine_selected_mask(tissue_id, measured_mask, selected_mask)
        if refined is selected_mask or refined.sum() == selected_mask.sum():
            return x_t, refined

        keep_in_selected = np.asarray(refined[selected_mask], dtype=bool)
        if keep_in_selected.all():
            return x_t, refined

        return x_t[torch.from_numpy(keep_in_selected)], refined


class IdentityStandardizer(BaseStandardizer):
    def __init__(self, **kwargs) -> None:
        super().__init__(name="identity", **kwargs)

    def apply(
        self,
        x: np.ndarray | torch.Tensor,
        tissue_id: str,
        measured_mask: np.ndarray,
        selected_mask: np.ndarray,
    ) -> Tuple[torch.Tensor, np.ndarray]:
        x_t = self._ensure_tensor(x)
        x_t = self._apply_optional_filters(x_t)
        return x_t, selected_mask


class StatsBackedStandardizer(BaseStandardizer):
    """Shared loader for the new parquet-based standardization outputs."""

    def __init__(
        self,
        *,
        spec: str,
        modality_dir: str | Path,
        channels_per_image: pd.DataFrame,
        disable_quantile_mask: bool = False,
        filter_factory: FilterFactory | None = None,
        quantile_level: str = "image",
        stats_level: str = "global",
        verbose: bool = True,
        use_mean_std: bool = True,
    ) -> None:
        self.spec = spec.strip().strip("/")
        self.method = Path(self.spec).parts[0]
        super().__init__(
            name=self.spec,
            modality_dir=modality_dir,
            channels_per_image=channels_per_image,
            disable_quantile_mask=disable_quantile_mask,
            filter_factory=filter_factory,
            quantile_level=quantile_level,
            stats_level=stats_level,
            verbose=verbose,
        )
        self.use_mean_std = use_mean_std

        self.stats_dir = self._resolve_stats_dir()
        self.upper_quantiles = self._load_table(
            self.stats_dir / f"{self.quantile_level}_level_upper_quantiles.parquet"
        )
        self.lower_quantiles = self._load_optional_table(
            self.stats_dir / f"{self.quantile_level}_level_lower_quantiles.parquet"
        )
        if use_mean_std:
            self.means_df: pd.DataFrame | None = self._load_table(
                self.stats_dir / f"{self.stats_level}_level_means.parquet"
            )
            self.stds_df: pd.DataFrame | None = self._load_table(
                self.stats_dir / f"{self.stats_level}_level_stds.parquet"
            )
        else:
            self.means_df = None
            self.stds_df = None

        self._stats_cache: dict[tuple, tuple[torch.Tensor, ...]] = {}
        self.valid_mask_df = self._build_valid_mask()

        if self.verbose:
            logger.info(
                f"Loaded standardization stats '{self.spec}' from {self.stats_dir} "
                f"(quantile_level={self.quantile_level}, stats_level={self.stats_level}, "
                f"use_mean_std={self.use_mean_std})"
            )

    def _resolve_stats_dir(self) -> Path:
        direct = self.modality_dir / "standardization" / Path(self.spec)
        if direct.exists():
            return direct

        nested = self.modality_dir / "standardization" / self.method / Path(self.spec)
        if nested.exists():
            return nested

        raise FileNotFoundError(
            f"Could not resolve standardization stats directory for '{self.spec}'. "
            f"Tried {direct} and {nested}."
        )

    def _load_table(self, path: Path) -> pd.DataFrame:
        if not path.exists():
            raise FileNotFoundError(f"Expected stats file at {path}")
        df = pd.read_parquet(path)
        return df.reindex(columns=self.channels_per_image.columns)

    def _load_optional_table(self, path: Path) -> pd.DataFrame | None:
        if not path.exists():
            return None
        return self._load_table(path)

    def _expand_to_image_space(self, table: pd.DataFrame | None) -> pd.DataFrame | None:
        if table is None:
            return None

        aligned = table.reindex(columns=self.channels_per_image.columns)
        target_index = self.channels_per_image.index

        if len(aligned) == 1 and aligned.index.tolist() != list(target_index):
            row = aligned.iloc[0].to_numpy()
            repeated = np.repeat(row[None, :], len(target_index), axis=0)
            return pd.DataFrame(repeated, index=target_index, columns=aligned.columns)

        return aligned.reindex(index=target_index)

    def _build_valid_mask(self) -> pd.DataFrame:
        upper = self._expand_to_image_space(self.upper_quantiles)
        lower = self._expand_to_image_space(self.lower_quantiles)

        assert upper is not None
        valid = upper.notna() & np.isfinite(upper) & (upper > 0)
        if lower is not None:
            valid &= lower.notna() & np.isfinite(lower)

        if self.use_mean_std:
            means = self._expand_to_image_space(self.means_df)
            stds = self._expand_to_image_space(self.stds_df)
            assert means is not None
            assert stds is not None
            valid &= means.notna() & np.isfinite(means)
            valid &= stds.notna() & np.isfinite(stds) & (stds > 0)

        return valid

    @abstractmethod
    def _transform(
        self,
        x_t: torch.Tensor,
        upper_t: torch.Tensor,
        lower_t: torch.Tensor,
    ) -> torch.Tensor:
        raise NotImplementedError

    def _get_cached_stats(
        self,
        tissue_id: str,
        measured_mask: np.ndarray,
        refined: np.ndarray,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
        cache_key = (tissue_id, measured_mask.tobytes(), refined.tobytes())
        if cache_key in self._stats_cache:
            return self._stats_cache[cache_key]

        combined = measured_mask.copy()
        combined[measured_mask] = refined

        upper_vals = self._table_row(self.upper_quantiles, tissue_id)[combined]
        lower_vals = (
            self._table_row(self.lower_quantiles, tissue_id)[combined]
            if self.lower_quantiles is not None
            else np.zeros_like(upper_vals)
        )

        if self.use_mean_std:
            mean_vals = self._table_row(self.means_df, tissue_id)[combined]
            std_vals = self._table_row(self.stds_df, tissue_id)[combined]
            stacked = np.stack([upper_vals, lower_vals, mean_vals, std_vals]).astype(
                np.float32, copy=False
            )  # (4, C)
            stats_t = torch.from_numpy(stacked).unsqueeze(-1).unsqueeze(-1)  # (4, C, 1, 1)
            upper_t, lower_t, mean_t, std_t = stats_t.unbind(0)
        else:
            stacked = np.stack([upper_vals, lower_vals]).astype(np.float32, copy=False)  # (2, C)
            stats_t = torch.from_numpy(stacked).unsqueeze(-1).unsqueeze(-1)  # (2, C, 1, 1)
            upper_t, lower_t = stats_t.unbind(0)
            mean_t, std_t = None, None

        result = (upper_t, lower_t, mean_t, std_t)
        self._stats_cache[cache_key] = result
        return result

    def apply(
        self,
        x: np.ndarray | torch.Tensor,
        tissue_id: str,
        measured_mask: np.ndarray,
        selected_mask: np.ndarray,
    ) -> Tuple[torch.Tensor, np.ndarray]:
        x_t = self._ensure_tensor(x)
        x_t, refined = self._refine_and_slice(x_t, tissue_id, measured_mask, selected_mask)

        upper_t, lower_t, mean_t, std_t = self._get_cached_stats(
            tissue_id, measured_mask, refined
        )

        x_t = self._transform(x_t, upper_t, lower_t)
        if self.use_mean_std:
            x_t = (x_t - mean_t) / (std_t + 1e-8)
        x_t = self._apply_optional_filters(x_t)
        return x_t, refined


class QuantileClippingStandardizer(StatsBackedStandardizer):
    def _transform(
        self,
        x_t: torch.Tensor,
        upper_t: torch.Tensor,
        lower_t: torch.Tensor,
    ) -> torch.Tensor:
        x_t = torch.clamp(x_t, min=lower_t, max=upper_t)
        return (x_t - lower_t) / (upper_t - lower_t + 1e-8)


class QuantileClippingLog1PStandardizer(StatsBackedStandardizer):
    def _transform(
        self,
        x_t: torch.Tensor,
        upper_t: torch.Tensor,
        lower_t: torch.Tensor,
    ) -> torch.Tensor:
        x_t = torch.clamp(x_t, min=lower_t, max=upper_t)
        return torch.log1p(x_t)


def build_standardizer(
    *,
    standardization: str,
    modality_dir: str | Path,
    channels_per_image: pd.DataFrame,
    disable_quantile_mask: bool = False,
    filter_factory: FilterFactory | None = None,
    quantile_level: str = "image",
    stats_level: str = "global",
    verbose: bool = True,
    use_mean_std: bool = True,
) -> BaseStandardizer:
    spec = standardization.strip().strip("/")
    kwargs_common = dict(
        modality_dir=modality_dir,
        channels_per_image=channels_per_image,
        disable_quantile_mask=disable_quantile_mask,
        filter_factory=filter_factory,
        quantile_level=quantile_level,
        stats_level=stats_level,
        verbose=verbose,
    )

    if spec == "identity":
        return IdentityStandardizer(**kwargs_common)

    method = Path(spec).parts[0]
    if method == "quantile_clipping":
        return QuantileClippingStandardizer(spec=spec, use_mean_std=use_mean_std, **kwargs_common)
    if method == "quantile_clipping_log1p":
        return QuantileClippingLog1PStandardizer(spec=spec, use_mean_std=use_mean_std, **kwargs_common)

    raise NotImplementedError(
        f"Standardization '{standardization}' not implemented for the new stats layout."
    )