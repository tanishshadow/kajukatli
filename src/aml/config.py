"""Validated configuration boundaries; derivation policies are configurable."""

from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class DataSettings(StrictModel):
    transactions: Path = Path("HI-Small_Trans.csv")
    patterns: Path = Path("HI-Small_Patterns.txt")
    processed_dir: Path = Path("data/processed")
    chunk_rows: int = Field(default=100_000, gt=0)


class DerivationSettings(StrictModel):
    duration_quantile: float = Field(default=0.9, ge=0, le=1)
    size_quantile: float = Field(default=0.25, ge=0, le=1)
    fallback_window_fraction: float = Field(default=0.25, gt=0, le=1)
    merge_window_fraction: float = Field(default=0.25, gt=0, le=1)
    minimum_counterparties: int = Field(default=2, ge=2)
    minimum_window_seconds: int = Field(default=1, ge=1)
    degree_quantiles: list[float] = [0.0, 0.25, 0.5, 0.75, 0.9, 0.99, 1.0]
    regularity_bins: int = Field(default=6, ge=2)
    active_bin_fraction: float = Field(default=0.8, gt=0, le=1)
    maximum_count_cv: float = Field(default=0.5, ge=0)


class Settings(StrictModel):
    seed: int = 42
    data: DataSettings = Field(default_factory=DataSettings)
    output_dir: Path = Path("outputs")
    split_fractions: tuple[float, float, float] = (0.6, 0.2, 0.2)
    split_method: Literal["whole_days", "timestamp_quantiles"] = "whole_days"
    derivation: DerivationSettings = Field(default_factory=DerivationSettings)
    detectors: dict[str, Any] = Field(default_factory=dict)
    baseline_grid: dict[str, Any] = Field(default_factory=dict)
    evaluation: dict[str, Any] = Field(default_factory=dict)
    scoring: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_fractions(self) -> "Settings":
        if any(x <= 0 for x in self.split_fractions) or abs(sum(self.split_fractions) - 1) > 1e-9:
            raise ValueError("split fractions must be positive and sum to one")
        if any(not 0 <= q <= 1 for q in self.derivation.degree_quantiles):
            raise ValueError("degree quantiles must lie in [0, 1]")
        return self

    @classmethod
    def from_yaml(cls, path: Path) -> "Settings":
        return cls.model_validate(yaml.safe_load(path.read_text(encoding="utf-8")))
