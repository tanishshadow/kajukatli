from typing import Mapping
from pydantic import BaseModel, Field


class RiskConfig(BaseModel):
    weights: dict[str, float]
    active_bin_fraction: float = Field(ge=0, le=1)
    maximum_count_cv: float = Field(ge=0)
    steady_activity_factor: float = Field(ge=0, le=1)
    top_features: int = Field(gt=0)
    export_limit: int = Field(gt=0)


def transparent_score(features: Mapping[str, float], config: RiskConfig) -> float:
    value = min(100., sum(config.weights[t] * features.get(f"pattern_{t}", 0.) for t in config.weights))
    if features.get("active_bin_fraction", 0) >= config.active_bin_fraction and features.get("count_cv", 0) <= config.maximum_count_cv:
        value *= config.steady_activity_factor
    return max(0., value)
