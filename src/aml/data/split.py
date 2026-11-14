"""Half-open temporal periods; attempts are attributed by start time."""

from typing import Literal

import numpy as np
import pandas as pd
from pydantic import BaseModel, model_validator

from aml.data.patterns_parser import Attempt

Period = Literal["train", "validation", "test"]
SECONDS_PER_DAY = 86_400


class TemporalSplit(BaseModel):
    start: int
    train_end: int
    validation_end: int
    end: int
    method: str

    @model_validator(mode="after")
    def ordered(self) -> "TemporalSplit":
        if not self.start < self.train_end < self.validation_end < self.end:
            raise ValueError("split boundaries must be strictly ordered")
        return self

    def bounds(self, period: Period) -> tuple[int, int]:
        return {"train": (self.start, self.train_end),
                "validation": (self.train_end, self.validation_end),
                "test": (self.validation_end, self.end)}[period]

    def period_of(self, timestamp: int) -> Period:
        if not self.start <= timestamp < self.end:
            raise ValueError("timestamp outside split")
        return "train" if timestamp < self.train_end else "validation" if timestamp < self.validation_end else "test"


def make_split(ts: np.ndarray, fractions: tuple[float, float, float],
               *, method: Literal["whole_days", "timestamp_quantiles"] = "whole_days") -> TemporalSplit:
    if len(ts) == 0 or any(f <= 0 for f in fractions) or not np.isclose(sum(fractions), 1):
        raise ValueError("invalid timestamps or split fractions")
    unique = np.unique(ts)
    if len(unique) < 3:
        raise ValueError("need at least three distinct timestamps")
    first_day, last_day = int(unique[0]) // SECONDS_PER_DAY, int(unique[-1]) // SECONDS_PER_DAY
    days = last_day - first_day + 1
    if days >= 3 and method == "whole_days":
        train_days = min(max(1, int(days * fractions[0])), days - 2)
        validation_days = min(max(1, int(days * fractions[1])), days - train_days - 1)
        boundaries = [(first_day + train_days) * SECONDS_PER_DAY,
                      (first_day + train_days + validation_days) * SECONDS_PER_DAY]
        if all(np.any((ts >= lo) & (ts < hi)) for lo, hi in zip(
                [first_day * SECONDS_PER_DAY, *boundaries], [*boundaries, (last_day + 1) * SECONDS_PER_DAY])):
            return TemporalSplit(start=first_day * SECONDS_PER_DAY, train_end=boundaries[0],
                                 validation_end=boundaries[1], end=(last_day + 1) * SECONDS_PER_DAY,
                                 method="whole_days")
    ordered = np.sort(ts)
    cut1 = int(ordered[min(len(ordered) - 1, int(len(ordered) * fractions[0]))])
    cut2 = int(ordered[min(len(ordered) - 1, int(len(ordered) * sum(fractions[:2])))])
    # Shift boundaries only when timestamp ties would empty a period.
    i = min(max(1, int(np.searchsorted(unique, cut1))), len(unique) - 2)
    j = min(max(i + 1, int(np.searchsorted(unique, cut2))), len(unique) - 1)
    return TemporalSplit(start=int(unique[0]), train_end=int(unique[i]), validation_end=int(unique[j]),
                         end=int(unique[-1]) + 1, method="timestamp_quantiles")


def select_period(frame: pd.DataFrame, split: TemporalSplit, period: Period,
                  *, for_tuning: bool = False) -> pd.DataFrame:
    if for_tuning:
        assert period != "test", "test period access forbidden during tuning"
    start, end = split.bounds(period)
    return frame.loc[(frame.ts >= start) & (frame.ts < end)].copy()


def train_attempts(attempts: list[Attempt], split: TemporalSplit) -> list[Attempt]:
    """Use fully contained train attempts for fitting; exclude straddlers."""
    return [a for a in attempts if split.start <= a.t_start <= a.t_end < split.train_end]
