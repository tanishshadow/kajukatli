"""Period-local labels and an explicit, conservative typology coverage map."""

import numpy as np
import pandas as pd

from aml.data.patterns_parser import Attempt
from aml.data.split import Period, TemporalSplit

TYPOLOGY_MAPPING: dict[str, dict[str, str | None]] = {
    "FAN-OUT": {"detector": "fan_out", "coverage": "direct"},
    "FAN-IN": {"detector": "fan_in", "coverage": "direct"},
    "SCATTER-GATHER": {"detector": "scatter_gather", "coverage": "direct"},
    "GATHER-SCATTER": {"detector": "gather_scatter", "coverage": "direct"},
    "CYCLE": {"detector": "cycle", "coverage": "direct"},
    "STACK": {"detector": "pass_through", "coverage": "partial: only qualifying temporal paths"},
    "BIPARTITE": {"detector": None, "coverage": "not covered"},
    "RANDOM": {"detector": None, "coverage": "not covered"},
}


def account_labels(frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Fixed period population: all accounts touching any period transaction."""
    accounts = np.union1d(frame.src.to_numpy(), frame.dst.to_numpy())
    positive = frame.loc[frame.label == 1]
    positive_accounts = np.union1d(positive.src.to_numpy(), positive.dst.to_numpy())
    return accounts, np.isin(accounts, positive_accounts)


def period_counts(frame: pd.DataFrame, attempts: list[Attempt], split: TemporalSplit,
                  period: Period) -> dict[str, object]:
    start, end = split.bounds(period)
    local = frame.loc[(frame.ts >= start) & (frame.ts < end)]
    accounts, labels = account_labels(local)
    positives = int(local.label.sum())
    return {"transactions": len(local), "positive_transactions": positives,
            "positive_rate": positives / len(local) if len(local) else None,
            "accounts": len(accounts), "positive_accounts": int(labels.sum()),
            "attempts": sum(start <= a.t_start < end for a in attempts),
            "attempts_by_typology": {name: sum(a.typology == name and start <= a.t_start < end for a in attempts)
                                     for name in sorted({a.typology for a in attempts})}}
