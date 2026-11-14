"""Train-only implied currency rates, shared by amount-aware detectors."""
from collections import deque
import numpy as np
from pydantic import BaseModel
from aml.graph.build import TemporalGraph


class FXModel(BaseModel):
    base_currency: str
    factors: dict[str, float]
    fit_max_ts: int


def fit_fx(graph: TemporalGraph, train_end: int) -> FXModel:
    mask = graph.ts < train_end
    if not mask.any():
        raise ValueError("no train transactions for FX")
    names = graph.currency_names
    counts = np.bincount(np.r_[graph.cur_out[mask], graph.cur_in[mask]], minlength=len(names))
    base = int(np.argmax(counts))  # alphabetical tie break from graph vocabulary
    paired = mask & (graph.cur_in != graph.cur_out) & (graph.amt_out > 0) & (graph.amt_in > 0)
    adjacency: dict[int, dict[int, float]] = {i: {} for i in range(len(names))}
    for source, target in sorted(set(zip(graph.cur_out[paired].tolist(), graph.cur_in[paired].tolist()))):
        rows = paired & (graph.cur_out == source) & (graph.cur_in == target)
        adjacency[source][target] = float(np.median(graph.amt_in[rows].astype(float) / graph.amt_out[rows]))
    for source in sorted(adjacency):
        for target, rate in list(adjacency[source].items()):
            adjacency[target].setdefault(source, 1 / rate)
    factors = {base: 1.}
    queue = deque([base])
    while queue:
        source = queue.popleft()
        for target, rate in sorted(adjacency[source].items()):
            if target not in factors:
                factors[target] = factors[source] / rate
                queue.append(target)
    return FXModel(base_currency=names[base], factors={names[k]: v for k, v in factors.items()},
                   fit_max_ts=int(graph.ts[mask].max()))


def normalized_amounts(graph: TemporalGraph, fx: FXModel | None) -> tuple[np.ndarray, np.ndarray]:
    factors = np.array([fx.factors.get(name, np.nan) if fx else (1. if len(graph.currency_names) == 1 else np.nan)
                        for name in graph.currency_names])
    return graph.amt_out.astype(float) * factors[graph.cur_out], graph.amt_in.astype(float) * factors[graph.cur_in]
