from typing import Any, Literal
import numpy as np
from pydantic import BaseModel, ConfigDict, Field
from aml.graph.build import TemporalGraph
from aml.data.fx import FXModel, normalized_amounts

Typology = Literal["fan_out", "fan_in", "scatter_gather", "gather_scatter", "cycle", "pass_through"]


class Hit(BaseModel):
    model_config = ConfigDict(extra="forbid")
    typology: Typology
    accounts: list[int]
    edge_ids: list[int]  # original transaction IDs, never CSR positions
    t_start: int
    t_end: int
    total_value: float = Field(ge=0)
    meta: dict[str, Any]


def make_hit(graph: TemporalGraph, positions: list[int] | np.ndarray, typology: Typology,
             meta: dict[str, Any], fx: FXModel | None = None) -> Hit:
    edges = np.unique(positions).astype(np.int64)
    # Normalize just the hit values, avoiding a full-edge allocation per hit.
    factors = np.array([fx.factors.get(name, np.nan) if fx else (1. if len(graph.currency_names) == 1 else np.nan)
                        for name in graph.currency_names])
    values = graph.amt_out[edges].astype(float) * factors[graph.cur_out[edges]]
    missing = int(np.isnan(values).sum())
    return Hit(typology=typology, accounts=np.union1d(graph.src[edges], graph.dst[edges]).tolist(),
        edge_ids=graph.tx_id[edges].tolist(), t_start=int(graph.ts[edges[0]]), t_end=int(graph.ts[edges[-1]]),
        total_value=float(np.nansum(values)), meta={**meta, "value_currency": fx.base_currency if fx else graph.currency_names[0],
            "fx_missing": bool(meta.get("fx_missing", False) or missing), "unconverted_edges": missing,
            "total_value_is_partial": missing > 0,
            "native_totals": {graph.currency_names[int(c)]: float(graph.amt_out[edges[graph.cur_out[edges] == c]].astype(float).sum())
                              for c in np.unique(graph.cur_out[edges])}})
