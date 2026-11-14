from collections import Counter
from typing import Sequence
import numpy as np
import pandas as pd
from pydantic import BaseModel, Field
from aml.graph.build import TemporalGraph
from aml.detect.common import Hit
from aml.data.fx import FXModel, normalized_amounts

TYPOLOGIES = ("fan_out", "fan_in", "scatter_gather", "gather_scatter", "cycle", "pass_through")


class FeatureConfig(BaseModel):
    size_bins: list[int]
    regularity_bins: int = Field(ge=2)


def features_for_groups(graph: TemporalGraph, groups: Sequence[Sequence[int]], hit_groups: Sequence[list[Hit]],
                         banks: dict[int, str], config: FeatureConfig, fx: FXModel | None = None) -> pd.DataFrame:
    paid, _ = normalized_amounts(graph, fx)
    degrees = np.diff(graph.out_idx.indptr) + np.diff(graph.in_idx.indptr)
    activity_range = (int(graph.ts[0]), int(graph.ts[-1]) + 1) if graph.edge_count else (0, 1)
    rows: list[dict[str, float]] = []
    for ids, hits in zip(groups, hit_groups, strict=True):
        positions = graph.positions_for_tx_ids(np.array(ids, dtype=np.int64))
        positions.sort()
        nodes = np.union1d(graph.src[positions], graph.dst[positions])
        times = graph.ts[positions]
        duration = int(times[-1] - times[0]) if len(times) else 0
        value = float(np.nansum(paid[positions]))
        counts = Counter(h.typology for h in hits)
        row = {f"pattern_{t}": float(counts[t]) for t in TYPOLOGIES}
        for typology in TYPOLOGIES:
            for size in config.size_bins:
                row[f"pattern_{typology}_size_ge_{size}"] = float(sum(h.typology == typology and len(h.accounts) >= size for h in hits))
        local = {int(graph.tx_id[e]): int(e) for e in positions}
        flow_dwells: list[int] = []
        for hit in hits:
            path = sorted(local[e] for e in hit.edge_ids if e in local)
            flow_dwells.extend(int(graph.ts[b] - graph.ts[a]) for a, b in zip(path[:-1], path[1:])
                               if graph.dst[a] == graph.src[b])
        gaps = np.array(flow_dwells)
        activity = np.histogram(times, bins=config.regularity_bins, range=activity_range)[0] if len(times) else np.zeros(config.regularity_bins)
        row.update({"distinct_typologies": float(len(counts)), "accounts": float(len(nodes)), "edges": float(len(positions)),
            "total_value": value, "value_per_hour": value * 3600 / max(duration, 1), "activity_span": float(duration),
            "max_fan_degree": float(max((h.meta.get("distinct_counterparties", 0) for h in hits), default=0)),
            "cycle_length": float(max((h.meta.get("length", 0) for h in hits if h.typology == "cycle"), default=0)),
            "chain_length": float(max((h.meta.get("links", 0) for h in hits if h.typology == "pass_through"), default=0)),
            "cross_bank_hops": float(sum(banks.get(int(graph.src[e]), "") != banks.get(int(graph.dst[e]), "") for e in positions)),
            "cross_currency_hops": float(np.sum(graph.cur_out[positions] != graph.cur_in[positions])),
            "dwell_mean": float(gaps.mean()) if len(gaps) else 0., "dwell_max": float(gaps.max()) if len(gaps) else 0.,
            "degree_mean": float(degrees[nodes].mean()) if len(nodes) else 0., "degree_max": float(degrees[nodes].max()) if len(nodes) else 0.,
            "active_bin_fraction": float(np.mean(activity > 0)), "count_cv": float(activity.std() / activity.mean()) if activity.mean() else 0.,
            "fx_missing_fraction": float(np.isnan(paid[positions]).mean()) if len(positions) else 0.})
        rows.append(row)
    return pd.DataFrame(rows).astype(np.float32)
