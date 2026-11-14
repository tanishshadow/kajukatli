from collections import Counter
import numpy as np
from pydantic import BaseModel, Field
from aml.graph.build import TemporalGraph
from aml.data.fx import FXModel
from aml.detect.common import Hit, make_hit


class FanConfig(BaseModel):
    window_seconds: int = Field(ge=0)
    k: int = Field(ge=2)
    fx: FXModel | None = None


def detect_fans(graph: TemporalGraph, config: FanConfig) -> list[Hit]:
    hits: list[Hit] = []
    for direction, index, counterparties in (("fan_out", graph.out_idx, graph.dst), ("fan_in", graph.in_idx, graph.src)):
        for node in np.flatnonzero(np.diff(index.indptr) >= config.k):
            edges = index[int(node)]
            edges = edges[~graph.is_self_loop[edges]]
            counts: Counter[int] = Counter()
            left = 0
            intervals: list[list[int]] = []
            for right, edge in enumerate(edges):
                counts[int(counterparties[edge])] += 1
                while graph.ts[edge] - graph.ts[edges[left]] > config.window_seconds:
                    other = int(counterparties[edges[left]])
                    counts[other] -= 1
                    if counts[other] == 0:
                        del counts[other]
                    left += 1
                if len(counts) >= config.k:
                    if intervals and graph.ts[edges[left]] <= graph.ts[edges[intervals[-1][1]]]:
                        intervals[-1][1] = right
                    else:
                        intervals.append([left, right])
            for start, end in intervals:
                members = edges[start:end + 1]
                hits.append(make_hit(graph, members, direction, {"hub": int(node),
                    "distinct_counterparties": len(np.unique(counterparties[members])),
                    "window_seconds": config.window_seconds}, config.fx))
    return sorted(hits, key=lambda hit: (hit.t_start, hit.typology, hit.edge_ids))
