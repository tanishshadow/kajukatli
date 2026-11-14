import logging
from collections import defaultdict
import numpy as np
from pydantic import BaseModel, Field
from aml.detect.common import Hit
from aml.graph.build import TemporalGraph

logger = logging.getLogger(__name__)


class MergeConfig(BaseModel):
    window_seconds: int = Field(ge=0)
    hub_degree: int = Field(gt=0)
    max_accounts: int = Field(ge=2)


class EvidenceCluster(BaseModel):
    accounts: list[int]
    edge_ids: list[int]
    hit_ids: list[int]
    t_start: int
    t_end: int


def merge_hits(graph: TemporalGraph, hits: list[Hit], config: MergeConfig) -> list[EvidenceCluster]:
    parent = list(range(len(hits)))
    def root(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i
    degree = np.diff(graph.out_idx.indptr) + np.diff(graph.in_idx.indptr)
    by_account: dict[int, list[int]] = defaultdict(list)
    for i, hit in enumerate(hits):
        for node in hit.accounts:
            if degree[node] < config.hub_degree:
                by_account[node].append(i)
    for indices in by_account.values():
        indices.sort(key=lambda i: (hits[i].t_start, i))
        frontier = indices[0]
        for i in indices[1:]:
            if hits[i].t_start <= hits[frontier].t_end + config.window_seconds:
                a, b = root(i), root(frontier)
                parent[max(a, b)] = min(a, b)
                if hits[i].t_end > hits[frontier].t_end:
                    frontier = i
            else:
                frontier = i
    components: dict[int, list[int]] = defaultdict(list)
    for i in range(len(hits)):
        components[root(i)].append(i)
    output: list[EvidenceCluster] = []
    for indices in components.values():
        nodes = {a for i in indices for a in hits[i].accounts}
        groups = [indices]
        if len(nodes) > config.max_accounts:
            logger.warning("merge max_accounts exceeded accounts=%d hits=%d; retaining separate hits", len(nodes), len(indices))
            groups = [[i] for i in indices]
        for group in groups:
            output.append(EvidenceCluster(accounts=sorted({a for i in group for a in hits[i].accounts}),
                edge_ids=sorted({e for i in group for e in hits[i].edge_ids}), hit_ids=group,
                t_start=min(hits[i].t_start for i in group), t_end=max(hits[i].t_end for i in group)))
    return sorted(output, key=lambda c: (c.t_start, c.edge_ids))
