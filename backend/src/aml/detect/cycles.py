"""Earliest-edge anchored Numba DFS with overlapping-window SCC prefilter."""
import logging
import numpy as np
from numba import njit
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from pydantic import BaseModel, Field
from aml.graph.build import TemporalGraph
from aml.data.fx import FXModel, normalized_amounts
from aml.detect.common import Hit, make_hit

logger = logging.getLogger(__name__)


class CycleConfig(BaseModel):
    window_seconds: int = Field(gt=0)
    max_length: int = Field(ge=3)
    rho_lo: float = Field(gt=0, le=1)
    eps: float = Field(ge=0)
    deg_cap: int = Field(gt=0)
    expansion_budget: int = Field(gt=0)
    fx: FXModel | None = None


@njit
def _search(src: np.ndarray, dst: np.ndarray, ts: np.ndarray, paid: np.ndarray,
            ptr: np.ndarray, adj: np.ndarray, components: np.ndarray, sizes: np.ndarray,
            begin: int, end: int, window: int, length: int, cap: int, budget: int,
            rho: float, eps: float) -> tuple:
    results = [np.empty(0, dtype=np.int64)]
    results.pop()
    budget_starts = [np.int64(-1)]
    budget_starts.pop()
    skipped = np.zeros(len(ptr) - 1, dtype=np.bool_)
    excluded = np.zeros(len(src), dtype=np.bool_)
    missing_count = 0
    for anchor in range(begin, end):
        origin = src[anchor]
        if origin == dst[anchor] or components[origin] != components[dst[anchor]] or sizes[components[origin]] < 3:
            continue
        path = np.empty(length, dtype=np.int64)
        cursors = np.empty(length, dtype=np.int64)
        path[0] = anchor
        cursors[0] = ptr[dst[anchor]]
        depth = 1
        expansions = 0
        deadline = ts[anchor] + window
        while depth > 0:
            previous = path[depth - 1]
            node = dst[previous]
            lo = np.searchsorted(ts[adj[ptr[node]:ptr[node + 1]]], ts[anchor]) + ptr[node]
            hi = np.searchsorted(ts[adj[ptr[node]:ptr[node + 1]]], deadline, side="right") + ptr[node]
            if hi - lo > cap:
                skipped[node] = True
                for k in range(lo, hi):
                    excluded[adj[k]] = True
                depth -= 1
                continue
            cursor = max(cursors[depth - 1], lo)
            descended = False
            while cursor < hi:
                edge = adj[cursor]
                cursor += 1
                cursors[depth - 1] = cursor
                if edge <= previous:
                    continue
                expansions += 1
                if expansions > budget:
                    budget_starts.append(anchor)
                    depth = 0
                    break
                target = dst[edge]
                if target == node or components[target] != components[origin]:
                    continue
                missing = not (np.isfinite(paid[previous]) and np.isfinite(paid[edge]))
                if missing:
                    missing_count += 1
                elif paid[previous] <= 0 or not rho <= paid[edge] / paid[previous] <= 1 + eps:
                    continue
                if target == origin:
                    if depth + 1 >= 3:
                        cycle = np.empty(depth + 1, dtype=np.int64)
                        cycle[:depth] = path[:depth]
                        cycle[depth] = edge
                        results.append(cycle)
                    continue
                if depth + 1 >= length:
                    continue
                repeated = False
                for k in range(depth):
                    if dst[path[k]] == target:
                        repeated = True
                if repeated:
                    continue
                path[depth] = edge
                cursors[depth] = ptr[target]
                depth += 1
                descended = True
                break
            if depth and not descended:
                depth -= 1
    return results, budget_starts, skipped, excluded, missing_count


def detect_cycles(graph: TemporalGraph, config: CycleConfig) -> list[Hit]:
    if not graph.edge_count:
        return []
    paid, _ = normalized_amounts(graph, config.fx)
    hits: list[Hit] = []
    all_excluded = np.zeros(graph.edge_count, dtype=bool)
    for start in range(int(graph.ts[0]), int(graph.ts[-1]) + 1, config.window_seconds):
        begin, end, stop = np.searchsorted(graph.ts, [start, start + config.window_seconds, start + 2 * config.window_seconds], side="left")
        stop = int(np.searchsorted(graph.ts, start + 2 * config.window_seconds, side="right"))
        indices = np.arange(begin, stop)
        indices = indices[~graph.is_self_loop[indices]]
        adjacency = coo_matrix((np.ones(len(indices)), (graph.src[indices], graph.dst[indices])),
                               shape=(graph.node_count, graph.node_count)).tocsr()
        count, components = connected_components(adjacency, directed=True, connection="strong")
        results, budgets, skipped, excluded, missing = _search(graph.src, graph.dst, graph.ts, paid,
            graph.out_idx.indptr, graph.out_idx.edge_ids, components, np.bincount(components, minlength=count),
            int(begin), int(end), config.window_seconds, config.max_length, config.deg_cap,
            config.expansion_budget, config.rho_lo, config.eps)
        for node in np.flatnonzero(skipped):
            rows = graph.out_idx[int(node)]
            positives = int(graph.label[rows[excluded[rows]]].sum())
            logger.warning("deg_cap skipped account=%d window_start=%d excluded_positives=%d", node, start, positives)
        all_excluded |= excluded
        for anchor in budgets:
            logger.warning("cycle expansion_budget triggered tx_id=%d budget=%d", graph.tx_id[anchor], config.expansion_budget)
        if missing:
            logger.warning("cycle fx_missing amount tests skipped=%d", missing)
        hits.extend(make_hit(graph, edges, "cycle", {"length": len(edges), "anchor_tx_id": int(graph.tx_id[edges[0]]),
                    "fx_missing": bool(np.isnan(paid[edges]).any())}, config.fx) for edges in results)
    positives = int(graph.label[all_excluded].sum())
    logger.warning("cycle deg_cap unique excluded positive edges=%d total_positive_edges=%d conservative_recall_ceiling=%s",
                   positives, int(graph.label.sum()), 1 - positives / graph.label.sum() if graph.label.sum() else None)
    return sorted(hits, key=lambda h: (h.t_start, h.edge_ids))
