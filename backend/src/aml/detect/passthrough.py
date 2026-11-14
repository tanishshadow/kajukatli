"""Numba link search and DP-pruned enumeration of maximal temporal paths."""
import logging
import numpy as np
from numba import njit
from pydantic import BaseModel, Field
from aml.graph.build import TemporalGraph
from aml.data.fx import FXModel, normalized_amounts
from aml.detect.common import Hit, make_hit

logger = logging.getLogger(__name__)


class PassConfig(BaseModel):
    dwell_seconds: int = Field(ge=0)
    rho_lo: float = Field(gt=0, le=1)
    eps: float = Field(ge=0)
    top_m: int = Field(gt=0)
    min_links: int = Field(ge=1)
    max_paths_per_component: int = Field(gt=0)
    fx: FXModel | None = None


@njit
def _links(src: np.ndarray, dst: np.ndarray, ts: np.ndarray, paid: np.ndarray, received: np.ndarray,
           ptr: np.ndarray, adj: np.ndarray, dwell: int, rho: float, eps: float, top_m: int) -> tuple:
    links = np.full((len(src), top_m), -1, dtype=np.int64)
    truncated = np.zeros(len(src), dtype=np.int64)
    parent = np.arange(len(src))
    indegree = np.zeros(len(src), dtype=np.int64)
    missing = 0
    for first in range(len(src)):
        if src[first] == dst[first]:
            continue
        node = dst[first]
        candidates = adj[ptr[node]:ptr[node + 1]]
        lo = np.searchsorted(candidates, first, side="right")
        hi = np.searchsorted(ts[candidates], ts[first] + dwell, side="right")
        chosen = [np.int64(-1)]
        chosen.pop()
        errors = [0.]
        errors.pop()
        for second in candidates[lo:hi]:
            if src[second] == dst[second]:
                continue
            unknown = not (np.isfinite(received[first]) and np.isfinite(paid[second]))
            error = 0.
            if unknown:
                missing += 1
            else:
                if received[first] <= 0:
                    continue
                ratio = paid[second] / received[first]
                if not rho <= ratio <= 1 + eps:
                    continue
                error = abs(ratio - 1)
            pos = len(chosen)
            for k in range(len(chosen)):
                if (ts[second], error, second) < (ts[chosen[k]], errors[k], chosen[k]):
                    pos = k
                    break
            chosen.insert(pos, second)
            errors.insert(pos, error)
            if len(chosen) > top_m:
                chosen.pop()
                errors.pop()
                truncated[first] += 1
        for k, second in enumerate(chosen):
            links[first, k] = second
            indegree[second] += 1
            a, b = first, second
            while parent[a] != a:
                parent[a] = parent[parent[a]]
                a = parent[a]
            while parent[b] != b:
                parent[b] = parent[parent[b]]
                b = parent[b]
            if a != b:
                parent[max(a, b)] = min(a, b)
    for edge in range(len(src)):
        node = edge
        while parent[node] != node:
            node = parent[node]
        parent[edge] = node
    longest = np.ones(len(src), dtype=np.int64)
    for edge in range(len(src) - 1, -1, -1):
        for child in links[edge]:
            if child >= 0:
                longest[edge] = max(longest[edge], 1 + longest[child])
    return links, indegree, parent, longest, truncated, missing


def detect_passthrough(graph: TemporalGraph, config: PassConfig) -> list[Hit]:
    paid, received = normalized_amounts(graph, config.fx)
    links, indegree, components, longest, truncated, missing = _links(graph.src, graph.dst, graph.ts,
        paid, received, graph.out_idx.indptr, graph.out_idx.edge_ids, config.dwell_seconds,
        config.rho_lo, config.eps, config.top_m)
    for edge in np.flatnonzero(truncated):
        logger.warning("pass_through top_m truncated tx_id=%d matches_dropped=%d", graph.tx_id[edge], truncated[edge])
    if missing:
        logger.warning("pass_through fx_missing amount tests skipped=%d", missing)
    hits: list[Hit] = []
    emitted: dict[int, int] = {}
    capped: set[int] = set()
    for root in np.flatnonzero((indegree == 0) & (longest > config.min_links)):
        component = int(components[root])
        stack = [(int(root), [])]
        while stack:
            node, prefix = stack.pop()
            if emitted.get(component, 0) >= config.max_paths_per_component:
                if component not in capped:
                    logger.warning("pass_through max_paths_per_component triggered component=%d cap=%d", component, config.max_paths_per_component)
                    capped.add(component)
                break
            path = prefix + [node]
            children = links[node][links[node] >= 0]
            if not len(children):
                if len(path) > config.min_links:
                    emitted[component] = emitted.get(component, 0) + 1
                    hits.append(make_hit(graph, path, "pass_through", {"links": len(path) - 1,
                        "likely_mules": graph.dst[np.array(path[:-1])].tolist(),
                        "fx_missing": bool(np.isnan(paid[path]).any() or np.isnan(received[path]).any())}, config.fx))
            else:
                for child in reversed(children):
                    if len(path) + longest[child] > config.min_links:
                        stack.append((int(child), path))
    return sorted(hits, key=lambda h: (h.t_start, h.edge_ids))
