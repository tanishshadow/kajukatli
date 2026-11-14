"""Direct two-stage motifs; no dependency on fan hits or fan thresholds."""
import logging
from collections import defaultdict
import numpy as np
from numba import njit
from pydantic import BaseModel, Field
from aml.graph.build import TemporalGraph
from aml.data.fx import FXModel, normalized_amounts
from aml.detect.common import Hit, make_hit

logger = logging.getLogger(__name__)


@njit
def _source_paths(source: int, src: np.ndarray, dst: np.ndarray, ts: np.ndarray,
                  ptr: np.ndarray, adj: np.ndarray, paid: np.ndarray, received: np.ndarray,
                  window: int, rho: float, eps: float) -> tuple:
    paths = [(np.int64(0), np.int64(0), np.int64(0), np.int64(0), False)]
    paths.pop()
    missing_tests = 0
    for first in adj[ptr[source]:ptr[source + 1]]:
        middle = dst[first]
        if middle == source:
            continue
        following = adj[ptr[middle]:ptr[middle + 1]]
        lo = np.searchsorted(following, first, side="right")
        hi = np.searchsorted(ts[following], ts[first] + window, side="right")
        for second in following[lo:hi]:
            target = dst[second]
            if target == source or target == middle:
                continue
            missing = not (np.isfinite(received[first]) and np.isfinite(paid[second]))
            if missing:
                missing_tests += 1
            elif received[first] <= 0 or not rho <= paid[second] / received[first] <= 1 + eps:
                continue
            paths.append((target, first, second, middle, missing))
    return paths, missing_tests


class ScatterConfig(BaseModel):
    window_seconds: int = Field(ge=0)
    m: int = Field(ge=2)
    m_in: int = Field(ge=2)
    m_out: int = Field(ge=2)
    rho_lo: float = Field(gt=0, le=1)
    eps: float = Field(ge=0)
    fx: FXModel | None = None


@njit
def _gather_paths(hubs: np.ndarray, src: np.ndarray, dst: np.ndarray, ts: np.ndarray,
                  in_ptr: np.ndarray, in_adj: np.ndarray, out_ptr: np.ndarray, out_adj: np.ndarray,
                  paid: np.ndarray, received: np.ndarray, window: int, m_in: int, m_out: int,
                  rho: float, eps: float) -> tuple:
    matches = [(np.int64(0), np.empty(0, np.int64), np.empty(0, np.int64), False)]
    matches.pop()
    missing_tests = 0
    for hub in hubs:
        incoming = in_adj[in_ptr[hub]:in_ptr[hub + 1]]
        incoming = incoming[src[incoming] != dst[incoming]]
        outgoing = out_adj[out_ptr[hub]:out_ptr[hub + 1]]
        outgoing = outgoing[src[outgoing] != dst[outgoing]]
        outgoing_times = ts[outgoing]
        if len(np.unique(src[incoming])) < m_in or len(np.unique(dst[outgoing])) < m_out:
            continue
        pi = np.zeros(len(incoming) + 1)
        po = np.zeros(len(outgoing) + 1)
        ni = np.zeros(len(incoming) + 1, np.int64)
        no = np.zeros(len(outgoing) + 1, np.int64)
        for k, edge in enumerate(incoming):
            known = np.isfinite(received[edge])
            pi[k + 1] = pi[k] + (received[edge] if known else 0.)
            ni[k + 1] = ni[k] + (not known)
        for k, edge in enumerate(outgoing):
            known = np.isfinite(paid[edge])
            po[k + 1] = po[k] + (paid[edge] if known else 0.)
            no[k + 1] = no[k] + (not known)
        imbalance = np.empty(len(outgoing))
        for k, edge in enumerate(outgoing):
            imbalance[k] = po[k + 1] - (1 + eps) * pi[np.searchsorted(incoming, edge)]
        maximum_queue = np.empty(len(outgoing), np.int64)
        head = tail = 0
        inc = {np.int32(-1): 0}
        outc = {np.int32(-1): 0}
        del inc[np.int32(-1)]
        del outc[np.int32(-1)]
        il = ih = ol = oh = 0
        for end_index, end in enumerate(outgoing):
            lower = ts[end] - window
            new_ih = np.searchsorted(incoming, end)
            while ih < new_ih:
                node = src[incoming[ih]]
                inc[node] = inc.get(node, 0) + 1
                ih += 1
            while il < ih and ts[incoming[il]] < lower:
                node = src[incoming[il]]
                inc[node] -= 1
                if inc[node] == 0:
                    del inc[node]
                il += 1
            if il == ih:
                continue
            new_ol = max(np.searchsorted(outgoing_times, lower), np.searchsorted(outgoing, incoming[il], side="right"))
            while oh <= end_index:
                node = dst[outgoing[oh]]
                outc[node] = outc.get(node, 0) + 1
                while tail > head and imbalance[maximum_queue[tail - 1]] <= imbalance[oh]:
                    tail -= 1
                maximum_queue[tail] = oh
                tail += 1
                oh += 1
            while ol < new_ol:
                node = dst[outgoing[ol]]
                outc[node] -= 1
                if outc[node] == 0:
                    del outc[node]
                ol += 1
            while head < tail and maximum_queue[head] < ol:
                head += 1
            if len(inc) < m_in or len(outc) < m_out:
                continue
            missing = (ni[ih] != ni[il]) or (no[oh] != no[ol])
            if missing:
                missing_tests += 1
            else:
                total_in, total_out = pi[ih] - pi[il], po[oh] - po[ol]
                if total_in <= 0 or not rho <= total_out / total_in <= 1 + eps:
                    continue
                if imbalance[maximum_queue[head]] > po[ol] - (1 + eps) * pi[il]:
                    continue
            matches.append((hub, incoming[il:ih].copy(), outgoing[ol:oh].copy(), missing))
    return matches, missing_tests


def detect_scatter_gather(graph: TemporalGraph, config: ScatterConfig) -> list[Hit]:
    paid, received = normalized_amounts(graph, config.fx)
    hits: list[Hit] = []
    seen: set[tuple[str, tuple[int, ...]]] = set()
    missing_tests = 0

    def emit(positions: list[int], typology: str, meta: dict) -> None:
        key = (typology, tuple(sorted(set(positions))))
        if key not in seen:
            seen.add(key)
            hits.append(make_hit(graph, np.array(key[1]), typology, meta, config.fx))

    # Enumerate direct S->M->T paths, grouped per source to bound working memory.
    for source in np.flatnonzero(np.diff(graph.out_idx.indptr) >= config.m):
        paths: dict[int, list[tuple[int, int, int, bool]]] = defaultdict(list)
        candidates, skipped = _source_paths(int(source), graph.src, graph.dst, graph.ts,
            graph.out_idx.indptr, graph.out_idx.edge_ids, paid, received, config.window_seconds,
            config.rho_lo, config.eps)
        missing_tests += skipped
        for target, first, second, middle, missing in candidates:
            paths[target].append((first, second, middle, missing))
        for target, candidates in paths.items():
            candidates.sort()
            left = 0
            for right in range(len(candidates)):
                while graph.ts[candidates[right][0]] - graph.ts[candidates[left][0]] > config.window_seconds:
                    left += 1
                active = candidates[left:right + 1]
                middles = {p[2] for p in active}
                if len(middles) >= config.m:
                    emit([edge for p in active for edge in p[:2]], "scatter_gather", {
                        "source": int(source), "sink": target, "middle_accounts": sorted(middles),
                        "stages": {"placement": sorted({int(graph.tx_id[p[0]]) for p in active}),
                                   "integration": sorted({int(graph.tx_id[p[1]]) for p in active})},
                        "fx_missing": any(p[3] for p in active)})
    logger.warning("scatter-gather direct stage complete hits=%d", len(hits))
    # Gather-scatter uses aggregate conservation, with causality checked at each
    # outgoing event. Stages may overlap; outgoing value cannot use future inflow.
    hubs = np.flatnonzero((np.diff(graph.in_idx.indptr) >= config.m_in) &
                         (np.diff(graph.out_idx.indptr) >= config.m_out))
    matches, skipped = _gather_paths(hubs, graph.src, graph.dst, graph.ts, graph.in_idx.indptr,
        graph.in_idx.edge_ids, graph.out_idx.indptr, graph.out_idx.edge_ids, paid, received,
        config.window_seconds, config.m_in, config.m_out, config.rho_lo, config.eps)
    missing_tests += skipped
    for hub, ins, outs, missing in matches:
        emit(np.r_[ins, outs].tolist(), "gather_scatter", {"hub": int(hub), "fx_missing": missing,
            "stages": {"placement": graph.tx_id[ins].tolist(), "integration": graph.tx_id[outs].tolist()}})
    if missing_tests:
        logger.warning("fx_missing amount tests skipped: %d", missing_tests)
    return sorted(hits, key=lambda h: (h.t_start, h.typology, h.edge_ids))
