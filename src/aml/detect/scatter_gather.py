"""Direct two-stage motifs; no dependency on fan hits or fan thresholds."""
import logging
from collections import defaultdict
import numpy as np
from pydantic import BaseModel, Field
from aml.graph.build import TemporalGraph
from aml.data.fx import FXModel, normalized_amounts
from aml.detect.common import Hit, make_hit

logger = logging.getLogger(__name__)


class ScatterConfig(BaseModel):
    window_seconds: int = Field(ge=0)
    m: int = Field(ge=2)
    m_in: int = Field(ge=2)
    m_out: int = Field(ge=2)
    rho_lo: float = Field(gt=0, le=1)
    eps: float = Field(ge=0)
    fx: FXModel | None = None


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
        for first in graph.out_idx[int(source)]:
            middle = int(graph.dst[first])
            if middle == source:
                continue
            following = graph.out_idx[middle]
            lo = np.searchsorted(following, first, side="right")
            hi = np.searchsorted(graph.ts[following], graph.ts[first] + config.window_seconds, side="right")
            for second in following[lo:hi]:
                target = int(graph.dst[second])
                if target in (source, middle):
                    continue
                missing = not (np.isfinite(received[first]) and np.isfinite(paid[second]))
                if missing:
                    missing_tests += 1
                elif received[first] <= 0 or not config.rho_lo <= paid[second] / received[first] <= 1 + config.eps:
                    continue
                paths[target].append((int(first), int(second), middle, missing))
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
    # Gather-scatter uses aggregate conservation, with causality checked at each
    # outgoing event. Stages may overlap; outgoing value cannot use future inflow.
    hubs = np.flatnonzero((np.diff(graph.in_idx.indptr) >= config.m_in) &
                         (np.diff(graph.out_idx.indptr) >= config.m_out))
    for hub in hubs:
        incoming = graph.in_idx[int(hub)]
        incoming = incoming[~graph.is_self_loop[incoming]]
        outgoing = graph.out_idx[int(hub)]
        outgoing = outgoing[~graph.is_self_loop[outgoing]]
        for end in outgoing:
            lower = graph.ts[end] - config.window_seconds
            ins = incoming[(incoming < end) & (graph.ts[incoming] >= lower)]
            outs = outgoing[(outgoing <= end) & (graph.ts[outgoing] >= lower)]
            if len(ins) == 0:
                continue
            outs = outs[outs > ins[0]]
            if len(np.unique(graph.src[ins])) < config.m_in or len(np.unique(graph.dst[outs])) < config.m_out:
                continue
            missing = not (np.isfinite(received[ins]).all() and np.isfinite(paid[outs]).all())
            if missing:
                missing_tests += 1
            else:
                total_in, total_out = received[ins].sum(), paid[outs].sum()
                if total_in <= 0 or not config.rho_lo <= total_out / total_in <= 1 + config.eps:
                    continue
                available = np.r_[0., np.cumsum(received[ins])][np.searchsorted(ins, outs)]
                if np.any(np.cumsum(paid[outs]) > available * (1 + config.eps)):
                    continue
            emit(np.r_[ins, outs].tolist(), "gather_scatter", {"hub": int(hub), "fx_missing": missing,
                "stages": {"placement": graph.tx_id[ins].tolist(), "integration": graph.tx_id[outs].tolist()}})
    if missing_tests:
        logger.warning("fx_missing amount tests skipped: %d", missing_tests)
    return sorted(hits, key=lambda h: (h.t_start, h.typology, h.edge_ids))
