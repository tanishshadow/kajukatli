from typing import Any
import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from pydantic import BaseModel, ConfigDict, Field
from numba import njit

from aml.graph.build import TemporalGraph
from aml.data.split import TemporalSplit
from aml.eval.schemas import Cluster
from aml.eval.metrics import binary_metrics


class BaselineConfig(BaseModel):
    amount: float = Field(gt=0)
    count: int = Field(gt=0)
    volume: float = Field(gt=0)
    window_seconds: int = Field(gt=0)
    cluster_stride_seconds: int = Field(default=86400, gt=0)


class BaselineResult(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    edge_scores: np.ndarray
    account_scores: np.ndarray
    edge_pred: np.ndarray
    account_pred: np.ndarray
    clusters: list[Cluster]


def rolling_clusters(graph: TemporalGraph, scores: np.ndarray, config: BaselineConfig) -> list[Cluster]:
    flagged = np.flatnonzero(scores >= 1)
    if not len(flagged):
        return []
    stride = min(config.cluster_stride_seconds, config.window_seconds)
    seen: set[tuple[int, ...]] = set()
    clusters: list[Cluster] = []
    # Fixed, overlapping rolling windows. Include the prefix and final tail.
    for end in range(int(graph.ts[0]), int(graph.ts[-1]) + stride + 1, stride):
        indices = flagged[(graph.ts[flagged] >= end - config.window_seconds) & (graph.ts[flagged] <= end)]
        if not len(indices):
            continue
        nodes = np.union1d(graph.src[indices], graph.dst[indices])
        src, dst = np.searchsorted(nodes, graph.src[indices]), np.searchsorted(nodes, graph.dst[indices])
        adjacency = coo_matrix((np.ones(len(indices)), (src, dst)), shape=(len(nodes), len(nodes))).tocsr()
        _, labels = connected_components(adjacency, directed=False)
        edge_labels = labels[src]
        order = np.argsort(edge_labels, kind="stable")
        boundaries = np.r_[0, np.flatnonzero(np.diff(edge_labels[order])) + 1, len(order)]
        for lo, hi in zip(boundaries[:-1], boundaries[1:]):
            edges = np.sort(indices[order[lo:hi]])
            key = tuple(graph.tx_id[edges].tolist())
            if key in seen:
                continue
            seen.add(key)
            clusters.append(Cluster(edge_ids=list(key), accounts=np.union1d(graph.src[edges], graph.dst[edges]).tolist(),
                                    score=float(scores[edges].max())))
    return clusters


@njit
def _rolling(indptr: np.ndarray, edges: np.ndarray, ts: np.ndarray, amounts: np.ndarray,
             currencies: np.ndarray, currency_count: int, window: int) -> tuple[np.ndarray, np.ndarray]:
    counts = np.zeros(len(indptr) - 1)
    volumes = np.zeros(len(indptr) - 1)
    for node in range(len(counts)):
        left = indptr[node]
        sums = np.zeros(currency_count)
        for right in range(left, indptr[node + 1]):
            edge = edges[right]
            sums[currencies[edge]] += amounts[edge]
            while ts[edge] - ts[edges[left]] > window:
                old = edges[left]
                sums[currencies[old]] -= amounts[old]
                left += 1
            counts[node] = max(counts[node], right - left + 1)
            volumes[node] = max(volumes[node], sums[currencies[edge]])
    return counts, volumes


def baseline(graph: TemporalGraph, config: BaselineConfig, *, with_clusters: bool = True) -> BaselineResult:
    account_scores = np.zeros(graph.node_count)
    transaction_scores = graph.amt_out.astype(np.float64) / config.amount
    np.maximum.at(account_scores, graph.src, transaction_scores)
    np.maximum.at(account_scores, graph.dst, transaction_scores)
    for index, amounts, currencies, outgoing in ((graph.out_idx, graph.amt_out, graph.cur_out, True),
                                                (graph.in_idx, graph.amt_in, graph.cur_in, False)):
        counts, volumes = _rolling(index.indptr, index.edge_ids, graph.ts, amounts, currencies,
                                    len(graph.currency_names), config.window_seconds)
        if outgoing:
            np.maximum(account_scores, counts / config.count, out=account_scores)
        np.maximum(account_scores, volumes / config.volume, out=account_scores)
    edge_scores = np.maximum(transaction_scores, np.maximum(account_scores[graph.src], account_scores[graph.dst]))
    return BaselineResult(edge_scores=edge_scores, account_scores=account_scores, edge_pred=edge_scores >= 1,
                          account_pred=account_scores >= 1,
                          clusters=rolling_clusters(graph, edge_scores, config) if with_clusters else [])


def tune_baseline(graph: TemporalGraph, split: TemporalSplit, candidates: list[BaselineConfig]) -> tuple[BaselineConfig, list[dict[str, Any]]]:
    assert graph.edge_count and np.all((graph.ts >= split.start) & (graph.ts < split.train_end)), "tuning accepts train inputs only; test access forbidden"
    truth = np.zeros(graph.node_count, dtype=bool)
    truth[graph.src[graph.label == 1]] = True
    truth[graph.dst[graph.label == 1]] = True
    population = np.union1d(graph.src, graph.dst)
    trials: list[dict[str, Any]] = []
    for candidate in candidates:
        result = baseline(graph, candidate, with_clusters=False)
        metrics = binary_metrics(truth[population], result.account_pred[population], result.account_scores[population])
        trials.append({**candidate.model_dump(), **metrics.model_dump()})
    best = max(range(len(trials)), key=lambda i: (trials[i]["f1"], -i))
    return candidates[best], trials
