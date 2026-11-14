from typing import Any
import numpy as np
from sklearn.metrics import average_precision_score

from aml.eval.schemas import BinaryMetrics, Cluster
from aml.graph.build import TemporalGraph
from aml.data.patterns_parser import Attempt
from aml.data.split import TemporalSplit, Period


def binary_metrics(truth: np.ndarray, predicted: np.ndarray, scores: np.ndarray) -> BinaryMetrics:
    if truth.shape != predicted.shape or truth.shape != scores.shape or truth.ndim != 1:
        raise ValueError("metric arrays must be aligned vectors")
    if not np.isin(truth, [0, 1]).all() or not np.isin(predicted, [0, 1]).all() or not np.isfinite(scores).all():
        raise ValueError("invalid metric labels/scores")
    y, p = truth.astype(bool), predicted.astype(bool)
    tp, fp, fn, tn = (int(x.sum()) for x in (y & p, ~y & p, y & ~p, ~y & ~p))
    precision, recall = tp / (tp + fp) if tp + fp else 0., tp / (tp + fn) if tp + fn else 0.
    return BinaryMetrics(tp=tp, fp=fp, fn=fn, tn=tn, precision=precision, recall=recall,
        f1=2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.,
        pr_auc=float(average_precision_score(y, scores)) if y.any() else None)


def cluster_metrics(graph: TemporalGraph, clusters: list[Cluster], tau: float, k: int) -> dict[str, Any]:
    ranked = sorted(clusters, key=lambda c: (-c.score, c.edge_ids))
    positive = [float(graph.label[graph.positions_for_tx_ids(np.array(c.edge_ids, dtype=np.int64))].mean()) >= tau
                for c in ranked if c.edge_ids]
    return {"precision": float(np.mean(positive)) if positive else None,
            "precision_at_k": float(np.mean(positive[:k])) if positive else None,
            "alert_volume": len(positive), "true_positive_clusters": sum(positive)}


def attempt_recall(attempts: list[Attempt], clusters: list[Cluster], tau: float) -> dict[str, Any]:
    accounts = [set(c.accounts) for c in clusters]
    result: dict[str, Any] = {}
    for attempt in attempts:
        group = result.setdefault(attempt.typology, {"attempts": 0, "detected": 0})
        group["attempts"] += 1
        group["detected"] += int(any(len(set(attempt.accounts) & nodes) >= tau * len(attempt.accounts) for nodes in accounts))
    for group in result.values():
        group["recall"] = group["detected"] / group["attempts"]
    return result


def evaluate(graph: TemporalGraph, clusters: list[Cluster], attempts: list[Attempt], split: TemporalSplit,
             period: Period, tau: float, tau_a: float, k: int,
             edge_scores: np.ndarray | None = None, account_scores: np.ndarray | None = None,
             cutoff: float = 1.) -> dict[str, Any]:
    start, end = split.bounds(period)
    mask = (graph.ts >= start) & (graph.ts < end)
    population = np.union1d(graph.src[mask], graph.dst[mask])
    positive = np.union1d(graph.src[mask & (graph.label == 1)], graph.dst[mask & (graph.label == 1)])
    es = np.zeros(graph.edge_count) if edge_scores is None else edge_scores
    acs = np.zeros(graph.node_count) if account_scores is None else account_scores
    local: list[Cluster] = []
    for cluster in clusters:
        positions = graph.positions_for_tx_ids(np.array(cluster.edge_ids, dtype=np.int64))
        kept = positions[mask[positions]]
        if len(kept):
            local.append(cluster.model_copy(update={"edge_ids": graph.tx_id[kept].tolist()}))
        if edge_scores is None:
            np.maximum.at(es, positions, cluster.score)
        if account_scores is None:
            np.maximum.at(acs, np.array(cluster.accounts, dtype=int), cluster.score)
    selected_attempts = [a for a in attempts if start <= a.t_start < end]
    return {"period": period, "bounds": [start, end], "edge": binary_metrics(graph.label[mask], es[mask] >= cutoff, es[mask]).model_dump(),
        "account": binary_metrics(np.isin(population, positive), acs[population] >= cutoff, acs[population]).model_dump(),
        "cluster": cluster_metrics(graph, local, tau, k), "attempt": attempt_recall(selected_attempts, clusters, tau_a),
        "positive_edges": int(graph.label[mask].sum()), "positive_accounts": len(positive), "account_population": len(population)}


def choose_cutoff(truth: np.ndarray, scores: np.ndarray) -> tuple[float, dict[str, Any]]:
    if truth.shape != scores.shape or truth.ndim != 1 or not np.isfinite(scores).all():
        raise ValueError("cutoff inputs must be aligned finite vectors")
    candidates = np.unique(np.r_[0., scores, 1.])
    trials = [(float(c), binary_metrics(truth, scores >= c, scores)) for c in candidates]
    cutoff, metrics = max(trials, key=lambda pair: (pair[1].f1, pair[0]))
    return cutoff, {"cutoff": cutoff, "metrics": metrics.model_dump(), "candidates": len(candidates)}
