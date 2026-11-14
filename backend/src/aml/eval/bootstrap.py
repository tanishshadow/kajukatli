"""Attempt-dependent cluster bootstrap for account-level headline metrics."""
import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from aml.data.patterns_parser import Attempt


def dependent_blocks(population: np.ndarray, attempts: list[Attempt], edges: list[tuple[int, int]]) -> list[np.ndarray]:
    population = np.asarray(population, dtype=np.int64)
    index = {int(node): i for i, node in enumerate(population)}
    parent = np.arange(len(population))
    def root(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return int(i)
    for attempt in attempts:
        ids = [index[a] for a in attempt.accounts if a in index]
        for i in ids[1:]:
            parent[root(i)] = root(ids[0])
    for a, b in edges:
        if a in index and b in index:
            parent[root(index[a])] = root(index[b])
    groups: dict[int, list[int]] = {}
    for i, account in enumerate(population):
        groups.setdefault(root(i), []).append(int(account))
    return [np.asarray(groups[key], dtype=np.int64) for key in sorted(groups)]


def bootstrap_account_f1(population: np.ndarray, truth: np.ndarray, prediction: np.ndarray,
                         blocks: list[np.ndarray], *, replicates: int, seed: int,
                         confidence: float) -> dict[str, float]:
    lookup = {int(a): i for i, a in enumerate(population)}
    rng = np.random.default_rng(seed)
    values = np.empty(replicates)
    for r in range(replicates):
        sampled = rng.integers(0, len(blocks), size=len(blocks))
        ids = np.concatenate([blocks[i] for i in sampled])
        ix = np.array([lookup[int(a)] for a in ids])
        y, p = truth[ix].astype(bool), prediction[ix].astype(bool)
        tp, fp, fn = np.sum(y & p), np.sum(~y & p), np.sum(y & ~p)
        values[r] = 2 * tp / max(1, 2 * tp + fp + fn)
    alpha = (1 - confidence) / 2
    return {"f1": float(2 * np.sum(truth.astype(bool) & prediction.astype(bool)) /
                         max(1, 2*np.sum(truth.astype(bool) & prediction.astype(bool)) +
                             np.sum(~truth.astype(bool) & prediction.astype(bool)) + np.sum(truth.astype(bool) & ~prediction.astype(bool)))),
            "lower": float(np.quantile(values, alpha)), "upper": float(np.quantile(values, 1-alpha)),
            "blocks": len(blocks), "replicates": replicates}
