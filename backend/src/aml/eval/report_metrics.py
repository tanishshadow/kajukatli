"""Pure attempt-coverage summaries shared by reports and planted tests."""
from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Set
from typing import Any

from aml.data.patterns_parser import Attempt

SUPPORTED_TYPOLOGIES = {"FAN-OUT", "FAN-IN", "SCATTER-GATHER", "GATHER-SCATTER", "CYCLE", "STACK"}


def attempt_coverage_report(attempts: list[Attempt], alerted_edges: Set[int],
                            alerted_accounts: Set[int], taus: Iterable[float] = (.3, .5, .7)) -> dict[str, Any]:
    thresholds = tuple(float(tau) for tau in taus)
    if any(not 0 < tau <= 1 for tau in thresholds):
        raise ValueError("attempt coverage thresholds must be in (0, 1]")
    by_typology: dict[str, list[Attempt]] = defaultdict(list)
    for attempt in attempts:
        by_typology[attempt.typology.upper()].append(attempt)
    result: dict[str, Any] = {}
    for typology, members in sorted(by_typology.items()):
        covered = typology in SUPPORTED_TYPOLOGIES
        rows = []
        for tau in thresholds:
            edge_detected = sum(bool(a.edge_ids) and len(set(a.edge_ids) & alerted_edges) / len(a.edge_ids) >= tau for a in members)
            account_detected = sum(bool(a.accounts) and len(set(a.accounts) & alerted_accounts) / len(a.accounts) >= tau for a in members)
            rows.append({"tau": tau, "edge_detected": edge_detected if covered else None,
                "edge_recall": edge_detected / len(members) if covered and members else None,
                "account_detected": account_detected if covered else None,
                "account_recall": account_detected / len(members) if covered and members else None})
        result[typology] = {"status": "covered" if covered else "not covered", "attempts": len(members), "thresholds": rows}
    return result


def straddling_attempts(attempts: list[Attempt], boundaries: Iterable[int]) -> list[dict[str, Any]]:
    edges = sorted(set(int(boundary) for boundary in boundaries))
    result = []
    for attempt in attempts:
        crossed = [boundary for boundary in edges if attempt.t_start < boundary <= attempt.t_end]
        if crossed:
            result.append({"attempt_id": attempt.attempt_id, "typology": attempt.typology,
                "t_start": attempt.t_start, "t_end": attempt.t_end, "boundaries": crossed})
    return result


def cluster_precision_summary(positive_fractions: list[float], scores: list[float], cutoff: float,
                             taus: Iterable[float] = (.3, .5, .7), k: int = 20) -> dict[str, Any]:
    if len(positive_fractions) != len(scores) or k < 1:
        raise ValueError("cluster fractions/scores must align and k must be positive")
    thresholds = tuple(float(tau) for tau in taus)
    if any(not 0 < tau <= 1 for tau in thresholds):
        raise ValueError("cluster precision thresholds must be in (0, 1]")
    selected = [i for i, score in enumerate(scores) if score >= cutoff]
    top = sorted(range(len(scores)), key=lambda i: (-scores[i], i))[:k]
    result = {}
    for tau in thresholds:
        result[str(tau)] = {
            "alert_count": len(selected),
            "precision": (sum(positive_fractions[i] >= tau for i in selected) / len(selected)) if selected else 0.0,
            "precision_at_k": (sum(positive_fractions[i] >= tau for i in top) / len(top)) if top else 0.0,
            "top_k": len(top),
        }
    return result
