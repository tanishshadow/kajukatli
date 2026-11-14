from aml.data.patterns_parser import Attempt
from aml.eval.report_metrics import attempt_coverage_report, cluster_precision_summary, straddling_attempts


def test_attempt_edge_and_account_coverage_sensitivity_and_not_covered_typology():
    attempts = [
        Attempt(attempt_id="C1", typology="CYCLE", description="", edge_ids=[1, 2, 3],
                accounts=[10, 11, 12], t_start=100, t_end=200),
        Attempt(attempt_id="B1", typology="BIPARTITE", description="", edge_ids=[4, 5],
                accounts=[20, 21], t_start=100, t_end=200),
    ]
    report = attempt_coverage_report(attempts, {1, 2}, {10, 11}, (.3, .5, .7))
    cycle = report["CYCLE"]["thresholds"]
    assert [(row["edge_recall"], row["account_recall"]) for row in cycle] == [(1.0, 1.0), (1.0, 1.0), (0.0, 0.0)]
    assert report["BIPARTITE"]["status"] == "not covered"
    assert all(row["edge_recall"] is None for row in report["BIPARTITE"]["thresholds"])


def test_lists_attempts_crossing_period_boundary():
    attempts = [Attempt(attempt_id="S", typology="STACK", description="", edge_ids=[1],
        accounts=[1, 2], t_start=9, t_end=11)]
    assert straddling_attempts(attempts, (10, 20)) == [{"attempt_id": "S", "typology": "STACK",
        "t_start": 9, "t_end": 11, "boundaries": [10]}]


def test_cluster_report_has_precision_only_and_threshold_sensitivity():
    metrics = cluster_precision_summary([.8, .4, .1], [.9, .8, .1], cutoff=.5, taus=(.3, .5, .7), k=2)
    assert metrics["0.5"]["precision"] == .5
    assert metrics["0.7"]["precision"] == .5
    assert metrics["0.3"]["precision_at_k"] == 1.0
    assert all("recall" not in row and "tn" not in row for row in metrics.values())
