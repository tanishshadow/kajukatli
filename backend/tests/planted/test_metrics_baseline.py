import numpy as np
import pytest

from aml.eval.metrics import binary_metrics, cluster_metrics, attempt_recall, evaluate
from aml.eval.schemas import Cluster
from aml.baseline.threshold import baseline, BaselineConfig, tune_baseline
from aml.data.patterns_parser import Attempt
from aml.data.split import TemporalSplit
from aml.graph.build import build_graph
from tests_helpers import frame


def test_confusion_matrix_and_average_precision() -> None:
    result = binary_metrics(np.array([1, 1, 0, 0]), np.array([1, 0, 1, 0]), np.array([.9, .4, .8, .1]))
    assert (result.tp, result.fp, result.fn, result.tn) == (1, 1, 1, 1)
    assert result.precision == result.recall == result.f1 == .5
    assert result.pr_auc == pytest.approx((1 + 2 / 3) / 2)
    assert binary_metrics(np.array([0]), np.array([0]), np.array([0.])).pr_auc is None


def test_cluster_precision_and_attempt_single_cluster_coverage() -> None:
    g = build_graph(frame([(0, 1, 0, 10, 1), (1, 2, 1, 10, 0), (3, 4, 2, 10, 0)]))
    clusters = [Cluster(edge_ids=[0, 1], accounts=[0, 1, 2], score=.8),
                Cluster(edge_ids=[2], accounts=[3, 4], score=.2)]
    result = cluster_metrics(g, clusters, tau=.5, k=1)
    assert result == {"precision": .5, "precision_at_k": 1., "alert_volume": 2, "true_positive_clusters": 1}
    attempt = Attempt(attempt_id="P", typology="CYCLE", description="", edge_ids=[0, 1], accounts=[0, 1, 2, 3], t_start=0, t_end=1)
    assert attempt_recall([attempt], clusters, .75)["CYCLE"]["recall"] == 1
    assert attempt_recall([attempt], [Cluster(edge_ids=[0], accounts=[0, 1], score=1), Cluster(edge_ids=[2], accounts=[2, 3], score=1)], .75)["CYCLE"]["recall"] == 0


def test_baseline_rolling_window_not_lifetime_count_and_near_miss() -> None:
    g = build_graph(frame([(0, 1, 0, 2, 0), (0, 2, 3, 2, 1), (0, 3, 20, 2, 0)]))
    cfg = BaselineConfig(amount=100, count=2, volume=100, window_seconds=5)
    result = baseline(g, cfg)
    assert result.account_pred.tolist() == [True, False, False, False]
    assert result.edge_pred.tolist() == [True, True, True]
    assert len(result.clusters) > 0
    miss = baseline(g, cfg.model_copy(update={"count": 3}))
    assert not miss.edge_pred.any()


def test_tuning_rejects_test_or_validation_input() -> None:
    g = build_graph(frame([(0, 1, 0, 10, 0), (0, 2, 3, 10, 1), (0, 3, 8, 10, 0)]))
    split = TemporalSplit(start=0, train_end=3, validation_end=6, end=10, method="fixture")
    cfg = BaselineConfig(amount=100, count=2, volume=100, window_seconds=5)
    with pytest.raises(AssertionError, match="train"):
        tune_baseline(g, split, [cfg])


def test_common_evaluation_population_and_period_for_both_systems() -> None:
    g = build_graph(frame([(0, 1, 0, 10, 1), (2, 3, 3, 10, 0), (0, 4, 8, 10, 1)]))
    split = TemporalSplit(start=0, train_end=3, validation_end=6, end=10, method="fixture")
    clusters = [Cluster(edge_ids=[0], accounts=[0, 1], score=1)]
    main = evaluate(g, clusters, [], split, "train", .5, .5, 1)
    baseline_report = evaluate(g, clusters, [], split, "train", .5, .5, 1,
        np.array([1., 0., 0.]), np.array([1., 1., 0., 0., 0.]))
    assert main == baseline_report
    validation = evaluate(g, clusters, [], split, "validation", .5, .5, 1)
    assert validation["edge"]["tn"] == 1 and validation["positive_accounts"] == 0


def test_baseline_currency_volumes_do_not_mix_and_parallel_edges_count() -> None:
    rows = frame([(0, 1, 0, 60, 0), (0, 1, 0, 60, 1), (2, 2, 2, 10, 1)])
    rows.loc[1, ["cur_out", "cur_in"]] = "EUR"
    g = build_graph(rows)
    cfg = BaselineConfig(amount=1000, count=3, volume=100, window_seconds=10)
    assert not baseline(g, cfg).edge_pred.any()
    result = baseline(g, cfg.model_copy(update={"count": 2}))
    assert result.edge_pred.tolist() == [True, True, False]
