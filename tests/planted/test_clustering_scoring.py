import numpy as np
import pytest
from aml.cluster.merge import merge_hits, MergeConfig
from aml.detect.common import make_hit
from aml.graph.build import build_graph
from aml.score.features import features_for_groups, FeatureConfig
from aml.score.model import train_model, probabilities, ModelConfig
from tests_helpers import frame


def test_tiny_end_to_end_ranked_alerts_have_real_probabilities(tmp_path) -> None:
    import json
    import pandas as pd
    from aml.config import Settings
    from aml.data.split import TemporalSplit
    from aml.score.run import run
    from aml.api.schemas import Alert
    settings = Settings.from_yaml(__import__("pathlib").Path("configs/default.yaml"))
    settings.data.processed_dir = tmp_path / "data" / "processed"
    settings.output_dir = tmp_path / "outputs"
    settings.scoring["model"]["n_estimators"] = 5
    settings.scoring["risk"]["export_limit"] = 5
    directory = settings.data.processed_dir
    directory.mkdir(parents=True)
    edges = [(i * 3, i * 3 + 1 + j, i * 2 + j, 100. if i % 3 == 0 else 1., int(i % 3 == 0))
             for i in range(150) for j in range(2)]
    rows = frame(edges)
    rows.to_parquet(directory / "transactions.parquet", index=False)
    pd.DataFrame({"account_id": range(450), "bank": ["001"] * 450, "account": [str(i) for i in range(450)]}).to_parquet(directory / "accounts.parquet", index=False)
    (directory / "split.json").write_text(TemporalSplit(start=0, train_end=180, validation_end=240, end=301, method="fixture").model_dump_json())
    graph = build_graph(rows)
    hits = [make_hit(graph, [2*i, 2*i+1], "fan_out", {}) for i in range(150)]
    for detector in ("fan", "scatter_gather", "cycle", "pass_through"):
        output = settings.output_dir / detector
        output.mkdir(parents=True)
        (output / "hits.json").write_text("[\n" + ",\n".join(h.model_dump_json() for h in hits if detector == "fan") + "\n]\n")
    report = run(settings)
    assert report["model_score_available"] and not report["test_evaluated"]
    alerts = [Alert.model_validate(a) for a in json.loads((settings.output_dir / "alerts.json").read_text())]
    assert len(alerts) == 5
    assert all(a.evidence["features"]["model_score_available"] and a.top_features for a in alerts)
    assert [a.risk_score for a in alerts] == sorted((a.risk_score for a in alerts), reverse=True)
    from aml.narrative.run import run as narrate
    assert narrate(settings)["model_score_available"] is True
    restored = json.loads((settings.output_dir / "alerts.json").read_text())
    assert [a["risk_score"] for a in restored] == [a.risk_score for a in alerts]


def test_merge_requires_shared_nonhub_account_and_time_and_size_guard(caplog) -> None:
    g = build_graph(frame([(0, 1, 0, 10, 1), (1, 2, 1, 10, 1), (2, 3, 100, 10, 0)]))
    hits = [make_hit(g, [i], "fan_out", {}) for i in range(3)]
    cfg = MergeConfig(window_seconds=5, hub_degree=10, max_accounts=10)
    clusters = merge_hits(g, hits, cfg)
    assert [c.edge_ids for c in clusters] == [[0, 1], [2]]
    assert len(merge_hits(g, hits, cfg.model_copy(update={"hub_degree": 2}))) == 3
    assert len(merge_hits(g, hits, cfg.model_copy(update={"max_accounts": 2}))) == 3
    assert "max_accounts" in caplog.text


def test_features_do_not_use_labels_and_model_blocks_nontrain() -> None:
    rows = frame([(0, 1, 0, 10, 1), (1, 2, 1, 10, 1), (3, 4, 2, 1, 0), (4, 5, 3, 1, 0)])
    g = build_graph(rows)
    cfg = FeatureConfig(size_bins=[2, 4], regularity_bins=3)
    groups = [[0, 1], [2, 3]]
    x = features_for_groups(g, groups, [[], []], {}, cfg)
    rows.label = 1 - rows.label
    y = features_for_groups(build_graph(rows), groups, [[], []], {}, cfg)
    np.testing.assert_array_equal(x.values, y.values)
    config = ModelConfig(n_estimators=5, max_depth=2, learning_rate=.2, n_jobs=1, seed=42)
    with pytest.raises(AssertionError, match="train"):
        train_model(x, np.array([1, 0]), np.array([1, 10]), train_end=5, config=config)
    a = train_model(x, np.array([1, 0]), np.array([1, 3]), train_end=5, config=config)
    b = train_model(x, np.array([1, 0]), np.array([1, 3]), train_end=5, config=config)
    np.testing.assert_array_equal(probabilities(a, x), probabilities(b, x))
