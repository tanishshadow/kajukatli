"""Train-only baseline grid; descriptive train/validation evaluation only."""
import json
from itertools import product
from pathlib import Path
import pandas as pd
import pyarrow.parquet as pq

from aml.config import Settings
from aml.baseline.threshold import BaselineConfig, baseline, tune_baseline
from aml.data.patterns_parser import Attempt
from aml.data.split import TemporalSplit
from aml.graph.build import build_graph, load_graph
from aml.eval.metrics import evaluate


def run(settings: Settings) -> dict:
    directory = settings.data.processed_dir
    split = TemporalSplit.model_validate_json((directory / "split.json").read_text())
    # Arrow predicate excludes validation/test rows before any tuning input exists.
    train = build_graph(pq.read_table(directory / "transactions.parquet", filters=[("ts", ">=", split.start), ("ts", "<", split.train_end)]).to_pandas())
    grid = settings.baseline_grid
    candidates = [BaselineConfig(amount=a, count=n, volume=s, window_seconds=w,
                                cluster_stride_seconds=grid["cluster_stride_seconds"])
                  for a, n, s, w in product(grid["amount"], grid["count"], grid["volume"], grid["window_seconds"])]
    selected, trials = tune_baseline(train, split, candidates)
    del train
    graph = load_graph(directory / "transactions.parquet", directory / "accounts.parquet")
    predictions = baseline(graph, selected)
    attempts = [Attempt.model_validate({key: row[key] for key in Attempt.model_fields})
                for row in pq.read_table(directory / "attempts.parquet").to_pylist()]
    evaluation = settings.evaluation
    report = {"selected": selected.model_dump(), "trials": len(trials),
        "currency_policy": "transaction threshold in paid native units; rolling volume separately per currency",
        "cluster_policy": "connected flagged edges in overlapping rolling windows on a fixed stride; identical edge sets deduplicated",
        "prediction_policy": "transaction rule flags both endpoints; rolling rule flags its account; incident edges flagged without recursively flagging their other endpoints",
        "detector_scope": "fixed baseline runs on full timeline, as required; only metrics are period-restricted",
        "test_evaluated": False,
        "metrics": {period: evaluate(graph, predictions.clusters, attempts, split, period,
            evaluation["cluster_tau"], evaluation["attempt_tau"], evaluation["precision_k"],
            predictions.edge_scores, predictions.account_scores) for period in ("train", "validation")}}
    output = settings.output_dir / "m3"
    output.mkdir(parents=True, exist_ok=True)
    (output / "baseline.json").write_text(json.dumps(report, indent=2) + "\n")
    pd.DataFrame(trials).to_csv(output / "baseline_trials.csv", index=False)
    (output / "report.md").write_text("# M3 baseline\n\nCommand: `aml baseline`\n\n```json\n" + json.dumps(report, indent=2) + "\n```\n")
    return report
