"""Train-only scorer grid, validation-only cutoff, and validation ablations."""
import csv
import json
from pathlib import Path
import numpy as np
import pandas as pd
from xgboost import XGBClassifier
from aml.config import Settings
from aml.data.split import TemporalSplit
from aml.eval.metrics import binary_metrics, choose_cutoff


def tune(settings: Settings) -> dict:
    split = TemporalSplit.model_validate_json((settings.data.processed_dir / "split.json").read_text())
    parquet = settings.output_dir / "m8" / "cluster_features.parquet"
    ranked = [json.loads(line) for line in (settings.output_dir / "m8" / "ranked_clusters.jsonl").open()]
    features = pd.read_parquet(parquet)
    ends = np.array([row["t_end"] for row in ranked])
    labels = np.array([0] * len(ranked), dtype=np.int8)
    # Recover labels from period-local edges in compact Parquet; never labels from scores.
    import pyarrow.parquet as pq
    graph = pq.read_table(settings.data.processed_dir / "transactions.parquet",
                          filters=[("ts", "<", split.validation_end)]).to_pandas()
    labels_by_tx = dict(zip(graph.tx_id.astype(int), graph.label.astype(int)))
    for i, row in enumerate(ranked):
        if ends[i] < split.validation_end:
            member_labels = [labels_by_tx[e] for e in row["edge_ids"] if e in labels_by_tx]
            labels[i] = bool(member_labels) and np.mean(member_labels) >= settings.evaluation["cluster_tau"]
    train_mask = (ends >= split.start) & (ends < split.train_end)
    val_mask = (ends >= split.train_end) & (ends < split.validation_end)
    assert not np.any((ends[train_mask] >= split.validation_end) | (ends[train_mask] >= split.train_end)), "test leakage in tuning inputs"
    if not train_mask.any() or not val_mask.any():
        raise ValueError("train or validation clusters missing")
    ytrain, yval = labels[train_mask], labels[val_mask]
    if len(np.unique(ytrain)) < 2:
        raise ValueError("train clusters need both classes")
    grid = settings.scoring["tuning_grid"]
    columns = list(features.columns)
    trials = []
    output = settings.output_dir / "m9"
    output.mkdir(parents=True, exist_ok=True)
    for n_estimators in grid["n_estimators"]:
        for max_depth in grid["max_depth"]:
            model = XGBClassifier(n_estimators=n_estimators, max_depth=max_depth,
                learning_rate=settings.scoring["model"]["learning_rate"], n_jobs=settings.scoring["model"]["n_jobs"],
                random_state=settings.seed, tree_method="hist", objective="binary:logistic", eval_metric="aucpr",
                scale_pos_weight=float((ytrain == 0).sum() / max(1, (ytrain == 1).sum())))
            model.fit(features.loc[train_mask], ytrain)
            probabilities = model.predict_proba(features.loc[val_mask])[:, 1]
            metrics = binary_metrics(yval, probabilities >= .5, probabilities)
            trials.append({"n_estimators": n_estimators, "max_depth": max_depth, **metrics.model_dump()})
            model.save_model(output / f"model_{n_estimators}_{max_depth}.json")
    best = max(trials, key=lambda t: (t["f1"], t["pr_auc"] or 0, -t["n_estimators"], -t["max_depth"]))
    best_model = XGBClassifier()
    best_model.load_model(output / f"model_{best['n_estimators']}_{best['max_depth']}.json")
    val_scores = best_model.predict_proba(features.loc[val_mask])[:, 1]
    cutoff, cutoff_report = choose_cutoff(yval, val_scores)
    ablations = {}
    for name in ("patterns_only", "no_pattern_features", "no_legitimacy_features"):
        if name == "patterns_only":
            mask_cols = [c.startswith("pattern_") for c in columns]
        elif name == "no_pattern_features":
            mask_cols = [not c.startswith("pattern_") for c in columns]
        else:
            mask_cols = [c not in {"active_bin_fraction", "count_cv", "activity_span"} for c in columns]
        model = XGBClassifier(n_estimators=best["n_estimators"], max_depth=best["max_depth"],
            learning_rate=settings.scoring["model"]["learning_rate"], n_jobs=settings.scoring["model"]["n_jobs"],
            random_state=settings.seed, tree_method="hist", objective="binary:logistic", eval_metric="aucpr",
            scale_pos_weight=float((ytrain == 0).sum() / max(1, (ytrain == 1).sum())))
        model.fit(features.loc[train_mask].loc[:, np.array(mask_cols)], ytrain)
        pred = model.predict_proba(features.loc[val_mask].loc[:, np.array(mask_cols)])[:, 1]
        ablations[name] = binary_metrics(yval, pred >= cutoff, pred).model_dump()
    report = {"command": "aml tune", "trials": trials, "best": best,
        "cutoff_selected_on": "validation", "cutoff": cutoff_report,
        "validation_ablations": ablations, "train_rows": int(train_mask.sum()),
        "validation_rows": int(val_mask.sum()), "test_access": False,
        "scope": "cluster ranking tuning; account scorer retained from M8 unless separately tuned"}
    (output / "tuning.json").write_text(json.dumps(report, indent=2))
    with (output / "trials.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(trials[0])); writer.writeheader(); writer.writerows(trials)
    (output / "frozen.yaml").write_text(json.dumps({"model": {"n_estimators": best["n_estimators"],
        "max_depth": best["max_depth"]}, "cutoff": cutoff, "seed": settings.seed}, indent=2))
    return report
