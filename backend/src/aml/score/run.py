"""M8 train-only model fitting, full-timeline ranking, bounded alert export."""
import json
import logging
import time
from pathlib import Path
from collections import defaultdict
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from aml.config import Settings
from aml.data.split import TemporalSplit
from aml.data.fx import fit_fx
from aml.graph.build import build_graph, load_graph
from aml.detect.common import Hit, make_hit
from aml.cluster.merge import MergeConfig, merge_hits
from aml.score.features import FeatureConfig, features_for_groups
from aml.score.model import ModelConfig, train_model, probabilities, contributions
from aml.score.risk import RiskConfig, transparent_score
from aml.narrative.template import alert_from_hit
from aml.api.schemas import Contribution

logger = logging.getLogger(__name__)


def read_hits(path: Path) -> list[Hit]:
    with path.open() as stream:
        return [Hit.model_validate_json(line.strip().rstrip(",")) for line in stream if line.strip() not in ("[", "]", "")]


def run(settings: Settings) -> dict[str, object]:
    started = time.perf_counter()
    output = settings.output_dir / "m8"
    output.mkdir(parents=True, exist_ok=True)
    handler = logging.FileHandler(output / "scoring.log", mode="w")
    logging.getLogger("aml").addHandler(handler)
    directory = settings.data.processed_dir
    split = TemporalSplit.model_validate_json((directory / "split.json").read_text())
    graph = load_graph(directory / "transactions.parquet", directory / "accounts.parquet")
    hits = [h for detector in ("fan", "scatter_gather", "cycle", "pass_through")
            for h in read_hits(settings.output_dir / detector / "hits.json")]
    print(f"Loaded {len(hits)} hits", flush=True)
    merge_config = MergeConfig(**settings.detectors["merge"])
    feature_config = FeatureConfig(**settings.scoring["features"])
    model_config = ModelConfig(**settings.scoring["model"], seed=settings.seed)
    risk_config = RiskConfig(**settings.scoring["risk"])
    account_table = pq.read_table(directory / "accounts.parquet").to_pandas()
    banks = dict(zip(account_table.account_id, account_table.bank))
    names = {int(row.account_id): f"{row.bank}:{row.account}" for row in account_table.itertuples()}
    fx = fit_fx(graph, split.train_end)
    # Independent train graph and clustering prevent future degrees, activity,
    # amounts or merged components from entering training features.
    train = build_graph(pq.read_table(directory / "transactions.parquet", filters=[("ts", ">=", split.start),
                        ("ts", "<", split.train_end)]).to_pandas(), node_count=graph.node_count)
    train_hits = [h for h in hits if split.start <= h.t_start <= h.t_end < split.train_end]
    training_clusters = merge_hits(train, train_hits, merge_config)
    training_groups = [c.edge_ids for c in training_clusters]
    train_features = features_for_groups(train, training_groups, [[train_hits[i] for i in c.hit_ids] for c in training_clusters], banks, feature_config, fx)
    labels = np.array([train.label[train.positions_for_tx_ids(np.array(c.edge_ids))].mean() >= settings.evaluation["cluster_tau"] for c in training_clusters], dtype=np.int8)
    ends = np.array([c.t_end for c in training_clusters])
    cluster_model = train_model(train_features, labels, ends, split.train_end, model_config)
    cluster_model.save_model(output / "cluster_model.json")
    print(f"Trained cluster model: {len(labels)} clusters, {labels.sum()} positives", flush=True)
    cluster_train_count, cluster_positive_count = len(labels), int(labels.sum())
    del train_features, training_groups, training_clusters
    by_account: dict[int, list[Hit]] = defaultdict(list)
    for hit in train_hits:
        for node in hit.accounts:
            by_account[node].append(hit)
    population = np.union1d(train.src, train.dst)
    account_groups = [train.tx_id[np.union1d(train.in_idx[int(node)], train.out_idx[int(node)])].tolist() for node in population]
    account_features = features_for_groups(train, account_groups, [by_account[int(node)] for node in population], banks, feature_config, fx)
    positives = np.union1d(train.src[train.label == 1], train.dst[train.label == 1])
    account_labels = np.isin(population, positives).astype(np.int8)
    account_ends = np.array([int(train.ts[train.positions_for_tx_ids(np.array(ids))].max()) for ids in account_groups])
    account_model = train_model(account_features, account_labels, account_ends, split.train_end, model_config)
    account_model.save_model(output / "account_model.json")
    print(f"Trained account model: {len(population)} accounts, {account_labels.sum()} positives", flush=True)
    account_train_count, account_positive_count = len(population), int(account_labels.sum())
    del account_features, account_groups, by_account, train, train_hits
    clusters = merge_hits(graph, hits, merge_config)
    features = features_for_groups(graph, [c.edge_ids for c in clusters], [[hits[i] for i in c.hit_ids] for c in clusters], banks, feature_config, fx)
    scores = probabilities(cluster_model, features)
    order = sorted(range(len(clusters)), key=lambda i: (-float(scores[i]), clusters[i].t_start, clusters[i].edge_ids))
    # Preserve every ranked cluster and its features for M9/M11. Only the frontend
    # payload is limited; the export limit is not an alert cutoff or evaluation rule.
    features.assign(model_probability=scores).to_parquet(output / "cluster_features.parquet", index=False)
    with (output / "ranked_clusters.jsonl").open("w") as stream:
        for rank, index in enumerate(order, 1):
            stream.write(json.dumps({**clusters[index].model_dump(), "rank": rank, "model_probability": float(scores[index])}) + "\n")
    chosen = order[:risk_config.export_limit]
    if len(chosen) < len(order):
        logger.warning("alert export_limit triggered total=%d exported=%d; full ranking retained", len(order), len(chosen))
    shap = contributions(cluster_model, features.iloc[chosen])
    alerts = []
    for rank, (index, explanations) in enumerate(zip(chosen, shap), 1):
        cluster = clusters[index]
        row = features.iloc[index].to_dict()
        members = [hits[i] for i in cluster.hit_ids]
        types = sorted({h.typology for h in members})
        feature_contributions = dict(zip(features.columns, explanations))
        def strength(typology: str) -> tuple[float, str]:
            return (sum(float(v) for k, v in feature_contributions.items() if k == f"pattern_{typology}" or k.startswith(f"pattern_{typology}_size_")), typology)
        primary = max(types, key=strength)
        positions = graph.positions_for_tx_ids(np.array(cluster.edge_ids))
        combined = make_hit(graph, positions, primary, {"contributing_hits": len(members)}, fx)
        alert = alert_from_hit(graph, combined, names, float(scores[index]), rank, True)
        alert.secondary_typologies = [t for t in types if t != primary]
        alert.evidence["hits"] = [h.model_dump() for h in members]
        alert.evidence["features"].update(row)
        alert.evidence["features"]["transparent_score"] = transparent_score(row, risk_config)
        alert.evidence["features"]["contribution_units"] = "model log-odds"
        top = np.argsort(-np.abs(explanations), kind="stable")[:risk_config.top_features]
        alert.top_features = [Contribution(name=features.columns[i], contribution=float(explanations[i])) for i in top]
        alerts.append(alert.model_dump(mode="json"))
    payload = json.dumps(alerts, indent=2)
    (output / "alerts.json").write_text(payload + "\n")
    (settings.output_dir / "alerts.json").write_text(payload + "\n")
    train_risk = probabilities(cluster_model, features.iloc[[i for i, c in enumerate(clusters)
        if split.start <= c.t_start <= c.t_end < split.train_end]])
    report = {"command": "aml cluster", "clusters": len(clusters), "exported_alerts": len(alerts),
        "training_clusters": cluster_train_count, "positive_training_clusters": cluster_positive_count,
        "training_accounts": account_train_count, "positive_training_accounts": account_positive_count,
        "model_score_available": True, "test_evaluated": False, "cutoff_selected": False,
        "runtime_seconds": time.perf_counter() - started,
        "dampening": "explicit transparent-score reduction; legitimacy features supplied to learned models",
        "scope": "initial M8 models, not M9 tuned/frozen models; export limit is not a decision threshold",
        "ranked_cluster_scores": str(output / "cluster_features.parquet")}
    (output / "report.json").write_text(json.dumps(report, indent=2))
    logging.getLogger("aml").removeHandler(handler)
    handler.close()
    return report
