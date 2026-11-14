"""Final one-time test evaluation using the frozen M9 cutoff."""
import json
import numpy as np
import pyarrow.parquet as pq
from aml.config import Settings
from aml.data.split import TemporalSplit
from aml.eval.metrics import binary_metrics
from aml.eval.schemas import Cluster
from aml.eval.bootstrap import dependent_blocks, bootstrap_account_f1
from aml.eval.report_metrics import attempt_coverage_report, cluster_precision_summary, straddling_attempts
from aml.graph.build import load_graph
from aml.eval.frozen import check_final_guard


def run(settings: Settings) -> dict:
    config_hash = check_final_guard(settings)
    split = TemporalSplit.model_validate_json((settings.data.processed_dir / "split.json").read_text())
    ranked = [json.loads(line) for line in (settings.output_dir / "m8" / "ranked_clusters.jsonl").open()]
    frozen = json.loads((settings.output_dir / "m9" / "frozen.yaml").read_text())
    cutoff = float(frozen["cutoff"])
    graph = load_graph(settings.data.processed_dir / "transactions.parquet", settings.data.processed_dir / "accounts.parquet")
    # First and only test evaluation. All model choices/cutoff were frozen in M9.
    test_mask = (graph.ts >= split.validation_end) & (graph.ts < split.end)
    test_ids = set(graph.tx_id[test_mask].tolist())
    positive_fractions, scores = [], []
    edges = []
    alerted_test_edges: set[int] = set()
    alerted_accounts: set[int] = set()
    account_score = np.zeros(graph.node_count)
    for row in ranked:
        ids = [int(tx) for tx in row["edge_ids"] if int(tx) in test_ids]
        if not ids:
            continue
        positions = graph.positions_for_tx_ids(np.array(ids))
        positive_fractions.append(float(graph.label[positions].mean()))
        scores.append(float(row["model_probability"]))
        if float(row["model_probability"]) >= cutoff:
            alerted_test_edges.update(ids)
            alerted_accounts.update(int(node) for node in np.union1d(graph.src[positions], graph.dst[positions]))
        nodes = np.union1d(graph.src[positions], graph.dst[positions])
        np.maximum.at(account_score, nodes, float(row["model_probability"]))
        edges.extend((int(graph.src[p]), int(graph.dst[p])) for p in positions)
    cluster_metrics = cluster_precision_summary(positive_fractions, scores, cutoff,
        taus=(.3, .5, .7), k=int(settings.evaluation["precision_k"]))
    population = np.union1d(graph.src[test_mask], graph.dst[test_mask])
    positive = np.union1d(graph.src[test_mask & (graph.label == 1)], graph.dst[test_mask & (graph.label == 1)])
    account_truth = np.isin(population, positive)
    account_prediction = account_score[population] >= cutoff
    account_metrics = binary_metrics(account_truth.astype(np.int8), account_prediction.astype(np.int8), account_score[population])
    pattern_rows = pq.read_table(settings.data.processed_dir / "attempts.parquet").to_pylist()
    from aml.data.patterns_parser import Attempt
    all_attempts = [Attempt.model_validate({k:r[k] for k in Attempt.model_fields}) for r in pattern_rows]
    attempts = [a for a in all_attempts if split.validation_end <= a.t_start < split.end]
    attempt_recall = attempt_coverage_report(attempts, alerted_test_edges, alerted_accounts)
    straddlers = straddling_attempts(all_attempts, (split.validation_end, split.end))
    blocks = dependent_blocks(population, attempts, edges)
    ci = bootstrap_account_f1(population, account_truth, account_prediction, blocks,
        replicates=settings.evaluation["bootstrap_replicates"], seed=settings.seed,
        confidence=settings.evaluation["confidence_level"])
    report = {"command": "aml report", "test_evaluated_once": True, "frozen_hash": config_hash, "cutoff_frozen": cutoff,
        "cluster": cluster_metrics.model_dump(), "account": account_metrics.model_dump(),
        "account_f1_block_bootstrap": ci, "test_transactions": int(test_mask.sum()),
        "positive_test_transactions": int(graph.label[test_mask].sum()),
        "attempts_by_start_period": len(attempts), "attempt_recall_edge_coverage_tau_0_5": attempt_recall,
        "straddling_attempts": straddlers,
        "test_access_in_tuning": False}
    output = settings.output_dir / "m11"
    output.mkdir(parents=True, exist_ok=True)
    (output / "final_report.json").write_text(json.dumps(report, indent=2))
    (output / "final_report.md").write_text("# Final test report\n\nCommand: `aml report`\n\n```json\n" + json.dumps(report, indent=2) + "\n```\n")
    return report
