"""M4 backend slice, using already detected fan evidence."""
import json
import pyarrow.parquet as pq
from aml.config import Settings
from aml.graph.build import load_graph
from aml.detect.common import Hit
from aml.narrative.template import alert_from_hit
from aml.api.schemas import Alert
from aml.narrative.render import OfflineLLMClient, render_narrative


def run(settings: Settings) -> dict[str, object]:
    ranked_path = settings.output_dir / "m8" / "alerts.json"
    if ranked_path.exists():
        alerts = [Alert.model_validate(value) for value in json.loads(ranked_path.read_text())]
        failures = 0
        client = OfflineLLMClient()
        for alert in alerts:
            template = alert.summary
            try:
                alert.summary = render_narrative(client.generate(alert.evidence, "AML patterns"), alert.evidence)
                alert.summary_source = "llm_validated"
            except Exception:
                failures += 1
                alert.summary = template
                alert.summary_source = "template"
        # M8 already renders deterministic templates. Preserve learned scores
        # when this command is called after the earlier vertical-slice milestone.
        destination = settings.output_dir / "alerts.json"
        destination.write_text(json.dumps([a.model_dump(mode="json") for a in alerts], indent=2) + "\n")
        stats = {"alerts": len(alerts), "failure_count": failures,
                 "failure_rate": failures / len(alerts) if alerts else 0.0}
        (settings.output_dir / "narrative_stats.json").write_text(json.dumps(stats, indent=2) + "\n")
        return {"alerts": len(alerts), "model_score_available": True, "summary_source": "llm_validated",
                "narrative_failures": failures, "narrative_failure_rate": failures / max(1, len(alerts)),
                "command": "aml narrate", "output": str(destination)}
    directory = settings.data.processed_dir
    graph = load_graph(directory / "transactions.parquet", directory / "accounts.parquet")
    # Detector JSON is one hit per line; do not load the full candidate set for one alert.
    with (settings.output_dir / "fan" / "hits.json").open() as stream:
        next(stream)
        hit = Hit.model_validate_json(next(stream).strip().rstrip(","))
    accounts = pq.read_table(directory / "accounts.parquet").to_pandas()
    names = {int(row.account_id): f"{row.bank}:{row.account}" for row in accounts.itertuples()}
    alert = alert_from_hit(graph, hit, names, probability=0., rank=1, score_available=False)
    output = settings.output_dir / "m4"
    output.mkdir(parents=True, exist_ok=True)
    data = json.dumps([alert.model_dump(mode="json")], indent=2)
    (output / "alerts.json").write_text(data + "\n")
    (settings.output_dir / "alerts.json").write_text(data + "\n")
    report = {"alerts": 1, "model_score_available": False, "summary_source": "template",
              "command": "aml narrate", "output": str(output / "alerts.json")}
    (output / "report.json").write_text(json.dumps(report, indent=2))
    return report
