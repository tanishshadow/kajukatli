"""M4 backend slice, using already detected fan evidence."""
import json
import pyarrow.parquet as pq
from aml.config import Settings
from aml.graph.build import load_graph
from aml.detect.common import Hit
from aml.narrative.template import alert_from_hit


def run(settings: Settings) -> dict[str, object]:
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
