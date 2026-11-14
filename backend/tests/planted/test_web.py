import json
from pathlib import Path
from threading import Thread
from urllib.request import urlopen
from urllib.error import HTTPError
from http.server import ThreadingHTTPServer

from aml.config import Settings
from aml.web import api_data, make_handler


def put_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def test_page_api_exposes_only_project_summaries(tmp_path):
    settings = Settings(output_dir=tmp_path / "outputs")
    put_json(settings.output_dir / "m1" / "checklist.json", {
        "transactions": 100, "accounts": 12, "positive_transactions": 4,
        "split": {"iso_boundaries": {"train": ["a", "b"], "validation": ["b", "c"]}}})
    put_json(settings.output_dir / "m8" / "report.json", {"clusters": 7})
    put_json(settings.output_dir / "m11" / "final_report.json", {
        "account": {"f1": .5, "recall": .6}, "attempts_by_start_period": 3,
        "test_transactions": 20, "positive_test_transactions": 2,
        "attempt_recall_edge_coverage_tau_0_5": {"CYCLE": {"status": "covered", "recall": .5}}})
    put_json(settings.output_dir / "fan" / "report.json", {"hits": 11})
    (settings.output_dir / "evasion.csv").write_text(
        "kind,strength,main_attempt_recall,baseline_attempt_recall\n"
        "amount_split,0.5,0.2,0.3\ndelays,0.5,0.1,0.25\n", encoding="utf-8")
    overview = api_data("overview", settings)
    assert overview == {"transactions": 100, "positive_transactions": 4, "accounts": 12,
        "detectors": 4, "ranked_clusters": 7, "test_report_available": True}
    evaluation = api_data("evaluation", settings)
    assert evaluation["account_f1"] == .5
    assert evaluation["attempt_recall_by_typology"]["CYCLE"]["recall"] == .5
    assert [point["kind"] for point in evaluation["evasion"]] == ["amount_split", "delays"]
    encoded = json.dumps(overview)
    assert "account_id" not in encoded and "transaction_id" not in encoded


def test_web_server_serves_page_and_same_origin_api(tmp_path):
    settings = Settings(output_dir=tmp_path / "outputs")
    frontend = tmp_path / "frontend"
    frontend.mkdir()
    (frontend / "index.html").write_text("<h1>AML Graph</h1>", encoding="utf-8")
    server = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(settings, frontend))
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        base = f"http://127.0.0.1:{server.server_port}"
        with urlopen(base + "/") as response:
            assert b"AML Graph" in response.read()
        with urlopen(base + "/api/overview") as response:
            assert json.loads(response.read())["transactions"] is None
        try:
            urlopen(base + "/api/nope")
        except HTTPError as error:
            assert error.code == 404
        else:
            raise AssertionError("unknown API route should return 404")
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)
