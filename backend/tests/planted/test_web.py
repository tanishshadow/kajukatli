import json
from pathlib import Path
from threading import Thread
from urllib.request import urlopen
from urllib.error import HTTPError
from http.server import ThreadingHTTPServer

from aml.config import Settings
from aml.web import api_data, make_handler
import pandas as pd


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


def test_investigation_api_returns_real_hit_summaries_and_exact_subgraph(tmp_path):
    settings = Settings(output_dir=tmp_path / "outputs", data={"processed_dir": tmp_path / "processed"},
        detectors={"fan": {"k": 2}})
    settings.data.processed_dir.mkdir(parents=True)
    hits = [
        {"typology": "fan_in", "accounts": [1, 2, 3], "edge_ids": [10, 11],
         "t_start": 100, "t_end": 200, "total_value": 75.0,
         "meta": {"hub": 1, "distinct_counterparties": 2, "value_currency": "USD", "fx_missing": False}},
        {"typology": "fan_out", "accounts": [1, 4], "edge_ids": [12],
         "t_start": 300, "t_end": 300, "total_value": 12.0, "meta": {"hub": 1}},
    ]
    path = settings.output_dir / "fan" / "hits.json"
    path.parent.mkdir(parents=True)
    path.write_text("[\n" + ",\n".join(json.dumps(hit) for hit in hits) + "\n]\n", encoding="utf-8")
    pd.DataFrame({"tx_id": [10, 11, 12, 99], "ts": [200, 100, 300, 150],
        "src": [2, 3, 1, 8], "dst": [1, 1, 4, 9], "amt_out": [25., 50., 12., 1.],
        "cur_out": ["USD"] * 4, "label": [1, 0, 1, 0]}).to_parquet(settings.data.processed_dir / "transactions.parquet")
    pd.DataFrame({"account_id": [1, 2, 3, 4, 8, 9], "bank": ["A", "B", "C", "D", "X", "Y"]}).to_parquet(settings.data.processed_dir / "accounts.parquet")

    cases = api_data("investigation/cases", settings, {"detector": ["fan"], "offset": ["0"], "limit": ["1"]})
    assert cases["total"] == 2 and cases["has_more"] is True
    assert cases["cases"][0]["edge_count"] == 2
    subgraph = api_data("investigation/case", settings, {"detector": ["fan"], "id": ["0"]})
    assert {edge["id"] for edge in subgraph["edges"]} == {10, 11}
    assert [(edge["src"], edge["dst"]) for edge in subgraph["edges"]] == [(3, 1), (2, 1)]
    assert "2 distinct counterparties" in subgraph["conditions"][0]
    assert [node["id"] for node in subgraph["nodes"]] == [1, 2, 3]
    assert set(subgraph["currencies"]) == {"USD"}
    assert all(edge["id"] != 99 for edge in subgraph["edges"])


def test_investigation_account_finds_cases_with_case_endpoint_ids(tmp_path):
    settings = Settings(output_dir=tmp_path / "outputs")
    fan_path = settings.output_dir / "fan" / "hits.json"
    fan_hits = [
        {"typology": "unrelated", "accounts": [7]},
        {"typology": "fan_out", "accounts": ["7", 8], "edge_ids": [1, 2],
         "t_start": 10, "t_end": 20, "total_value": 50, "meta": {"value_currency": "USD"}},
    ]
    fan_path.parent.mkdir(parents=True)
    fan_path.write_text("[\n" + ",\n".join(json.dumps(hit) for hit in fan_hits) + "\n]\n", encoding="utf-8")
    cycle_hits = [
        {"typology": "cycle", "accounts": [7, 9], "edge_ids": [3],
         "t_start": 30, "t_end": 40, "total_value": 25, "meta": {"value_currency": "EUR"}},
    ]
    cycle_path = settings.output_dir / "cycle" / "hits.json"
    cycle_path.parent.mkdir(parents=True)
    cycle_path.write_text("[\n" + ",\n".join(json.dumps(hit) for hit in cycle_hits) + "\n]\n", encoding="utf-8")

    result = api_data("investigation/account", settings, {"id": ["7"]})
    fan_cases = api_data("investigation/cases", settings, {"detector": ["fan"], "offset": ["0"], "limit": ["30"]})
    cycle_cases = api_data("investigation/cases", settings, {"detector": ["cycle"], "offset": ["0"], "limit": ["30"]})
    assert result["account"] == 7 and result["total"] == 2 and result["truncated"] is False
    assert [(case["detector"], case["id"]) for case in result["cases"]] == [("fan", 1), ("cycle", 0)]
    assert result["cases"][0]["id"] == fan_cases["cases"][0]["id"]
    assert result["cases"][1]["id"] == cycle_cases["cases"][0]["id"]
    assert result["cases"][0]["currency"] == "USD"


def test_investigation_account_handles_missing_invalid_and_unmatched_ids(tmp_path):
    settings = Settings(output_dir=tmp_path / "outputs")
    missing = api_data("investigation/account", settings, {})
    invalid = api_data("investigation/account", settings, {"id": ["7x"]})
    absent = api_data("investigation/account", settings, {"id": ["7"]})
    assert missing == {"error": "invalid account id", "cases": [], "total": 0}
    assert invalid == {"error": "invalid account id", "cases": [], "total": 0}
    assert absent == {"account": 7, "cases": [], "total": 0, "truncated": False}


def test_investigation_account_caps_results_and_only_reads_hits(tmp_path):
    settings = Settings(output_dir=tmp_path / "outputs")
    hits_path = settings.output_dir / "fan" / "hits.json"
    hits = [{"typology": "fan_out", "accounts": [12], "edge_ids": [index],
             "t_start": index, "t_end": index, "total_value": index, "meta": {}}
            for index in range(101)]
    hits_path.parent.mkdir(parents=True)
    hits_path.write_text("[\n" + ",\n".join(json.dumps(hit) for hit in hits) + "\n]\n", encoding="utf-8")
    before = {path: path.read_bytes() for path in settings.output_dir.rglob("*") if path.is_file()}

    result = api_data("investigation/account", settings, {"id": ["12"]})

    after = {path: path.read_bytes() for path in settings.output_dir.rglob("*") if path.is_file()}
    assert result["total"] == 101 and result["truncated"] is True
    assert len(result["cases"]) == 100
    assert [case["id"] for case in result["cases"]] == list(range(100))
    assert after == before
