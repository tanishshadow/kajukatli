import pytest
import json
from datetime import datetime, timezone
from aml.config import Settings
from aml.api.schemas import Alert
import aml.narrative.run as narrative_run
from aml.narrative.render import render_narrative, OfflineLLMClient


def evidence():
    return {"entities": {"E1": "bank:account"}, "transactions": {
        "T1": {"currency": "USD", "amount": 10, "ts": "2022-09-01T00:00:00Z"},
        "T2": {"currency": "EUR", "amount": 20, "ts": "2022-09-01T00:01:00Z"}}}


def test_offline_stub_renders_resolved_placeholders():
    bundle = evidence()
    assert "USD" in render_narrative(OfflineLLMClient().generate(bundle, ""), bundle)


@pytest.mark.parametrize("text,ids", [("Invented 999 {T1.amount}", ["T1"]),
    ("Unknown {T9.amount}", ["T9"]), ("Bad {T1.missing}", ["T1"])])
def test_narrative_rejects_invalid_claims(text, ids):
    with pytest.raises(ValueError):
        render_narrative({"sentences": [{"text": text, "evidence_ids": ids}]}, evidence())


def test_narrative_requires_ids_and_chronological_event_order():
    bundle = evidence()
    with pytest.raises(ValueError):
        render_narrative({"sentences": [{"text": "First {T2.amount}", "evidence_ids": ["T2"]},
          {"text": "Then {T1.amount}", "evidence_ids": ["T1"]}]}, bundle)


def test_run_falls_back_to_template_and_writes_failure_stats(tmp_path, monkeypatch):
    now = datetime(2022, 9, 1, tzinfo=timezone.utc).isoformat()
    alert = Alert(alert_id="A-1", rank=1, risk_score=50, primary_typology="cycle", secondary_typologies=[],
        accounts=["a", "b"], edges=[{"tx_id": 1, "src": "a", "dst": "b", "ts": now, "amount": 10, "currency": "USD"}],
        time_span={"start": now, "end": now}, total_value=10, top_features=[], summary="template stays",
        summary_source="template", evidence=evidence())
    (tmp_path / "m8").mkdir()
    (tmp_path / "m8" / "alerts.json").write_text(json.dumps([alert.model_dump(mode="json")]))
    monkeypatch.setattr(narrative_run.OfflineLLMClient, "generate", lambda *_: {"sentences": [{"text": "Bad 999 {T1.amount}", "evidence_ids": ["T1"]}]})
    narrative_run.run(Settings(output_dir=tmp_path))
    saved = json.loads((tmp_path / "alerts.json").read_text())[0]
    stats = json.loads((tmp_path / "narrative_stats.json").read_text())
    assert saved["summary"] == "template stays" and saved["summary_source"] == "template"
    assert stats == {"alerts": 1, "failure_count": 1, "failure_rate": 1.0}
