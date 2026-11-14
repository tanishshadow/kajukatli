import pytest
from aml.api.schemas import Alert
from aml.narrative.template import alert_from_hit
from aml.detect.fan import FanConfig, detect_fans
from aml.graph.build import build_graph
from tests_helpers import frame


def test_fan_to_valid_evidence_backed_template_alert() -> None:
    g = build_graph(frame([(0, 1, 0, 10, 1), (0, 2, 1, 10, 1), (0, 3, 2, 10, 1)]))
    hit = detect_fans(g, FanConfig(window_seconds=3, k=3))[0]
    # Supplied probability is a test input, not a claim of a trained M4 model.
    alert = alert_from_hit(g, hit, {0: "001:A", 1: "001:B", 2: "002:C", 3: "002:D"},
                           probability=.87, rank=1, score_available=True)
    assert alert.risk_score == 87
    assert alert.summary_source == "template" and "consistent with" in alert.summary
    assert [edge.tx_id for edge in alert.edges] == [0, 1, 2]
    assert set(alert.evidence["entities"]) == {"E1", "E2", "E3", "E4"}
    assert set(alert.evidence["transactions"]) == {"T1", "T2", "T3"}
    assert Alert.model_validate_json(alert.model_dump_json()) == alert
    bad = alert.model_dump()
    bad["risk_score"] = 101
    with pytest.raises(ValueError):
        Alert.model_validate(bad)
