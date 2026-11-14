from datetime import datetime, timezone
import numpy as np
from aml.api.schemas import Alert, AlertEdge, TimeSpan
from aml.detect.common import Hit
from aml.graph.build import TemporalGraph


def alert_from_hit(graph: TemporalGraph, hit: Hit, account_names: dict[int, str],
                   probability: float, rank: int, score_available: bool) -> Alert:
    positions = graph.positions_for_tx_ids(np.array(hit.edge_ids, dtype=np.int64))
    edges = [AlertEdge(tx_id=int(graph.tx_id[e]), src=account_names[int(graph.src[e])],
        dst=account_names[int(graph.dst[e])], ts=datetime.fromtimestamp(int(graph.ts[e]), timezone.utc),
        amount=float(graph.amt_out[e]), currency=graph.currency_names[int(graph.cur_out[e])]) for e in positions]
    start, end = edges[0].ts, edges[-1].ts
    typology = hit.typology.replace("_", "-")
    summary = (f"Between {start.isoformat()} and {end.isoformat()}, {len(hit.accounts)} accounts "
               f"participated in {len(edges)} transactions showing patterns consistent with {typology}.")
    features = {"model_score_available": score_available, "transparent_score": None,
                "total_value_is_partial": hit.meta.get("total_value_is_partial", False)}
    return Alert(alert_id=f"A-{rank:06d}", rank=rank, risk_score=100 * probability,
        primary_typology=hit.typology, secondary_typologies=[], accounts=[account_names[a] for a in hit.accounts],
        edges=edges, time_span=TimeSpan(start=start, end=end), total_value=hit.total_value, top_features=[],
        summary=summary, summary_source="template", evidence={"hits": [hit.model_dump()], "features": features,
            "entities": {f"E{i + 1}": account_names[a] for i, a in enumerate(hit.accounts)},
            "transactions": {f"T{i + 1}": edge.model_dump(mode="json") for i, edge in enumerate(edges)}})
