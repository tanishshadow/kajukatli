import json
import time
import logging
import pyarrow.parquet as pq
from aml.config import Settings
from aml.graph.build import load_graph
from aml.data.split import TemporalSplit
from aml.data.fx import fit_fx
from aml.detect.fan import FanConfig, detect_fans


def run(settings: Settings, detector: str) -> dict:
    start = time.perf_counter()
    output = settings.output_dir / detector
    output.mkdir(parents=True, exist_ok=True)
    handler = logging.FileHandler(output / "detector.log", mode="w")
    logging.getLogger("aml.detect").addHandler(handler)
    directory = settings.data.processed_dir
    graph = load_graph(directory / "transactions.parquet", directory / "accounts.parquet")
    split = TemporalSplit.model_validate_json((directory / "split.json").read_text())
    fx = fit_fx(graph, split.train_end)
    if detector == "fan":
        hits = detect_fans(graph, FanConfig(**settings.detectors["fan"], fx=fx))
    elif detector == "scatter_gather":
        from aml.detect.scatter_gather import ScatterConfig, detect_scatter_gather
        hits = detect_scatter_gather(graph, ScatterConfig(**settings.detectors["scatter_gather"], fx=fx))
    elif detector == "cycle":
        from aml.detect.cycles import CycleConfig, detect_cycles
        hits = detect_cycles(graph, CycleConfig(**settings.detectors["cycle"], fx=fx))
    else:
        raise ValueError("detector not implemented yet")
    output = settings.output_dir / detector
    output.mkdir(parents=True, exist_ok=True)
    with (output / "hits.json").open("w") as stream:
        stream.write("[")
        for index, hit in enumerate(hits):
            stream.write((",\n" if index else "\n") + hit.model_dump_json())
        stream.write("\n]\n")
    (output / "fx.json").write_text(fx.model_dump_json(indent=2))
    report = {"detector": detector, "hits": len(hits), "runtime_seconds": time.perf_counter() - start,
              "fx_missing_hits": sum(h.meta["fx_missing"] for h in hits), "base_currency": fx.base_currency,
              "mapped_currencies": len(fx.factors), "test_evaluated": False,
              "command": f"aml detect --detector {detector}"}
    (output / "report.json").write_text(json.dumps(report, indent=2))
    logging.getLogger("aml.detect").removeHandler(handler)
    handler.close()
    return report
