"""Frozen artifact hash and one-shot final-evaluation guard."""
import hashlib
import json
from pathlib import Path
import yaml


def frozen_hash(settings) -> str:
    frozen_path = settings.output_dir / "m9" / "frozen.yaml"
    frozen = yaml.safe_load(frozen_path.read_text(encoding="utf-8"))
    model = frozen["model"]
    paths = [(str(frozen_path), frozen_path),
             (f"outputs/m9/model_{model['n_estimators']}_{model['max_depth']}.json",
              settings.output_dir / "m9" / f"model_{model['n_estimators']}_{model['max_depth']}.json"),
             ("outputs/m8/cluster_model.json", settings.output_dir / "m8" / "cluster_model.json"),
             ("outputs/m8/account_model.json", settings.output_dir / "m8" / "account_model.json"),
             ("outputs/m3/baseline.json", settings.output_dir / "m3" / "baseline.json"),
             ("configs/default.yaml", Path("backend/configs/default.yaml")),
             ("src/aml/detect/fan.py", Path("backend/src/aml/detect/fan.py")),
             ("src/aml/detect/scatter_gather.py", Path("backend/src/aml/detect/scatter_gather.py")),
             ("src/aml/detect/cycles.py", Path("backend/src/aml/detect/cycles.py")),
             ("src/aml/detect/passthrough.py", Path("backend/src/aml/detect/passthrough.py")),
             ("src/aml/cluster/merge.py", Path("backend/src/aml/cluster/merge.py")),
             ("src/aml/score/features.py", Path("backend/src/aml/score/features.py")),
             ("src/aml/score/model.py", Path("backend/src/aml/score/model.py")),
             ("src/aml/score/risk.py", Path("backend/src/aml/score/risk.py"))]
    digest = hashlib.sha256()
    for logical_path, path in paths:
        if not path.is_file():
            raise FileNotFoundError(f"frozen artifact missing: {path}")
        digest.update(str(Path(logical_path)).encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def check_final_guard(settings, *, expected_hash: str | None = None) -> str:
    marker = settings.output_dir / "m11" / "FINAL_RUN_STARTED.json"
    if marker.exists():
        raise RuntimeError("final evaluation marker exists; test evaluation is one-shot")
    actual = frozen_hash(settings)
    manifest = settings.output_dir / "m9" / "frozen.yaml"
    record_path = settings.output_dir / "m9" / "frozen_hash.json"
    record = json.loads(record_path.read_text()) if record_path.exists() else None
    expected = expected_hash or (record or {}).get("sha256")
    if expected and expected != actual:
        raise RuntimeError("frozen config/model hash differs from recorded hash")
    if not record_path.exists():
        record_path.write_text(json.dumps({"sha256": actual, "artifacts": [
            "frozen.yaml", "cluster_model.json", "account_model.json", "backend/configs/default.yaml"]}, indent=2))
    marker.parent.mkdir(parents=True, exist_ok=True)
    marker.write_text(json.dumps({"sha256": actual, "state": "started"}, indent=2))
    return actual
