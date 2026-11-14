"""Frozen artifact hash and one-shot final-evaluation guard."""
import hashlib
import json
from pathlib import Path


def frozen_hash(settings) -> str:
    paths = [settings.output_dir / "m9" / "frozen.yaml",
             settings.output_dir / "m8" / "cluster_model.json",
             settings.output_dir / "m8" / "account_model.json",
             Path("configs/default.yaml")]
    digest = hashlib.sha256()
    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(f"frozen artifact missing: {path}")
        digest.update(path.name.encode())
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
            "frozen.yaml", "cluster_model.json", "account_model.json", "configs/default.yaml"]}, indent=2))
    marker.parent.mkdir(parents=True, exist_ok=True)
    marker.write_text(json.dumps({"sha256": actual, "state": "started"}, indent=2))
    return actual
