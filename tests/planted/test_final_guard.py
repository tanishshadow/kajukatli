import json
import pytest
from aml.config import Settings
from aml.eval.frozen import check_final_guard


def test_final_guard_blocks_second_run_and_changed_hash(tmp_path):
    settings = Settings(output_dir=tmp_path)
    m8, m9 = tmp_path / "m8", tmp_path / "m9"
    m8.mkdir(); m9.mkdir()
    (m8 / "cluster_model.json").write_text("cluster")
    (m8 / "account_model.json").write_text("account")
    (m9 / "frozen.yaml").write_text('{"cutoff":0.8}')
    config = tmp_path / "configs"
    config.mkdir()
    from pathlib import Path
    old = Path("configs/default.yaml")
    # Hash helper resolves repository config, so exercise guard hash seam directly.
    import aml.eval.frozen as frozen
    monkey = frozen.frozen_hash
    frozen.frozen_hash = lambda _: "a" * 64
    try:
        first = check_final_guard(settings)
        with pytest.raises(RuntimeError, match="one-shot"):
            check_final_guard(settings)
        (tmp_path / "m11" / "FINAL_RUN_STARTED.json").unlink()
        (tmp_path / "m9" / "frozen_hash.json").write_text(json.dumps({"sha256": "b" * 64}))
        with pytest.raises(RuntimeError, match="differs"):
            check_final_guard(settings)
    finally:
        frozen.frozen_hash = monkey
