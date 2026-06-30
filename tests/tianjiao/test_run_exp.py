import os
import sys
import json
from datetime import datetime
import importlib.util
from pathlib import Path

os.environ.setdefault("TORCH_COMPILE_DISABLE", "1")

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from pydantic import ValidationError

from spargel_llm.train import TrainInfo

RUN_EXP_PATH = Path(__file__).resolve().parents[1] / "scripts" / "tianjiao" / "run_exp.py"
SPEC = importlib.util.spec_from_file_location("run_exp_under_test", RUN_EXP_PATH)
assert SPEC is not None
assert SPEC.loader is not None
run_exp = importlib.util.module_from_spec(SPEC)
sys.modules["run_exp_under_test"] = run_exp
SPEC.loader.exec_module(run_exp)
assert "scripts.tool" not in sys.modules


def _config_dict(**overrides):
    config = {
        "schema_version": 1,
        "model": {
            "vocab_size": 32,
            "max_seq_len": 8,
            "num_layer": 1,
            "num_head": 1,
            "dim": 8,
            "dim_key": 4,
            "dim_value": 4,
            "dim_ff_hidden": 16,
            "use_rope": True,
            "ff_activation": "relu",
        },
        "tokenizer": "data/tokenizer.json",
        "data": {
            "train_path": "data/train.parquet",
            "validation_path": None,
            "start_index": 0,
            "start_offset": 0,
            "add_sot": False,
            "add_eot": False,
        },
        "train": {
            "steps": 3,
            "seq_len": 4,
            "batch_size": 2,
            "learning_rate": 0.001,
            "weight_decay": 0.1,
            "micro_batches": 1,
            "log_period": 1,
            "checkpoint_interval": 2,
            "validation_batches": None,
            "loop_dataset": False,
            "seed": 123,
            "use_bf16": False,
            "float32_precision": "high",
        },
    }
    for section, values in overrides.items():
        config[section].update(values)
    return config


def _write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2) + "\n")


def _write_tokens(path: Path, rows: list[list[int]], dataset_id: str = "ds") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    schema = pa.schema([pa.field("tokens", pa.list_(pa.uint32()))])
    schema = schema.with_metadata({"dataset_id": dataset_id})
    table = pa.table(
        {"tokens": [pa.array(row, type=pa.uint32()) for row in rows]},
        schema=schema,
    )
    pq.write_table(table, path)


def _state(tag: str = "run") -> run_exp.RunState:
    now = run_exp.utc_now()
    return run_exp.RunState(
        tag=tag,
        status="running",
        git_commit="abcdef1234567890",
        source_config_path="exps/a.json",
        source_config_hash="0" * 64,
        created_at=now,
        updated_at=now,
        host="host",
        device="cpu",
        steps_trained=0,
        train_info=TrainInfo(),
    )


def test_config_validation_accepts_valid_config():
    config = run_exp.ExperimentConfig.model_validate(_config_dict())
    assert config.schema_version == 1
    assert config.train.validation_batches is None
    assert config.train.batch_size == 2


def test_config_validation_rejects_invalid_batch_split():
    data = _config_dict(train={"batch_size": 3, "micro_batches": 2})
    with pytest.raises(ValidationError, match="batch_size must be divisible"):
        run_exp.ExperimentConfig.model_validate(data)


def test_config_validation_rejects_seq_len_beyond_model_limit():
    data = _config_dict(train={"seq_len": 16})
    with pytest.raises(ValidationError, match="model.max_seq_len"):
        run_exp.ExperimentConfig.model_validate(data)


def test_tag_generation_uses_config_stem_timestamp_and_commit():
    tag = run_exp.make_run_tag(
        Path("exps/tiny.json"),
        "abcdef1234567890",
        now=datetime(2026, 1, 2, 3, 4, 5),
    )
    assert tag == "tiny-20260102-030405-abcdef12"


def test_git_gate_accepts_clean_tracked_pushed_config(monkeypatch, tmp_path):
    config_path = tmp_path / "exps" / "a.json"
    config_path.parent.mkdir()
    config_path.write_text("{}")

    def fake_git(args, *, repo_root):
        if args[:2] == ["ls-files", "--error-unmatch"]:
            return "exps/a.json"
        if args == ["status", "--porcelain"]:
            return ""
        if args[:2] == ["rev-parse", "--abbrev-ref"]:
            return "origin/main"
        if args == ["log", "@{u}..HEAD", "--oneline"]:
            return ""
        raise AssertionError(args)

    monkeypatch.setattr(run_exp, "git_stdout", fake_git)
    run_exp.ensure_git_ready(config_path, repo_root=tmp_path)


def test_git_gate_rejects_dirty_tree(monkeypatch, tmp_path):
    config_path = tmp_path / "exps" / "a.json"
    config_path.parent.mkdir()
    config_path.write_text("{}")

    def fake_git(args, *, repo_root):
        if args[:2] == ["ls-files", "--error-unmatch"]:
            return "exps/a.json"
        if args == ["status", "--porcelain"]:
            return " M src/spargel_llm/model.py"
        raise AssertionError(args)

    monkeypatch.setattr(run_exp, "git_stdout", fake_git)
    with pytest.raises(run_exp.ExperimentError, match="working tree"):
        run_exp.ensure_git_ready(config_path, repo_root=tmp_path)


def test_git_gate_rejects_unpushed_commits(monkeypatch, tmp_path):
    config_path = tmp_path / "exps" / "a.json"
    config_path.parent.mkdir()
    config_path.write_text("{}")

    def fake_git(args, *, repo_root):
        if args[:2] == ["ls-files", "--error-unmatch"]:
            return "exps/a.json"
        if args == ["status", "--porcelain"]:
            return ""
        if args[:2] == ["rev-parse", "--abbrev-ref"]:
            return "origin/main"
        if args == ["log", "@{u}..HEAD", "--oneline"]:
            return "abc123 local commit"
        raise AssertionError(args)

    monkeypatch.setattr(run_exp, "git_stdout", fake_git)
    with pytest.raises(run_exp.ExperimentError, match="unpushed"):
        run_exp.ensure_git_ready(config_path, repo_root=tmp_path)


def test_atomic_state_serialization(tmp_path):
    state_path = tmp_path / "state.json"
    state = _state()
    run_exp.save_state(state_path, state)

    loaded = run_exp.load_state(state_path)
    assert loaded.tag == "run"
    assert loaded.git_commit == "abcdef1234567890"


def test_duplicate_run_directory_is_refused(tmp_path):
    runs_dir = tmp_path / "runs"
    run_exp.create_run_dir("a", runs_dir=runs_dir)
    with pytest.raises(run_exp.ExperimentError, match="already exists"):
        run_exp.create_run_dir("a", runs_dir=runs_dir)


def test_resume_state_loading(tmp_path):
    run_dir = tmp_path / "runs" / "a"
    run_dir.mkdir(parents=True)
    config_path = run_dir / "config.json"
    _write_json(config_path, _config_dict())

    state = _state(tag="a")
    state.source_config_hash = run_exp.sha256_file(config_path)
    run_exp.save_state(run_dir / "state.json", state)

    config, loaded_state = run_exp.load_run_files(run_dir)
    assert config.train.steps == 3
    assert loaded_state.tag == "a"


def test_smoke_run_creates_state_and_checkpoints(tmp_path):
    assert "scripts.tool" not in sys.modules

    rows = [
        [4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
        [5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16],
        [6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17],
    ]
    _write_tokens(tmp_path / "data" / "train.parquet", rows)

    config_path = tmp_path / "exps" / "tiny.json"
    _write_json(config_path, _config_dict())

    run_dir = run_exp.start_experiment(
        config_path,
        repo_root=tmp_path,
        runs_dir=tmp_path / "runs",
        enforce_git=False,
        git_commit="abcdef1234567890",
        device="cpu",
        now=datetime(2026, 1, 2, 3, 4, 5),
        threads=1,
    )

    assert run_dir.name == "tiny-20260102-030405-abcdef12"
    assert (run_dir / "config.json").is_file()
    assert (run_dir / "iter0" / "model_state.pth").is_file()
    assert (run_dir / "iter0" / "optimizer_state.pth").is_file()
    assert (run_dir / "iter2" / "model_state.pth").is_file()
    assert (run_dir / "iter3" / "model_state.pth").is_file()

    state = run_exp.load_state(run_dir / "state.json")
    assert state.status == "completed"
    assert state.steps_trained == 3
    assert state.latest_checkpoint == "iter3"
    assert [record.step for record in state.checkpoint_records] == [0, 2, 3]
