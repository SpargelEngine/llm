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

RUN_EXP_PATH = (
    Path(__file__).resolve().parents[2] / "scripts" / "tianjiao" / "run_exp.py"
)
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
    assert config.train.lr_schedule.type == "constant"
    assert config.train.optimizer.type == "adamw"
    assert config.train.optimizer.beta1 == pytest.approx(0.9)
    assert config.train.optimizer.beta2 == pytest.approx(0.999)
    assert config.train.optimizer.eps == pytest.approx(1e-8)
    assert config.train.gradient_clip_norm is None


def test_config_validation_accepts_nested_lr_schedule():
    config = run_exp.ExperimentConfig.model_validate(
        _config_dict(
            train={
                "lr_schedule": {
                    "type": "warmup_constant_cooldown",
                    "warmup_steps": 1,
                    "cooldown_steps": 2,
                    "min_lr": 0.0001,
                }
            }
        )
    )

    schedule = run_exp.build_lr_schedule(config.train)
    assert isinstance(schedule, run_exp.LinearWarmupConstantCooldownSchedule)
    assert schedule.peak_lr == pytest.approx(0.001)
    assert schedule.total_steps == 3
    assert schedule.warmup_steps == 1
    assert schedule.cooldown_steps == 2
    assert schedule.min_lr == pytest.approx(0.0001)


def test_config_validation_accepts_warmup_step_decay_schedule():
    config = run_exp.ExperimentConfig.model_validate(
        _config_dict(
            train={
                "lr_schedule": {
                    "type": "warmup_step_decay",
                    "warmup_steps": 1,
                    "decay_steps": [2],
                    "decay_factor": 0.316,
                }
            }
        )
    )

    schedule = run_exp.build_lr_schedule(config.train)
    assert isinstance(schedule, run_exp.LinearWarmupStepDecaySchedule)
    assert schedule.peak_lr == pytest.approx(0.001)
    assert schedule.total_steps == 3
    assert schedule.warmup_steps == 1
    assert schedule.decay_steps == (2,)
    assert schedule.decay_factor == pytest.approx(0.316)


def test_config_validation_rejects_invalid_nested_lr_schedule():
    data = _config_dict(
        train={
            "lr_schedule": {
                "type": "warmup_constant_cooldown",
                "warmup_steps": 2,
                "cooldown_steps": 2,
                "min_lr": 0.0001,
            }
        }
    )
    with pytest.raises(ValidationError, match="warmup_steps \\+ cooldown_steps"):
        run_exp.ExperimentConfig.model_validate(data)


@pytest.mark.parametrize(
    "lr_schedule, match",
    [
        (
            {
                "type": "warmup_step_decay",
                "warmup_steps": 1,
                "decay_steps": [2, 2],
                "decay_factor": 0.316,
            },
            "decay_steps",
        ),
        (
            {
                "type": "warmup_step_decay",
                "warmup_steps": 2,
                "decay_steps": [1],
                "decay_factor": 0.316,
            },
            "warmup_steps",
        ),
        (
            {
                "type": "warmup_step_decay",
                "warmup_steps": 1,
                "decay_steps": [3],
                "decay_factor": 0.316,
            },
            "< steps",
        ),
        (
            {
                "type": "warmup_step_decay",
                "warmup_steps": 1,
                "decay_steps": [2],
                "decay_factor": 1.1,
            },
            "decay_factor",
        ),
    ],
)
def test_config_validation_rejects_invalid_warmup_step_decay_schedule(
    lr_schedule, match
):
    data = _config_dict(train={"lr_schedule": lr_schedule})
    with pytest.raises(ValidationError, match=match):
        run_exp.ExperimentConfig.model_validate(data)


def test_default_lr_schedule_builds_constant_schedule():
    config = run_exp.ExperimentConfig.model_validate(_config_dict())

    schedule = run_exp.build_lr_schedule(config.train)

    assert isinstance(schedule, run_exp.ConstantLearningRateSchedule)
    assert schedule.lr_at_step(0) == pytest.approx(0.001)


def test_config_validation_rejects_invalid_batch_split():
    data = _config_dict(train={"batch_size": 3, "micro_batches": 2})
    with pytest.raises(ValidationError, match="batch_size must be divisible"):
        run_exp.ExperimentConfig.model_validate(data)


def test_config_validation_rejects_seq_len_beyond_model_limit():
    data = _config_dict(train={"seq_len": 16})
    with pytest.raises(ValidationError, match="model.max_seq_len"):
        run_exp.ExperimentConfig.model_validate(data)


def test_config_validation_accepts_adamw_optimizer_config():
    config = run_exp.ExperimentConfig.model_validate(
        _config_dict(
            train={
                "optimizer": {
                    "type": "adamw",
                    "beta1": 0.9,
                    "beta2": 0.95,
                    "eps": 1e-8,
                }
            }
        )
    )

    assert config.train.optimizer.beta1 == pytest.approx(0.9)
    assert config.train.optimizer.beta2 == pytest.approx(0.95)
    assert config.train.optimizer.eps == pytest.approx(1e-8)


def test_create_optimizer_uses_adamw_config():
    config = run_exp.ExperimentConfig.model_validate(
        _config_dict(
            train={
                "optimizer": {
                    "type": "adamw",
                    "beta1": 0.9,
                    "beta2": 0.95,
                    "eps": 1e-8,
                }
            }
        )
    )
    model = run_exp.torch.nn.Linear(2, 2)

    optimizer = run_exp.create_optimizer(
        model,
        learning_rate=config.train.learning_rate,
        weight_decay=config.train.weight_decay,
        optimizer_config=config.train.optimizer,
    )

    assert optimizer.defaults["betas"] == (0.9, 0.95)
    assert optimizer.defaults["eps"] == pytest.approx(1e-8)


@pytest.mark.parametrize(
    "train_update, match",
    [
        ({"optimizer": {"type": "adamw", "beta1": 1.0}}, "beta1"),
        ({"optimizer": {"type": "adamw", "beta2": -0.1}}, "beta2"),
        ({"optimizer": {"type": "adamw", "eps": 0.0}}, "eps"),
        ({"gradient_clip_norm": 0.0}, "gradient_clip_norm"),
    ],
)
def test_config_validation_rejects_invalid_optimizer_and_clip_config(
    train_update, match
):
    data = _config_dict(train=train_update)
    with pytest.raises(ValidationError, match=match):
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


def test_smoke_run_creates_state_and_checkpoints(tmp_path, capsys):
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

    captured = capsys.readouterr()
    assert "lr=0.001" in captured.out


def test_smoke_run_with_warmup_cooldown_finishes_at_scheduled_lr(tmp_path, capsys):
    rows = [
        [4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
        [5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16],
        [6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17],
    ]
    _write_tokens(tmp_path / "data" / "train.parquet", rows)

    config_path = tmp_path / "exps" / "tiny.json"
    _write_json(
        config_path,
        _config_dict(
            train={
                "lr_schedule": {
                    "type": "warmup_constant_cooldown",
                    "warmup_steps": 1,
                    "cooldown_steps": 2,
                    "min_lr": 0.0001,
                }
            }
        ),
    )

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

    state = run_exp.load_state(run_dir / "state.json")
    assert state.status == "completed"
    assert state.steps_trained == 3
    assert state.latest_checkpoint == "iter3"

    optimizer_state = run_exp.torch.load(
        run_dir / "iter3" / "optimizer_state.pth",
        weights_only=True,
        map_location="cpu",
    )
    assert optimizer_state["param_groups"][0]["lr"] == pytest.approx(0.0001)

    captured = capsys.readouterr()
    assert "lr=0.0001" in captured.out


def test_smoke_run_with_warmup_step_decay_adamw_and_clip(tmp_path, capsys):
    rows = [
        [4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
        [5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16],
        [6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17],
    ]
    _write_tokens(tmp_path / "data" / "train.parquet", rows)

    config_path = tmp_path / "exps" / "tiny.json"
    _write_json(
        config_path,
        _config_dict(
            train={
                "lr_schedule": {
                    "type": "warmup_step_decay",
                    "warmup_steps": 1,
                    "decay_steps": [2],
                    "decay_factor": 0.316,
                },
                "optimizer": {
                    "type": "adamw",
                    "beta1": 0.9,
                    "beta2": 0.95,
                    "eps": 1e-8,
                },
                "gradient_clip_norm": 1.0,
            }
        ),
    )

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

    state = run_exp.load_state(run_dir / "state.json")
    assert state.status == "completed"
    assert state.steps_trained == 3
    assert state.latest_checkpoint == "iter3"

    optimizer_state = run_exp.torch.load(
        run_dir / "iter3" / "optimizer_state.pth",
        weights_only=True,
        map_location="cpu",
    )
    param_group = optimizer_state["param_groups"][0]
    assert param_group["lr"] == pytest.approx(0.001 * 0.316)
    assert param_group["betas"] == (0.9, 0.95)
    assert param_group["eps"] == pytest.approx(1e-8)

    captured = capsys.readouterr()
    assert "lr=0.000316" in captured.out


def test_deepseek_v2_lite_hparams_config_loads():
    path = Path(__file__).resolve().parents[2] / "exps" / (
        "0701-deepseek-v2-lite-hparams-100M-3B-climbmix.json"
    )

    config = run_exp.load_experiment_config(path)

    assert config.train.steps == 11445
    assert config.train.seq_len == 256
    assert config.train.batch_size == 1024
    assert config.train.micro_batches == 2
    assert config.train.learning_rate == pytest.approx(0.00042)
    assert config.train.weight_decay == pytest.approx(0.1)
    assert config.train.optimizer.beta1 == pytest.approx(0.9)
    assert config.train.optimizer.beta2 == pytest.approx(0.95)
    assert config.train.gradient_clip_norm == pytest.approx(1.0)
    assert config.train.lr_schedule.type == "warmup_step_decay"
    assert config.train.lr_schedule.warmup_steps == 2000
    assert config.train.lr_schedule.decay_steps == [9156, 10301]
    assert config.train.lr_schedule.decay_factor == pytest.approx(0.316)
