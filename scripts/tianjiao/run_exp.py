#!/usr/bin/env python3
"""Reproducible experiment runner.

This script intentionally does not import ``scripts.tool``.  It uses the
package training APIs directly and keeps run metadata in ``runs/<tag>``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import socket
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

import pyarrow.parquet as pq
import torch
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    NonNegativeFloat,
    NonNegativeInt,
    PositiveFloat,
    PositiveInt,
    model_validator,
)
from torch.optim import AdamW, Optimizer
from torch.utils.tensorboard import SummaryWriter

from spargel_llm.lr_schedule import (
    ConstantLearningRateSchedule,
    LearningRateSchedule,
    LinearWarmupStepDecaySchedule,
    LinearWarmupConstantCooldownSchedule,
)
from spargel_llm.model import Config, Model
from spargel_llm.parquet_utils import get_dataset_id
from spargel_llm.train import (
    StepInfo,
    TrainInfo,
    TrainTracker,
    compute_validation_metrics,
    iter_batches,
    train,
)

PAD = 1
SOT = 2
EOT = 3

REPO_ROOT = Path(__file__).resolve().parents[2]
RUNS_DIR = REPO_ROOT / "runs"

Float32Precision = Literal["highest", "high", "medium"]
RunStatus = Literal["running", "completed", "stopped", "failed"]


class ExperimentError(RuntimeError):
    """Raised for expected runner failures that should not print a traceback."""


class DataConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    train_path: str = Field(min_length=1)
    validation_path: str | None = None
    start_index: NonNegativeInt = 0
    start_offset: NonNegativeInt = 0
    add_sot: bool = False
    add_eot: bool = False


class ConstantLrScheduleConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    type: Literal["constant"] = "constant"


class WarmupConstantCooldownLrScheduleConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    type: Literal["warmup_constant_cooldown"]
    warmup_steps: NonNegativeInt
    cooldown_steps: NonNegativeInt
    min_lr: NonNegativeFloat


class WarmupStepDecayLrScheduleConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    type: Literal["warmup_step_decay"]
    warmup_steps: NonNegativeInt
    decay_steps: list[NonNegativeInt]
    decay_factor: PositiveFloat


LrScheduleConfig = (
    ConstantLrScheduleConfig
    | WarmupConstantCooldownLrScheduleConfig
    | WarmupStepDecayLrScheduleConfig
)


class AdamWOptimizerConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    type: Literal["adamw"] = "adamw"
    beta1: float = Field(default=0.9, ge=0, lt=1)
    beta2: float = Field(default=0.999, ge=0, lt=1)
    eps: PositiveFloat = 1e-8


OptimizerConfig = AdamWOptimizerConfig


class TrainRunConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    steps: PositiveInt
    seq_len: PositiveInt
    batch_size: PositiveInt
    learning_rate: PositiveFloat
    weight_decay: NonNegativeFloat
    micro_batches: PositiveInt = 1
    log_period: PositiveInt = 10
    checkpoint_interval: PositiveInt
    validation_batches: PositiveInt | None = None
    loop_dataset: bool = False
    seed: int = 0
    use_bf16: bool = True
    float32_precision: Float32Precision = "high"
    optimizer: OptimizerConfig = Field(default_factory=AdamWOptimizerConfig)
    gradient_clip_norm: PositiveFloat | None = None
    lr_schedule: LrScheduleConfig = Field(
        default_factory=ConstantLrScheduleConfig,
        discriminator="type",
    )

    @model_validator(mode="after")
    def _validate_train_config(self) -> "TrainRunConfig":
        if self.batch_size % self.micro_batches != 0:
            raise ValueError(
                "batch_size must be divisible by micro_batches "
                f"({self.batch_size} % {self.micro_batches} != 0)"
            )
        if self.lr_schedule.type == "warmup_constant_cooldown":
            scheduled_steps = (
                self.lr_schedule.warmup_steps + self.lr_schedule.cooldown_steps
            )
            if scheduled_steps > self.steps:
                raise ValueError("warmup_steps + cooldown_steps must be <= steps")
            if self.lr_schedule.min_lr > self.learning_rate:
                raise ValueError("lr_schedule.min_lr must be <= learning_rate")
        elif self.lr_schedule.type == "warmup_step_decay":
            if self.lr_schedule.decay_factor > 1:
                raise ValueError(
                    "lr_schedule.decay_factor must satisfy 0 < decay_factor <= 1"
                )
            previous = -1
            for decay_step in self.lr_schedule.decay_steps:
                if decay_step <= previous:
                    raise ValueError("lr_schedule.decay_steps must be sorted")
                if decay_step < self.lr_schedule.warmup_steps:
                    raise ValueError(
                        "lr_schedule.decay_steps must be >= warmup_steps"
                    )
                if decay_step >= self.steps:
                    raise ValueError("lr_schedule.decay_steps must be < steps")
                previous = decay_step
        return self


def build_lr_schedule(config: TrainRunConfig) -> LearningRateSchedule:
    lr_schedule = config.lr_schedule
    if lr_schedule.type == "constant":
        return ConstantLearningRateSchedule(config.learning_rate)
    if lr_schedule.type == "warmup_constant_cooldown":
        return LinearWarmupConstantCooldownSchedule(
            peak_lr=config.learning_rate,
            total_steps=config.steps,
            warmup_steps=lr_schedule.warmup_steps,
            cooldown_steps=lr_schedule.cooldown_steps,
            min_lr=lr_schedule.min_lr,
        )
    if lr_schedule.type == "warmup_step_decay":
        return LinearWarmupStepDecaySchedule(
            peak_lr=config.learning_rate,
            total_steps=config.steps,
            warmup_steps=lr_schedule.warmup_steps,
            decay_steps=lr_schedule.decay_steps,
            decay_factor=lr_schedule.decay_factor,
        )
    raise ValueError(f"unknown lr_schedule type: {lr_schedule.type}")


class ExperimentConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: Literal[1]
    model: Config
    tokenizer: str = Field(min_length=1)
    data: DataConfig
    train: TrainRunConfig

    @model_validator(mode="after")
    def _validate_model_shape(self) -> "ExperimentConfig":
        if self.train.seq_len > self.model.max_seq_len:
            raise ValueError(
                "train.seq_len must be <= model.max_seq_len "
                f"({self.train.seq_len} > {self.model.max_seq_len})"
            )
        return self


class CheckpointRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")

    step: NonNegativeInt
    path: str
    created_at: str
    train_info: TrainInfo


class RunState(BaseModel):
    model_config = ConfigDict(extra="forbid")

    tag: str
    status: RunStatus
    git_commit: str
    source_config_path: str
    source_config_hash: str
    created_at: str
    updated_at: str
    host: str
    device: str
    device_name: str | None = None
    steps_trained: NonNegativeInt = 0
    train_info: TrainInfo
    latest_checkpoint: str | None = None
    checkpoint_records: list[CheckpointRecord] = Field(default_factory=list)
    failure: str | None = None


@dataclass
class LogState:
    sum_loss: float = 0.0
    sum_time: float = 0.0
    tokens: int = 0
    tokens_non_pad: int = 0
    val_exhausted_warned: bool = False


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def make_run_tag(
    config_path: Path, git_commit: str, *, now: datetime | None = None
) -> str:
    timestamp = (now or datetime.now()).strftime("%Y%m%d-%H%M%S")
    return f"{config_path.stem}-{timestamp}-{git_commit[:8]}"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def relative_or_absolute(path: Path, root: Path) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return str(path.resolve())


def atomic_write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp-{os.getpid()}-{time.time_ns()}")
    try:
        with tmp.open("w") as f:
            json.dump(data, f, indent=2)
            f.write("\n")
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    finally:
        if tmp.exists():
            tmp.unlink()


def save_state(path: Path, state: RunState) -> None:
    state.updated_at = utc_now()
    atomic_write_json(path, state.model_dump(mode="json"))


def load_state(path: Path) -> RunState:
    return RunState.model_validate_json(path.read_text())


def load_experiment_config(path: Path) -> ExperimentConfig:
    return ExperimentConfig.model_validate_json(path.read_text())


def git_stdout(args: list[str], *, repo_root: Path = REPO_ROOT) -> str:
    result = subprocess.run(
        ["git", *args],
        cwd=repo_root,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        detail = result.stderr.strip() or result.stdout.strip()
        raise ExperimentError(f"git {' '.join(args)} failed: {detail}")
    return result.stdout.strip()


def ensure_config_in_exps(
    config_path: str | Path, *, repo_root: Path = REPO_ROOT
) -> Path:
    path = Path(config_path)
    if not path.is_absolute():
        path = repo_root / path
    path = path.resolve()

    exps_dir = (repo_root / "exps").resolve()
    try:
        path.relative_to(exps_dir)
    except ValueError as exc:
        raise ExperimentError(
            f"experiment config must be under {exps_dir}: {path}"
        ) from exc

    if path.suffix != ".json":
        raise ExperimentError(f"experiment config must be a JSON file: {path}")
    if not path.is_file():
        raise ExperimentError(f"experiment config does not exist: {path}")
    return path


def ensure_run_dir_in_runs(
    run_path: str | Path,
    *,
    repo_root: Path = REPO_ROOT,
    runs_dir: Path | None = None,
) -> Path:
    runs_dir = (runs_dir or repo_root / "runs").resolve()
    path = Path(run_path)
    if not path.is_absolute():
        path = repo_root / path
    path = path.resolve()

    try:
        path.relative_to(runs_dir)
    except ValueError as exc:
        raise ExperimentError(
            f"run directory must be under {runs_dir}: {path}"
        ) from exc
    if not path.is_dir():
        raise ExperimentError(f"run directory does not exist: {path}")
    return path


def ensure_git_ready(config_path: Path, *, repo_root: Path = REPO_ROOT) -> None:
    rel_config = relative_or_absolute(config_path, repo_root)

    git_stdout(["ls-files", "--error-unmatch", "--", rel_config], repo_root=repo_root)

    status = git_stdout(["status", "--porcelain"], repo_root=repo_root)
    if status:
        raise ExperimentError(
            "working tree is not clean; commit all changes before running"
        )

    try:
        git_stdout(
            ["rev-parse", "--abbrev-ref", "--symbolic-full-name", "@{u}"],
            repo_root=repo_root,
        )
    except ExperimentError as exc:
        raise ExperimentError(
            "current branch has no upstream; push the branch before running"
        ) from exc

    unpushed = git_stdout(["log", "@{u}..HEAD", "--oneline"], repo_root=repo_root)
    if unpushed:
        raise ExperimentError(
            "current branch has unpushed commits; push before running"
        )


def get_git_commit(*, repo_root: Path = REPO_ROOT) -> str:
    return git_stdout(["rev-parse", "HEAD"], repo_root=repo_root)


def create_run_dir(tag: str, *, runs_dir: Path = RUNS_DIR) -> Path:
    run_dir = runs_dir / tag
    try:
        run_dir.mkdir(parents=True, exist_ok=False)
    except FileExistsError as exc:
        raise ExperimentError(f"run directory already exists: {run_dir}") from exc
    return run_dir


def resolve_repo_path(path: str, *, repo_root: Path = REPO_ROOT) -> Path:
    result = Path(path)
    if result.is_absolute():
        return result
    return repo_root / result


def load_parquet(path: Path) -> pq.ParquetFile:
    if not path.is_file():
        raise ExperimentError(f"Parquet file does not exist: {path}")
    return pq.ParquetFile(str(path))


def choose_device(device: str | None = None) -> str:
    if device is not None:
        return device
    return "cuda" if torch.cuda.is_available() else "cpu"


def get_device_name(device: str) -> str | None:
    if device == "cuda":
        return torch.cuda.get_device_name()
    return None


def configure_torch(
    train_config: TrainRunConfig, *, device: str, threads: int | None = None
) -> None:
    torch.set_printoptions(linewidth=160)
    if threads is not None:
        torch.set_num_threads(threads)
    if device == "cuda":
        torch.set_float32_matmul_precision(train_config.float32_precision)


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def create_optimizer(
    model: Model,
    *,
    learning_rate: float,
    weight_decay: float,
    optimizer_config: OptimizerConfig | None = None,
) -> AdamW:
    optimizer_config = optimizer_config or AdamWOptimizerConfig()
    if optimizer_config.type == "adamw":
        return AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
            betas=(optimizer_config.beta1, optimizer_config.beta2),
            eps=optimizer_config.eps,
        )
    raise ValueError(f"unknown optimizer type: {optimizer_config.type}")


def build_model_and_optimizer(
    config: ExperimentConfig, *, device: str
) -> tuple[Model, Optimizer]:
    model = Model(config.model).to(device)
    optimizer = create_optimizer(
        model,
        learning_rate=config.train.learning_rate,
        weight_decay=config.train.weight_decay,
        optimizer_config=config.train.optimizer,
    )
    return model, optimizer


def save_checkpoint(
    run_dir: Path,
    *,
    step: int,
    model: Model,
    optimizer: Optimizer,
    train_info: TrainInfo,
) -> CheckpointRecord:
    target = run_dir / f"iter{step}"
    if target.exists():
        raise ExperimentError(f"checkpoint already exists: {target}")

    tmp = run_dir / f".iter{step}.tmp-{os.getpid()}-{time.time_ns()}"
    tmp.mkdir()
    try:
        torch.save(model.state_dict(), tmp / "model_state.pth")
        torch.save(optimizer.state_dict(), tmp / "optimizer_state.pth")
        os.replace(tmp, target)
    except Exception:
        if tmp.exists():
            shutil.rmtree(tmp)
        raise

    return CheckpointRecord(
        step=step,
        path=target.name,
        created_at=utc_now(),
        train_info=train_info.model_copy(deep=True),
    )


def load_checkpoint(
    run_dir: Path,
    record: CheckpointRecord,
    *,
    model: Model,
    optimizer: Optimizer,
    device: str,
) -> None:
    checkpoint_dir = run_dir / record.path
    model_path = checkpoint_dir / "model_state.pth"
    optimizer_path = checkpoint_dir / "optimizer_state.pth"
    if not model_path.is_file() or not optimizer_path.is_file():
        raise ExperimentError(f"checkpoint is incomplete: {checkpoint_dir}")

    model.load_state_dict(
        torch.load(model_path, weights_only=True, map_location=device)
    )
    optimizer.load_state_dict(torch.load(optimizer_path, map_location=device))


def latest_checkpoint_record(state: RunState) -> CheckpointRecord | None:
    if not state.checkpoint_records:
        return None
    if state.latest_checkpoint is not None:
        for record in state.checkpoint_records:
            if record.path == state.latest_checkpoint:
                return record
        raise ExperimentError(
            f"latest checkpoint is missing from state: {state.latest_checkpoint}"
        )
    return max(state.checkpoint_records, key=lambda record: record.step)


def record_checkpoint(state: RunState, record: CheckpointRecord) -> None:
    state.checkpoint_records = [
        old for old in state.checkpoint_records if old.step != record.step
    ]
    state.checkpoint_records.append(record)
    state.checkpoint_records.sort(key=lambda item: item.step)
    state.latest_checkpoint = record.path


def make_initial_state(
    *,
    tag: str,
    git_commit: str,
    source_config_path: Path,
    source_config_hash: str,
    config: ExperimentConfig,
    repo_root: Path,
    device: str,
) -> RunState:
    now = utc_now()
    return RunState(
        tag=tag,
        status="running",
        git_commit=git_commit,
        source_config_path=relative_or_absolute(source_config_path, repo_root),
        source_config_hash=source_config_hash,
        created_at=now,
        updated_at=now,
        host=socket.gethostname(),
        device=device,
        device_name=get_device_name(device),
        steps_trained=0,
        train_info=TrainInfo(
            index=config.data.start_index,
            offset=config.data.start_offset,
        ),
    )


def load_run_files(run_dir: Path) -> tuple[ExperimentConfig, RunState]:
    config_path = run_dir / "config.json"
    state_path = run_dir / "state.json"
    if not config_path.is_file():
        raise ExperimentError(f"run config is missing: {config_path}")
    if not state_path.is_file():
        raise ExperimentError(f"run state is missing: {state_path}")

    config = load_experiment_config(config_path)
    state = load_state(state_path)
    if state.tag != run_dir.name:
        raise ExperimentError(
            f"state tag {state.tag!r} does not match run directory {run_dir.name!r}"
        )
    if sha256_file(config_path) != state.source_config_hash:
        raise ExperimentError("run config hash does not match state.json")
    return config, state


def _persist_progress(state_path: Path, state: RunState, train_info: TrainInfo) -> None:
    state.train_info = train_info.model_copy(deep=True)
    save_state(state_path, state)


def _log_step(
    step_info: StepInfo,
    *,
    state: RunState,
    train_info: TrainInfo,
    log_state: LogState,
    writer: SummaryWriter | None,
    model: Model,
    val_dataset: pq.ParquetFile | None,
    val_batches: int,
    config: ExperimentConfig,
    micro_batch_size: int,
    device: str,
) -> None:
    token_count = train_info.token_count

    if writer is not None:
        writer.add_scalar("loss/train", step_info.loss, token_count)
        writer.add_scalar("metric/time/elapsed", train_info.time, token_count)
        if step_info.learning_rate is not None:
            writer.add_scalar("lr/train", step_info.learning_rate, token_count)

    log_state.sum_loss += step_info.loss * step_info.tokens_non_pad
    log_state.sum_time += step_info.time
    log_state.tokens += step_info.tokens
    log_state.tokens_non_pad += step_info.tokens_non_pad

    completed_step = step_info.step + 1
    if completed_step % config.train.log_period != 0:
        return

    avg_loss = log_state.sum_loss / max(log_state.tokens_non_pad, 1)
    avg_time = log_state.sum_time / config.train.log_period

    val_metrics = None
    val_time = 0.0
    if val_dataset is not None:
        t_val_start = time.perf_counter()
        val_loss, val_entropy, actual_batches = compute_validation_metrics(
            model=model,
            dataset=val_dataset,
            seq_len=config.train.seq_len,
            batch_size=micro_batch_size,
            pad_index=PAD,
            device=device,
            num_batches=val_batches,
            eot_index=EOT if config.data.add_eot else None,
            sot_index=SOT if config.data.add_sot else None,
            use_bf16=config.train.use_bf16,
        )
        if actual_batches < val_batches and not log_state.val_exhausted_warned:
            print(
                "warning: validation dataset exhausted early "
                f"({actual_batches}/{val_batches} batches)",
                file=sys.stderr,
            )
            log_state.val_exhausted_warned = True
        if device == "cuda":
            torch.cuda.synchronize()
        val_time = time.perf_counter() - t_val_start
        val_metrics = (val_loss, val_entropy)

    if writer is not None:
        writer.add_scalar("metric/time/avg", avg_time, token_count)
        if val_metrics is not None:
            writer.add_scalar("metric/time/val", val_time, token_count)

    lr_text = (
        "unknown"
        if step_info.learning_rate is None
        else f"{step_info.learning_rate:.6g}"
    )

    if val_metrics is None:
        print(
            f"  {completed_step}: avg_loss={avg_loss:.6f}, "
            f"lr={lr_text}, avg_time={avg_time:.6f}"
        )
    else:
        val_loss, val_entropy = val_metrics
        print(
            f"  {completed_step}: avg_loss={avg_loss:.6f}, "
            f"val_loss={val_loss:.6f}, val_entropy={val_entropy:.6f}, "
            f"lr={lr_text}, avg_time={avg_time:.6f}, "
            f"val_time={val_time:.6f}"
        )
        if writer is not None:
            writer.add_scalar("loss/val", val_loss, token_count)
            writer.add_scalar("entropy/val", val_entropy, token_count)

    log_state.sum_loss = 0.0
    log_state.sum_time = 0.0
    log_state.tokens = 0
    log_state.tokens_non_pad = 0


def _finalize_success(
    *,
    run_dir: Path,
    state_path: Path,
    state: RunState,
    model: Model,
    optimizer: Optimizer,
    train_info: TrainInfo,
    target_steps: int,
) -> None:
    latest_record = latest_checkpoint_record(state)
    latest_step = latest_record.step if latest_record is not None else None
    if state.steps_trained > 0 and latest_step != state.steps_trained:
        record = save_checkpoint(
            run_dir,
            step=state.steps_trained,
            model=model,
            optimizer=optimizer,
            train_info=train_info,
        )
        record_checkpoint(state, record)
        _persist_progress(state_path, state, train_info)

    if state.steps_trained >= target_steps:
        state.status = "completed"
    elif state.status != "failed":
        state.status = "stopped"
    state.failure = None
    _persist_progress(state_path, state, train_info)


def _train_to_target(
    *,
    run_dir: Path,
    state: RunState,
    config: ExperimentConfig,
    model: Model,
    optimizer: Optimizer,
    repo_root: Path,
    device: str,
) -> None:
    state_path = run_dir / "state.json"

    train_dataset = load_parquet(
        resolve_repo_path(config.data.train_path, repo_root=repo_root)
    )
    dataset_id = get_dataset_id(train_dataset)
    if state.train_info.dataset_id and dataset_id:
        if state.train_info.dataset_id != dataset_id:
            raise ExperimentError(
                "training dataset_id changed "
                f"({state.train_info.dataset_id!r} -> {dataset_id!r})"
            )
    if dataset_id:
        state.train_info.dataset_id = dataset_id
        _persist_progress(state_path, state, state.train_info)

    val_dataset = None
    if config.data.validation_path is not None:
        val_dataset = load_parquet(
            resolve_repo_path(config.data.validation_path, repo_root=repo_root)
        )

    micro_batch_size = config.train.batch_size // config.train.micro_batches
    val_batches = config.train.validation_batches
    if val_batches is None:
        val_batches = max(config.train.log_period // 10, 1)

    train_info = state.train_info.model_copy(deep=True)
    log_state = LogState()
    writer = SummaryWriter(str(run_dir / "tensorboard"))
    lr_schedule = build_lr_schedule(config.train)

    try:
        while state.steps_trained < config.train.steps:
            start_index = train_info.index
            start_offset = train_info.offset
            train_tracker = TrainTracker(index=start_index, offset=start_offset)

            def make_iterator():
                return iter_batches(
                    train_dataset,
                    config.train.seq_len,
                    micro_batch_size,
                    PAD,
                    start_index=start_index,
                    start_offset=start_offset,
                    tracker=train_tracker,
                    eot_index=EOT if config.data.add_eot else None,
                    sot_index=SOT if config.data.add_sot else None,
                )

            def step_callback(step_info: StepInfo) -> None:
                completed_step = step_info.step + 1

                train_info.index = train_tracker.index
                train_info.offset = train_tracker.offset
                state.steps_trained = completed_step
                _persist_progress(state_path, state, train_info)

                _log_step(
                    step_info,
                    state=state,
                    train_info=train_info,
                    log_state=log_state,
                    writer=writer,
                    model=model,
                    val_dataset=val_dataset,
                    val_batches=val_batches,
                    config=config,
                    micro_batch_size=micro_batch_size,
                    device=device,
                )

                if completed_step % config.train.checkpoint_interval == 0:
                    record = save_checkpoint(
                        run_dir,
                        step=completed_step,
                        model=model,
                        optimizer=optimizer,
                        train_info=train_info,
                    )
                    record_checkpoint(state, record)
                    _persist_progress(state_path, state, train_info)

            remaining = config.train.steps - state.steps_trained
            actual_steps = train(
                info=train_info,
                model=model,
                optimizer=optimizer,
                batch_iterator=make_iterator(),
                pad_index=PAD,
                steps=remaining,
                device=device,
                step_callback=step_callback,
                micro_batches=config.train.micro_batches,
                step_offset=state.steps_trained,
                use_bf16=config.train.use_bf16,
                lr_schedule=lr_schedule,
                gradient_clip_norm=config.train.gradient_clip_norm,
            )

            if actual_steps == remaining:
                break

            if not config.train.loop_dataset:
                print("training dataset exhausted; stopping early", file=sys.stderr)
                break

            if actual_steps == 0 and start_index == 0 and start_offset == 0:
                raise ExperimentError("training dataset produced no full batches")

            print("training dataset exhausted; restarting from beginning")
            train_info.index = 0
            train_info.offset = 0
            state.train_info = train_info.model_copy(deep=True)
            _persist_progress(state_path, state, train_info)
    finally:
        writer.close()

    _finalize_success(
        run_dir=run_dir,
        state_path=state_path,
        state=state,
        model=model,
        optimizer=optimizer,
        train_info=train_info,
        target_steps=config.train.steps,
    )


def start_experiment(
    config_path: str | Path,
    *,
    repo_root: Path = REPO_ROOT,
    runs_dir: Path | None = None,
    enforce_git: bool = True,
    git_commit: str | None = None,
    device: str | None = None,
    now: datetime | None = None,
    threads: int | None = None,
) -> Path:
    source_config_path = ensure_config_in_exps(config_path, repo_root=repo_root)
    if enforce_git:
        ensure_git_ready(source_config_path, repo_root=repo_root)

    config = load_experiment_config(source_config_path)
    device = choose_device(device)
    configure_torch(config.train, device=device, threads=threads)
    set_seed(config.train.seed)

    commit = git_commit or get_git_commit(repo_root=repo_root)
    tag = make_run_tag(source_config_path, commit, now=now)
    run_dir = create_run_dir(tag, runs_dir=(runs_dir or repo_root / "runs"))
    shutil.copyfile(source_config_path, run_dir / "config.json")

    state = make_initial_state(
        tag=tag,
        git_commit=commit,
        source_config_path=source_config_path,
        source_config_hash=sha256_file(source_config_path),
        config=config,
        repo_root=repo_root,
        device=device,
    )
    state_path = run_dir / "state.json"
    save_state(state_path, state)

    try:
        model, optimizer = build_model_and_optimizer(config, device=device)
        record = save_checkpoint(
            run_dir,
            step=0,
            model=model,
            optimizer=optimizer,
            train_info=state.train_info,
        )
        record_checkpoint(state, record)
        save_state(state_path, state)

        _train_to_target(
            run_dir=run_dir,
            state=state,
            config=config,
            model=model,
            optimizer=optimizer,
            repo_root=repo_root,
            device=device,
        )
    except Exception as exc:
        state.status = "failed"
        state.failure = str(exc)
        save_state(state_path, state)
        raise

    return run_dir


def resume_experiment(
    run_path: str | Path,
    *,
    repo_root: Path = REPO_ROOT,
    runs_dir: Path | None = None,
    device: str | None = None,
    threads: int | None = None,
) -> Path:
    run_dir = ensure_run_dir_in_runs(run_path, repo_root=repo_root, runs_dir=runs_dir)
    config, state = load_run_files(run_dir)
    device = choose_device(device)
    configure_torch(config.train, device=device, threads=threads)
    set_seed(config.train.seed)

    state.status = "running"
    state.failure = None
    state.host = socket.gethostname()
    state.device = device
    state.device_name = get_device_name(device)
    state_path = run_dir / "state.json"
    save_state(state_path, state)

    try:
        model, optimizer = build_model_and_optimizer(config, device=device)
        checkpoint = latest_checkpoint_record(state)
        if checkpoint is None:
            raise ExperimentError("run has no checkpoint to resume from")
        load_checkpoint(
            run_dir,
            checkpoint,
            model=model,
            optimizer=optimizer,
            device=device,
        )
        if state.steps_trained != checkpoint.step:
            state.steps_trained = checkpoint.step
            state.train_info = checkpoint.train_info.model_copy(deep=True)
            save_state(state_path, state)

        if state.steps_trained >= config.train.steps:
            state.status = "completed"
            save_state(state_path, state)
            return run_dir

        _train_to_target(
            run_dir=run_dir,
            state=state,
            config=config,
            model=model,
            optimizer=optimizer,
            repo_root=repo_root,
            device=device,
        )
    except Exception as exc:
        state.status = "failed"
        state.failure = str(exc)
        save_state(state_path, state)
        raise

    return run_dir


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Reproducible experiment runner")
    parser.add_argument(
        "--device",
        choices=["cpu", "cuda"],
        default=None,
        help="training device (default: cuda if available, else cpu)",
    )
    parser.add_argument(
        "--threads",
        type=int,
        default=None,
        help="number of PyTorch CPU threads",
    )

    subparsers = parser.add_subparsers(dest="action", required=True)

    run_parser = subparsers.add_parser("run", help="start a new experiment")
    run_parser.add_argument("config", help="tracked JSON config under exps/")

    resume_parser = subparsers.add_parser("resume", help="resume an existing run")
    resume_parser.add_argument("run_dir", help="run directory under runs/")

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = create_parser()
    args = parser.parse_args(argv)

    try:
        if args.action == "run":
            run_dir = start_experiment(
                args.config,
                device=args.device,
                threads=args.threads,
            )
        elif args.action == "resume":
            run_dir = resume_experiment(
                args.run_dir,
                device=args.device,
                threads=args.threads,
            )
        else:
            raise ExperimentError(f"unknown action: {args.action}")
    except ExperimentError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    print(run_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
