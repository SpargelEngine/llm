import pytest

from spargel_llm.lr_schedule import (
    ConstantLearningRateSchedule,
    LinearWarmupStepDecaySchedule,
    LinearWarmupConstantCooldownSchedule,
)


def test_constant_returns_configured_lr_for_nonnegative_steps():
    schedule = ConstantLearningRateSchedule(0.001)

    assert schedule.lr_at_step(0) == pytest.approx(0.001)
    assert schedule.lr_at_step(10) == pytest.approx(0.001)


def test_constant_rejects_negative_steps():
    schedule = ConstantLearningRateSchedule(0.001)

    with pytest.raises(ValueError, match="step must be non-negative"):
        schedule.lr_at_step(-1)


def test_warmup_reaches_peak_on_last_warmup_update():
    schedule = LinearWarmupConstantCooldownSchedule(
        peak_lr=0.001,
        total_steps=10,
        warmup_steps=4,
        cooldown_steps=2,
        min_lr=0.0001,
    )

    assert schedule.lr_at_step(0) == pytest.approx(0.00025)
    assert schedule.lr_at_step(3) == pytest.approx(0.001)


def test_plateau_remains_at_peak_lr():
    schedule = LinearWarmupConstantCooldownSchedule(
        peak_lr=0.001,
        total_steps=10,
        warmup_steps=2,
        cooldown_steps=3,
        min_lr=0.0001,
    )

    assert schedule.lr_at_step(2) == pytest.approx(0.001)
    assert schedule.lr_at_step(6) == pytest.approx(0.001)


def test_cooldown_reaches_min_lr_on_final_update():
    schedule = LinearWarmupConstantCooldownSchedule(
        peak_lr=0.001,
        total_steps=10,
        warmup_steps=2,
        cooldown_steps=3,
        min_lr=0.0001,
    )

    assert schedule.lr_at_step(7) == pytest.approx(0.001)
    assert schedule.lr_at_step(8) == pytest.approx(0.00055)
    assert schedule.lr_at_step(9) == pytest.approx(0.0001)


def test_single_step_cooldown_reaches_min_lr():
    schedule = LinearWarmupConstantCooldownSchedule(
        peak_lr=0.001,
        total_steps=3,
        warmup_steps=1,
        cooldown_steps=1,
        min_lr=0.0001,
    )

    assert schedule.lr_at_step(2) == pytest.approx(0.0001)


@pytest.mark.parametrize(
    "kwargs, match",
    [
        (
            {
                "peak_lr": 0.001,
                "total_steps": 0,
                "warmup_steps": 0,
                "cooldown_steps": 0,
                "min_lr": 0.0,
            },
            "total_steps must be positive",
        ),
        (
            {
                "peak_lr": 0.001,
                "total_steps": 10,
                "warmup_steps": 8,
                "cooldown_steps": 3,
                "min_lr": 0.0,
            },
            "warmup_steps \\+ cooldown_steps",
        ),
        (
            {
                "peak_lr": 0.001,
                "total_steps": 10,
                "warmup_steps": 0,
                "cooldown_steps": 0,
                "min_lr": 0.01,
            },
            "min_lr",
        ),
    ],
)
def test_invalid_warmup_cooldown_configs_raise(kwargs, match):
    with pytest.raises(ValueError, match=match):
        LinearWarmupConstantCooldownSchedule(**kwargs)


def test_warmup_cooldown_rejects_negative_steps():
    schedule = LinearWarmupConstantCooldownSchedule(
        peak_lr=0.001,
        total_steps=10,
        warmup_steps=2,
        cooldown_steps=2,
        min_lr=0.0001,
    )

    with pytest.raises(ValueError, match="step must be non-negative"):
        schedule.lr_at_step(-1)


def test_warmup_step_decay_warms_up_to_peak():
    schedule = LinearWarmupStepDecaySchedule(
        peak_lr=0.001,
        total_steps=10,
        warmup_steps=4,
        decay_steps=[8, 9],
        decay_factor=0.316,
    )

    assert schedule.lr_at_step(0) == pytest.approx(0.00025)
    assert schedule.lr_at_step(3) == pytest.approx(0.001)


def test_warmup_step_decay_applies_each_decay_from_decay_step():
    schedule = LinearWarmupStepDecaySchedule(
        peak_lr=0.00042,
        total_steps=11445,
        warmup_steps=2000,
        decay_steps=[9156, 10301],
        decay_factor=0.316,
    )

    assert schedule.lr_at_step(9155) == pytest.approx(0.00042)
    assert schedule.lr_at_step(9156) == pytest.approx(0.00042 * 0.316)
    assert schedule.lr_at_step(10300) == pytest.approx(0.00042 * 0.316)
    assert schedule.lr_at_step(10301) == pytest.approx(0.00042 * 0.316 * 0.316)


@pytest.mark.parametrize(
    "kwargs, match",
    [
        (
            {
                "peak_lr": 0.001,
                "total_steps": 10,
                "warmup_steps": 4,
                "decay_steps": [5, 5],
                "decay_factor": 0.316,
            },
            "strictly increasing",
        ),
        (
            {
                "peak_lr": 0.001,
                "total_steps": 10,
                "warmup_steps": 4,
                "decay_steps": [3],
                "decay_factor": 0.316,
            },
            ">= warmup_steps",
        ),
        (
            {
                "peak_lr": 0.001,
                "total_steps": 10,
                "warmup_steps": 4,
                "decay_steps": [10],
                "decay_factor": 0.316,
            },
            "< total_steps",
        ),
        (
            {
                "peak_lr": 0.001,
                "total_steps": 10,
                "warmup_steps": 4,
                "decay_steps": [8],
                "decay_factor": 0,
            },
            "decay_factor",
        ),
        (
            {
                "peak_lr": 0.001,
                "total_steps": 10,
                "warmup_steps": 4,
                "decay_steps": [8],
                "decay_factor": 1.1,
            },
            "decay_factor",
        ),
    ],
)
def test_invalid_warmup_step_decay_configs_raise(kwargs, match):
    with pytest.raises(ValueError, match=match):
        LinearWarmupStepDecaySchedule(**kwargs)


def test_warmup_step_decay_rejects_negative_steps():
    schedule = LinearWarmupStepDecaySchedule(
        peak_lr=0.001,
        total_steps=10,
        warmup_steps=2,
        decay_steps=[8],
        decay_factor=0.316,
    )

    with pytest.raises(ValueError, match="step must be non-negative"):
        schedule.lr_at_step(-1)
