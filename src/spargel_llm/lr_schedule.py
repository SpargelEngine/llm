from dataclasses import dataclass
from typing import Protocol

from torch.optim import Optimizer


class LearningRateSchedule(Protocol):
    def lr_at_step(self, step: int) -> float: ...


@dataclass(frozen=True)
class ConstantLearningRateSchedule:
    learning_rate: float

    def __post_init__(self) -> None:
        if self.learning_rate < 0:
            raise ValueError("learning_rate must be non-negative")

    def lr_at_step(self, step: int) -> float:
        _validate_step(step)
        return self.learning_rate


@dataclass(frozen=True)
class LinearWarmupConstantCooldownSchedule:
    peak_lr: float
    total_steps: int
    warmup_steps: int
    cooldown_steps: int
    min_lr: float

    def __post_init__(self) -> None:
        if self.total_steps <= 0:
            raise ValueError("total_steps must be positive")
        if self.warmup_steps < 0:
            raise ValueError("warmup_steps must be non-negative")
        if self.cooldown_steps < 0:
            raise ValueError("cooldown_steps must be non-negative")
        if self.warmup_steps + self.cooldown_steps > self.total_steps:
            raise ValueError("warmup_steps + cooldown_steps must be <= total_steps")
        if not 0 <= self.min_lr <= self.peak_lr:
            raise ValueError("min_lr must satisfy 0 <= min_lr <= peak_lr")

    def lr_at_step(self, step: int) -> float:
        _validate_step(step)

        if self.warmup_steps > 0 and step < self.warmup_steps:
            return self.peak_lr * (step + 1) / self.warmup_steps

        cooldown_start = self.total_steps - self.cooldown_steps
        if self.cooldown_steps > 0 and step >= cooldown_start:
            if self.cooldown_steps == 1 or step >= self.total_steps - 1:
                return self.min_lr
            progress = (step - cooldown_start) / (self.cooldown_steps - 1)
            return self.peak_lr + (self.min_lr - self.peak_lr) * progress

        return self.peak_lr


def set_optimizer_learning_rate(optimizer: Optimizer, learning_rate: float) -> None:
    for group in optimizer.param_groups:
        group["lr"] = learning_rate


def _validate_step(step: int) -> None:
    if step < 0:
        raise ValueError("step must be non-negative")
