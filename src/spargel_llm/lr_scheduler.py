from typing import Annotated, Literal, override

from pydantic import BaseModel, Discriminator
from torch.optim import Optimizer
from torch.optim.lr_scheduler import (
    ConstantLR,
    CosineAnnealingLR,
    LinearLR,
    LRScheduler,
    SequentialLR,
)


class LRSchedulerBaseModel(BaseModel):
    def build(self, optimizer: Optimizer, last_step: int) -> LRScheduler:
        raise NotImplementedError


class ConstantLRModel(LRSchedulerBaseModel):
    name: Literal["constant"] = "constant"
    factor: float = 1.0
    total_iters: int

    @override
    def build(self, optimizer, last_step) -> LRScheduler:
        return ConstantLR(
            optimizer, self.factor, self.total_iters, last_epoch=last_step
        )


class CosineAnnealingLRModel(LRSchedulerBaseModel):
    name: Literal["cosine_annealing"] = "cosine_annealing"
    T_max: int
    eta_min: float = 0.0

    @override
    def build(self, optimizer, last_step):
        return CosineAnnealingLR(
            optimizer, self.T_max, self.eta_min, last_epoch=last_step
        )


class LinearLRModel(LRSchedulerBaseModel):
    name: Literal["linear"] = "linear"
    start_factor: float = 1.0
    end_factor: float = 1.0
    total_iters: int

    @override
    def build(self, optimizer, last_step):
        return LinearLR(
            optimizer,
            self.start_factor,
            self.end_factor,
            self.total_iters,
            last_epoch=last_step,
        )


class SequentialLRModel(LRSchedulerBaseModel):
    name: Literal["sequential"] = "sequential"
    schedulers: list[LRSchedulerModel]
    milestones: list[int]

    @override
    def build(self, optimizer, last_step):
        return SequentialLR(
            optimizer,
            [model.build(optimizer, -1) for model in self.schedulers],
            self.milestones,
            last_epoch=last_step,
        )


type LRSchedulerModelUnion = ConstantLRModel | CosineAnnealingLRModel | LinearLRModel | SequentialLRModel
type LRSchedulerModel = Annotated[LRSchedulerModelUnion, Discriminator("name")]
