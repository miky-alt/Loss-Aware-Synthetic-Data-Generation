"""Experiment configuration: every knob a training run needs, in one place."""

from dataclasses import dataclass, field
from enum import auto, Enum
from typing import Any


class ExperimentMode(Enum):
    TRAIN_AND_EVALUATE = auto()
    EVALUATE_ONLY = auto()  # load a pre-trained generator; skip fit


@dataclass
class TrainingConfig:
    dataset_name: str  # one of src.data.loader.LOADERS keys
    generator_name: str  # one of src.generators.registry.GENERATORS keys
    num_samples: int
    seed: int = 42
    generator_kwargs: dict[str, Any] = field(default_factory=dict)

    @property
    def run_name(self) -> str:
        return f"{self.dataset_name}_{self.generator_name}_seed{self.seed}"
