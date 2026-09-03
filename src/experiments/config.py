"""Experiment configuration: every knob a training run needs, in one place."""

import hashlib
import json
import secrets
from dataclasses import dataclass, field
from datetime import datetime
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
    test_size: float = 0.2  # fraction held out from generator.fit(), used for evaluation
    generator_kwargs: dict[str, Any] = field(default_factory=dict)
    transformer_specs: dict[str, Any] = field(default_factory=dict)
    # random suffix guarantees run_name uniqueness even if two configs with
    # identical hyperparameters are created within the same second
    created_at: str = field(
        default_factory=lambda: datetime.now().strftime("%Y%m%d-%H%M%S") + "-" + secrets.token_hex(3)
    )

    @property
    def _kwargs_hash(self) -> str:
        """Short deterministic hash of the hyperparameters that vary per run.

        Ensures different --kwarg / --test-size combinations never collide in
        run_name, even before created_at is considered.
        """
        payload = json.dumps(
            {"generator_kwargs": self.generator_kwargs, "test_size": self.test_size, "transformer_specs": self.transformer_specs},
            sort_keys=True,
        )
        return hashlib.sha256(payload.encode()).hexdigest()[:6]

    @property
    def run_name(self) -> str:
        return f"{self.dataset_name}_{self.generator_name}_seed{self.seed}_{self._kwargs_hash}_{self.created_at}"
