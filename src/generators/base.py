"""Common interface that every synthetic data generator must implement."""

from abc import ABC, abstractmethod

import pandas as pd


class SyntheticGenerator(ABC):
    """Minimal fit/sample contract shared by all generators.

    Keeping this interface stable lets the evaluation pipeline (utility and
    privacy metrics) and, later, the loss-aware training loop treat any
    generator - baseline or custom - interchangeably.
    """

    @abstractmethod
    def fit(self, real_data: pd.DataFrame) -> "SyntheticGenerator":
        """Train the generator on real tabular data. Returns self."""
        raise NotImplementedError

    @abstractmethod
    def sample(self, num_rows: int) -> pd.DataFrame:
        """Draw `num_rows` synthetic records from the fitted generator."""
        raise NotImplementedError

    def update_transformers(self, transformer_specs: dict) -> None:
        """Configure preprocessing transformers before fitting, when supported."""
        if transformer_specs:
            raise NotImplementedError("this generator does not support custom transformers")

    def get_training_diagnostics(self) -> dict:
        """Generator-specific post-fit artifacts included in the experiment report.

        Override in subclasses to expose training history, learned distributions,
        or any other generator-specific output. Default: no artifacts.
        """
        return {}
