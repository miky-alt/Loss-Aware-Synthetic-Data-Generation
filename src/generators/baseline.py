"""Baseline synthetic data generators, wrapping the SDV library's
single-table synthesizers behind the project's `SyntheticGenerator` interface.
"""

import pandas as pd
from sdv.metadata import Metadata
from sdv.single_table import CTGANSynthesizer, GaussianCopulaSynthesizer, TVAESynthesizer

from src.generators.base import SyntheticGenerator


class _SDVSynthesizerGenerator(SyntheticGenerator):
    """Shared fit/sample logic for SDV single-table synthesizers.

    Subclasses only need to set `_synthesizer_cls`.
    """

    _synthesizer_cls = None

    def __init__(self, **synthesizer_kwargs):
        if self._synthesizer_cls is None:
            raise NotImplementedError("subclasses must set _synthesizer_cls")
        self._synthesizer_kwargs = synthesizer_kwargs
        self._synthesizer = None

    def fit(self, real_data: pd.DataFrame) -> "SyntheticGenerator":
        metadata = Metadata.detect_from_dataframe(real_data)
        boolean_columns = real_data.select_dtypes(include="bool").columns
        metadata.update_columns_metadata(
            {column: {"sdtype": "boolean"} for column in boolean_columns}
        )
        self._synthesizer = self._synthesizer_cls(metadata, **self._synthesizer_kwargs)
        self._synthesizer.fit(real_data)
        return self

    def sample(self, num_rows: int) -> pd.DataFrame:
        if self._synthesizer is None:
            raise RuntimeError("call fit() before sample()")
        return self._synthesizer.sample(num_rows=num_rows)

    def _preprocessing_diagnostics(self) -> dict:
        if self._synthesizer is None:
            return {}
        transformers = self._synthesizer.get_transformers()
        diagnostics = {}
        for column, transformer in transformers.items():
            if isinstance(transformer, str):
                diagnostics[str(column)] = {
                    "class": transformer,
                    "module": "sdv",
                    "parameters": transformer,
                }
                continue
            transformer_type = type(transformer)
            diagnostics[str(column)] = {
                "class": transformer_type.__name__,
                "module": transformer_type.__module__,
                "parameters": f"{transformer_type.__module__}.{transformer_type.__name__}",
            }
        return diagnostics


class CTGANGenerator(_SDVSynthesizerGenerator):
    """Baseline generator using SDV's CTGAN synthesizer.

    Exposes per-epoch generator/discriminator loss via get_training_diagnostics().
    """

    _synthesizer_cls = CTGANSynthesizer

    def get_training_diagnostics(self) -> dict:
        if self._synthesizer is None:
            return {}
        return {
            "loss_values": self._synthesizer.get_loss_values().to_dict(orient="records"),
            "preprocessing_transformers": self._preprocessing_diagnostics(),
        }


class TVAEGenerator(_SDVSynthesizerGenerator):
    """Baseline generator using SDV's TVAE synthesizer.

    Exposes per-epoch/batch loss via get_training_diagnostics() (no plot, unlike CTGAN).
    """

    _synthesizer_cls = TVAESynthesizer

    def get_training_diagnostics(self) -> dict:
        if self._synthesizer is None:
            return {}
        return {
            "loss_values": self._synthesizer.get_loss_values().to_dict(orient="records"),
            "preprocessing_transformers": self._preprocessing_diagnostics(),
        }


class GaussianCopulaGenerator(_SDVSynthesizerGenerator):
    """Baseline generator using SDV's GaussianCopula synthesizer.

    Exposes per-column learned distributions via get_training_diagnostics().
    Does not accept an `epochs` parameter (statistical model, not neural network).
    """

    _synthesizer_cls = GaussianCopulaSynthesizer

    def get_training_diagnostics(self) -> dict:
        if self._synthesizer is None:
            return {}
        return {
            "learned_distributions": self._synthesizer.get_learned_distributions(),
            "preprocessing_transformers": self._preprocessing_diagnostics(),
        }
