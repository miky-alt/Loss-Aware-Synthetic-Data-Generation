"""Baseline synthetic data generators, wrapping the SDV library's
single-table synthesizers behind the project's `SyntheticGenerator` interface.
"""

import pandas as pd
from rdt.transformers.boolean import BinaryEncoder
from rdt.transformers.categorical import (
    CustomLabelEncoder,
    FrequencyEncoder,
    LabelEncoder,
    OneHotEncoder,
    OrderedLabelEncoder,
    OrderedUniformEncoder,
    UniformEncoder,
)
from rdt.transformers.numerical import (
    ClusterBasedNormalizer,
    FloatFormatter,
    GaussianNormalizer,
    LogScaler,
    LogitScaler,
)
from sdv.single_table import CTGANSynthesizer, GaussianCopulaSynthesizer, TVAESynthesizer

from src.data.loader import build_sdv_metadata
from src.generators.base import SyntheticGenerator


RDT_TRANSFORMERS = {
    cls.__name__: cls
    for cls in (
        BinaryEncoder,
        ClusterBasedNormalizer,
        CustomLabelEncoder,
        FloatFormatter,
        FrequencyEncoder,
        GaussianNormalizer,
        LabelEncoder,
        LogScaler,
        LogitScaler,
        OneHotEncoder,
        OrderedLabelEncoder,
        OrderedUniformEncoder,
        UniformEncoder,
    )
}


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
        self._transformer_specs = {}

    def fit(self, real_data: pd.DataFrame) -> "SyntheticGenerator":
        metadata = build_sdv_metadata(real_data)
        self._synthesizer = self._synthesizer_cls(metadata, **self._synthesizer_kwargs)
        self._synthesizer._data_processor.prepare_for_fitting(real_data)
        self._configure_transformers()
        self._synthesizer.fit(real_data)
        return self

    def sample(self, num_rows: int) -> pd.DataFrame:
        if self._synthesizer is None:
            raise RuntimeError("call fit() before sample()")
        return self._synthesizer.sample(num_rows=num_rows)

    def update_transformers(self, transformer_specs: dict) -> None:
        if self._synthesizer is not None:
            raise RuntimeError("custom transformers must be configured before fit()")
        self._transformer_specs = transformer_specs

    def _configure_transformers(self) -> None:
        if not getattr(self, "_transformer_specs", None):
            return
        transformers = {}
        for column, spec in self._transformer_specs.items():
            if not isinstance(spec, dict) or "name" not in spec:
                raise ValueError(f"Transformer spec for '{column}' must contain a 'name'.")
            name = spec["name"]
            if name not in RDT_TRANSFORMERS:
                raise ValueError(f"Unknown RDT transformer '{name}'.")
            transformers[column] = RDT_TRANSFORMERS[name](**spec.get("kwargs", {}))
        self._synthesizer.update_transformers(transformers)

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
