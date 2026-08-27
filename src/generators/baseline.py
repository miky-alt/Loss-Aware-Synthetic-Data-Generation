"""Baseline synthetic data generators, wrapping the SDV library's
single-table synthesizers behind the project's `SyntheticGenerator` interface.

These are plug-and-play baselines (Week 1 deliverable). The loss-aware
variants that add penalty terms to the training objective will subclass
or replace these later, without changing the fit/sample contract.
"""

import pandas as pd
from sdv.metadata import SingleTableMetadata
from sdv.single_table import CTGANSynthesizer, TVAESynthesizer

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
        metadata = SingleTableMetadata()
        metadata.detect_from_dataframe(real_data)
        self._synthesizer = self._synthesizer_cls(metadata, **self._synthesizer_kwargs)
        self._synthesizer.fit(real_data)
        return self

    def sample(self, num_rows: int) -> pd.DataFrame:
        if self._synthesizer is None:
            raise RuntimeError("call fit() before sample()")
        return self._synthesizer.sample(num_rows=num_rows)


class CTGANGenerator(_SDVSynthesizerGenerator):
    """Baseline generator using SDV's CTGAN synthesizer."""

    _synthesizer_cls = CTGANSynthesizer


class TVAEGenerator(_SDVSynthesizerGenerator):
    """Baseline generator using SDV's TVAE synthesizer."""

    _synthesizer_cls = TVAESynthesizer
