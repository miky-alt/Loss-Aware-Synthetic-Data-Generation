import pandas as pd
import pytest

from src.generators.base import SyntheticGenerator


def test_cannot_instantiate_abstract_generator():
    with pytest.raises(TypeError):
        SyntheticGenerator()


def test_concrete_subclass_respects_fit_sample_contract():
    class _EchoGenerator(SyntheticGenerator):
        def fit(self, real_data: pd.DataFrame) -> "SyntheticGenerator":
            self._data = real_data
            return self

        def sample(self, num_rows: int) -> pd.DataFrame:
            return self._data.head(num_rows)

    real_data = pd.DataFrame({"a": [1, 2, 3]})
    generator = _EchoGenerator().fit(real_data)

    assert generator.sample(2).shape == (2, 1)
