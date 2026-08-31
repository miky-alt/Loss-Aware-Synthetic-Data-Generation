import pandas as pd
import pytest

from src.generators.baseline import CTGANGenerator, TVAEGenerator
from src.generators.registry import GENERATORS, build_generator


@pytest.fixture
def tiny_real_data() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "age": [23, 45, 31, 52, 29, 40, 36, 61, 27, 33],
            "income": [2100.0, 5400.0, 3200.0, 6100.0, 2900.0, 4800.0, 3900.0, 7000.0, 2500.0, 3300.0],
            "job": ["a", "b", "a", "c", "b", "a", "c", "b", "a", "c"],
        }
    )


def test_sample_before_fit_raises():
    generator = CTGANGenerator(epochs=1)
    with pytest.raises(RuntimeError):
        generator.sample(1)


def test_build_generator_returns_correct_type():
    assert isinstance(build_generator("ctgan"), CTGANGenerator)
    assert isinstance(build_generator("tvae"), TVAEGenerator)


def test_build_generator_unknown_name_raises():
    with pytest.raises(ValueError, match="Unknown generator"):
        build_generator("unknown")


def test_build_generator_covers_all_registered():
    # ensures every key in GENERATORS is constructable
    for name, cls in GENERATORS.items():
        assert isinstance(build_generator(name), cls)


# epochs=1 keeps these fast, but they still train real torch models, so they
# are marked slow and excluded by default (run with `pytest -m slow` to include them).
@pytest.mark.slow
@pytest.mark.parametrize("generator_cls", [CTGANGenerator, TVAEGenerator])
def test_fit_sample_roundtrip(generator_cls, tiny_real_data):
    generator = generator_cls(epochs=1)
    generator.fit(tiny_real_data)

    synthetic = generator.sample(5)

    assert list(synthetic.columns) == list(tiny_real_data.columns)
    assert len(synthetic) == 5
