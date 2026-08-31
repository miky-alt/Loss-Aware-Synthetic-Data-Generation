"""Registry mapping generator names to constructors.

Adding a new generator (baseline or the future loss-aware one) means adding
one line here — nothing in `training/` needs to change.
"""

from src.generators.baseline import CTGANGenerator, TVAEGenerator
from src.generators.base import SyntheticGenerator

GENERATORS: dict[str, type[SyntheticGenerator]] = {
    "ctgan": CTGANGenerator,
    "tvae": TVAEGenerator,
}


def build_generator(name: str, **kwargs) -> SyntheticGenerator:
    if name not in GENERATORS:
        raise ValueError(f"Unknown generator '{name}'. Choose from: {list(GENERATORS.keys())}")
    return GENERATORS[name](**kwargs)
