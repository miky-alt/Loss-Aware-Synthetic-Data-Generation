"""Registry mapping generator names to constructors.

Adding a new generator (baseline or the future loss-aware one) means adding
one line here — nothing in `training/` needs to change.
"""

from src.generators.base import SyntheticGenerator
from src.generators.baseline import CTGANGenerator, GaussianCopulaGenerator, TVAEGenerator
from src.generators.loss_aware import LossAwareTVAEGenerator

GENERATORS: dict[str, type[SyntheticGenerator]] = {
    "ctgan": CTGANGenerator,
    "tvae": TVAEGenerator,
    "gaussian_copula": GaussianCopulaGenerator,
    "tvae_loss_aware": LossAwareTVAEGenerator,
}


def build_generator(name: str, **kwargs) -> SyntheticGenerator:
    if name not in GENERATORS:
        raise ValueError(f"Unknown generator '{name}'. Choose from: {list(GENERATORS.keys())}")
    return GENERATORS[name](**kwargs)
