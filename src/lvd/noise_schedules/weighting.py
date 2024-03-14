from dataclasses import dataclass

from jax.scipy import stats
from jax import numpy as jnp
from jax import Array

from flax import linen as nn

from lvd.config.noise_schedule_config import NoiseScheduleConfig, WeightingType


class BaseWeighting(nn.Module):
    def __call__(self, gamma: Array):
        raise NotImplementedError()


class UnitWeighting(BaseWeighting):
    @nn.compact
    def __call__(self, gamma: Array):
        return jnp.ones_like(gamma)


class CosineWeighting(BaseWeighting):
    @nn.compact
    def __call__(self, gamma: Array):
        return 1.0 / jnp.cosh(-gamma / 2)


class EDMWeighting(BaseWeighting):
    @nn.compact
    def __call__(self, gamma: Array):
        w1 = stats.norm.pdf(-gamma, 2.4, 2.4)
        w2 = jnp.exp(gamma) + 0.25

        return 5.0 * w1 * w2


class SigmoidWeighting(BaseWeighting):
    offset: float

    @nn.compact
    def __call__(self, gamma: Array):
        return nn.sigmoid(gamma + self.offset)


def Weighting(config: NoiseScheduleConfig) -> BaseWeighting:
    match config.weighting:
        case WeightingType.EDM:
            return EDMWeighting()
        case WeightingType.Unit:
            return UnitWeighting()
        case WeightingType.Cosine:
            return CosineWeighting()
        case WeightingType.Sigmoid:
            return SigmoidWeighting(config.sigmoid_weighting_offset)
