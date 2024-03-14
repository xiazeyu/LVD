from typing import Any, NamedTuple
from enum import Enum

from jax import Array
from jax import numpy as jnp

from flax import linen as nn

from lvd.config.noise_schedule_config import NoiseScheduleConfig, NoiseScheduleType

from lvd.noise_schedules.conditional_network_schedule import ConditionalNetworkSchedule


def softclip(arr: Array, min: float) -> Array:
    """ Clips the tensor values at the minimum value min in a softway. Taken from Handful of Trials """
    return min + nn.softplus(arr - min)


class GammaLimits(nn.Module):
    config: NoiseScheduleConfig

    def setup(self) -> None:
        self.gamma_min = self.param(
            "gamma_min",
            nn.initializers.constant(self.config.initial_gamma_min),
            (),
        )

        self.gamma_max = self.param(
            "gamma_max",
            nn.initializers.constant(self.config.initial_gamma_max),
            (),
        )
        
        self.limit_gamma_min = self.config.limit_gamma_min
        self.limit_gamma_max = self.config.limit_gamma_max

    def __call__(self) -> Any:
        gamma_min = self.gamma_min
        gamma_max = self.gamma_max

        if self.limit_gamma_min is not None:
            gamma_min = softclip(gamma_min, self.limit_gamma_min)

        if self.limit_gamma_max is not None:
            gamma_max = -softclip(-gamma_max, -self.limit_gamma_max)
        
        return gamma_min, gamma_max