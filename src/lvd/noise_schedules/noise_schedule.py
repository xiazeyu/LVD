from typing import NamedTuple
from enum import Enum

import jax
from jax import Array
from jax import numpy as jnp

from flax import linen as nn

from lvd.config.noise_schedule_config import NoiseScheduleConfig, NoiseScheduleType

from lvd.noise_schedules.conditional_network_schedule import ConditionalNetworkSchedule
from lvd.noise_schedules.network_schedule import NetworkSchedule

class NoiseSchedule(nn.Module):
    class Statistics(NamedTuple):
        gamma: Array
        log_alpha_squared: Array
        log_sigma_squared: Array

    config: NoiseScheduleConfig

    def setup(self) -> None:
        match self.config.noise_schedule:
            case NoiseScheduleType.ConditionalNetwork:
                self.noise_schedule = ConditionalNetworkSchedule(self.config)
            case NoiseScheduleType.Network:
                self.noise_schedule = NetworkSchedule(self.config)
                
    def __call__(self, t: Array, gamma_min: Array, gamma_max: Array) -> Array:
        if jnp.isscalar(t) and t == 0.0:
            return gamma_min
        elif jnp.isscalar(t) and t == 1.0:
            return gamma_max
        else:
            return self.noise_schedule(t, gamma_min, gamma_max)

    def SNR(self, t: Array, gamma_min, gamma_max):
        return jnp.exp(0.5 * self(t, gamma_min, gamma_max))

    def alpha_squared(self, t, gamma_min, gamma_max):
        return nn.sigmoid(-self(t, gamma_min, gamma_max))

    def alpha(self, t, gamma_min, gamma_max):
        return jnp.sqrt(self.alpha_squared(t, gamma_min, gamma_max))

    def log_var(self, t, gamma_min, gamma_max):
        return nn.log_sigmoid(self(t, gamma_min, gamma_max))

    def var(self, t, gamma_min, gamma_max):
        return nn.sigmoid(self(t, gamma_min, gamma_max))

    def sigma(self, t, gamma_min, gamma_max):
        return jnp.sqrt(self.var(t, gamma_min, gamma_max))

    def statistics(self, t, gamma_min, gamma_max):
        gamma = self(t, gamma_min, gamma_max)

        return NoiseSchedule.Statistics(
            gamma=gamma,
            log_alpha_squared=nn.log_sigmoid(-gamma),
            log_sigma_squared=nn.log_sigmoid(gamma),
        )

    def gamma(self, t, gamma_min, gamma_max):
        return self(t, gamma_min, gamma_max)

    def prime(self, t: Array, gamma_min: Array, gamma_max: Array):
        return self.noise_schedule.prime(t, gamma_min, gamma_max)
    
    def g_squared(self, t: Array, gamma_min: Array, gamma_max: Array):
        return self.noise_schedule.g_squared(t, gamma_min, gamma_max)
        # def g(t):
        #     gamma = self.noise_schedule.noise_schedule
        #     return jnp.logaddexp(0, gamma(t, gamma_min, gamma_max)).sum()

        # g2 = jax.vmap(jax.vmap(jax.grad(g)))(t)

        # return g2[..., None]
