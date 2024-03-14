from typing import Optional
from dataclasses import dataclass

from functools import partial

import jax
from jax import numpy as jnp
from jax import Array

from flax import linen as nn

from lvd.config import NoiseScheduleConfig
from lvd.noise_schedules.dense_monotone import DenseMonotone


class ConditionalMonotonicNetwork(nn.Module):
    output_dim: int = 1
    hidden_dim: int = 1024

    def setup(self):
        self.l1 = DenseMonotone(self.hidden_dim)
        self.l2 = DenseMonotone(self.output_dim)

    def __call__(
        self, 
        c: Array, # [B?, T?, D] 
        t: Array  # [B?, T?]
    ) ->  Array:
        t = 2 * t[..., None] - 1
        h = jnp.concatenate([t, c], axis=-1)

        h = self.l1(h)
        h = nn.tanh(h)

        h = self.l2(h)

        return h

    def prime(
        self, 
        c: Array, # [B, T, D] 
        t: Array  # [B, T]
    ) -> Array:
        return jax.jacobian(self.__call__, argnums=1)(c, t)


class ConditionalNoiseSchedule(nn.Module):
    output_dim: int = 1
    hidden_dim: int = 1024

    def setup(self) -> None:
        self.noise_schedule = ConditionalMonotonicNetwork(self.output_dim, self.hidden_dim)

    def __call__(
        self, 
        c: Array,         # [B, T, D] 
        t: Array,         # [B, T]
        gamma_min: Array, # []
        gamma_max: Array  # []
    ) -> Array:  
        """
        Parameters
        ----------
        t: (B, T)

        Returns
        -------
        (B, D) or (D,)
        """      
        gamma_0 = self.noise_schedule(c, jnp.zeros_like(t))
        gamma_1 = self.noise_schedule(c, jnp.ones_like(t))
        gamma_t = self.noise_schedule(c, t)

        scaled_gamma_t = (gamma_t - gamma_0) / (gamma_1 - gamma_0)
        return (gamma_max - gamma_min) * scaled_gamma_t + gamma_min

    def prime(self, c, t, gamma_min, gamma_max):
        gamma_0 = self.noise_schedule(c, jnp.zeros_like(t))
        gamma_1 = self.noise_schedule(c, jnp.ones_like(t))

        scale = (gamma_max - gamma_min) / (gamma_1 - gamma_0)

        return scale * self.noise_schedule.prime(c, t)


class ConditionalNetworkSchedule(nn.Module):
    config: NoiseScheduleConfig
    
    def setup(self) -> None:
        self.noise_schedule = ConditionalNoiseSchedule(self.config.output_dim, self.config.hidden_dim)
        self.position_encoding = nn.Embed(1024, self.config.conditioning_dim)
    
    def conditioning(self, t):
        B, T = t.shape

        c = self.position_encoding(jnp.arange(T))
        c = jnp.broadcast_to(c, (B, T, self.config.conditioning_dim))

        return c

    def __call__(
        self, 
        t: Array, 
        gamma_min: Array, 
        gamma_max: Array,
    ) -> Array:    
        """
        Parameters
        ----------
        t: (B, T)
        gamma_min: (D)
        gamma_max: (D)

        Returns
        -------
        (B, T, D)
        """   
        c = self.conditioning(t)

        f = self.noise_schedule
        f = jax.vmap(f, in_axes=(0, 0, None, None))
        f = jax.vmap(f, in_axes=(0, 0, None, None))

        return f(c, t, gamma_min, gamma_max)

    def prime(
        self, 
        t: Array, 
        gamma_min: Array,
        gamma_max: Array,
    ) -> Array:
        """
        Parameters
        ----------
        t: (B, T)
        gamma_min: (D)
        gamma_max: (D)

        Returns
        -------
        (B, T, D)
        """   
        
        c = self.conditioning(t)

        f = self.noise_schedule.prime
        f = jax.vmap(f, in_axes=(0, 0, None, None))
        f = jax.vmap(f, in_axes=(0, 0, None, None))

        return f(c, t, gamma_min, gamma_max)
    
    def g_squared(self, t: Array, gamma_min: Array, gamma_max: Array):
        c = self.conditioning(t)

        def g(c, t):
            return jnp.logaddexp(0, self.noise_schedule(c, t, gamma_min, gamma_max)).sum()

        g2 = jax.grad(g, argnums=1)
        g2 = jax.vmap(g2)
        g2 = jax.vmap(g2)

        return g2(c, t)[..., None]

