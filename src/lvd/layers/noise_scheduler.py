from typing import Optional
from dataclasses import dataclass

import jax
from jax import numpy as jnp
from jax import Array, nn, lax

import haiku as hk

from lvd.noise_schedules.conditional_network_schedule import ConditionalNetworkSchedule

START_GAMMA_MIN = -15.0
START_GAMMA_MAX = 5.0

FLOAT_GAMMA_MIN = -20.0
FLOAT_GAMMA_MAX = 20.0


def make_noise_schedule(name: str, outputs: int):
    name == name.lower()

    if name == "cosine":
        return NoiseScheduleCosine(outputs)
    elif name == "edm":
        return EDMNoiseSchedule(outputs)
    elif name == "nnet":
        return ScaledNoiseSchedule(NoiseScheduleNNet(outputs))
    elif name == "cnnet":
        return ConditionalNetworkSchedule(outputs)
    else:
        raise ValueError(f"Unkown noise scheduler: {name}")


def softclip(arr: Array, min: float) -> Array:
    """ Clips the tensor values at the minimum value min in a softway. Taken from Handful of Trials """
    return min + nn.softplus(arr - min)


class DenseMonotone(hk.Linear):
    """Strictly increasing Dense layer."""

    def __call__(
        self,
        inputs: jnp.ndarray,
        *,
        precision: Optional[lax.Precision] = None,
    ) -> jnp.ndarray:
        if not inputs.shape:
            raise ValueError("Input must not be scalar.")

        input_size = self.input_size = inputs.shape[-1]
        output_size = self.output_size
        dtype = inputs.dtype

        w_init = self.w_init
        if w_init is None:
            stddev = 1. / jnp.sqrt(self.input_size)
            w_init = hk.initializers.TruncatedNormal(stddev=stddev)

        w = hk.get_parameter("w", [input_size, output_size], dtype, init=w_init)
        w = jnp.square(w)

        out = jnp.dot(inputs, w, precision=precision)

        if self.with_bias:
            b = hk.get_parameter("b", [self.output_size], dtype, init=self.b_init)
            b = jnp.broadcast_to(b, out.shape)
            out = out + b

        return out


class NoiseScheduleNNet(hk.Module):
    def __init__(self, num_outputs: int = 1, num_features: int = 1024):
        super(NoiseScheduleNNet, self).__init__()

        self.num_outputs = num_outputs
        self.num_features = num_features

        self.l1 = DenseMonotone(num_outputs)
        self.l2 = DenseMonotone(num_features)
        self.l3 = DenseMonotone(num_outputs, with_bias=False)

    def __call__(self, t: Array) -> Array:
        """
        Parameters
        ----------
        t: (B) or ()

        Returns
        -------
        (B, D) or (D,)
        """   
        t = jnp.expand_dims(t, -1)

        t = 2.0 * t - 1.0

        hidden = self.l1(t)

        nonlinearity = self.l2(hidden)
        nonlinearity = nn.sigmoid(nonlinearity)
        nonlinearity = self.l3(nonlinearity)

        hidden = hidden + nonlinearity

        return hidden    


class ScaledNoiseSchedule(hk.Module):
    def __init__(
        self, 
        noise_schedule: NoiseScheduleNNet
    ):
        super(ScaledNoiseSchedule, self).__init__()

        self.noise_schedule = noise_schedule

    def __call__(
        self, 
        t: Array, 
        gamma_min: Array, 
        gamma_max: Array,
    ) -> Array:  
        """
        Parameters
        ----------
        t: (B) or ()

        Returns
        -------
        (B, D) or (D,)
        """      
        gamma_0 = self.noise_schedule(jnp.zeros_like(t))
        gamma_1 = self.noise_schedule(jnp.ones_like(t))
        gamma_t = self.noise_schedule(t)

        scaled_gamma_t = (gamma_t - gamma_0) / (gamma_1 - gamma_0)
        return (gamma_max - gamma_min) * scaled_gamma_t + gamma_min
    
@dataclass
class EDMNoiseSchedule(hk.Module):
    num_outputs: int = 1
    
    def __call__(
        self, 
        t: Array,
        gamma_min: Array, 
        gamma_max: Array,
    ) -> Array:  
        
        t0 = jax.scipy.stats.norm.cdf(gamma_min, 2.4, 2.4)
        t1 = jax.scipy.stats.norm.cdf(gamma_max, 2.4, 2.4)

        t = jnp.expand_dims(t, -1)
        t = t0 + (t1 - t0) * t

        return jax.scipy.stats.norm.ppf(t, 2.4, 2.4)
    
@dataclass
class NoiseScheduleCosine(hk.Module):
    num_outputs: int = 1

    def inverse(self, gamma):
        return (2 / jnp.pi) * jnp.arctan(jnp.exp(gamma / 2.0))
    
    def __call__(
        self, 
        t: Array,
        gamma_min: Array, 
        gamma_max: Array,
    ) -> Array:  
        
        t0 = self.inverse(gamma_min)
        t1 = self.inverse(gamma_max)

        t = jnp.expand_dims(t, -1)
        t = t0 + (t1 - t0) * t

        return 2.0 * jnp.log(jnp.tan(jnp.pi * t / 2.0)) 
        
