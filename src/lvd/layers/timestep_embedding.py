from dataclasses import dataclass

import jax
from jax import numpy as jnp
from jax import Array

from flax import linen as nn

class TimestepEmbedding(nn.Module):
  embedding_dim: int

  @nn.compact
  def __call__(self, timesteps: Array) -> Array:
    """
    Parameters
    ----------
    t: (B, T, 1) or (B, T, D)

    Returns
    -------
    (B, T, D)
    """    
    # Handle the case where we have many different scehdulers
    if timesteps.shape[-1] > 1:
      return 2 * timesteps - 1
    
    # Scale from [0, 1] to [0, 1_000]
    timesteps = 1_0000.0 * timesteps

    # Create the fourier evaluation points, shape (1, 1, D // 2)
    cosine_dim = self.embedding_dim // 2
    cosine_time = jnp.arange(cosine_dim)
    cosine_time = cosine_time[None, None, :]

    embedding = jnp.log(10_000) / (cosine_dim - 1)
    embedding = jnp.exp(-cosine_time * embedding)
    embedding = timesteps * embedding
    embedding = jnp.concatenate((jnp.sin(embedding), jnp.cos(embedding)), axis=-1)

    return embedding