import numpy as np

import jax
from jax import Array

from flax import linen as nn


class PositionalEncoding(nn.Module):
    d_model: int         # Hidden dimensionality of the input.
    max_len: int = 1024  # Maximum length of a sequence to expect.

    def setup(self):
        # Create matrix of [SeqLen, HiddenDim] representing the positional encoding for max_len inputs
        pe = np.zeros((self.max_len, self.d_model))
        position = np.arange(0, self.max_len, dtype=np.float32)[:, None]
        div_term = np.exp(np.arange(0, self.d_model, 2) *
                          (-np.log(10000.0) / self.d_model))
        pe[:, 0::2] = np.sin(position * div_term)
        pe[:, 1::2] = np.cos(position * div_term)
        pe = pe[None]
        self.pe = jax.device_put(pe)

    def __call__(self, x: Array, *, training: bool = False) -> Array:
        x = x + self.pe[:, :x.shape[1]]
        return x
