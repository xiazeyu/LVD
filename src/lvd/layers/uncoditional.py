from typing import Tuple

import jax
from jax import numpy as jnp
from jax import random, nn

from flax import linen as nn

from lvd.networks.detector_encoder import DetectorEncoderOutputs


class Uncoditional(nn.Module):
    hidden_dim: int
    unconditional_probability: float = 0.0

    def setup(self) -> None:
        self.unconditional_vector = self.param(
            "unconditional_vector",
            nn.initializers.normal(stddev=1.0),
            (1, 1, self.hidden_dim)
        )

    def __call__(self, encoded_detector: DetectorEncoderOutputs, all_masked: bool = False) -> DetectorEncoderOutputs:
        detector_features = encoded_detector.vectors
        detector_mask = encoded_detector.mask

        B, T, D = detector_features.shape

        unconditional_features = jnp.broadcast_to(self.unconditional_vector, (B, T, self.hidden_dim))
        unconditional_mask = jnp.concatenate((
            jnp.ones_like(detector_mask[:, :1]),
            jnp.zeros_like(detector_mask[:, 1:])
        ), axis=1)


        uncond_mask = jax.random.uniform(self.make_rng("uncond"), (B,))
        uncond_mask = uncond_mask < self.unconditional_probability
        if all_masked:
            uncond_mask = jnp.ones_like(uncond_mask)
            

        detector_features = jnp.where(
            uncond_mask[:, None, None],
            unconditional_features,
            detector_features
        )

        detector_mask = jnp.where(
            uncond_mask[:, None],
            unconditional_mask,
            detector_mask
        )
        
        return DetectorEncoderOutputs(
            summary=encoded_detector.summary,
            vectors=detector_features,
            mask=detector_mask
        )