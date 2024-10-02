from dataclasses import dataclass

import jax
from jax import Array
from jax import numpy as jnp

from flax import linen as nn

from lvd.config import NetworkConfig
from lvd.layers import Embedding
from tensorflow_probability.substrates.jax import distributions


class MultiplicityPredictor(nn.Module):
    OutputType = distributions.Gamma
    
    config: NetworkConfig

    @property
    def embedding_config(self):
        return (
            self.config.hidden_dim, 
            self.config.num_multiplicity_layers, 
            self.config.transformer_expansion,
            self.config.dropout,
            self.config.skip_connection_type,
        )
    
    @nn.compact
    def __call__(
        self, 
        embeddings: Array,  # [B, D]
        *,
        training: bool = True,
    ) -> distributions.Gamma: 
          
        embeddings = Embedding(*self.embedding_config)(embeddings)
        log_alpha, log_beta = jnp.transpose(nn.Dense(2)(embeddings))

        if self.config.discrete_multiplicity_predictor:
            return distributions.NegativeBinomial(
                total_count=jnp.exp(log_alpha),
                logits=log_beta,
                validate_args=False,
                require_integer_total_count=False
            )
        else:
            return distributions.Gamma(
                concentration=jnp.exp(log_alpha / 10.0), 
                log_rate=log_beta
            )
        