from dataclasses import dataclass

import jax
from jax import Array
from jax import numpy as jnp

from flax import linen as nn

from lvd.utils import masked_fill
from lvd.layers.linear_block import LinearBlock
from lvd.layers.attention_block import AttentionBlock


@dataclass
class TransformerBlock(nn.Module):
    hidden_dim: int
    num_heads: int
    expansion: int = 2
    dropout: float = 0.0
    skip_connection_type: str = "gru"
    
    def setup(self):
        self.attention = AttentionBlock(
            self.hidden_dim, 
            self.num_heads, 
            self.dropout, 
            self.skip_connection_type
        )

        self.linear = LinearBlock(
            self.hidden_dim, 
            self.expansion, 
            self.dropout,
            self.skip_connection_type
        )
        
    def __call__(
        self, 
        embeddings: Array,  # [B, T, D]
        mask: Array,  # [B, T]
        *,
        training: bool = False,
    ) -> Array:
        hidden = self.attention(embeddings, mask, training = training)
        hidden = self.linear(hidden, training = training)
        hidden = masked_fill(hidden, mask)

        return hidden