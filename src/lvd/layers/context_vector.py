from jax import numpy as jnp

from flax import linen as nn

from lvd.layers.positional_encoding import PositionalEncoding


class ContextVector(nn.Module):
    hidden_dim: int

    ordered: bool = True
    max_length: int = 1024

    def setup(self):
        self.positional_encoding = PositionalEncoding(
            self.hidden_dim, 
            self.max_length
        )

        self.context_embedding = nn.Dense(
            2 * self.hidden_dim, 
        )

        self.context_vector = self.param(
            "context_vector",
            nn.initializers.normal(stddev=1.0),
            (1, 1, self.hidden_dim)
        )

    def __call__(
        self, 
        embeddings, 
        *,
        training: bool = False
    ):
        B, T, D = embeddings.shape

        # Context vectors to add to different inputs to differentiate them.
        context_vector = jnp.broadcast_to(self.context_vector, (B, T, self.hidden_dim))

        # Modify context vectors with position information
        if self.ordered:
            context_vector = self.positional_encoding(context_vector)

        # Add the context vectors to the inputs.
        embeddings = self.context_embedding(jnp.concatenate(axis=-1, arrays=(
            embeddings,
            context_vector   
        )))

        return embeddings

        
