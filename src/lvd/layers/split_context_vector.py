from jax import Array
from jax import numpy as jnp

from flax import linen as nn

from lvd.layers.context_vector import ContextVector


class SplitContextVector(nn.Module):
    hidden_dim: int

    ordered: bool = True
    max_length: int = 1024

    def setup(self):
        self.event_context_vector = ContextVector(
            self.hidden_dim, 
            self.ordered,
            self.max_length,
            name="event_context_vector"
        )

        self.vector_context_vector = ContextVector(
            self.hidden_dim, 
            self.ordered,
            self.max_length,
            name="vector_context_vector"
        )

    def __call__(self, embeddings: Array, *, training: bool = False) -> Array:
        event_embeddings, jet_embeddings = embeddings[:, :1], embeddings[:, 1:]

        event_embeddings = self.event_context_vector(event_embeddings, training=training)
        jet_embeddings = self.vector_context_vector(jet_embeddings, training=training)

        return jnp.concatenate(axis=1, arrays=(event_embeddings, jet_embeddings))

