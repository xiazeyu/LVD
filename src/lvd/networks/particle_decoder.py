from typing import Tuple, NamedTuple
from dataclasses import dataclass

import jax
from jax import Array
from jax import numpy as jnp

from flax import linen as nn

from lvd.utils import masked_fill
from lvd.config import NetworkConfig, DatasetConfig
from lvd.layers import (
    Embedding, 
    TransformerBlock, 
    SplitContextVector
)


class ParticleDecoderOutput(NamedTuple):
    vectors: Array
    type_logits: Array
    mask: Array
    event: Array

class ParticleDecoder(nn.Module):
    OutputType = ParticleDecoderOutput

    # =============================================================================================
    # Options
    # region --------------------------------------------------------------------------------------
    config: NetworkConfig
    dataset_config: DatasetConfig

    @property
    def embedding_config(self):
        return (
            self.config.hidden_dim, 
            self.config.num_linear_layers, 
            self.config.transformer_expansion,
            self.config.dropout,
            self.config.skip_connection_type,
        )
    
    @property
    def context_vector_config(self):
        return (
            self.config.hidden_dim, 
            self.config.ordered_particle_encoder
        )
    
    @property
    def transformer_config(self):
        return (
            2 * self.config.hidden_dim,
            self.config.transformer_heads,
            self.config.transformer_expansion,
            self.config.dropout,
            self.config.skip_connection_type
        )
    
    @property
    def hidden_dim(self):
        return self.config.hidden_dim
    
    @property
    def conditional(self):
        return self.config.conditional_particle_decoder
    
    @property
    def num_transformer_layers(self):
        return self.config.num_particle_decoder_layers    
    # endregion ===================================================================================
    
    # =============================================================================================
    # Network Blocks
    # region --------------------------------------------------------------------------------------
    @property
    def ParticleInputEmbedding(self):
        return Embedding(*self.embedding_config, name="particle_input_embedding")

    @property
    def ParticleContextVector(self):
        return SplitContextVector(*self.context_vector_config, name="particle_context_vector")
    
    @property
    def DetectorInputEmbedding(self):
        return Embedding(*self.embedding_config, name="detector_input_embedding")

    @property
    def DetectorContextVector(self):
        return SplitContextVector(*self.context_vector_config, name="detector_context_vector")
    
    @property
    def EventOutputEmbedding(self):
        return Embedding(*self.embedding_config, name="event_output_embedding")
    
    @property
    def EventOutput(self):
        return nn.Dense(self.dataset_config.particle_event_dim, name="event_output")
    
    @property
    def JetOutputEmbedding(self):
        return Embedding(*self.embedding_config, name="jet_output_embedding")

    @property
    def JetFeatureOutput(self):
        return nn.Dense(self.dataset_config.particle_vector_dim, name="jet_feature_output")

    @property
    def JetTypeOutput(self):
        return nn.Dense(self.dataset_config.num_particle_types, name="jet_type_output")
    # endregion ===================================================================================

    @nn.compact
    def __call__(
        self, 

        particle_embeddings: Array,  # [B, 1 + T, D]
        particle_mask: Array,        # [B, 1 + T],

        detector_embeddings: Array, # [B, 1 + C, D]
        detector_mask: Array,       # [B, 1 + C]

        *,
        training: bool = True,
    ) -> Tuple[Array, Array]:  # ([B, T, D], [B, D])
        B, T, D = particle_embeddings.shape

        # Initial embedding into the hidden dimension.
        particle_embeddings = self.ParticleInputEmbedding(particle_embeddings, training = training)
        particle_embeddings = masked_fill(particle_embeddings, particle_mask)

        particle_embeddings = self.ParticleContextVector(particle_embeddings, training = training)
        particle_embeddings = masked_fill(particle_embeddings, particle_mask)

        # Combine the two inputs into a single sequence.
        if self.conditional:
            detector_embeddings = self.DetectorInputEmbedding(detector_embeddings, training = training)
            detector_embeddings = masked_fill(detector_embeddings, detector_mask)

            detector_embeddings = self.DetectorContextVector(detector_embeddings, training = training)
            detector_embeddings = masked_fill(detector_embeddings, detector_mask)
    
            combined_embeddings = jnp.concatenate(axis=1, arrays=(particle_embeddings, detector_embeddings))
            combined_mask = jnp.concatenate(axis=1, arrays=(particle_mask, detector_mask))
        else:
            combined_embeddings = particle_embeddings
            combined_mask = particle_mask

        # Transformer Encoders to contextualize embeddings.
        for _ in range(self.num_transformer_layers):
            combined_embeddings = TransformerBlock(*self.transformer_config)(
                combined_embeddings, 
                combined_mask,
                training = training
            )

        # Split off the special event embedding for met output
        event_embeddings = self.EventOutputEmbedding(combined_embeddings[:, 0], training = training)
        event_output = self.EventOutput(event_embeddings)

        # Predict continous and discrete jet features.
        jet_mask = particle_mask[:, 1:T]
        jet_embeddings = self.JetOutputEmbedding(combined_embeddings[:, 1:T, :], training = training)
        jet_embeddings = masked_fill(jet_embeddings, jet_mask)

        jet_feature_output = self.JetFeatureOutput(jet_embeddings)
        jet_type_output = self.JetTypeOutput(jet_embeddings)

        return ParticleDecoderOutput(
            masked_fill(jet_feature_output, jet_mask), 
            masked_fill(jet_type_output, jet_mask), 
            jet_mask,
            event_output
        )
        