from typing import Tuple, NamedTuple
from dataclasses import dataclass

import jax
from jax import Array
from jax import numpy as jnp

from flax import linen as nn
from tensorflow_probability.substrates.jax import distributions

from lvd.config import NetworkConfig, DatasetConfig
from lvd.utils import masked_fill

from lvd.layers import (
    Embedding,
    TransformerBlock,

    ContextVector,
    SplitContextVector
)


class ParticleEncoderOutputs(NamedTuple):
    vector_distribution: distributions.Normal # [B, 1 + T, D]
    mean: Array                               # [B, 1 + T, D]
    log_std: Array                            # [B, 1 + T, D]
    masks: Array                              # [B, 1 + T]


class ParticleEncoder(nn.Module):
    OutputType = ParticleEncoderOutputs
    
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
        return self.config.conditional_particle_encoder
    
    @property
    def normalized(self):
        return self.config.normalized_particle_encoder
    
    @property
    def num_transformer_layers(self):
        return self.config.num_particle_encoder_layers    
    # endregion ===================================================================================
    
    # =============================================================================================
    # Network Blocks
    # region --------------------------------------------------------------------------------------
    @property
    def JetInputEmbedding(self):
        return Embedding(*self.embedding_config, name="jet_input_embedding")

    @property
    def JetContextVector(self):
        return ContextVector(*self.context_vector_config, name="jet_context_vector")
    
    @property
    def EventInputEmbedding(self):
        return Embedding(*self.embedding_config, name="event_input_embedding")

    @property
    def EventContextVector(self):
        return ContextVector(*self.context_vector_config, name="event_context_vector")

    @property
    def DetectorInputEmbedding(self):
        return Embedding(*self.embedding_config, name="detector_input_embedding")

    @property
    def DetectorContextVector(self):
        return SplitContextVector(*self.context_vector_config, name="detector_context_vector")

    @property
    def MeanLatentEmbedding(self):
        return Embedding(*self.embedding_config, name="mean_latent_embedding")

    @property
    def STDLatentEmbedding(self):
        return Embedding(*self.embedding_config, name="std_latent_embedding")
    # endregion ===================================================================================

    # =============================================================================================
    # Network Functions
    # region --------------------------------------------------------------------------------------
    def create_particle_embeddings(
        self, 

        jet_features: Array,   # [B, T, D]
        jet_types: Array,      # [B, T]
        jet_mask: Array,       # [B, T]

        event_features: Array, # [B, D]
        *,
        training: bool = False
    ) -> Tuple[Array, Array]: # ([B, 1 + T, D], [B, 1 + T])
        B, T, D = jet_features.shape

        # Append the particle types to the particle features.
        # Embedd the particle features and types into the initial latent space.
        jet_types = jax.nn.one_hot(jet_types, self.dataset_config.num_particle_types)
        jet_embeddings = jnp.concatenate(axis=2, arrays=(jet_features, jet_types))
        jet_embeddings = self.JetInputEmbedding(jet_embeddings, training = training)
        jet_embeddings = masked_fill(jet_embeddings, jet_mask)
        
        # Embedd the event features and add the extra time dimension.
        event_embeddings = self.EventInputEmbedding(event_features, training = training)
        event_embeddings = jnp.expand_dims(event_embeddings, axis=1)
        event_mask = jnp.ones((B, 1), dtype=jet_mask.dtype)

        # Add context vectors to the different types of inputs
        jet_embeddings = self.JetContextVector(jet_embeddings, training = training)
        event_embeddings = self.EventContextVector(event_embeddings, training = training)

        # Create a combined particle list with event and jet features.
        combined_embeddings = jnp.concatenate(axis=1, arrays=(event_embeddings, jet_embeddings))
        combined_mask = jnp.concatenate(axis=1, arrays=(event_mask, jet_mask))
        combined_embeddings = masked_fill(combined_embeddings, combined_mask)

        return combined_embeddings, combined_mask

    def create_detector_embeddings(
        self,
        embeddings: Array, # [B, 1 + C, D]
        mask: Array,       # [B, 1 + C]
        *,
        training: bool = False
    ) -> Tuple[Array, Array]: # ([B, 1 + C, D], [B, 1 + C])
        # Preprocess the detector variables for the encoder.
        embeddings = self.DetectorInputEmbedding(embeddings, training = training)
        embeddings = masked_fill(embeddings, mask)
        
        embeddings = self.DetectorContextVector(embeddings, training = training)
        embeddings = masked_fill(embeddings, mask)

        return embeddings, mask

    def extract_latent_mean(
        self, 
        embeddings: Array, # [B, 1 + T, D]
        mask: Array,       # [B, 1 + T]
        *, 
        training: bool = False
    )  -> Array: # [B, 1 + T, D]
        latent_mean = self.MeanLatentEmbedding(embeddings, training = training)
        latent_mean = masked_fill(latent_mean, mask)

        if self.normalized:
            # scale = jnp.max(jnp.abs(latent_mean), axis=-1, keepdims=True)
            # norms = scale * jnp.sqrt(jnp.mean(jnp.square(latent_mean / scale), axis=-1, keepdims=True))

            norms = jnp.sqrt(jnp.mean(jnp.square(latent_mean), axis=-1, keepdims=True))
            # norms = jnp.mean(jnp.abs(latent_mean), axis=-1, keepdims=True)
            norms = jnp.where(mask[:, :, None], norms, 1.0)

            latent_mean = latent_mean / norms

            # latent_mean = jnp.where(mask[..., None], normalized_latent_mean, latent_mean)

        return masked_fill(latent_mean, mask)

    def extract_latent_log_std( 
        self, 
        embeddings: Array, # [B, 1 + T, D]
        mask: Array,       # [B, 1 + T]
        *, 
        training: bool = False
    )  -> Array: # [B, 1 + T, D]
        latent_log_std = self.STDLatentEmbedding(embeddings, training = training)

        return masked_fill(latent_log_std, mask)
    # endregion ===================================================================================
    
    @nn.compact
    def __call__(
        self, 

        particle_jet_features: Array,   # [B, T, D]
        particle_jet_types: Array,      # [B, T]
        particle_jet_mask: Array,       # [B, T]

        particle_event_features: Array, # [B, D]

        detector_embeddings: Array,     # [B, 1 + C, D]
        detector_mask: Array,           # [B, 1 + C]
        *,
        training: bool = False,
    ) -> ParticleEncoderOutputs:
        B, T, D = particle_jet_features.shape

        particle_embeddings, particle_mask = self.create_particle_embeddings(
            particle_jet_features, 
            particle_jet_types, 
            particle_jet_mask, 
            particle_event_features,
            training = training
        )

        # Combine the two inputs into a single sequence.
        if self.conditional:
            detector_embeddings, detector_mask = self.create_detector_embeddings(
                detector_embeddings, 
                detector_mask,
                training = training
            )

            combined_embeddings = jnp.concatenate(axis=1, arrays=(particle_embeddings, detector_embeddings))
            combined_mask = jnp.concatenate(axis=1, arrays=(particle_mask, detector_mask))
        else:
            combined_embeddings = particle_embeddings
            combined_mask = particle_mask

        # Transformer Decoders to contextualize embeddings.
        for _ in range(self.num_transformer_layers):
            combined_embeddings = TransformerBlock(*self.transformer_config)(
                combined_embeddings, 
                combined_mask,
                training = training
            )

        # Extract the particle jet embeddings from the combined embeddings.
        particle_embeddings = combined_embeddings[:, :T + 1, :]

        particle_mean = self.extract_latent_mean(particle_embeddings, particle_mask, training=training)
        particle_log_std = self.extract_latent_log_std(particle_embeddings, particle_mask, training=training)
        particle_distribution = distributions.Normal(particle_mean, jnp.exp(particle_log_std))

        return ParticleEncoderOutputs(
            particle_distribution,
            particle_mean,
            particle_log_std,
            particle_mask
        )
        