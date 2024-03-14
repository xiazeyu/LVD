from dataclasses import dataclass
from typing import Any, Optional

import jax
from jax import Array
from jax import numpy as jnp

from flax import linen as nn

from lvd.config import NetworkConfig
from lvd.utils import masked_fill

from lvd.layers import (
    TransformerBlock,
    SplitContextVector,
    Embedding,
    TimestepEmbedding
)

class DenoisingNetwork(nn.Module):
    # =============================================================================================
    # Options
    # region --------------------------------------------------------------------------------------
    config: NetworkConfig

    @property
    def embedding_config(self):
        return (
            self.config.denoising_expansion * self.config.hidden_dim, 
            self.config.num_linear_layers, 
            self.config.transformer_expansion,
            self.config.dropout,
            self.config.skip_connection_type,
        )
    
    @property
    def context_vector_config(self):
        return (
            self.config.denoising_expansion * self.config.hidden_dim, 
            self.config.ordered_denoising_network
        )
    
    @property
    def transformer_config(self):
        return (
            2 * self.config.denoising_expansion * self.config.hidden_dim,
            self.config.transformer_heads,
            self.config.transformer_expansion,
            self.config.dropout,
            self.config.skip_connection_type
        )
    
    @property
    def hidden_dim(self):
        return self.config.hidden_dim
    
    @property
    def num_transformer_layers(self):
        return self.config.num_denoising_layers
    
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
    def TimestepEmbedding(self):
        return TimestepEmbedding(self.hidden_dim)

    @property
    def OutputEmbedding(self):
        return nn.Dense(self.hidden_dim)
    
    # endregion ===================================================================================
    
    @nn.compact
    def __call__(
        self, 

        particle_vectors: Array,  # [B, T, D]
        particle_masks: Array, # [B, T]

        detector_vectors: Array, # [B, C, D]
        detector_masks: Array, # [B, C]

        alpha_squared: Array, # [], [B], [B, T], or [B, T, 1]
        *,
        training: bool = True,
    ) -> Array:  # [B, D]  
        B, T, D = particle_vectors.shape

        # Handle the different types of inputs for the timestep
        if alpha_squared.ndim == 0:
            alpha_squared = alpha_squared[None, None, None]
        elif alpha_squared.ndim == 1:
            alpha_squared = alpha_squared[:   , None, None]
        elif alpha_squared.ndim == 2:
            alpha_squared = alpha_squared[:   , :   , None]
        
        alpha_squared = jnp.broadcast_to(alpha_squared, (B, T, 1))
            
        # Compute the time embedding and add to the particles
        timestep_jets = self.TimestepEmbedding(alpha_squared)
        particle_vectors = jnp.concatenate(axis=2, arrays=(particle_vectors, timestep_jets))

        # Embedd the particle inputs. [B, T, 2 * E * D]
        particle_vectors = self.ParticleInputEmbedding(particle_vectors, training=training)
        particle_vectors = masked_fill(particle_vectors, particle_masks)

        particle_vectors = self.ParticleContextVector(particle_vectors, training=training)
        particle_vectors = masked_fill(particle_vectors, particle_masks)

        # Embedd the detector inputs. [B, T, 2 * E * D]
        detector_vectors = self.DetectorInputEmbedding(detector_vectors, training=training)
        detector_vectors = masked_fill(detector_vectors, detector_masks)

        detector_vectors = self.DetectorContextVector(detector_vectors, training=training)
        detector_vectors = masked_fill(detector_vectors, detector_masks)

        # Combine the inputs and conditioning into a single sequence.
        combined_jets = jnp.concatenate(axis=1, arrays=(particle_vectors, detector_vectors))
        combined_mask = jnp.concatenate(axis=1, arrays=(particle_masks, detector_masks))

        for _ in range(self.num_transformer_layers):
            combined_jets = TransformerBlock(*self.transformer_config)(
                combined_jets, 
                combined_mask,
                training=training
            )

        # Split the combined sequence back into the original inputs.
        noise_predictions = combined_jets[:, :T, :]
        noise_predictions = self.OutputEmbedding(noise_predictions)
        noise_predictions = masked_fill(noise_predictions, particle_masks)

        return noise_predictions
            
        