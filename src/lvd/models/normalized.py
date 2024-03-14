from typing import Any, Tuple

from jax import Array

import flax.linen as nn

from lvd.config import Config
from lvd.dataset import Batch
from lvd.networks.particle_decoder import ParticleDecoderOutput

class NormalizedModule(nn.Module):
    config: Config

        
    def setup(self):
        self.detector_vector_mean = self.variable(
            "normalization", 
            "detector_vector_mean",
            nn.initializers.constant(0.0),
            None,
            (self.config.dataset.detector_vector_dim,),
        )

        self.detector_vector_std = self.variable(
            "normalization", 
            "detector_vector_std",
            nn.initializers.constant(1.0),
            None,
            (self.config.dataset.detector_vector_dim,)
        )

        self.detector_event_mean = self.variable(
            "normalization", 
            "detector_event_mean",
            nn.initializers.constant(0.0),
            None,
            (self.config.dataset.detector_event_dim,)
        )

        self.detector_event_std = self.variable(
            "normalization", 
            "detector_event_std",
            nn.initializers.constant(1.0),
            None,
            (self.config.dataset.detector_event_dim,)
        )

        self.particle_vector_mean = self.variable(
            "normalization", 
            "particle_vector_mean",
            nn.initializers.constant(0.0),
            None,
            (self.config.dataset.particle_vector_dim,)
        )

        self.particle_vector_std = self.variable(
            "normalization", 
            "particle_vector_std",
            nn.initializers.constant(1.0),
            None,
            (self.config.dataset.particle_vector_dim,)
        )

        self.particle_event_mean = self.variable(
            "normalization", 
            "particle_event_mean",
            nn.initializers.constant(0.0),
            None,
            (self.config.dataset.particle_event_dim,)
        )

        self.particle_event_std = self.variable(
            "normalization", 
            "particle_event_std",
            nn.initializers.constant(1.0),
            None,
            (self.config.dataset.particle_event_dim,)
        )

        self.squared_mass_mean = self.variable(
            "normalization", 
            "squared_mass_mean",
            nn.initializers.constant(0.0),
            None,
            tuple()
        )

        self.squared_mass_std = self.variable(
            "normalization", 
            "squared_mass_std",
            nn.initializers.constant(1.0),
            None,
            tuple()
        )

    def __call__(self, batch: Batch, *, training: bool = False) -> Any:        
        return self.normalize_batch(batch)
    
    @staticmethod
    def normalize(data: Array, mean: nn.Variable, std: nn.Variable) -> Array:
        return (data - mean.value) / std.value
    
    @staticmethod
    def denormalize(data: Array, mean: nn.Variable, std: nn.Variable) -> Array:
        return (data * std.value) + mean.value
    
    def normalize_batch(self, batch: Batch) -> Batch:
        return Batch(
            detector_vectors=self.normalize(batch.detector_vectors, self.detector_vector_mean, self.detector_vector_std),
            detector_event=self.normalize(batch.detector_event, self.detector_event_mean, self.detector_event_std),
            detector_mask=batch.detector_mask,

            particle_vectors=self.normalize(batch.particle_vectors, self.particle_vector_mean, self.particle_vector_std),
            particle_event=self.normalize(batch.particle_event, self.particle_event_mean, self.particle_event_std),
            particle_mask=batch.particle_mask,
            particle_types=batch.particle_types,

            particle_weight=batch.particle_weight
        )         
    
    def normalize_detector(
            self, 
            detector_vectors: Array, 
            detector_mask: Array, 
            detector_event: Array
        ) -> Tuple[Array, Array, Array]:
        return (
            self.normalize(detector_vectors, self.detector_vector_mean, self.detector_vector_std),
            detector_mask,
            self.normalize(detector_event, self.detector_event_mean, self.detector_event_std),
        )
    
    def denormalize_particle(
        self,
        decoded_particle: ParticleDecoderOutput
    ) -> ParticleDecoderOutput:
        return ParticleDecoderOutput(
            vectors=self.denormalize(decoded_particle.vectors, self.particle_vector_mean, self.particle_vector_std),
            type_logits=decoded_particle.type_logits,
            mask=decoded_particle.mask,
            event=self.denormalize(decoded_particle.event, self.particle_event_mean, self.particle_event_std)
        )
    
    def normalize_squared_mass(self, squared_mass):
        return self.normalize(squared_mass, self.squared_mass_mean, self.squared_mass_std)