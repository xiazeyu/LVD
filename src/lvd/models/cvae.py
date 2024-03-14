from typing import Any, Tuple, NamedTuple

import jax
import jax.numpy as jnp
from jax import Array

import flax.linen as nn

from lvd.dataset import Batch
from lvd.utils import masked_fill

from lvd.networks import (
    DetectorEncoder,
    ParticleEncoder,
    ParticleDecoder,
    MultiplicityPredictor
)

from lvd.models.normalized import NormalizedModule


class CVAEOutputs(NamedTuple):
    batch: Batch
    encoded_particles: ParticleEncoder.OutputType
    decoded_particle: ParticleDecoder.OutputType
    multiplicity: MultiplicityPredictor.OutputType


class CVAE(NormalizedModule):
    OutputType = CVAEOutputs
    
    def setup(self):
        super().setup()

        self.detector_encoder = DetectorEncoder(self.config.network)
        self.particle_encoder = ParticleEncoder(self.config.network, self.config.dataset)
        self.particle_decoder = ParticleDecoder(self.config.network, self.config.dataset)
        self.multiplicity_predictor = MultiplicityPredictor(self.config.network)
    
    def __call__(self, batch: Batch, *, training: bool = False) -> Any:
        batch = super().__call__(batch, training = training)

        encoded_detector = self.detector_encoder(
            batch.detector_vectors,
            batch.detector_mask,
            batch.detector_event,

            training = training
        )

        encoded_particles = self.particle_encoder(
            batch.particle_vectors,
            batch.particle_types,
            batch.particle_mask,
            batch.particle_event,

            encoded_detector.vectors,
            encoded_detector.mask,

            training = training
        )

        if self.config.network.deterministic_particle_encoder:
            encoded_particle_vector = encoded_particles.mean
        else:
            encoded_particle_vector = encoded_particles.vector_distribution.sample(seed=self.make_rng("latent"))
        
        decoded_particle = self.particle_decoder(
            encoded_particle_vector,
            encoded_particles.masks,

            encoded_detector.vectors,
            encoded_detector.mask,

            training = training
        )

        multiplicity = self.multiplicity_predictor(encoded_detector.summary, training = training)

        return CVAEOutputs(batch, encoded_particles, decoded_particle, multiplicity)

    def generate(
            self,
            detector_vectors: Array,
            detector_mask: Array,
            detector_event: Array,
            max_particle_vectors: int
    ) -> ParticleDecoder.OutputType:
        batch_size = detector_event.shape[0]

        encoded_detector = self.detector_encoder(
            *self.normalize_detector(detector_vectors, detector_mask, detector_event),
            training = False
        )

        multiplicity = self.multiplicity_predictor(
            encoded_detector.summary,
            training = False
        )

        # Sample the multiplicites and round to the nearest integer.
        multiplicity = multiplicity.sample(seed=self.make_rng("multiplicity"))
        multiplicity = multiplicity.round().astype(jnp.int32) + 1

        output_shape = (batch_size, max_particle_vectors, self.config.network.hidden_dim)
        latent_mask = jnp.repeat(jnp.arange(max_particle_vectors)[None, :], batch_size, axis=0) < multiplicity[:, None]
        latent_vectors = masked_fill(jax.random.normal(self.make_rng("latent"), output_shape), latent_mask)
        
        decoded_particles = self.particle_decoder(
            latent_vectors,
            latent_mask,

            encoded_detector.vectors,
            encoded_detector.mask,

            training = False
        )

        return self.denormalize_particle(decoded_particles)
    
    def rngs(self, key: jax.random.PRNGKey):
        dropout_key, latent_key, multiplicity_key = jax.random.split(key, 3)

        return {
            "dropout": dropout_key,
            "latent": latent_key,
            "multiplicity": multiplicity_key
        }