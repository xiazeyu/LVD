from typing import Any, Tuple

import jax
import jax.numpy as jnp
from jax import Array

from functools import partial

from flax.training.train_state import TrainState, core, struct
import flax.linen as nn

from lvd.config import Config
from lvd.dataset import Batch
from lvd.models.cvae import CVAE, DetectorEncoder, ParticleEncoder, ParticleDecoder, MultiplicityPredictor
from lvd.noise_schedules import NoiseSchedule

from tensorflow_probability.substrates.jax import distributions

def multiplicity_loss(batch: Batch, multiplicity: MultiplicityPredictor.OutputType):
    multiplicity_target = 1.0 * batch.particle_mask.sum(1)
    multiplicity_loss = -multiplicity.log_prob(multiplicity_target)

    return jnp.mean(multiplicity_loss)

def vector_reconstruction_loss(batch: Batch, decoded_particle: ParticleDecoder.OutputType):
    loss = jnp.square(decoded_particle.vectors - batch.particle_vectors).mean(-1)

    return jnp.mean(loss * batch.particle_weight, where=batch.particle_mask)

def type_reconstruction_loss(batch: Batch, decoded_particle: ParticleDecoder.OutputType):
    type_distribution = distributions.Categorical(logits=decoded_particle.type_logits)
    type_reconstruction_loss = -type_distribution.log_prob(batch.particle_types)
    
    return jnp.mean(type_reconstruction_loss * batch.particle_weight, where=batch.particle_mask)

def event_reconstruction_loss(batch: Batch, decoded_particle: ParticleDecoder.OutputType):
    event_reconstruction_loss = jnp.square(decoded_particle.event - batch.particle_event).mean(-1)

    return jnp.mean(event_reconstruction_loss)

def consistency_loss(
        batch: Batch,
        explicit_squared_mass: Array,
        derived_square_mass: Array
    ):
    consistency_loss = jnp.abs(explicit_squared_mass - derived_square_mass)
    return jnp.mean(consistency_loss * batch.particle_weight, where=batch.particle_mask)

def unit_latent_prior_loss(
    batch: Batch,
    encoded_particles: ParticleEncoder.OutputType,
    prior_mean: Array,
    prior_log_var: Array
):
    mean_squared_1 = prior_mean * prior_mean
    log_var_1 = prior_log_var
    var_1 = jnp.exp(log_var_1)

    prior_loss = 0.5 * (
        + mean_squared_1
        + var_1 
        - log_var_1 
        - 1.0
    ).mean(2)

    padded_weight = jnp.pad(batch.particle_weight, ((0, 0), (1, 0)), mode="constant", constant_values=1.0)

    return jnp.mean(prior_loss * padded_weight, where=encoded_particles.masks)


def coupled_latent_prior_loss(
    batch: Batch, 
    encoded_particles: ParticleEncoder.OutputType, 
    prior_mean: Array,
    prior_log_var: Array
):
    log_var_x = 2.0 * encoded_particles.log_std
    log_var_y = prior_log_var
    var_y = jnp.exp(log_var_y)
    
    prior_loss = 0.5 * (
        + jnp.exp(log_var_x - log_var_y)
        + jnp.square(encoded_particles.mean - prior_mean) / var_y
        + log_var_y
        - log_var_x
        - 1.0
    ).mean(2)

    padded_weight = jnp.pad(batch.particle_weight, ((0, 0), (1, 0)), mode="constant", constant_values=1.0)

    return jnp.mean(prior_loss * padded_weight, where=encoded_particles.masks)

def latent_prior_loss(
    batch: Batch, 
    encoded_particles: ParticleEncoder.OutputType, 
    prior_mean: Array,
    prior_log_var: Array
):
    return unit_latent_prior_loss(batch, encoded_particles, prior_mean, prior_log_var)
    
def diffusion_prior_loss(
    batch: Batch, 
    mean: Array,
    masks: Array,
    gamma_1: NoiseSchedule.Statistics
):
    mean_squared_1 = jnp.exp(gamma_1.log_alpha_squared) * jnp.square(mean)

    log_var_1 = gamma_1.log_sigma_squared
    var_1 = jnp.exp(log_var_1)

    prior_loss = 0.5 * (
        + mean_squared_1 
        + var_1 
        - log_var_1 
        - 1.0
    ).mean(2)

    padded_weight = jnp.pad(batch.particle_weight, ((0, 0), (1, 0)), mode="constant", constant_values=1.0)

    return jnp.mean(prior_loss * padded_weight, where=masks)

def norm_prior_loss(
    encoded_particle_vector: Array,
    encoded_particle_mask: Array
):
    norm_prior_distribution = distributions.Chi(df=encoded_particle_vector.shape[-1])
    particle_norms = jnp.linalg.norm(encoded_particle_vector, axis=-1)

    norm_prior_loss = -norm_prior_distribution.log_prob(particle_norms)
    
    return jnp.mean(norm_prior_loss, where=encoded_particle_mask)


def diffusion_loss(
    eps_t: Array,
    eps_hat: Array,
    eps_weighting: Array,
):
    loss = eps_weighting * jnp.square(eps_hat - eps_t)

    return loss.mean(-1)

def chamfer_loss(
    eps_t: Array,
    eps_hat: Array,
    eps_weighting: Array,
    masks: Array
):
    pairwise_distance = (eps_hat[:, None, :, :] - eps_t[:, :, None, :])
    pairwise_distance = jnp.square(pairwise_distance).mean(-1)
    pairwise_mask = masks[:, :, None] & masks[:, None, :]

    l1 = jnp.min(pairwise_distance, where=pairwise_mask, initial=10, axis=1)
    l2 = jnp.min(pairwise_distance, where=pairwise_mask, initial=10, axis=2)

    return 0.5 * eps_weighting[..., 0] * (l1 + l2)

def diffusion_loss_mean(
    batch: Batch,
    eps_t: Array,
    eps_hat: Array,
    eps_weighting: Array,
    masks: Array,
    ordered_denoising_network: bool
):
    if ordered_denoising_network:
        loss = diffusion_loss(eps_t, eps_hat, eps_weighting)
    else:
        loss = chamfer_loss(eps_t, eps_hat, eps_weighting, masks)
    padded_weight = jnp.pad(batch.particle_weight, ((0, 0), (1, 0)), mode="constant", constant_values=1.0)

    return jnp.mean(loss * padded_weight, where=masks)

def diffusion_loss_variance(
    batch: Batch,
    eps_t: Array,
    eps_hat: Array,
    eps_weighting: Array,
    masks: Array,
    ordered_denoising_network: bool
):
    if ordered_denoising_network:
        loss = diffusion_loss(eps_t, eps_hat, eps_weighting)
    else:
        loss = chamfer_loss(eps_t, eps_hat, eps_weighting, masks)
    
    padded_weight = jnp.pad(batch.particle_weight, ((0, 0), (1, 0)), mode="constant", constant_values=1.0)
    return jnp.var(loss * padded_weight, where=masks)