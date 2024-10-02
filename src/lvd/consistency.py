import numpy as np

import jax
import jax.numpy as jnp
from jax import Array

from lvd.networks.particle_decoder import ParticleDecoderOutput

from lvd.config import Config
from lvd.config.training_config import ConsistencyType
from lvd.dataset import Batch


ZJET_MASSES = np.array([
    0.00000000,
    0.13957000,
    0.13957000,
    0.49761000,
    0.00051100,
    0.00051100,
    0.10566000,
    0.10566000,
    0.49368000,
    0.49368000,
    0.93827000,
    0.93827000,
    0.93957000,
    0.93957000,
])


def get_consistency_loss(config: Config):
    if config.training.consistency_loss_type == ConsistencyType.Identity:
        return identity
    elif config.training.consistency_loss_type == ConsistencyType.ZJetsAbsoluteLog:
        return zjets_absolute_log
    else:
        raise ValueError(
            f"Unknown consistency loss type: {config.training.consistency_loss_type}")


def identity(true: ParticleDecoderOutput, pred: ParticleDecoderOutput):
    return 0.0


def zjets_absolute_log_invariant_mass_squared(output: ParticleDecoderOutput):
    types = output.type_logits.argmax(-1)
    mass = jnp.array(ZJET_MASSES)[types]

    pt = jnp.exp(output.vectors[..., 0])
    y = output.vectors[..., 1]
    sin_phi = jnp.tanh(output.vectors[..., 2])
    cos_phi = jnp.tanh(output.vectors[..., 3])

    mt = jnp.sqrt(pt ** 2 + mass ** 2)

    # max_val = jnp.maximum(pt, mass)
    # min_val = jnp.minimum(pt, mass)
    # mt = max_val * jnp.sqrt(1 + (min_val / max_val) ** 2)

    # Momentum Components
    # px, py, pz, E
    momentum = jnp.where(output.mask[..., None], jnp.stack([
        pt * cos_phi,
        pt * sin_phi,
        mt * jnp.sinh(y),
        mt * jnp.cosh(y)
    ], axis=-1), 0.0)

    jet = jnp.sum(momentum, axis=1)

    p = jnp.linalg.norm(jet[..., :3], axis=-1)
    E = jet[..., 3]

    mass_squared = (E - p) * (E + p)

    return mass_squared


def zjets_absolute_log(true: ParticleDecoderOutput, pred: ParticleDecoderOutput):
    true_mass_squared = zjets_absolute_log_invariant_mass_squared(true)
    pred_mass_squared = zjets_absolute_log_invariant_mass_squared(pred)

    return jnp.mean(jnp.abs(true_mass_squared - pred_mass_squared))

def consistency_loss(
    batch: Batch,
    explicit_squared_mass: Array,
    derived_square_mass: Array
):
    consistency_loss = jnp.abs(explicit_squared_mass - derived_square_mass)
    return jnp.mean(consistency_loss * batch.particle_weight, where=batch.particle_mask)
