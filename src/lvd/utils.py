from glob import glob
from os import makedirs, path
import jax
from jax import Array
from jax import numpy as jnp

import optax

from lvd.config.training_config import TrainingConfig

def masked_fill(array: Array, mask: Array):
    return jnp.where(mask[:, :, None], array, 0.0)


def create_optimizer(config: TrainingConfig):
    optimizer = optax.adam(learning_rate=create_learning_rate_fn(config))

    if config.gradient_clipping is not None:
        optimizer = optax.chain(
            optax.clip_by_global_norm(config.gradient_clipping),
            optimizer
        )

    return optimizer


def create_learning_rate_fn(config: TrainingConfig):
    num_cycles = config.training_steps // config.cosine_steps

    schedule = []

    for i in range(num_cycles):
        schedule.append({
            "init_value": config.learning_rate_minimum,
            "peak_value": config.learning_rate * (config.learning_rate_decay ** i),
            "end_value": config.learning_rate_minimum,
            "warmup_steps": config.warmup_steps,
            "decay_steps": config.cosine_steps,
        })

    return optax.sgdr_schedule(schedule)


def create_log_folder(logdir: str, name: str):
    base_dir = f"{logdir}/{name}"
    makedirs(base_dir, exist_ok=True)

    next_version = len(glob(f"{base_dir}/version*"))
    log_folder = f"{base_dir}/version_{next_version}"
    makedirs(log_folder, exist_ok=True)

    return path.abspath(log_folder)