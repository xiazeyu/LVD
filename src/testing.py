
# %%
import os
os.environ["JAX_DISABLE_JIT"] = "1"

# %%
import jax
from jax import numpy as jnp

from flax import linen as nn

from lvd.config import Config, NetworkConfig, TrainingConfig
from lvd.dataset import Dataset

from lvd.layers.embedding import Embedding
from lvd.layers.transformer_block import TransformerBlock
from lvd.layers.context_vector import ContextVector

from lvd.networks.detector_encoder import DetectorEncoder
from lvd.networks.particle_encoder import ParticleEncoder

from lvd.trainers.cvae import create_trainer

from dataclasses import asdict, replace
from rich import print as rprint
import rich

import optax

from functools import partial

# %%
config = Config.load("./test.yaml")
config.display()

# %%
dataset = Dataset("../data/ZJetsNormed.Herwig.small.npz")
config = replace(config, dataset=dataset.config)
batch = next(iter(dataset.single_device_dataloader(batch_size=config.training.batch_size)))

# %%
network, TrainingState, update, generate = create_trainer(config)

# %%
key = jax.random.PRNGKey(0)

key, params_key, latent_key, dropout_key, multiplicity_key = jax.random.split(key, 5)

variables = network.init({"params": params_key, "latent": latent_key, "multiplicity": multiplicity_key}, batch)
predict = jax.jit(network.apply, static_argnames={"training"})

# %%
def create_cosine_learning_rate_fn(config: TrainingConfig):
    num_cycles = config.training_steps // config.cosine_steps

    single_cycle_options = {
        "init_value": 0.0,
        "peak_value": config.learning_rate,
        "end_value": 0.0,
        "warmup_steps": config.warmup_steps,
        "decay_steps": config.cosine_steps - config.warmup_steps,
    }

    return optax.sgdr_schedule([
        single_cycle_options.copy()
        for _ in range(num_cycles)
    ])

# %%
state = TrainingState.create(
    apply_fn = network.apply,
    learning_rate_fn = create_cosine_learning_rate_fn(config.training),

    params = variables["params"],
    normalization = dataset.statistics,

    dropout_key = dropout_key,
    latent_key = latent_key,
    multiplicity_key = multiplicity_key,

    tx = optax.adam(learning_rate=1e-3)
)

# %%
# network.apply(
#     {"params": state.params, "normalization": state.normalization},
#     batch,
#     training=True,
#     rngs={
#         "dropout": state.dropout_key,
#         "latent": state.latent_key,
#         "multiplicity": state.multiplicity_key
#     }
# )

# %%
state, metrics = update(state, batch)

# %%
metrics

# %%
A = jax.tree_util.tree_flatten_with_path(jax.tree_map(jnp.all, jax.tree_map(jnp.isfinite, state.params)))[0]
list(map(lambda x: "/".join( map(str, x[0])), filter(lambda x: not x[1], A)))


