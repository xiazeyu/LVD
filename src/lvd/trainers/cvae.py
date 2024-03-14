from typing import Any, Tuple, NamedTuple, Callable, Dict, Optional
from functools import partial

import jax
from jax import Array

from flax.training.train_state import TrainState, core, struct

from lvd.utils import create_learning_rate_fn, create_optimizer
from lvd.config import Config
from lvd.dataset import Dataset, Batch
from lvd.models.cvae import CVAE

from lvd.trainers.trainer import Trainer


from lvd.losses import (
    multiplicity_loss,
    vector_reconstruction_loss,
    type_reconstruction_loss,
    event_reconstruction_loss,
    latent_prior_loss
)

class CVAEState(TrainState):
    normalization: core.FrozenDict[str, Any] = struct.field(pytree_node=True)
    seed: jax.random.PRNGKey = struct.field(pytree_node=True)
    
    learning_rate_fn: Callable = struct.field(pytree_node=False)


class CVAEMetrics(NamedTuple):
    multiplicity: float
    vector_reconstruction: float
    type_reconstruction: float
    event_reconstruction: float
    latent_prior: float

def create_trainer(config: Config):
    model = CVAE(config)
    
    def initialize(
        key: jax.random.PRNGKey, 
        dataset: Dataset,
        batch: Batch
    ) -> CVAEState:
        key, params_key, init_key = jax.random.split(key, 3)

        variables = model.init(
            {
                "params": params_key, 
                **model.rngs(init_key)
            }, 
            batch
        )

        return CVAEState.create(
            apply_fn = model.apply,
            learning_rate_fn = create_learning_rate_fn(config.training),
            tx = create_optimizer(config.training),

            params = variables["params"],
            normalization = dataset.statistics,

            seed = key,
        )

    @partial(jax.pmap, axis_name="device")
    def update(state: CVAEState, batch: Batch) -> Tuple[CVAEState, Dict[str, Array]]:
        seed = jax.random.fold_in(state.seed, state.step)
        rngs = model.rngs(seed)

        def loss_fn(params):
            outputs: CVAE.OutputType = model.apply(
                {"params": params, "normalization": state.normalization}, 
                batch, 
                training = True,
                rngs = rngs
            )

            metrics = {
                "multiplicity": multiplicity_loss(outputs.batch, outputs.multiplicity),
                "vector_reconstruction": vector_reconstruction_loss(outputs.batch, outputs.decoded_particle),
                "type_reconstruction": type_reconstruction_loss(outputs.batch, outputs.decoded_particle),
                "event_reconstruction": event_reconstruction_loss(outputs.batch, outputs.decoded_particle),
                "latent_prior": latent_prior_loss(outputs.batch, outputs.encoded_particles, 0.0, 0.0)
            }

            total_loss = (
                + config.training.multiplicity_loss_scale * metrics["multiplicity"]

                + config.training.reconstruction_loss_scale * metrics["vector_reconstruction"]
                + config.training.reconstruction_loss_scale * metrics["event_reconstruction"]
                + config.training.reconstruction_loss_scale * metrics["type_reconstruction"]

                + config.training.latent_prior_loss_scale * metrics["latent_prior"]
            )

            return total_loss, metrics
        
        # Compute the loss and its gradient.
        grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
        (loss, metrics), grads = grad_fn(state.params)

        # Aggregate across devices.
        grads = jax.lax.pmean(grads, "device")
        metrics = jax.lax.pmean(metrics, "device")

        # Update the training state.
        state = state.apply_gradients(grads=grads)

        # Add extra metrics to the return.
        metrics['learning_rate'] = state.learning_rate_fn(state.step)
        metrics['loss'] = loss

        return state, metrics
    
    @partial(jax.jit, static_argnums=(4,))
    def generate(
        state: CVAEState,
        detector_vectors: Array,
        detector_mask: Array,
        detector_event: Array,
        max_particles: int,
        seed: Optional[jax.random.PRNGKey] = None
    ):
        if seed is None:
            seed = state.seed

        return model.apply(
            {"params": state.params, "normalization": state.normalization}, 
            detector_vectors, 
            detector_mask,
            detector_event,
            max_particles,
            rngs = model.rngs(seed),
            method=CVAE.generate
        )
    
    return Trainer(model, initialize, update, generate)

