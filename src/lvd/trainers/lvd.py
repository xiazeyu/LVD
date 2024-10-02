from typing import Any, Tuple, NamedTuple, Callable, Dict, Optional
from functools import partial

import jax
from jax import Array
from jax import numpy as jnp
from flax import struct

from flax.training.train_state import TrainState, core, struct
from diffrax import diffeqsolve, Dopri5, ODETerm, SaveAt, PIDController

from lvd.utils import create_learning_rate_fn, create_optimizer
from lvd.config import Config
from lvd.dataset import Dataset, Batch
from lvd.models.lvd import LVD
from lvd.utils import masked_fill

from lvd.trainers.trainer import Trainer
from diffusers.schedulers import FlaxDPMSolverMultistepScheduler


from lvd.losses import (
    multiplicity_loss,
    vector_reconstruction_loss,
    type_reconstruction_loss,
    event_reconstruction_loss,
    latent_prior_loss,
    diffusion_prior_loss,
    diffusion_loss_mean,
    diffusion_loss_variance,
    norm_prior_loss,
    # consistency_loss
)

from lvd.consistency import ConsistencyType, get_consistency_loss

class LVDState(struct.PyTreeNode):
    normalization: core.FrozenDict[str, Any] = struct.field(pytree_node=True)
    learning_rate_fn: Callable = struct.field(pytree_node=False)
    seed: jax.random.PRNGKey = struct.field(pytree_node=True)
    step: int

    lvd_state: TrainState = struct.field(pytree_node=True)
    gamma_state: TrainState = struct.field(pytree_node=True)



class CVAEMetrics(NamedTuple):
    multiplicity: float
    vector_reconstruction: float
    type_reconstruction: float
    event_reconstruction: float
    latent_prior: float


class GenerationOptions(NamedTuple):
    discrete: Callable
    ode: Callable

def create_trainer(config: Config):
    model = LVD(config)
    consistency_loss = get_consistency_loss(config)
    
    def initialize(
        key: jax.random.PRNGKey, 
        dataset: Dataset,
        batch: Batch
    ) -> LVDState:
        key, params_key, init_key = jax.random.split(key, 3)

        variables = model.init(
            {
                "params": params_key, 
                **model.rngs(init_key)
            }, 
            batch
        )

        # Separate out LVD and noise schedule parameters.
        lvd_params = variables["params"]
        gamma_params = {"noise_schedule": lvd_params.pop("noise_schedule")}

        return LVDState(
            learning_rate_fn = create_learning_rate_fn(config.training),
            normalization = dataset.statistics,
            seed = key,
            step = 0,

            lvd_state = TrainState.create(
                apply_fn = model.apply,
                params = lvd_params,
                tx = create_optimizer(config.training),
            ),

            gamma_state= TrainState.create(
                apply_fn = model.apply,
                params = gamma_params,
                tx = create_optimizer(config.training),
            )
        )

    @partial(jax.pmap, axis_name="device")
    def update(state: LVDState, batch: Batch) -> Tuple[LVDState, Dict[str, Array]]:
        seed = jax.random.fold_in(state.seed, state.step)

        lvd_rngs = model.rngs(jax.random.fold_in(seed, 0))
        gamma_rngs = model.rngs(jax.random.fold_in(seed, 1))

        @partial(jax.value_and_grad, has_aux=True)
        def lvd_loss_fn(lvd_params):
            params = {**lvd_params, **state.gamma_state.params}

            outputs: LVD.OutputType = model.apply(
                {"params": params, "normalization": state.normalization}, 
                batch, 
                training = True,
                rngs = lvd_rngs
            )

            metrics = {
                "multiplicity": multiplicity_loss(
                    outputs.batch, 
                    outputs.multiplicity
                ),

                "vector_reconstruction": vector_reconstruction_loss(
                    outputs.batch, 
                    outputs.decoded_particle
                ),

                "type_reconstruction": type_reconstruction_loss(
                    outputs.batch, 
                    outputs.decoded_particle
                ),

                "event_reconstruction": event_reconstruction_loss(
                    outputs.batch, 
                    outputs.decoded_particle
                ),

                "consistency": consistency_loss(
                    outputs.denormalized_true_particles,
                    outputs.denormalized_decoded_particles
                ),
                
                "latent_prior": latent_prior_loss(
                    outputs.batch, 
                    outputs.encoded_particles, 
                    outputs.z_0_over_alpha,
                    outputs.gamma_0.gamma
                ),

                # "norm_prior": norm_prior_loss(
                #     outputs.encoded_particles.mean,
                #     outputs.encoded_particles.masks,
                # ),

                "diffusion_prior": diffusion_prior_loss(
                    outputs.batch,
                    outputs.encoded_particles.mean, 
                    outputs.encoded_particles.masks, 
                    outputs.gamma_1
                ),

                "diffusion_loss": diffusion_loss_mean(
                    outputs.batch,
                    outputs.eps_t, 
                    outputs.eps_hat, 
                    outputs.eps_weighting, 
                    outputs.encoded_particles.masks,
                    config.network.ordered_denoising_network
                ),

                "gamma_min": outputs.gamma_0.gamma,
                "gamma_max": outputs.gamma_1.gamma
            }

            total_loss = (
                + config.training.multiplicity_loss_scale * metrics["multiplicity"]

                + config.training.reconstruction_loss_scale * metrics["vector_reconstruction"]
                + config.training.reconstruction_loss_scale * metrics["event_reconstruction"]
                + config.training.reconstruction_loss_scale * metrics["type_reconstruction"]

                + config.training.latent_prior_loss_scale * metrics["latent_prior"]
                + config.training.diffusion_prior_loss_scale * metrics["diffusion_prior"]

                + config.training.diffusion_loss_scale * metrics["diffusion_loss"]
                + config.training.consistency_loss_scale * metrics["consistency"]
            )

            return total_loss, metrics
        
        @partial(jax.value_and_grad, has_aux=True)
        def gamma_loss_fn(gamma_params):
            params = {**state.lvd_state.params, **gamma_params}

            outputs: LVD.OutputType = model.apply(
                {"params": params, "normalization": state.normalization}, 
                batch, 
                training = True,
                rngs = gamma_rngs
            )

            metrics = {
                "diffusion_variance": diffusion_loss_variance(
                    outputs.batch,
                    outputs.eps_t, 
                    outputs.eps_hat, 
                    outputs.eps_weighting, 
                    outputs.encoded_particles.masks,
                    config.network.ordered_denoising_network
                )
            }

            total_loss = config.training.diffusion_loss_scale * metrics["diffusion_variance"]

            return total_loss, metrics
        
        # Compute the loss and its gradient.
        (lvd_loss, lvd_metrics), lvd_grads = lvd_loss_fn(state.lvd_state.params)        
        (gamma_loss, gamma_metrics), gamma_grads = gamma_loss_fn(state.gamma_state.params)
        
        # Aggregate across devices.
        lvd_grads = jax.lax.pmean(lvd_grads, "device")
        gamma_grads = jax.lax.pmean(gamma_grads, "device")

        # Update the training state.
        new_lvd_state = state.lvd_state.apply_gradients(grads=lvd_grads)
        new_gamma_state = state.gamma_state.apply_gradients(grads=gamma_grads)

        state = state.replace(
            lvd_state=new_lvd_state, 
            gamma_state=new_gamma_state, 
            step=state.step + 1
        )

        # Add extra metrics to the return.
        metrics = {**lvd_metrics, **gamma_metrics}
        metrics['learning_rate'] = state.learning_rate_fn(state.step)
        metrics['loss'] = lvd_loss

        return state, metrics
    
    def reconstruct(
        state: LVDState,
        batch: Batch,
        seed: Optional[jax.random.PRNGKey] = None
    ):
        if seed is None:
            seed = state.seed

        seed, reconstruct_seed = jax.random.split(seed, 2)

        params = {**state.lvd_state.params, **state.gamma_state.params}
        apply_fn = partial(
            model.apply, 
            {"params": params, "normalization": state.normalization},
        )

        output = apply_fn(
            batch, 
            training = False,
            rngs = model.rngs(reconstruct_seed)
        )

        return apply_fn(
            output.decoded_particle,
            method=model.denormalize_particle
        ), seed
    
    @partial(jax.jit, static_argnums=(4, 5))
    def generate(
        state: LVDState,
        detector_vectors: Array,
        detector_mask: Array,
        detector_event: Array,
        max_particles: int,
        num_steps: int,
        guidance_scale: float = 0.0,
        betas: Optional[Array] = None,
        multiplicity: Optional[Array] = None,
        seed: Optional[jax.random.PRNGKey] = None
    ):
        if seed is None:
            seed = state.seed

        seed, encode_seed, schedule_seed, denoise_seed, decode_seed = jax.random.split(seed, 5)

        batch_size = detector_vectors.shape[0]

        params = {**state.lvd_state.params, **state.gamma_state.params}
        apply_fn = partial(
            model.apply, 
            {"params": params, "normalization": state.normalization},
        )

        encoded_detector, z_1, z_mask = apply_fn(
            detector_vectors,
            detector_mask,
            detector_event,
            max_particle_vectors=max_particles,
            training = False,
            rngs = model.rngs(encode_seed),
            method=model.sample_latent
        )

        if multiplicity is not None:
            z_mask = jnp.repeat(jnp.arange(max_particles)[None, :], batch_size, axis=0) < multiplicity[:, None]
            
        if betas is None:
            schedule = apply_fn(
                num_training_steps=num_steps,
                max_particle_vectors=max_particles,
                rngs = model.rngs(schedule_seed),
                method=model.create_schedule
            )
        else:
            schedule = FlaxDPMSolverMultistepScheduler(
                num_train_timesteps=num_steps,
                trained_betas=betas
            )

        # Unroll the reverse diffusion process.
        def loop_body(i, args):
            z_t, schedule_state = args
            
            t = schedule_state.timesteps[i]

            alpha_squared_t = schedule_state.common.alphas_cumprod[t]
            alpha_squared_t = jnp.broadcast_to(alpha_squared_t, (batch_size, max_particles, 1))

            eps_pred = eps_pred_cond = apply_fn(
                encoded_detector,
                z_t,
                z_mask,
                alpha_squared_t,
                rngs = model.rngs(denoise_seed),
                method=model.denoise_latent
            )

            # eps_pred_uncond = apply_fn(
            #     encoded_detector,
            #     z_t,
            #     z_mask,
            #     alpha_squared_t,
            #     rngs = model.rngs(denoise_seed),
            #     method=model.denoise_latent_unconditional
            # )

            # eps_pred = (1.0 + guidance_scale) * eps_pred_cond - guidance_scale * eps_pred_uncond

            return schedule.step(schedule_state, eps_pred, t, z_t, return_dict=False)

        schedule_state = schedule.set_timesteps(
            state=schedule.create_state(), 
            num_inference_steps=num_steps, 
            shape=z_1.shape
        )
        
        z_0, schedule_state = jax.lax.fori_loop(0, num_steps, loop_body, (z_1, schedule_state))

        alpha_0 = apply_fn(
            method=model.alpha_0
        )

        z_0 = z_0 / alpha_0
        if config.network.normalized_particle_encoder:
            norms = jnp.sqrt(jnp.mean(jnp.square(z_0), axis=-1, keepdims=True))
            norms = jnp.where(z_mask[:, :, None], norms, 1.0)
            z_0 = z_0 / norms
            
        output = apply_fn(
            encoded_detector,
            z_0,
            z_mask,
            rngs = model.rngs(decode_seed),
            method=model.decode_latent
        )

        return output, seed
    
    @partial(jax.jit, static_argnums=(4, 5))
    def generate_ode(
        state: LVDState,
        detector_vectors: Array,
        detector_mask: Array,
        detector_event: Array,
        max_particles: int,
        num_steps: int,
        guidance_scale: float = 0.0,
        seed: Optional[jax.random.PRNGKey] = None
    ):
        if seed is None:
            seed = state.seed

        seed, encode_seed, denoise_seed, decode_seed = jax.random.split(seed, 4)

        params = {**state.lvd_state.params, **state.gamma_state.params}
        apply_fn = partial(
            model.apply, 
            {"params": params, "normalization": state.normalization},
        )

        encoded_detector, z_1, z_mask = apply_fn(
            detector_vectors,
            detector_mask,
            detector_event,
            max_particle_vectors=max_particles,
            training = False,
            rngs = model.rngs(encode_seed),
            method=model.sample_latent
        )

        @jax.jit
        def dzdt(t, z, params = None):
            t = -jnp.broadcast_to(t, (z.shape[0], z.shape[1]))

            alpha_squared, sigma, g_squared = apply_fn(
                t,
                method=LVD.ode_params
            )
            
            f = -0.5 * g_squared * z

            eps_pred_cond = apply_fn(
                encoded_detector,
                z,
                z_mask,
                alpha_squared,
                training = False,
                rngs = model.rngs(denoise_seed),
                method=model.denoise_latent
            )

            eps_pred_uncond = apply_fn(
                encoded_detector,
                z,
                z_mask,
                alpha_squared,
                training = False,
                rngs = model.rngs(denoise_seed),
                method=model.denoise_latent_unconditional
            )

            eps_pred = (1.0 + guidance_scale) * eps_pred_cond - guidance_scale * eps_pred_uncond

            score = -eps_pred / sigma

            return -(f - 0.5 * g_squared * score)

        solution = diffeqsolve(
            ODETerm(dzdt),
            Dopri5(),
            -1.0,
            0.0,
            1 / num_steps,
            z_1,
            stepsize_controller=PIDController(rtol=1e-4, atol=1e-4)
        )

        z_0 = solution.ys[0]

        alpha_0 = apply_fn(
            method=model.alpha_0
        )
        
        output = apply_fn(
            encoded_detector,
            z_0 / alpha_0,
            z_mask,
            rngs = model.rngs(decode_seed),
            method=model.decode_latent
        )

        return output, seed
        
    return Trainer(model, initialize, update, GenerationOptions(generate, generate_ode), reconstruct)

