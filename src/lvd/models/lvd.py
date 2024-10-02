from typing import Any, Tuple, NamedTuple

import numpy as np

import jax
import jax.numpy as jnp
from jax import Array

import flax.linen as nn
from diffusers.schedulers import FlaxDPMSolverMultistepScheduler

from lvd.dataset import Batch
from lvd.utils import masked_fill
from lvd.networks.particle_decoder import ParticleDecoderOutput

from lvd.networks import (
    DetectorEncoder,
    ParticleEncoder,
    ParticleDecoder,
    MultiplicityPredictor,
    DenoisingNetwork
)

from lvd.layers.uncoditional import Uncoditional

from lvd.noise_schedules import NoiseSchedule, Weighting, GammaLimits

from lvd.models.normalized import NormalizedModule


class LVDOutputs(NamedTuple):
    batch: Batch

    encoded_particles: ParticleEncoder.OutputType
    decoded_particle: ParticleDecoder.OutputType
    multiplicity: MultiplicityPredictor.OutputType

    z_0_over_alpha: Array
    z_t: Array

    eps_0: Array
    eps_t: Array
    eps_hat: Array
    eps_weighting: Array

    gamma_0: NoiseSchedule.Statistics
    gamma_1: NoiseSchedule.Statistics
    gamma_t: NoiseSchedule.Statistics

    denormalized_true_particles: ParticleDecoder.OutputType
    denormalized_decoded_particles: ParticleDecoder.OutputType

    explicit_squared_mass: Array
    derived_squared_mass: Array


class LVD(NormalizedModule):
    OutputType = LVDOutputs

    RNGS = [
        "dropout",
        "latent",
        "multiplicity",
        "timestep",
        "noise",
        "uncond"
    ]
    
    def setup(self):
        super().setup()

        self.detector_encoder = DetectorEncoder(self.config.network)
        self.particle_encoder = ParticleEncoder(self.config.network, self.config.dataset)
        self.particle_decoder = ParticleDecoder(self.config.network, self.config.dataset)
        self.multiplicity_predictor = MultiplicityPredictor(self.config.network)
        self.denoising_network = DenoisingNetwork(self.config.network)

        self.gamma_limits = GammaLimits(self.config.noise_schedule)
        self.noise_schedule = NoiseSchedule(self.config.noise_schedule)
        self.weighting = Weighting(self.config.noise_schedule)

        self.unconditional = Uncoditional(
            self.config.network.hidden_dim, 
            self.config.training.unconditional_probability
        )

    
    def sample_particle(self, encoded_particles: ParticleEncoder.OutputType) -> Array:
        if self.config.network.deterministic_particle_encoder:
            return encoded_particles.mean
        else:
            return encoded_particles.vector_distribution.sample(seed=self.make_rng("latent"))
    
    def sample_noise(self, template: Array) -> Array:
        return jax.random.normal(
            self.make_rng("noise"), 
            template.shape, 
            template.dtype
        )

    def sample_timesteps(self, template: Array):
        B, T = template.shape[:2]

        t0 = jax.random.uniform(self.make_rng("timestep"))    
        timesteps = jnp.linspace(0.0, 1.0, B + 1)
        timesteps = jnp.mod(t0 + timesteps, 1.0)[:B]
        timesteps = jnp.broadcast_to(timesteps[:, None], (B, T))

        return timesteps


    def derived_squared_mass(self, particle_vectors):
        px, py, pz, log_energy, _ = particle_vectors.transpose(2, 0, 1)
        energy = jnp.expm1(log_energy)
        square_mass = energy ** 2 - px ** 2 - py ** 2 - pz ** 2

        return self.normalize_squared_mass(square_mass)
    
    def explicit_squared_mass(self, particle_vectors):
        _, _, _, _, log_mass = particle_vectors.transpose(2, 0, 1)
        mass = jnp.expm1(log_mass)
        square_mass = mass ** 2

        return self.normalize_squared_mass(square_mass)
    
    def __call__(self, batch: Batch, *, training: bool = False) -> Any:
        denormalized_true_particles = ParticleDecoderOutput(
            vectors=batch.particle_vectors,
            type_logits=jax.nn.one_hot(batch.particle_types, self.config.dataset.num_particle_types),
            mask=batch.particle_mask,
            event=batch.particle_event
        )

        batch = self.normalize_batch(batch)

        gamma_limits = self.gamma_limits()

        # Encode detector variables once and reuse for all of the networks.
        encoded_detector = self.encode_detector(
            batch, 
            training = training
        )

        # Mask out some of the detector varaibles to train the network unconditionally.
        # encoded_detector = self.unconditional(encoded_detector)

        # Encode the truth particles into the latent space.
        encoded_particles = self.encode_particles(
            batch,
            encoded_detector,
            training = training
        )

        # Sample a latent encoding of the particles.
        encoded_particle_vector = self.sample_particle(encoded_particles)
        
         # Sample the initial diffusion latent. This should be very close to the VAE latent.
        eps_0 = self.sample_noise(encoded_particle_vector)

        alpha_0 = self.noise_schedule.alpha(0.0, *gamma_limits)
        sigma_0 = self.noise_schedule.sigma(0.0, *gamma_limits)
        z_0 = alpha_0 * encoded_particle_vector + sigma_0 * eps_0
        z_0 = masked_fill(z_0, encoded_particles.masks)

        # Compute a more numerically stable version of z_0 / alpha which is used in prior loss.
        z_0_over_alpha = encoded_particle_vector + self.noise_schedule.SNR(0.0, *gamma_limits) * eps_0
        z_0_over_alpha = masked_fill(z_0_over_alpha, encoded_particles.masks)
        
        # Decode the non-noisy latent back into particle space.
        decoded_particle = self.particle_decoder(
            z_0_over_alpha if self.config.network.coupled_diffusion_particle_decoder else encoded_particle_vector,
            encoded_particles.masks,

            encoded_detector.vectors,
            encoded_detector.mask,

            training = training
        )

        denormalized_decoded_particles = self.denormalize_particle(decoded_particle)

        # Compute the two version of particle mass if it is relevant
        # if self.config.training.consistency_loss_scale > 0:
        #     explicit_squared_mass = self.explicit_squared_mass(denormalized_decoded_particles.vectors)
        #     derived_squared_mass = self.derived_squared_mass(denormalized_decoded_particles.vectors)
        # else:
        explicit_squared_mass = jnp.zeros(decoded_particle.mask.shape)
        derived_squared_mass = jnp.zeros(decoded_particle.mask.shape)

        # Sample a timestep to compute the diffusion loss for.
        timesteps = self.sample_timesteps(encoded_particle_vector)
        eps_t = self.sample_noise(encoded_particle_vector)

        # Noise schedule parameters for this at each point in time.
        alpha_t = self.noise_schedule.alpha(timesteps, *gamma_limits)
        sigma_t = self.noise_schedule.sigma(timesteps, *gamma_limits)

        # Compute the noisy version of the input
        z_t = alpha_t * encoded_particle_vector + sigma_t * eps_t
        z_t = masked_fill(z_t, encoded_particles.masks)

        # Ask the network for the estimate of the noise.
        eps_hat = self.denoising_network(
            z_t, 
            encoded_particles.masks,

            encoded_detector.vectors,
            encoded_detector.mask,

            self.noise_schedule.alpha_squared(timesteps, *gamma_limits),
            training = training
        )

        eps_weighting = (
            0.5 * 
            self.noise_schedule.prime(timesteps, *gamma_limits) * 
            self.weighting(self.noise_schedule(timesteps, *gamma_limits))
        )

        # Estimate the particle multiplicity.
        multiplicity = self.multiplicity_predictor(
            encoded_detector.summary, 
            training = training
        )

        return LVDOutputs(
            batch, 
            encoded_particles, 
            decoded_particle, 
            multiplicity,

            z_0_over_alpha, 
            z_t,

            eps_0, 
            eps_t, 
            eps_hat,
            eps_weighting,

            self.noise_schedule.statistics(0.0, *gamma_limits),
            self.noise_schedule.statistics(1.0, *gamma_limits),
            self.noise_schedule.statistics(timesteps, *gamma_limits),

            denormalized_true_particles,
            denormalized_decoded_particles,

            explicit_squared_mass,
            derived_squared_mass
        )

    def sample_latent(
        self,
        detector_vectors: Array,
        detector_mask: Array,
        detector_event: Array,
        max_particle_vectors: int,
        *,
        training = False
    ):
        batch_size = detector_event.shape[0]

        encoded_detector = self.detector_encoder(
            *self.normalize_detector(detector_vectors, detector_mask, detector_event),
            training = training
        )

        multiplicity = self.multiplicity_predictor(
            encoded_detector.summary,
            training = training
        )

        multiplicity = multiplicity.sample(seed=self.make_rng("multiplicity"))
        multiplicity = multiplicity.round().astype(jnp.int32) + 1

        output_shape = (batch_size, max_particle_vectors, self.config.network.hidden_dim)
        z_mask = jnp.repeat(jnp.arange(max_particle_vectors)[None, :], batch_size, axis=0) < multiplicity[:, None]
        z_1 = masked_fill(jax.random.normal(self.make_rng("latent"), output_shape), z_mask)

        return encoded_detector, z_1, z_mask
    
    def create_schedule(self, num_training_steps: int, max_particle_vectors: int):
        timesteps = jnp.linspace(0, 1, num_training_steps)
        timesteps = jnp.broadcast_to(timesteps[:, None], (num_training_steps, max_particle_vectors))

        alphas_cumprod = self.noise_schedule.alpha_squared(timesteps, *self.gamma_limits())
        # alphas_cumprod = np.asarray(alphas_cumprod).astype(np.float64)
        alphas = alphas_cumprod / jnp.pad(alphas_cumprod[:-1], ((1, 0), (0, 0), (0, 0)), constant_values=1.0)
        betas = (1.0 - alphas)[:, None, :]

        return FlaxDPMSolverMultistepScheduler(
            num_train_timesteps=num_training_steps,
            # beta_schedule="scaled_linear",
            # beta_start=0.00085,
            # beta_end=0.012,
            trained_betas=betas
        )
    
    def create_schedule_alpha(self, num_training_steps: int, max_particle_vectors: int):
        timesteps = jnp.linspace(0, 1, num_training_steps)
        timesteps = jnp.broadcast_to(timesteps[:, None], (num_training_steps, max_particle_vectors))

        alphas_cumprod = self.noise_schedule.alpha_squared(timesteps, *self.gamma_limits())
        return alphas_cumprod
    
    def create_schedule_numpy(self, num_training_steps, alphas_cumprod):
        alphas_cumprod = np.asarray(alphas_cumprod).astype(np.float64)
        alphas = alphas_cumprod / np.pad(alphas_cumprod[:-1], ((1, 0), (0, 0), (0, 0)), constant_values=1.0)
        betas = (1.0 - alphas)[:, None, :]
        betas = jnp.array(betas.astype(np.float32))

        return FlaxDPMSolverMultistepScheduler(
            num_train_timesteps=num_training_steps,
            # beta_schedule="scaled_linear",
            # beta_start=0.00085,
            # beta_end=0.012,
            trained_betas=betas
        )
    
    def denoise_latent(self, encoded_detector, z_t, z_mask, alpha_squared_t, *, training=False):
        return self.denoising_network(
            z_t,
            z_mask,
            encoded_detector.vectors,
            encoded_detector.mask,
            alpha_squared_t,
            training = training
        )
    
    def denoise_latent_unconditional(self, encoded_detector, z_t, z_mask, alpha_squared_t, *, training=False):
        encoded_detector = self.unconditional(encoded_detector, all_masked=True)

        return self.denoising_network(
            z_t,
            z_mask,
            encoded_detector.vectors,
            encoded_detector.mask,
            alpha_squared_t,
            training = training
        )

    def decode_latent(self, encoded_detector, z_0, z_mask):
        decoded_particles = self.particle_decoder(
            z_0,
            z_mask,

            encoded_detector.vectors,
            encoded_detector.mask,

            training = False
        )

        return self.denormalize_particle(decoded_particles)
    
    def ode_params(self, t):
        alpha_squared = self.noise_schedule.alpha_squared(t, *self.gamma_limits())
        sigma = self.noise_schedule.sigma(t, *self.gamma_limits())
        g_squared = self.noise_schedule.g_squared(t, *self.gamma_limits())

        return alpha_squared, sigma, g_squared
    
    def alpha_0(self):
        return self.noise_schedule.alpha(0.0, *self.gamma_limits())
    
    def make_dzdt(self, z_1, z_mask, encoded_detector):
        def dzdt(z, t, params = None):
            t = jnp.broadcast_to(t, (z.shape[0], z.shape[1]))

            alpha_squared = self.noise_schedule.alpha_squared(t, *self.gamma_limits())
            sigma = self.noise_schedule.sigma(t, *self.gamma_limits())
            g_squared = self.noise_schedule.g_squared(t, *self.gamma_limits())
            f = -0.5 * g_squared * z

            eps = self.denoising_network(
                z,
                z_mask,
                encoded_detector.vectors,
                encoded_detector.mask,
                alpha_squared,
                training = False
            )

            score = -eps / sigma

            return f - 0.5 * g_squared * score

        return dzdt
    
    def generate(
            self,
            detector_vectors: Array,
            detector_mask: Array,
            detector_event: Array,
            max_particle_vectors: int
    ) -> ParticleDecoder.OutputType:
        batch_size = detector_event.shape[0]
        num_inference_steps = 1000
        num_training_steps = 1000

        encoded_detector = self.detector_encoder(
            *self.normalize_detector(detector_vectors, detector_mask, detector_event),
            training = False
        )

        multiplicity = self.multiplicity_predictor(
            encoded_detector.summary,
            training = False
        )

        multiplicity = multiplicity.sample(seed=self.make_rng("multiplicity"))
        multiplicity = multiplicity.round().astype(jnp.int32) + 1

        output_shape = (batch_size, max_particle_vectors, self.config.network.hidden_dim)
        z_mask = jnp.repeat(jnp.arange(max_particle_vectors)[None, :], batch_size, axis=0) < multiplicity[:, None]
        z_1 = masked_fill(jax.random.normal(self.make_rng("latent"), output_shape), z_mask)

        timesteps = jnp.linspace(0, 1, num_training_steps)
        timesteps = jnp.broadcast_to(timesteps[:, None], (num_training_steps, max_particle_vectors))

        alphas_cumprod = self.noise_schedule.alpha_squared(timesteps, *self.gamma_limits())
        alphas_cumprod = np.asarray(alphas_cumprod).astype(np.float64)
        alphas = alphas_cumprod / np.pad(alphas_cumprod[:-1], ((1, 0), (0, 0), (0, 0)), constant_values=1.0)
        betas = jnp.asarray((1.0 - alphas)[:, None, :])

        scheduler = FlaxDPMSolverMultistepScheduler(
            num_train_timesteps=num_training_steps,
            # beta_schedule="scaled_linear",
            # beta_start=0.00085,
            # beta_end=0.012,
            # trained_betas=betas
        )

        scheduler_state = scheduler.set_timesteps(
            state=scheduler.create_state(), 
            num_inference_steps=num_inference_steps, 
            shape=output_shape
        )

        # Unroll the reverse diffusion process.
        def loop_body(i, args):
            z_t, state = args
            
            t = state.timesteps[i]

            alpha_squared_t = state.common.alphas_cumprod[t]
            alpha_squared_t = jnp.broadcast_to(alpha_squared_t, (batch_size, max_particle_vectors, 1))

            eps_pred = self.denoising_network(
                z_t,
                z_mask,
                encoded_detector.vectors,
                encoded_detector.mask,
                alpha_squared_t,
                training = False
            )

            return scheduler.step(state, eps_pred, t, z_t, return_dict=False)

        z_0, scheduler_state = nn.scan(0, num_inference_steps, loop_body, (z_1, scheduler_state))

        decoded_particles = self.particle_decoder(
            z_0,
            z_mask,

            encoded_detector.vectors,
            encoded_detector.mask,

            training = False
        )

        return self.denormalize_particle(decoded_particles)
    
    def rngs(self, key: jax.random.PRNGKey):
        
        return dict(zip(
            self.RNGS,
            jax.random.split(key, len(self.RNGS))
        ))
    
    def encode_detector(
        self, 
        batch: Batch, 
        *, 
        training: bool = True
    ) -> DetectorEncoder.OutputType:
        return self.detector_encoder(
            batch.detector_vectors,
            batch.detector_mask,
            batch.detector_event,

            training = training
        )
    
    def encode_particles(
        self, 
        batch: Batch, 
        encoded_detector: DetectorEncoder.OutputType,
        *,
        training: bool = True
    ) -> ParticleEncoder.OutputType:
        return self.particle_encoder(
            batch.particle_vectors,
            batch.particle_types,
            batch.particle_mask,
            batch.particle_event,

            encoded_detector.vectors,
            encoded_detector.mask,

            training = training
        )