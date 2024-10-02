from typing import Optional
from enum import Enum
from dataclasses import dataclass


class ConsistencyType(Enum):
    Identity = "identity"
    ZJetsAbsoluteLog = "zjets_absolute_log"


@dataclass
class TrainingConfig:
    # Loss scales
    diffusion_loss_scale: float = 1.0
    reconstruction_loss_scale: float = 1.0
    latent_prior_loss_scale: float = 0.0
    diffusion_prior_loss_scale: float = 0.0
    norm_prior_loss_scale: float = 0.0
    multiplicity_loss_scale: float = 0.0
    consistency_loss_scale: float = 0.0

    consistency_loss_type: ConsistencyType = ConsistencyType.Identity
    
    # Training loop options
    seed: int = 0
    batch_size: int = 1024
    gradient_clipping: float = 1.0
    unconditional_probability: float = 0.1

    # Learning rate options
    warmup_steps: int = 1_000
    cosine_steps: int = 100_000
    training_steps: int = 1_000_000

    # Learning rate schedule
    learning_rate: float = 1e-3
    learning_rate_minimum: float = 0.0
    learning_rate_decay: float = 1.0

    # Training loop iterations
    log_interval: int = 50
    checkpoint_interval: int = 1000

    checkpoint_mask: Optional[list[str]] = None
    negative_checkpoint_mask: bool = False