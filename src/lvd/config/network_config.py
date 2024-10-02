from dataclasses import dataclass
from enum import Enum


class SkipConnectionType(Enum):
    Identity = "identity"
    GRU = "gru"
    Simple = "simple"
    Output = "output"
    Highway = "highway"


@dataclass
class NetworkConfig:
    # Base network dimensionality
    hidden_dim: int = 64
    dropout: float = 0.1

    # Transformer options
    transformer_heads: int = 4
    transformer_expansion: int = 2
    skip_connection_type: SkipConnectionType = SkipConnectionType.GRU

    # Denoising layer dimensionality expansion
    denoising_expansion: int = 1

    # Network depth options
    num_denoising_layers: int = 8
    num_multiplicity_layers: int = 2
    num_detector_encoder_layers: int = 8    
    num_particle_encoder_layers: int = 8
    num_particle_decoder_layers: int = 8

    num_linear_layers: int = 2

    # Ordering
    ordered_detector_encoder: bool = True
    ordered_particle_encoder: bool = True
    ordered_denoising_network: bool = True

    # Conditioning
    conditional_particle_encoder: bool = True
    conditional_particle_decoder: bool = True

    # Normalized
    normalized_particle_encoder: bool = True

    # Deterministic
    deterministic_particle_encoder: bool = False

    coupled_diffusion_particle_decoder: bool = False

    discrete_multiplicity_predictor: bool = False