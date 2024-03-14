from dataclasses import dataclass
from enum import Enum
from typing import Optional

class NoiseScheduleType(Enum):
    ConditionalNetwork = "conditional_network"
    Network = "network"


class WeightingType(Enum):
    EDM = "edm"
    Unit = "unit"
    Cosine = "cosine"
    Sigmoid = "sigmoid"


@dataclass
class NoiseScheduleConfig:
    noise_schedule: NoiseScheduleType = NoiseScheduleType.ConditionalNetwork
    weighting: WeightingType = WeightingType.Sigmoid

    sigmoid_weighting_offset: float = 2.0

    initial_gamma_min: float = -13.3
    initial_gamma_max: float = 5.0

    limit_gamma_max: Optional[float] = None
    limit_gamma_min: Optional[float] = None

    hidden_dim: int = 1024
    output_dim: int = 1
    conditioning_dim: int = 128


