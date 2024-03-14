from dataclasses import dataclass


@dataclass
class DatasetConfig:
    detector_vector_dim: int = 12
    detector_event_dim: int = 3

    particle_vector_dim: int = 5
    particle_event_dim: int = 3

    num_particle_types: int = 3